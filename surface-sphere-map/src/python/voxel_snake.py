from __future__ import division, print_function

import argparse
import contextlib
import logging
import multiprocessing
import itertools
import sys

import numpy as np
from numpy import gradient
from numpy.linalg import norm
from scipy.ndimage.filters import (
    convolve,
    gaussian_filter,
    laplace,
)
from scipy.ndimage.morphology import (
    distance_transform_edt,
    generate_binary_structure,
    binary_dilation,
    binary_erosion,
)
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

import spatial
import sphere
import lattice
import image3d
import gvf
import voxelize


from mayavi import mlab

logging.basicConfig(level=logging.DEBUG)


def triangle_args(surface):
    return list(surface.vertices.transpose()) + [surface.faces]


@contextlib.contextmanager
def numpy_error_handling(**settings):
    old_settings = np.seterr(**settings)
    yield
    np.seterr(**old_settings)


def field_magnitudes(A, zero_to_one=False):
    magnitudes = np.sqrt(np.sum(A**2, axis=-1))
    if zero_to_one:
        magnitudes[magnitudes == 0] = 1
    return magnitudes


def to_field(components):
    C = np.array(components)
    D = len(C.shape)
    F = np.rollaxis(C, 0, D)
    return F


def to_components(field):
    F = np.array(field)
    D = len(F.shape)
    C = np.rollaxis(F, D-1, 0)
    return C


def normalize_field(field):
    F = field
    M = field_magnitudes(field, zero_to_one=True)
    sel = [slice(None, None, None)] * len(M.shape) + [np.newaxis]
    N = F / M[sel]
    return N


class NullLog(object):
    def __getattr__(self, name):
        return lambda x, *args, **kwargs: None


class ContourForce(object):
    def __init__(self, scale=1.0, pool=None, log=NullLog):
        self._scale = scale
        self._pool = pool
        self._log = log

    def __call__(self, mesh, voxels):
        return self.scaled_forces(mesh, voxels)

    def scaled_forces(self, mesh, voxels):
        return self._scale * self.calculate_forces(mesh, voxels)


class FieldForce(ContourForce):
    def __init__(self, field, scale=1.0, **kwargs):
        super(FieldForce, self).__init__(scale=scale, **kwargs)
        self._field = field

    def calculate_forces(self, mesh, voxels):
        vX, vY, vZ = voxels.transpose()
        force = self._field[vX, vY, vZ, :]
        return force


class SurfaceForce(ContourForce):
    def __init__(self, voxel_map, surface, image, scale=1.0, **kwargs):
        super(SurfaceForce, self).__init__(scale=scale, **kwargs)
        self._voxel_map = voxel_map
        self._surface = surface
        self._image = image
        self._nearby_tris = {}
        triangles = self._surface.triangles

        # TODO: This is waseful and sloppy
        for idx, faces in self._voxel_map.iteritems():
            tris = triangles[faces]
            normals = spatial.triangle_normals(tris)
            self._nearby_tris[idx] = tris, normals

    def calculate_forces(self, mesh, voxels):
        forces = np.zeros(mesh.shape)
        for idx, (vertex, voxel) in enumerate(itertools.izip(mesh, voxels)):
            voxel = tuple(voxel)
            if self._image[voxel]:
                nearest, min_dist, inside = None, np.inf, False
                nearby = self._nearby_tris[voxel]
                for triangle, normal in zip(*nearby):
                    dist, point = spatial.distance_to_triangle(vertex, triangle, with_point=True)
                    if np.dot(normal, point - vertex) > 0:
                        inside = True
                    if dist < min_dist:
                        nearest, min_dist = point, dist
                forces[idx] = nearest - vertex

                # Remove scaling for internal points
                # Not optimal and prone to instability
                if inside:
                    forces[idx] /= self._scale
        return forces


class Snake(object):

    def __init__(self, mesh,
                       resolution=1,
                       padding=10,
                       contour_faces=None,
                       contour_size=None,
                       normalize_contour_distance=False,
                       additional_normalization=0,
                       push_scale=1,
                       reduce_push_scale_step=0,
                       cvf_scale=1,
                       cvf_blur=None,
                       gvf_scale=1,
                       cvf_normalize_threshold=0,
                       gradient_scale=1,
                       snap_scale=1,
                       snap_at_stop=True,
                       internal_scale=1,
                       gvf_smoothness=0.15,
                       gvf_timestep=.75,
                       gvf_max_steps=None,
                       gvf_normalize_threshold=0,
                       tension=0.1,
                       stiffness=0.1,
                       remove_external_from_internal=False,
                       timestep=.2,
                       epsilon=0.01,
                       converge_on_snapping=True,
                       max_iterations=100,
                       log=logging):
        self.log = log

        # Target Surface
        self.original_mesh = mesh
        # Voxelization
        self.padding = padding
        self.resolution = resolution

        # Contour
        self.contour_faces = contour_faces
        self.contour_size = contour_size
        self.normalize_contour_distance = normalize_contour_distance
        self.additional_normalization = additional_normalization
    
        # Curvature Force
        self.cvf_scale = cvf_scale
        self.blur_curvature = cvf_blur
        self.min_curvature_propagation = 2
        self.normalize_cvf_distance = cvf_normalize_threshold

        # GVF Force
        self.gvf_scale = gvf_scale
        self.smoothness = gvf_smoothness
        self.gvf_timestep = gvf_timestep
        self.numsteps = gvf_max_steps
        self.normalize_gvf_distance = gvf_normalize_threshold
        
        # Scap Force
        self.snap_scale = snap_scale
        self.snap_at_stopping = snap_at_stop

        # Push Force
        self.push_scale = push_scale
        self.reduce_push_scale_step = reduce_push_scale_step

        # Internal Force
        self.tension = tension
        self.stiffness = stiffness
        self.remove_external_component = remove_external_from_internal

        # Force scaling
        self.internal_scale = internal_scale

        self.iterations = 0
        self.timestep = timestep
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.converge_on_snapping = converge_on_snapping

        self._analyze_mesh()
        self._create_contour()
        self._create_grid()
        self._voxelize_mesh()
        self._create_internal_system()
        self._create_boundary_force()
        self._calculate_gvf()
        self._calculate_cvf()

        if self.normalize_contour_distance:
            self._normalize_distance()

        self.forces = []
        self.log.debug("Creating and pre-caching forces")

        if self.gvf_scale is not None and self.gvf_scale > 0:
            self.forces.append(
                FieldForce(
                    field=self.gvf_field, 
                    scale=self.gvf_scale*self.timestep))

        if self.cvf_scale is not None and self.cvf_scale > 0:
            self.forces.append(
                FieldForce(
                    field=self.cvf_field, 
                    scale=self.cvf_scale*self.timestep))
        
        if self.push_scale is not None and self.push_scale > 0:
            self.forces.append(
                FieldForce(
                    field=self.boundary_push_field,
                    scale=self.push_scale*self.timestep))

        if self.snap_scale is not None and self.snap_scale > 0:
            self.forces.append(
                SurfaceForce(
                    voxel_map=self.voxel_triangles,
                    surface=self.mesh,
                    image=self.voxelized,
                    scale=self.snap_scale*self.timestep))


    def _analyze_mesh(self):
        self.log.debug("Analyzing source mesh")
        low = self.original_mesh.vertices.min(axis=0)
        high = self.original_mesh.vertices.max(axis=0)
        center = (low + high) / 2
        distances = self.original_mesh.vertices - center
        radius = np.apply_along_axis(norm, 1, distances).max()
        radius += self.resolution
        self.contour_center = center
        self.contour_radius = radius
        vertices = self.original_mesh.vertices
        faces = self.original_mesh.faces
        self.mesh = spatial.Surface(vertices, faces)

    def _create_contour(self):
        center = self.contour_center
        radius = self.contour_radius
        if self.contour_size is not None:
            radius = self.contour_size * radius
        iterations = self.contour_faces
        if iterations is None:
            num_tris = len(self.mesh.faces)
            iterations = sphere.iterations_needed_for_triangles(num_tris)
        else:
            num_tris = iterations
            iterations = sphere.iterations_needed_for_triangles(iterations)
        self.log.debug("Generating spherical contour with {} faces".format(num_tris))
        tessalation = sphere.Sphere.from_tessellation(radius=radius,
                                                      center=center,
                                                      iterations=iterations)
        self.contour = tessalation
        self.vertices = self.contour.vertices.copy()
        self.faces = self.contour.faces

    def _create_grid(self):
        extents = self.contour.extents
        axes = lattice.PseudoGrid.from_extents(extents,
                                               resolution=self.resolution,
                                               padding=self.padding)
        self.axes = axes

    def _voxelize_mesh(self):
        self.log.debug("Voxelizing mesh")
        voxel_triangles = {}
        voxelized, tri_map = voxelize.voxelize_mesh(self.mesh,
                                                    self.axes,
                                                    fill=False,
                                                    triangle_map=True)
        self.voxelized = np.array(voxelized, dtype=float)
        self.voxel_triangles = tri_map

    def _calculate_gvf(self):
        self.log.debug("Starting GVF calculation")
        dt = self.gvf_timestep
        ds = (self.resolution, self.resolution, self.resolution)
        raw_image = self.voxelized
        image = raw_image
#        if self.gvf_mode == 'ggvf':
#            u, v, w = gvf.reference_ggvf3d(image,
#                                           iter=self.numsteps,
#                                           K=self.smoothness,
#                                           dt=self.gvf_timestep,
#                                          ds=ds)
#       else:
        u, v, w = gvf.reference_gvf3d(image,
                                      iter=self.numsteps,
                                      mu=self.smoothness,
                                      dt=self.gvf_timestep,
                                      ds=ds)

        mask = 1 - raw_image
        u, v, w, = u * mask, v * mask , w * mask
        gvf_field = gvf.make_field_array(u, v, w)
        self._raw_field = gvf_field

        if self.normalize_gvf_distance > 0:
            speed = 10
            thresh = self.normalize_gvf_distance
            Minv = 1 / field_magnitudes(gvf_field, zero_to_one=True)[:,:,:,np.newaxis]
            D = distance_transform_edt(1-raw_image)
            H = .5 + .5 * np.tanh(speed * (D - thresh))
            mask = 1 + H[:,:,:,np.newaxis] * (Minv - 1)
            gvf_field = gvf_field * mask
            
        self.gvf_components = to_components(gvf_field)
        self.gvf_field = gvf_field

    def _scale_curvature_level(self, level, index):
        level[level > 0] = index + 1
        return level

    def _calculate_cvf(self):
        self.log.debug("Generating Curvature Flow")
        raw_image = self.voxelized
        image = raw_image.copy()
        max_steps = 3*max(image.shape)
        min_grow_curvature = self.min_curvature_propagation
        levels = image.copy()

        # Fill in concavities to form cube
        while min_grow_curvature >= 2:
            delta = np.inf
            i = 0
            while delta > 0 and i < max_steps:
                curvature = laplace(image)
                curvature[image > 0] = 0
                curvature[curvature < min_grow_curvature] = 0
                dilation = curvature.copy()
                dilation[dilation > 0] = 1
                delta = np.count_nonzero(dilation)
                image += dilation
                curvature = self._scale_curvature_level(curvature, i)
                levels += curvature
                i += 1
            min_grow_curvature -= 1

        self.cdt_temp = levels
        
        if self.blur_curvature:
            levels = gaussian_filter(levels, self.blur_curvature)

        cvf_field = -to_field(np.gradient(levels, *self.axes.resolution))
        mask = binary_erosion(image)
        cvf_field[mask == 0] = 0
        cvf_field[self.voxelized == 1] = 0

        if self.normalize_cvf_distance > 0:
            speed = 10
            thresh = self.normalize_cvf_distance
            Minv = 1 / field_magnitudes(cvf_field, zero_to_one=True)[:,:,:,np.newaxis]
            D = distance_transform_edt(1-raw_image)
            H = .5 + .5 * np.tanh(speed * (D - thresh))
            mask = 1 + H[:,:,:,np.newaxis] * (Minv - 1)
            cvf_field = cvf_field * mask

        self.curvature_transform = levels
        self.cvf_field = cvf_field
        self.cvf_components = to_components(cvf_field)

    def _create_boundary_force(self):
        self.log.debug("Generating outward boundary force")
        raw_image = self.voxelized 
        shape = list(raw_image.shape) + [3]
        force = np.zeros(shape)
        triangles = self.mesh.triangles
        
        for idx, faces in self.voxel_triangles.iteritems():
            voxel = tuple(idx)
            tris = triangles[faces]
            normal = spatial.triangle_normals(tris).mean(axis=0)
            force[voxel] = normal

#        cu, cv, cw = map(laplace, gradient(raw_image, *self.axes.resolution))
#        mask = raw_image
#        push_force = np.array([cu, cv, cw]).transpose()

#        push_force[raw_image == 0] = np.zeros(3)
        self.boundary_push_field = force

    def _create_internal_system(self):
        self.log.debug("Generating internal energy system")
        dt, ds = self.timestep, self.resolution
        a = self.internal_scale * self.tension * dt / (ds ** 2)
        b = self.internal_scale * self.stiffness * dt / (ds ** 2)
        N = len(self.contour.vertices)
        A = lil_matrix((N,N))
        for i, M1 in self.contour.neighbors.iteritems():
            M2 = set(self.contour.neighboring_vertices(i, 2))
            N1, N2 = len(M1), len(M2)
            p, q, r = b, -a - 2*4*b, 1 + N1*a + (2*4*N1-N2)*b
            A[i,i] = r
            for k1 in M1:
                A[i,k1] = q
            for k2 in M2:
                A[i,k2] = p

        self.log.debug("Factoring internal energy system")
        self._internal_system = A.tocsc()
        solver = factorized(self._internal_system)
        self.apply_internal_force = solver

    def _normalize_distance(self):
        self.log.debug("Normalizing contour distribution over distance")
        vertex_voxels = self.axes.map_to_axes(self.vertices)

        if self.normalize_contour_distance == 'curvature':
            distances = self.curvature_transform
        else:
            distances = distance_transform_edt(1-self.voxelized)

        DG = to_field(gradient(distances))
        direction = normalize_field(DG)

        # Two-pass for memory reasons
        avg_dist = 0

        for x,y,z in vertex_voxels:
            avg_dist += distances[x,y,z]
        avg_dist /= len(vertex_voxels)

        for vertex, (x, y, z) in itertools.izip(self.vertices, vertex_voxels):
            step = avg_dist - distances[x,y,z]
            vertex += step * direction[x,y,z,:]

        # Bring back inside bounds
        i = 0
        dummy_external = np.zeros((len(self.vertices), 3))
        while i < 100 and not self.axes.all_points_in_grid(self.vertices):
            self.vertices = self.relax_internal_force(self.vertices, dummy_external)
            i += 1
        if i >= 100:
            print("Overload!")

    def _reset(self):
        self.vertices = self.contour.vertices.copy()
        self.iterations = 0

    @property
    def starting_points(self):
        return self.contour.vertices

    @property
    def travel(self):
        return np.array(map(norm, self.starting_points - self.vertices))

    @property
    def unit_starting_points(self):
        return self.contour.unit_vertices

    def image_force(self, active=None):
        total_force = np.zeros(self.vertices.shape)
        vertices = self.vertices
        vertex_voxels = self.axes.map_to_axes(vertices)
        deltas = np.zeros(len(self.forces))
        active_forces = self.forces
        if indicies is not None:
            active_forces = [f for i,f in active_forces if i in active]
        for i, force in enumerate(active_forces):
            force_element = force(vertices, vertex_voxels)
            total_force += force_element
            deltas[i] = np.apply_along_axis(norm, 1, force_element).sum()
        return total_force, deltas

    def relax_internal_force(self, vertices, external, remove_external=True):
        new_vertices = np.apply_along_axis(self.apply_internal_force, 0, vertices)
        if self.remove_external_component and remove_external:
            raise NotImplementedError
            magnitudes = np.apply_along_axis(norm, 1, external)
            magnitudes[magnitudes==0] = 1
            directions = external / magnitudes[:,np.newaxis]
        return new_vertices

    def update(self):
        forces, deltas = self.image_force()
        updated = self.vertices + forces
        updated = self.relax_internal_force(updated, forces)
        delta = abs(self.vertices - updated).sum()
        self.vertices = updated
        self.iterations += 1
        self.log.info("Step {0}: {1:.0f} total, {2:.3f} average ({3})".format(
                        self.iterations,
                        delta,
                        delta / len(self.vertices),
                        " ".join("{:.1f}".format(d) for d in deltas)))
        return delta, deltas

    def run(self):
        # Initialization phase:
        # Place snake in better starting position
        delta = np.inf
        # Get snake closer to surface while maintaining total internal energy
        N = len(self.vertices)
        while delta > self.epsilon and self.iterations < self.max_iterations:
            # Step towards image
            change, force_changes = self.update()
            delta = change / N

        if self.snap_at_stopping:
            snap, movement = self.image_force(active=[3])
            self.vertices += snap

    def mapping_accuracy(self):
        # Find number of vertices in surface voxel
        voxels = self.axes.map_to_axes(self.vertices)
        on_surface = self.voxelized[voxels].sum()
        total = len(voxels)
        self.log.info("{0} of {1} vertices in surface voxel ({2:.2f})".format(
                        on_surface, total, on_surface / total))

        # Find total distance of voxels to surface
        vectors_to_surface = self.image_force(active=3)
        distance_to_surface = np.apply_along_axis(norm, 1, vectors_to_surface)
        total_distance = distance_to_surface.sum()
        self.log.info("{0} total angstroms off surface for 'on' voxels".format(
                        total_distance))

        mesh_offset = 0
        search_limit = 10
        left_out = 0
        self.log.debug("Computing mesh to contour offset...")
        for vertex in self.mesh.vertices:
            dist, _t, pt = self.contour.nearest_surface_point(vertex, 
                                                              search_limit=search_limit)
            if pt is None:
                left_out += 1
                dist += search_limit
            
            mesh_offset += dist
        self.log.info("Total distance from mesh to contour: {0:.2f}".format(mesh_offset))
        self.log.info("{0} points not found in search range of {1}".format(leftout, search_range))
        return (on_surface, total), total_distance, mesh_offset 


def load_sphmap(stream):
    """ Ad-hoc sphere mapping dump format:
        First Line: [# Vertices, Original Sphere Radius, Original Sphere Center (XYZ)]
        Others: [Shape (distance), 
                 Sphere Coords (XYZ),
                 Unit Sphere Coords (XYZ),
                 Surface Coords (XYZ)]
    """
    tokens = next(stream).split()
    nV, radius, center = int(tokens[0]), float(tokens[1]), np.array(tokens[2:], dtype=np.float)
    mappings = np.loadtxt(islice(stream, nV), dtype=np.float)
    return mappings, radius, center


def dump_sphmap(stream, snake):
    """ Ad-hoc sphere mapping dump format:
        First Line: [# Vertices, Original Sphere Radius, Original Sphere Center (XYZ)]
        Others: [Shape (distance), 
                 Sphere Coords (XYZ),
                 Unit Sphere Coords (XYZ),
                 Surface Coords (XYZ)]
    """
    dump_data = zip(snake.travel, 
                    snake.starting_points, 
                    snake.unit_starting_points, 
                    snake.vertices)
    num_vertices = len(snake.vertices)
    radius = snake.contour.radius
    cx, cy, cz = snake.contour.center
    print("{0}\t{1}\t{2}\t{3}\t{4}".format(num_vertices, radius, cx, cy, cz), file=stream)
    for idx, vertex_data in enumerate(dump_data):
        travel, points, unit_travel, unit_points, on_surf = vertex_data
        line = []
        line.append(travel)
        line.extend(points)
        line.extend(unit_points)
        line.extend(on_surf)
        format = ("{:.4f}\t" * len(line)).strip()
        print(format.format(*line), file=stream)


def show_travel(contour):
    from mayavi import mlab
    mlab.triangle_mesh(*triangle_args(contour), 
                        opacity=.75, 
                        scalars=contour.travel,
                        color='PUGr')
    mlab.clf()
    mlab.triangle_mesh(*triangle_args(contour.mesh), opacity=.25, color='Bone')
    mlab.show()


def show_mapping(contour):
    from mayavi import mlab
    mlab.clf()
    mlab.triangle_mesh(*triangle_args(contour.contour), scalars=contour.travel)
    mlab.show()


def main(args, stdin=None, stdout=None):
    import sys
    parser = argparse.ArgumentParser
    parser.add_argument('surface_file', 
                        type=argparse.FileType('r'),
                        default='-',
                        help="Vet (MSRoll) file to map [Default: %(default)s]")
    parser.add_argument('mapping_file',
                        type=argparse.FileType('w'),
                        default='-',
                        help="Destination to write sphere mapping [Default: %(default)s]")

    parser.add_argument('-r', '--voxel-resolution',
                        type=float,
                        default=1.0,
                        help="Resolution at which to voxelize surface (in Angstroms) [Default: %(default)s]")
    parser.add_argument('-p', '--voxel-padding',
                        type=float,
                        default=10.0,
                        help="Padding to add to voxelized image [Default: %(default)s]")
    parser.add_argument('-f', '--contour-faces',
                        type=int,
                        default=None,
                        help="Number of faces to generate in contour [Default: >= source mesh]")
    parser.add_argument('-r', '--contour-scale',
                        type=float,
                        default=1.0,
                        help="Scale factor for contour [Default: %(default)s]")
    parser.add_argument('-n', '--normalize-contour-distance',
                        type=bool,
                        default=False,
                        help="Normalize contour distance to protein surface [Default: False] (Warning: May require signifigant padding!)")
    parser.add_argument('--timestep',
                        type=float,
                        default=.15,
                        help="Global system timestep [Default: %(default)s]")
    parser.add_argument('--cvf-scale',
                        type=float,
                        default=.5,
                        help="Curvature Vector Flow scale factor [Default: %(default)s]")
    parser.add_argument('--gvf-scale',
                        type=float,
                        default=.5,
                        help="Gradient Vector Flow scale factor [Default: %(default)s]")
    parser.add_argument('--push-scale',
                        type=float,
                        default=1,
                        help="Boundary push scale factor [Default: %(default)s]")
    parser.add_argument('--snap-scale',
                        type=float,
                        default=1,
                        help="Surface snapping scale factor [Default: %(default)s]")
    parser.add_argument('--internal-scale',
                        type=float,
                        default=1,
                        help="Global internal energy scale factor [Default: %(default)s]")
    parser.add_argument('--internal-tension',
                        type=float,
                        default=.5,
                        help="Internal tension (elasticity) energy [Default: %(default)s]")
    parser.add_argument('--internal-stiffness',
                        type=float,
                        default=.25,
                        help="Internal stiffness (rigidity) energy [Default: %(default)s]")
    parser.add_argument('--gvf-smoothness',
                        type=float,
                        default=.15,
                        help="Gradient vector flow smoothness parameter [Default: %(default)s]")
    parser.add_argument('--gvf-timestep',
                        type=float,
                        default=.75,
                        help="Gradient vector flow generation timestep [Default: %(default)s]")
    parser.add_argument('--gvf-max-steps',
                        type=int,
                        default=None,
                        help="Gradient vector flow iterations [Default: Guessed]")
    parser.add_argument('--gvf-normalize_distance',
                        type=float,
                        default=10,
                        help="Normalize GVF after Euclidian distance from surface term [Default: %(default)s]")
    parser.add_argument('--gvf-normalize_distance',
                        type=float,
                        default=15,
                        help="Normalize CVF after Euclidian distance from surface term [Default: %(default)s]")
    parser.add_argument('--convergence-threshold',
                        type=float,
                        default=0.015,
                        help="Average movement threshold for termination [Default: %(default)s]")
    parser.add_argument('--converge-on-snapping',
                        type=bool,
                        default=False,
                        help="Only consider snapping in convergence [Default: %(default)s]")
    parser.add_argument('--max-iterations',
                        type=int,
                        default=500,
                        help="Maxiumum number of iterations to run before forced termination [Default: %(default)s]")

    parser.add_argument('--visualize',
                        choices=('convergence', 'mapping', 'none'),
                        default='none',
                        help="Display visualization (Requires MayaVI) [Default: none]")
    parser.add_argument('--visualize-every',
                        type=int,
                        default=None,
                        help="Interrupt iterations to visualize every N steps [Default: %(default)s]")
    params = parser.parse_args(args)

    source = spatial.Surface.from_vet_file(params.surface_file)

    if params.visualize_every is not None:
        max_step = params.visualize_every
        total_iterations = params.max_iterations
        params.max_iterations = max_step
    else:
        total_iterations = params.max_iterations

    contour = Snake(source,
                resolution=params.resolution,
                padding=params.padding,
                contour_faces=params.contour_faces,
                contour_size=params.contour_size,
                normalize_contour_distance=params.normalize_contour_distance,
                push_scale=params.push_scale,
                gvf_scale=params.gvf_scale,
                cvf_scale=params.cvf_scale,
                cvf_normalize_distance=params.cvf_normalize_distance,
                snap_scale=params.snap_scale,
                internal_scale=params.internal_scale,
                gvf_smoothness=params.gvf_smoothness,
                gvf_timestep=params.gvf_timestep,
                gvf_max_steps=params.gvf_max_steps,
                gvf_normalize_distance=params.gvf_normalize_distance,
                tension=params.internal_tension,
                stiffness=params.internal_stiffness,
                timestep=params.timestep,
                epsilon=params.epsilon,
                max_iterations=params.max_iterations)

    while contour.max_iterations > total_iterations:
        contour.run()
        if params.visualize_every is not None:
            contour.max_iteratons += max_step
            if params.visualize == 'convergence':
                show_travel(contour)
            elif params.visualize == 'mapping':
                show_mapping(contour)

    dump_sphmap(params.mapping_file, contour)
    
    if params.visualize == 'convergence':
        show_travel(contour)
    elif params.visualize == 'mapping':
        show_mapping(contour)
                        

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
