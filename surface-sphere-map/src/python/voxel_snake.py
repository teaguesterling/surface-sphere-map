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

        # TODO: This is waseful and sloppy
        for idx, faces in self._voxel_map.iteritems():
            triangles = self._surface.triangles[faces]
            normals = spatial.triangle_normals(triangles)
            self._nearby_tris[idx] = triangles, normals

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
                       cvf_scale=1,
                       cvf_blur=None,
                       gvf_scale=1,
                       gradient_scale=1,
                       snap_scale=1,
                       internal_scale=1,
                       gvf_smoothness=0.15,
                       gvf_timestep=.75,
                       gvf_max_steps=None,
                       gvf_curvature_on_boundary=None,
                       tension=0.1,
                       stiffness=0.1,
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

        # GVF Force
        self.gvf_scale = gvf_scale
        self.smoothness = gvf_smoothness
        self.gvf_timestep = gvf_timestep
        self.numsteps = gvf_max_steps
        self.curvature_on_boundary = gvf_curvature_on_boundary
        self.normalize_force_after_distance = False
        
        # Scap Force
        self.snap_scale = snap_scale

        # Internal Force
        self.tension = tension
        self.stiffness = stiffness

        # Force scaling
        self.internal_scale = internal_scale

        self.iterations = 0
        self.timestep = timestep
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.converge_on_snapping = converge_on_snapping

        self._translate_mesh()
        self._create_contour()
        self._create_grid()
        self._voxelize_mesh()
        self._normalize_distance()
        self._create_internal_system()
        self._calculate_gvf()
        self._calculate_cvf()



        self.forces = []

        self.log.debug("Creating and pre-caching forces")

        if self.gvf_scale is not None:
            self.forces.append(
                FieldForce(
                    field=self.gvf_field, 
                    scale=self.gvf_scale*self.timestep))

        if self.cvf_scale is not None:
            self.forces.append(
                FieldForce(
                    field=self.cvf_field, 
                    scale=self.cvf_scale*self.timestep))
        
        if self.snap_scale is not None:
            self.forces.append(
                SurfaceForce(
                    voxel_map=self.voxel_triangles,
                    surface=self.mesh,
                    image=self.voxelized,
                    scale=self.snap_scale*self.timestep))

    def _translate_mesh(self):
        self.log.debug("Repositioning source mesh")
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

        if self.curvature_on_boundary == 0:
            mask = 1 - raw_image
            u, v, w, = u * mask, v * mask , w * mask
        elif self.curvature_on_boundary is not None:
            cu, cv, cw = map(laplace, gradient(raw_image, *self.axes.resolution))
            mask = 1 - self.voxelized
            u += cu * mask * self.curvature_on_boundary
            v += cv * mask * self.curvature_on_boundary
            w += cw * mask * self.curvature_on_boundary
            
        gvf_field = gvf.make_field_array(u, v, w)
        self._raw_field = gvf_field

        if self.normalize_force_after_distance == 0:
            gvf_field /= field_magnitudes(gvf_field, zero_to_one=True)[:,:,:,np.newaxis]
        elif self.normalize_force_after_distance is not None:
            speed = 10
            thresh = self.normalize_force_after_distance
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
        image = self.voxelized.copy()
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

        field = -to_field(np.gradient(levels, *self.axes.resolution))
        mask = binary_erosion(image)
        field[mask == 0] = 0
        field[self.voxelized == 1] = 0
        self.curvature_transform = levels
        self.cvf_field = field
        self.cvf_components = to_components(field)

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
        if not self.normalize_contour_distance:
            return

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
        while i < 100 and not self.axes.all_points_in_grid(self.vertices):
            self.vertices = self.apply_internal_force(self.vertices)
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

    def image_force(self):
        total_force = np.zeros(self.vertices.shape)
        vertices = self.vertices
        vertex_voxels = self.axes.map_to_axes(vertices)
        deltas = np.zeros(len(self.forces))
        for i, force in enumerate(self.forces):
            force_element = force(vertices, vertex_voxels)
            total_force += force_element
            deltas[i] = np.apply_along_axis(norm, 1, force_element).sum()
        return total_force, deltas

    def relax_internal_force(self, vertices):
        new_vertices = np.apply_along_axis(self.apply_internal_force, 0, vertices)
        return new_vertices

    def update(self):
        forces, deltas = self.image_force()
        updated = self.vertices + forces
        updated = self.relax_internal_force(updated)
        delta = abs(self.vertices - updated).sum()
        self.vertices = updated
        self.iterations += 1
        self.log.info("Step {0} deltas ({1} total): {2}".format(self.iterations,
                                                                delta,
                                                                deltas))
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


def main(args, stdin=None, stdout=None):
    import sys
    parser = argparse.ArgumentParser
    parser.add_argument('surface_file', 
                        type=argparse.FileType('r'),
                        default='-',
                        help="Vet (MSRoll) file to map [Default: %(default)s]")
    parser.add_argument('-r', '--voxel-resolution',
                        type=float,
                        default=1.0,
                        help="Resolution at which to voxelize surface (in Angstroms) [Default: %(default)s]")
    parser.add_argument('-p', '--voxel-padding',
                        type=float,
                        default=5.0,
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
                        help="Normalize contour distance to protein surface [Default: False]")
    parser.add_argument('--timestep',
                        type=float,
                        default=.2,
                        help="Global system timestep [Default: %(default)s]")
    parser.add_argument('--cvf-scale',
                        type=float,
                        default=.5,
                        help="Curvature Vector Flow scale factor [Default: %(default)s]")
    parser.add_argument('--gvf-scale',
                        type=float,
                        default=1,
                        help="Gradient Vector Flow scale factor [Default: %(default)s]")
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
                        default=.2,
                        help="Internal tension (elasticity) energy [Default: %(default)s]")
    parser.add_argument('--internal-stiffness',
                        type=float,
                        default=.2,
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
    parser.add_argument('--gvf-curvature-on-boundary',
                        type=bool,
                        default=True,
                        help="Augment GVF field with balancing boundary term [Default: %(default)s]")
    parser.add_argument('--gvf-curvature-on-boundary',
                        type=bool,
                        default=True,
                        help="Augment GVF field with balancing boundary term [Default: %(default)s]")
    parser.add_argument('--convergence-threshold',
                        type=float,
                        default=0.001,
                        help="Average movement threshold for termination [Default: %(default)s]")
    parser.add_argument('--converge-on-snapping',
                        type=bool,
                        default=True,
                        help="Only consider snapping in convergence [Default: %(default)s]")
    parser.add_argument('--max-iterations',
                        type=int,
                        default=200,
                        help="Maxiumum number of iterations to run before forced termination [Default: %(default)s]")
                    
                        


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
