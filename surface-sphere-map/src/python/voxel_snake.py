from __future__ import division, print_function

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

    def __call__(self, mesh):
        return self.scaled_forces(mesh)

    def update_after_step(self, mesh):
        pass

    def scaled_forces(self, mesh):
        return self._scale * self.calculate_forces(mesh)

    def calculate_forces(self, mesh):
        self.update_after_step(mesh)
        if self._pool is None:
            return self.calculate_forces_serial(mesh)
        else:
            return self.calculate_forces_parallel(mesh)

    def calculate_forces_serial(self, mesh):
        forces = np.zeros(mesh.shape)
        for index, vertex in enumerate(mesh):
            forces[index] = self.force_at_vertex(vertex, index)
        return forces

    def calculate_forces_parallel(self, mesh):
        workers = self._pool._processes
        vertices = len(mesh)
        per_chunk = vertices // workers
        evenly_distributed = per_chunk * workers
        chunks = np.vsplit(mesh[:evenly_distributed], workers)
        for i, leftover in enumerate(mesh[evenly_distributed:]):
            chunks[i] = np.vstack([chunks[i], leftover])
        fn = self._apply_forces
        args = zip(chunks, [self]*len(chunks))
        print(fn, args)
        results = self._pool.map(fn, args)
        forces = np.vstack(results)
        return forces
        
    def force_at_vertex(vertex, index):
        raise NotImplementedError("Force not implemented for abstract base")

    @staticmethod
    def _apply_forces(instance, mesh):
        return instance.calculate_forces_serial(mesh)


class FieldForce(ContourForce):
    def __init__(self, field, axes, scale=1.0, **kwargs):
        super(GVFForce, self).__init__(scale=scale, **kwargs)
        self._field = field

    def calculate_forces(self, mesh, voxels):
        vX, vY, vZ = voxels
        force = self._field[vX, vY, vZ, :]
        return force


class SurfaceForce(ContourForce):
    def __init__(self, voxel_map, surface, scale=1.0, **kwargs):
        super(SurfaceForce, self).__init__(scale=scale, **kwargs)
        self._voxel_map = voxel_map
        self._surface = surface
        self._nearby_tris = {}

        # TODO: This is waseful!
        for idx, faces in self._voxel_map.iteritems():
            triangles = self._surface.triangles[faces]
            normals = spatial.triangle_normals(triangles)
            self._nearby_tris[idx] = triangles, normals


    def calculate_forces(self, mesh, voxels):
        forces = np.zeros(mesh.shape)
        for idx, (vertex, voxel) in enumerate(itertools.izip(mesh, voxels)):
            voxel = tuple(voxel)
            if voxel in self._nearby_tris:
                nearest, min_dist, inside = None, np.inf, False
                triangles = self._nearby_tris[voxel]
                for triangle, normal in triangles:
                    dist, point = spatial.distance_to_triangle(vertex, triangle, with_point=True)
                    if np.dot(normal, nearest - vertex) > 0:
                        inside = True
                    if dist < min_dist:
                        nearest, min_dist = point, dist
                forces[idx] = nearest - vertex
                if inside:
                    forces[idx] /= self.scale
        return forces


class Snake(object):

    def __init__(self, mesh,
                       resolution=1,
                       padding=15,
                       contour_faces=None,
                       contour_size=None,
                       normalize_contour_distance=False,
                       blur_radius=None,
                       cvf_scale=1,
                       gvf_scale=1,
                       gradient_scale=1,
                       snap_scale=1,
                       internal_scale=1,
                       gvf_mode=None,
                       smoothness=0.15,
                       gvf_timestep=.75,
                       numsteps=None,
                       curvature_on_boundary=None,
                       normalize_force_after_distance=None,
                       tension=0.1,
                       stiffness=0.1,
                       timestep=.2,
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
    
        # Curvature Force
        self.cvf_scale = cvf_scale
        self.blur_curvature = None

        # GVF Force
        self.gvf_scale = gvf_scale
        self.gvf_mode = gvf_mode
        self.blur_radius = blur_radius
        self.smoothness = smoothness
        self.gvf_timestep = gvf_timestep
        self.numsteps = numsteps
        self.curvature_on_boundary = curvature_on_boundary
        self.normalize_force_after_distance = normalize_force_after_distance
        
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
        self.epsilon = 0.001

        self._translate_mesh()
        self._prep_snake()
        self._create_contour()
        self._create_internal_system()
        self._calculate_gvf()
        self._calculate_cvf()

        self.vertices = self.contour.vertices
        self.faces = self.contour.faces

        self._normalize_distance()

        self.forces = []

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
                    voxels=self.voxelized,
                    voxel_map=self.voxel_triangles,
                    surface=self.mesh,
                    scale=self.snap_scale*self.timestep))

        if self.snap_scale is None:
            self.forces = self.forces[:1]
        
    def _translate_mesh(self):
        self.log.debug("Repositioning source mesh")
        low = self.original_mesh.vertices.min(axis=0)
        lower_bound = low - self.padding
        self.low_point = low
        vertices = self.original_mesh.vertices - lower_bound
        faces = self.original_mesh.faces
        self.mesh = spatial.Surface(vertices, faces)

    def _prep_snake(self):
        self.log.debug("Voxelizing mesh")
        voxel_triangles = {}
        voxelized, axes, tri_map = voxelize.voxelize_mesh(self.mesh,
                                                          resolution=self.resolution,
                                                          padding=self.padding,
                                                          cube=True,
                                                          fill=False,
                                                          triangle_map=True)
        self.voxelized = np.array(voxelized, dtype=float)
        self.axes = axes
        self.voxel_triangles = tri_map

    def _create_contour(self):
        center = np.array(map(lambda ax: ax[len(ax)//2], self.axes))
        dists = map(np.linalg.norm, self.mesh.vertices - center)
        radius = max(dists)
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

    def _calculate_gvf(self):
        self.log.debug("Starting GVF calculation")
        dt = self.gvf_timestep
        ds = (self.resolution, self.resolution, self.resolution)
        raw_image = self.voxelized
        image = raw_image
        if self.blur_radius is not None:
            image = gaussian_filter(image, self.blur_radius) * image
        if self.gvf_mode == 'ggvf':
            u, v, w = gvf.reference_ggvf3d(image,
                                           iter=self.numsteps,
                                           K=self.smoothness,
                                           dt=self.gvf_timestep,
                                           ds=ds)
        else:
            u, v, w = gvf.reference_gvf3d(image,
                                          iter=self.numsteps,
                                          mu=self.smoothness,
                                          dt=self.gvf_timestep,
                                          ds=ds)

        if self.curvature_on_boundary == 0:
            mask = 1 - raw_image
            u, v, w, = u * mask, v * mask , w * mask
        elif self.curvature_on_boundary is not None:
            cu, cv, cw = map(laplace, gradient(raw_image))
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
        max_steps = max(image.shape)
        min_grow_curvature = 2
        levels = image.copy()
        delta = np.inf
        i = 0

        # Fill in concavities to form cube
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

        # Fill in remainder of the grid 
        while delta > 0 and i < max_steps:
            offsets = convolve(levels, generate_binary_structure(3, 1))
            level = offsets + 1
            level[image > 0] = 0
            level[offsets == 0] = 0
            dilation = level.copy()
            dilation[dilation > 0] = 1
            delta = np.count_nonzero(dilation)
            image += dilation
            levels += level
            i += 1

        if self.blur_curvature:
            levels = gaussian_filter(levels, self.blur_curvature)

        field = -to_field(np.gradient(levels))
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
        vertex_voxels = self.axes.map_to_axes(self.vertices)
        if self.normalize_contour_distance == 'euclidian':
            distances = distance_transform_edt(1-self.voxelized)
        else:
            distances = self.curvature_transform

        DG = to_field(*gradient(distances))
        direction = normalize_field(DG)

        # Two-pass for memory reasons
        avg_dist = 0

        for v in vertex_voxels:
            avg_dist += distances[v]
        avg_dist /= len(vertex_voxels)

        for vertex, (x, y, z) in itertools.izip(self.vertices, vertex_voxels):
            step = avg_dist - distances[x,y,z]
            vertex += step * direction[x,y,z,:]

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

    @property
    def unit_travel(self):
        return self.travel / self.contour.radius

    def force(self):
        total_force = np.zeros(self.vertices.shape)
        verices = self.vertices
        vertex_voxels = self.axes.map_to_axes(vertices)
        for force in self.forces:
            force_element = force(vertices, vertex_voxels)
            total_force += force_element
        return total_force


    def update(self):
        print("Updating contour step {0}".format(self.iterations), file=sys.stderr)
        updated = np.array(self.vertices + self.force())
        updated = np.apply_along_axis(self.apply_internal_force, 0, updated)
        delta = abs(self.vertices - updated).sum()
        self.vertices = updated
        self.iterations += 1
        return delta

    def run(self):
        # Initialization phase:
        # Place snake in better starting position
        delta = np.inf
        # Get snake closer to surface while maintaining total internal energy
        N = len(self.vertices)
        while delta > self.epsilon and self.iterations < self.max_iterations:
            # Step towards image
            change = self.update()
            delta = change / N
            

def load_sphmap(stream):
    """ Ad-hoc sphere mapping dump format:
        First Line: [# Vertices, Original Sphere Radius, Original Sphere Center (XYZ)]
        Others: [Shape (distance), Sphere Coords (XYZ),
                 Unit shape (distance), Unit Sphere Coords (XYZ),
                 Surface Coords (XYZ)]
    """
    tokens = next(stream).split()
    nV, radius, center = int(tokens[0]), float(tokens[1]), np.array(tokens[2:], dtype=np.float)
    mappings = np.loadtxt(islice(stream, nV), dtype=np.float)
    return mappings, radius, center


def dump_sphmap(stream, snake):
    """ Ad-hoc sphere mapping dump format:
        First Line: [# Vertices, Original Sphere Radius, Original Sphere Center (XYZ)]
        Others: [Shape (distance), Sphere Coords (XYZ),
                 Unit shape (distance), Unit Sphere Coords (XYZ),
                 Surface Coords (XYZ)]
    """
    dump_data = zip(snake.travel, snake.starting_points, 
                    snake.unit_travel, snake.unit_starting_points, 
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
        line.append(unit_travel)
        line.extend(unit_points)
        line.extend(on_surf)
        format = ("{:.4f}\t" * len(line)).strip()
        print(format.format(*line), file=stream)


def main(args, stdin=None, stdout=None):
    import sys
    opts = [a for a in args if a.startswith('-')]
    args = [a for a in args if not a.startswith('-')]
    params = dict((k[2:], v) for k,v in (a.split('=') for a in opts if '=' in a))
    if stdin is None:
        stdin = sys.stdin
    if stdout is None:
        stdout = sys.stdout

    print("Loading vertices", file=sys.stderr) 
    if len(args) > 0:
        with open(args[0]) as f:
            surf = spatial.Surface.from_vet_file(f)
    else:
        surf = spatial.Surface.from_vet_file(stdin)
    iterations = params.get('iterations', None)
    if iterations is not None:
        iterations = int(iterations)
    if '--invert' in opts:
        snake = Snake.create_for_surface_invert(surf, sphere_iterations=iterations)
    else:
        snake = Snake.create_for_surface(surf, sphere_iterations=iterations)
    snake.run()
    if len(args) > 1:
        with open(args[1], 'w') as f:
            dump_sphmap(f, snake)
    else:
        dump_sphmap(stdout, snake)

    if '--show-embedding' in opts:
        show_embedding(snake.image, snake)
    if '--show-travel' in opts:
        show_travel(snake)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
