from __future__ import division, print_function

import contextlib
import logging
import multiprocessing
import itertools
import sys

import numpy as np
from scipy import gradient
from scipy.linalg import norm
from scipy.ndimage.filters import (
    laplace,
    gaussian_filter,
)
from scipy.ndimage.morphology import distance_transform_edt
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
    # Ignore divide by zero (becomes zero)
    with numpy_error_handling(divide='ignore'):
        magnitudes = np.sqrt(np.sum(A**2, axis=3))
    if zero_to_one:
        magnitudes[magnitudes == 0] = 1
    return magnitudes


def make_banded_matrix(N, bands):
    A = np.zeros((N,N))
    M = len(bands)
    for band in bands:
        np.fill_diagonal(A, band)
        A = np.roll(A, 1, axis=0)
    A = np.roll(A, -M//2, axis=0)
    return A


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


class GVFForce(ContourForce):
    def __init__(self, field, axes, scale=1.0, **kwargs):
        super(GVFForce, self).__init__(scale=scale, **kwargs)
        self._axes = axes
        self._field = field

    def calculate_forces(self, mesh):
        vX, vY, vZ = self._axes.map_to_axes(mesh).transpose()
        force = self._field[vX, vY, vZ, :]
        return force

    def force_at_vertex(self, vertex, index):
        voxel = self._axes.get_voxel_index(vertex)
        force = self._field[voxel]
        return force


class SurfaceForce(ContourForce):
    def __init__(self, axes, voxels, voxel_map, surface, scale=1.0, **kwargs):
        super(SurfaceForce, self).__init__(scale=scale, **kwargs)
        self._axes = axes
        self._voxels = voxels
        self._voxel_map = voxel_map
        self._surface = surface
        self._nearby_tris = {}
        for idx, faces in self._voxel_map.iteritems():
            self._nearby_tris[idx] = self._surface.triangles[faces]

    def calculate_forces(self, mesh):
        forces = np.zeros(mesh.shape)
        voxels = self._axes.map_to_axes(mesh)
        for idx, (vertex, voxel) in enumerate(itertools.izip(mesh, voxels)):
            voxel = tuple(voxel)
            if self._voxels[voxel]:
                nearest, min_dist = None, np.inf
                triangles = self._nearby_tris[voxel]
                for triangle in triangles:
                    dist, point = spatial.distance_to_triangle(vertex, triangle, with_point=True)
                    if dist < min_dist:
                        nearest, min_dist = point, dist
                forces[idx] = nearest - vertex
        return forces


class Snake(object):

    def __init__(self, mesh,
                       resolution=1,
                       padding=10,
                       contour_faces=None,
                       blur_radius=None,
                       external_scale=1,
                       internal_scale=.1,
                       ggvf=False,
                       smoothness=0.15,
                       gvf_timestep=.75,
                       numsteps=None,
                       on_boundary=False,
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
        # GVF Force
        self.ggvf = ggvf
        self.blur_radius = blur_radius
        self.smoothness = smoothness
        self.gvf_timestep = gvf_timestep
        self.numsteps = numsteps
        self.on_boundary = on_boundary
        self.normalize_force_after_distance = normalize_force_after_distance
        # Internal Force
        self.tension = tension
        self.stiffness = stiffness

        # Force scaling
        self.external_scale = external_scale
        self.internal_scale = internal_scale

        self.step = 1
        self.iterations = 0
        self.timestep = timestep
        self.max_iterations = max_iterations
        self.epsilon = 0.001

        self._translate_mesh()
        self._prep_snake()
        self._create_contour()
        self._calculate_gvf()
        self._create_internal_system()

        self.vertices = self.contour.vertices
        self.faces = self.contour.faces

        self.forces = [
            GVFForce(
                field=self.gvf_field, 
                axes=self.axes, 
                scale=self.external_scale*self.timestep),
            SurfaceForce(
                axes=self.axes, 
                voxels=self.voxelized,
                voxel_map=self.voxel_triangles,
                surface=self.mesh,
                scale=1*self.timestep)
        ]
        
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
        center = self.mesh.centroid
        dists = map(np.linalg.norm, self.mesh.vertices - center)
        radius = max(dists)
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
        if self.ggvf:
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
        if self.on_boundary == 'zero':
            mask = 1 - raw_image
            u, v, w, = u * mask, v * mask , w * mask
        elif self.on_boundary == 'curvature':
            cu, cv, cw = gradient(laplace(raw_image))
            mask = 1 - self.voxelized
            u += cu * mask
            v += cv * mask
            w += cw * mask
            
        gvf_field = gvf.make_field_array(u, v, w)

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
            
        self.gvf_components = gvf_field[:,:,:,0], gvf_field[:,:,:,1], gvf_field[:,:,:,2]
        self.gvf_field = gvf_field

    def _create_internal_system(self):
        self.log.debug("Generating internal energy system")
        dt, ds = self.timestep, self.resolution
        a = self.internal_scale * self.tension * dt / (ds ** 2)
        b = self.internal_scale * self.stiffness * dt / (ds ** 2)
        N = len(self.contour.vertices)
        A = lil_matrix((N,N))
        for i, M1 in self.contour.neighbors.iteritems():
            M2 = self.contour.neighboring_vertices(i, 2)
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
        for force in self.forces:
            force_element = force(self.vertices)
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


def show_embedding(s, m, fig=None, ax=None):
    from mayavi import mlab
    mlab.clf()
    if s is not None:
        args = list(s.vertices.transpose()) + [s.faces]
        sph = mlab.triangular_mesh(*args)
    else:
        sph = None
    if m is not None:
        args = list(m.vertices.transpose()) + [m.faces]
        mol = mlab.triangular_mesh(*args)
    else:
        mol = None
    mlab.show()
    return sph, mol


def show_travel(s, fig=None, ax=None):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
    ax.set_axis_off()
    ax.axison = False
    max_travel = s.contour.radius
    values = []
    for face in s.faces:
        travel = s.travel[face].mean()
        value = (max_travel - travel) / max_travel
        values.append(value)
    colors = map(lambda x: (x, x, x), values)
    sph = ax.plot_trisurf(*s.contour.vertices.transpose(), triangles=s.faces,
                                                         shade=True)
    sph.set_facecolors(colors)
    plt.show()
    return fig, ax, sph


def demo(path, invert=False, fig=None, ax=None):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    with open(path) as f:
        surf = spatial.Surface.from_vet_file(f)
    if invert:
        snake = Snake3D.create_for_surface_invert(surf, sphere_iterations=2)
    else:
        snake = Snake3D.create_for_surface(surf, sphere_iterations=2)
    for i in range(25):
       try:
            sph.remove()
       except:
         pass
       sph = ax.plot_trisurf(*snake.vertices.transpose(), triangles=snake.faces)
       sph.set_alpha(1)
       fig.canvas.draw()
       snake.update()
       sph.remove()
       #s.clip += i/.25
    sph = ax.plot_trisurf(*snake.vertices.transpose(), triangles=snake.faces)
    return fig, ax



if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
