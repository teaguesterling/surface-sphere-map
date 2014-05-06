from __future__ import division, print_function

import contextlib
import logging
import multiprocessing
import sys

import numpy as np
from scipy.linalg import norm
from scipy.ndimage.morphology import distance_transform_edt
from scipy.sparse import (
    lil_matrix,
    csc_matrix,
)
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

    def force_at_vertex(self, vertex, index):
        voxel = self._axes.get_voxel_index(vertex)
        force = self._field[voxel]
        return force


class ImplicitInternalForce(ContourForce):
    def __init__(self, scale=1.0, 
                       alpha=.2,
                       beta=.1,
                       dt=1,
                       ds=1,
                       n=None):
        super(ImplicitInternalForce, self).__init__(scale=scale)
        self._alpha = alpha
        self._beta = beta
        self._dt = dt
        self._ds = ds
        self._n = n
        self._setup_system()

    def _setup_system(self):
        a = (self._alpha * self._dt) / self._ds**2
        b = (self._beta * self._dt) / self._ds**4
        p = b
        q = -a - 4*b
        r = 1 + 2*a + 6*b
            

    def calculate_forces(self, vertices):
        X, Y, Z = vertices.transpose()
        Fx = solve_banded((2,2), self._M, X)
        Fy = solve_banded((2,2), self._M, Y)
        Fz = solve_banded((2,2), self._M, Z)
        forces = np.array([Fx, Fy, Fz]).transpose()
        return forces


class CurvatureRegluarizationForce(ContourForce):
    def __init__(self, contour, scale=1.0, grad=None, alpha=1.0, beta=0.5):
        super(CurvatureRegluarizationForce, self).__init__(scale=scale)
        self._contour = contour
        self._grad = grad
        self._alpha = alpha
        self._beta = beta

    def update_after_step(self, mesh):
        self._vertices = mesh

    def curvature_force(self, vertex, index):
        # From 3-D Active Contours (Dufor et. al.)
        n, k, un = self.contour.vertex_normal_curvature(index)
        voxel = self._axes.position_on_axes(vertex)
        f = self._grad[voxel]
        g = self._image[voxel]
        curvature = (g * n) - np.dot(f, un) * un
        return curvature

    def internal_force(self, vertex, index):
        c1, n1 = self._control_value(index, 1)
        c2, n2 = self._control_value(index, 2)
        elasticity = c1 - n1 * vertex
        rigidity = c2 -  4 * c1 - (4 * n1 - n2) * vertex
        force = self._alpha * elasticity + self._beta * rigidity
        return force

    def _control_value(self, index, dist):
        neighbors = self._contour.neighboring_vertices(index, dist, cycles=False)
        points = self._vertices[neighbors]
        control = points.sum(axis=0)
        N = len(points)
        return control, N

    def force_at_vertex(self, vertex, index):
        return self.internal_force(vertex, index)

    @classmethod
    def from_contour(cls, contour, scale=1.0, alpha=1.0, beta=0.5):
        force = cls(contour, image=None, scale=scale, alpha=alpha, beta=beta)
        return force


class Snake(object):

    def __init__(self, mesh,
                       resolution=1,
                       padding=10,
                       contour_faces=None,
                       external_scale=1,
                       internal_scale=.1,
                       smoothness=0.15,
                       timestep=.75,
                       numsteps=None,
                       zero_on_boundary=False,
                       normalize_force_after_distance=None,
                       tension=1,
                       stiffness=0.0,
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
        self.smoothness = smoothness
        self.timestep = timestep
        self.numsteps = numsteps
        self.zero_on_boundary = zero_on_boundary
        self.normalize_force_after_distance = normalize_force_after_distance
        # Internal Force
        self.tension = tension
        self.stiffness = stiffness

        # Force scaling
        self.external_scale = external_scale
        self.internal_scale = internal_scale

        self.step = 1
        self.iterations = 0
        self.max_iterations = max_iterations

        self._translate_mesh()
        self._prep_snake()
        self._create_contour()
        self._calculate_gvf()
        self._create_tension_system()

        self.vertices = self.contour.vertices
        self.faces = self.contour.faces

        self.forces = [
            GVFForce(
                self.gvf_field, 
                axes=self.axes, 
                scale=self.external_scale),
        ]
        
    def _translate_mesh(self):
        self.log.debug("Repositioning source mesh")
        low = self.original_mesh.vertices.min(axis=0)
        high = self.original_mesh.vertices.max(axis=0)
        lower_bound = low - self.padding
        self.low_point = low
        self.high_point = high
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
            iterations = sphere.iterations_needed_for_triangles(iterations)
        self.log.debug("Generating spherical contour with {} faces".format(num_tris))
        tessalation = sphere.Sphere.from_tessellation(radius=radius,
                                                      center=center,
                                                      iterations=iterations)
        self.contour = tessalation

    def _calculate_gvf(self):
        self.log.debug("Starting GVF calculation")
        u, v, w = gvf.reference_gvf3d(self.voxelized,
                                      k=self.numsteps,
                                      mu=self.smoothness,
                                      dt=self.timestep)
        if self.zero_on_boundary:
            mask = 1 - self.voxelized
            u, v, w, = u * mask, v * mask , w * mask
            
        gvf_field = gvf.make_field_array(u, v, w)

        if self.normalize_force_after_distance is not None:
            speed = 10
            thresh = self.normalize_force_after_distance
            Minv = 1 / magnitudes(gvf_field, zero_to_one=True)[:,:,:,np.newaxis]
            D = distance_transform_edt(1-image)
            H = .5 + .5 * np.tanh(speed * (D - thresh))
            mask = 1 + H * (Minv - 1)
            gvf_field = gvf_field * mask
            
        self.gvf_components = gvf_field[:,:,:,0], gvf_field[:,:,:,1], gvf_field[:,:,:,2]
        self.gvf_field = gvf_field

    def _create_tension_system(self):
        self.log.debug("Generating internal tension system")
        dt, ds = 1, 1
        a = self.tension * dt / (ds ** 2)
        q_r = {
            6: (-a / 6, 2),
            5: (-a*5 / 6, 2),
        }
        N = len(self.contour.vertices)
        A = lil_matrix((N,N))
        for i, M in enumerate(self.contour.neighbors):
            M = self.contour.neighbors[i]
            N = len(M)
            q, r = q_r[N]
            A[i,i] = r
            for k in M:
                A[i,k] = q
        solver = factorized(csc_matrix(A))
        self.apply_internal_forces = solver

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
            total_force += force(self.vertices)
        return total_force

    def update(self):
        print("Updating contour step {0}".format(self.iterations), file=sys.stderr)
        new_vertices = np.array(self.vertices)
        new_vertices += self.force()
        new_vertices = self.apply_internal_force(new_vertices)
        delta = abs(self.vertices - new_vertices).sum()
        self.vertices = new_vertices
        return delta

    def run(self):
        # Initialization phase:
        # Place snake in better starting position
        delta = np.inf
        # Get snake closer to surface while maintaining total internal energy
        N = len(self.vertices)
        while delta > 0.01 and self.iterations < self.max_iterations:
            # Step towards image
            self.iterations += 1
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
