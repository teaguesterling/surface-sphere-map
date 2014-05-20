from __future__ import division, print_function

from itertools import islice

import numpy as np
from numpy.linalg import norm

import spatial
import sphere


class Snake3D(object):

    def __init__(self, contour, image,
                       step=0.2,
                       scale=1.0,
                       tension=1,
                       stiffness=0.0,
                       push=0.01,
                       clip=np.inf,
                       threshold=None,
                       iterations=500):
        self.vertices = np.array(contour.vertices)
        self.faces = np.array(contour.faces)
        self.distances = np.zeros(len(contour.vertices))

        self.contour = contour
        self.image = image

        self.step = step
        self.scale = scale
        self.tension = tension
        self.stiffness = stiffness
        self.push = push
        self.clip = clip

        self.average_scale = False
        self.tangential_interal_forces = False
        self.search = 2  # Number of adjacent vertices to check

        self.threshold = threshold
        self.iterations = 0
        self.max_iterations = iterations
        self.neighbor_cache = {}
        self.normal_cache = {}

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

    def neighbors(self, idx, dist, cycles=False):
        if (idx,dist) in self.neighbor_cache:
            return self.neighbor_cache[idx,dist]
        neighbors = self.contour.neighboring_vertices(idx, dist, cycles=cycles)
        self.neighbor_cache[idx,dist] = neighbors
        return neighbors

    def control_value(self, idx, dist, cycles=False):
        neighbors = self.neighbors(idx, dist, cycles=cycles)
        points = self.vertices[neighbors]
        control = points.mean(axis=0)
        return control

    def nearest_on_surface(self, idx, point):
        closest_data = self.image.approximate_nearest_point(point, steps=self.search)
        distance, triangle, nearest = closest_data
        self.normal_cache[idx] = spatial.triangle_normal(triangle)
        return closest_data

    def internal_forces(self, idx, point):
        # Find 1st and 2nd degree neighbors
        control_1 = self.control_value(idx, 1)
        control_2 = self.control_value(idx, 2)

        # Calculate internal energies
        elasticity = 2 * (control_1 - point)
        rigidity = 2 * (control_2 + 3 * point - 4 * control_1)
        scaled_elacticity = self.tension * elasticity
        scaled_rigidity = self.stiffness * rigidity
        internal =  scaled_elacticity + scaled_rigidity

        return internal

    def external_forces(self, idx, point):
        # Find nearest point on surface and distance as attraction
        distance, triangle, nearest = self.nearest_on_surface(idx, point)
        direction = nearest - point
        external = self.scale * direction
        return external, triangle

    def adjust_internace_force(self, internal, surface_normal):
        # Compute internal energy only tangential to surface
        # (e.g. remove attractive aspects of elacticty)
        if self.tangential_interal_forces:
            internal = spatial.project_to_plane(internal, surface_normal)
        return internal

    def force(self, idx, point):
        internal = self.internal_forces(idx, point)
        external, triangle = self.external_forces(idx, point)
        normal = self.normal_cache[idx]
        internal = self.adjust_internace_force(internal, normal)
        total_force = external + internal
        magnitude = norm(total_force)
        if magnitude > self.clip:
            total_force = self.clip * (total_force / magnitude)
        return total_force

    def averaged_attractions(self):
        attractions = np.zeros(self.vertices.shape)
        for idx, point in enumerate(self.vertices):
            distance, triangle, nearest = self.nearest_on_surface(idx, point)
            attractions[idx] = nearest - point
        distances = np.apply_along_axis(norm, 1, attractions)
        directions = attractions / distances.reshape(1,-1).transpose()
        mean_distance = distances.mean()
        normalized = distances - mean_distance
        normalized_attractions = directions * normalized.reshape(1,-1).transpose()
        return normalized_attractions

    def update_vertex(self, idx, vertex):
        direction = self.force(idx, vertex)
        step = self.step * direction
        return vertex + step

    def update(self):
        delta = self.step_external_forces()
        for _ in range(10):
            self.stabilize_internal_forces(tangent_only=delta < 0.05)
        return
        delta_total = 0
        num_vertices = len(self.vertices)
        new_vertices = np.zeros(self.vertices.shape)
        self.normal_cache.clear()
        for idx, vertex in enumerate(self.vertices):
            new_vertex = self.update_vertex(idx, vertex)
            new_vertices[idx] = new_vertex
            delta = norm(vertex - new_vertex)
            delta_total += delta
        self.vertices = new_vertices
        average_delta = delta_total / num_vertices
        return average_delta

    def normalize_distances(self):
        offsets = self.averaged_attractions()
        self.vertices = self.vertices + offsets

    def decrease_internal_forces(self):
        self.tension /= 2
        self.stiffness /= 2

    def step_external_forces(self, step=None):
        if step is None:
            step = self.step
        self.normal_cache.clear()
        delta_sum = 0
        num_vertices = len(self.vertices)
        new_vertices = np.zeros(self.vertices.shape)
        for idx, vertex in enumerate(self.vertices):
            external, triangle = self.external_forces(idx, vertex)
            new_vertex = vertex + step * external
            distance = norm(external)
            delta = norm(vertex - new_vertex)
            delta_sum += delta
            self.distances[idx] = distance
            new_vertices[idx] = new_vertex
        self.vertices = new_vertices
        delta = delta_sum / num_vertices
        return delta

    def stabilize_internal_forces(self, iterations=5, step=None, tangent_only=True):
        if step is None:
            step = self.step
        for iteration in range(iterations):
            try:
                max_distance = self.contour.radius
            except AttributeError:
                small, large = self.contour.extents.transpose()
                max_distance = np.abs(large - small)
            new_vertices = np.zeros(self.vertices.shape)
            for idx, vertex in enumerate(self.vertices):
                internal = self.internal_forces(idx, vertex)
                if tangent_only:
                    normal = self.normal_cache[idx]
                    internal = self.adjust_internace_force(internal, normal)
                relative_distance = (max_distance - self.distances[idx]) / max_distance
                new_vertices[idx] = vertex + relative_distance * self.step * internal
            self.vertices = new_vertices

    def run(self):
        # Initialization phase:
        # Place snake in better starting position
        self.normalize_distances()
        self.step_external_forces()
        self.normalize_distances()
        delta = np.inf
        # Primary Refinement

        # Get snake closer to surface while maintaining total internal energy
        while delta > 0.01 and self.iterations < self.max_iterations:
            # Step towards image
            delta = self.step_external_forces()
            # As snake approaches begin decreasing internal energy
            if delta < 0.2:
                self.decrease_internal_forces()
                steps = 5
            else:
                steps = 10
            # Enforce only tangential component of engergy when very close
            self.stabilize_internal_forces(steps, tangent_only=delta < 0.1)
            self.iterations += 1
        # Push snake back up to surface if it has "fallen in"
        #self.step_external_forces(1)

    @classmethod
    def create_for_surface(cls, surf, sphere_iterations=2, *args, **kwargs):
        radius = (surf.extents.ptp() / 2) * 1.1  # Increase slightly
        tessalation = sphere.Sphere.from_tessellation(radius=radius,
                                                      center=surf.centroid,
                                                      iterations=sphere_iterations)
        snake = cls(tessalation, surf, *args, **kwargs)
        return snake

    @classmethod
    def create_for_surface_invert(cls, surf, sphere_iterations=2, *args, **kwargs):
        radius = (surf.extents.ptp() / 2) * 1.1  # Increase slightly
        tess = sphere.Sphere.from_tessellation(radius=radius,
                                               center=surf.centroid,
                                               iterations=sphere_iterations)
        snake = cls(surf, tess, *args, **kwargs)
        return snake


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
    if len(args) > 0:
        with open(args[0]) as f:
            surf = spatial.Surface.from_vet_file(f)
    else:
        surf = spatial.Surface.from_vet_file(stdin)
    iterations = int(params.get('iterations', 2))
    if '--invert' in opts:
        snake = Snake3D.create_for_surface_invert(surf, sphere_iterations=iterations)
    else:
        snake = Snake3D.create_for_surface(surf, sphere_iterations=iterations)
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
    if s is not None:
        sph = ax.plot_trisurf(*s.vertices.transpose(), triangles=s.faces)
        sph.set_alpha(.1)
    else:
        sph = None
    if m is not None:
        mol = ax.plot_trisurf(*m.vertices.transpose(), triangles=m.faces)
        mol.set_alpha(.9)
    else:
        mol = None
    plt.show()
    return fig, ax, sph, mol


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
