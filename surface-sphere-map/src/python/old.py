import numpy as np
from numpy.linalg import norm

# Sloppy way to do this, but easy to understand

# Only one step
six_adjacency = adjacency_6 = [(i,j,k) for i in (-1,0,1) 
                                       for j in (-1,0,1) 
                                       for k in (-1,0,1) 
                                       if not (i == j == k == 0) 
                                       and sum(map(abs, (i, j, k))) <= 1]
# Max two steps
eighteen_adjacency = adjacency_6 = [(i,j,k) for i in (-1,0,1) 
                                            for j in (-1,0,1) 
                                            for k in (-1,0,1) 
                                            if not (i == j == k == 0) 
                                            and sum(map(abs, (i, j, k))) <= 2]
# All steps
twentysix_adjacency = adjacency_6 = [(i,j,k) for i in (-1,0,1) 
                                             for j in (-1,0,1) 
                                             for k in (-1,0,1) 
                                             if not (i == j == k == 0)]


class PseudoGrid(list):
    WRAPPER = np.array
    def __init__(self, *args):
        super(PseudoGrid, self).__init__(*args)
        self.distance_field = None
        self.gradient_field = None

    def __getitem__(self, coords):
        point = [dim[coord] for coord, dim in zip(coords, self)]
        return self.WRAPPER(point)

    def axis(self, i):
        return super(PseudoGrid, self).__getitem__(i)

    def nearest(self, point):
        points_in_dim = zip(self, point)
        index = tuple(starmap(np.searchsorted, points_in_dim))
        return index

    def quantize(self, point):
        index = self.nearest(point)
        coords = self[index]
        return coords

    def enumerate(self):
        for idx in np.ndindex(self.shape):
            yield idx, self[idx]

    def enumerate_over(self, points):
        for point in points:
            index = self.nearest(point)
            coords = self[index]
            yield index, coords

    @property
    def shape(self):
        return tuple(map(len, self))

    @property
    def spacing(self):
        n_dims = len(self)
        zeros = [0] * n_dims
        ones = [1] * n_dims
        return abs(self[ones] - self[zeros])

    @classmethod
    def from_extents(cls, extents, resolution=100):
        dims = []
        for low, high in extents:
            dims.append(np.linspace(low, high, resolution))
        return cls(dims)


def enumerate_gridpoints(dims):
    for coords in np.ndindex(*map(len, dims)):
        point = np.array([dim[pos] for pos, dim in zip(coords, dims)])
        yield coords, point




def rotation_matrix(axis,theta):
    axis = axis/norm(axis)
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def adjacent_indicies(index, shape=None, step=1):
    index = tuple(index)
    n_dims = len(index)
    if shape is None:
        shape = [np.inf] * n_dims
    if isinstance(step, (int, np.int)):
        step = [step] * n_dims
    step = np.array(step, dtype=int)
    def gen(idx):
        for dim, offset in enumerate(idx):
            left, right = offset - 1, offset + 1
            if left >= 0:
                yield tuple(idx[0:dim] + (left,) + idx[dim+1:])
            if right < shape[dim]:
                yield tuple(idx[0:dim] + (right,) + idx[dim+1:])
    if np.any(step > 1):
        unique = set()
        for neighbor in adjacent_indicies(index, shape, step-1):
            next_neighbors = gen(neighbor)
            unique.update(next_neighbors)
        return list(unique)
    else:
        return list(gen(index))


def adjacent(index, shape=None, connectivity=6):
    index = np.array(index)
    n_dims = len(index)
    if connectivity == 6:
        adjacency = six_adjacency
    elif connectivity == 18:
        adjacency = eighteen_adjacency
    elif connectivity == 26:
        adjacency = twentysix_adjacency
    else:
        raise ValueError("No such thing as {} adjacnecy".format(connectivity))
    left = np.zeros(len(index))
    if shape is None:
        shape = [np.inf] * n_dims
    right = np.array(shape)
    neighbors = []
    for delta in adjacency:
        neighbor = index + delta
        if np.all(left <= neighbor) and np.all(neighbor < right):
            neighbors.append(neighbor)
    return np.array(neighbors)


class OldSnake(object):

    def elasticity(self, idx, point):
        control = self.control_value(idx, 1)
        return 2 * (control - point)

    def rigidity(self, idx, point):
        control_1 = self.control_value(idx, 1)
        control_2 = self.control_value(idx, 2)
        # Seems to be off by a factor of -2 according to EYWTKAS
        #return 4 * control_1 - control_2 - 3 * point
        return control_2 + 3 * point - 4 * control_1

    def attraction(self, idx, point):
        # This is a really naive method
        # Find the nearest vertex, find all neighbors within two steps
        distance, triangle, nearest = self.nearest_on_surface(point)
        direction = nearest - point
        force = direction
        return force

    def force(self, idx, point):
        elasticity = self.tension * self.elasticity(idx, point)
        rigidity = self.stiffness * self.rigidity(idx, point)
        attraction = self.scale * self.attraction(idx, point)
        total_force = elasticity + rigidity + attraction
        magnitude = norm(total_force)
        if magnitude > self.clip:
            clipped = (total_force / magnitude) * self.clip  # Scale unit vector to clip
        else:
            clipped = total_force
        return clipped

    def clipped_force(self, idx, point):
        total_force = self.force(idx, point)
        magnitude = norm(total_force)
        if magnitude > self.clip:
            clipped = (total_force / magnitude) * self.clip  # Scale unit vector to clip
        else:
            clipped = total_force
        return clipped


def getZincStructure(zinc_id, format='mol2'):
    import requests
    url = "http://zinc.docking.org/substance/{0}.{1}".format(zinc_id, format)
    struct = requests.get(url)
    return struct.text


def MolToPDBBlock_custom(mol, mol_id=None, connect=True, chain_id='A', seg_id=1, res_id=1):
    import numpy as np
    from Bio.PDB.PDBIO import PDBIO
    from Bio.PDB.StructureBuilder import StructureBuilder
    if mol_id is None:
        mol_id = "MOL"
    builder = StructureBuilder()
    builder.init_structure(mol_id)
    builder.init_model(mol_id, 0)
    builder.init_chain(chain_id)
    builder.init_seg(seg_id)
    builder.init_residue(mol_id, 'H', res_id, ' ')

    atoms = mol.GetAtoms()
    conformer = mol.GetConformer()
    connections = []
    for idx, atom in enumerate(atoms):
        atom_id = idx + 1
        element = atom.GetSymbol()
        name = "{0}{1}".format(element, atom_id)
        coords = np.array(conformer.GetAtomPosition(idx))
        builder.init_atom(name=name,
                          coord=coords,
                          b_factor=1.0,
                          occupancy=1.0,
                          altloc=' ',
                          fullname=name.ljust(4),
                          serial_number=atom_id,
                          element=element)
        bonds = [bond.GetOtherAtomIdx(idx) + 1 for bond in atom.GetBonds()]
        connections.append((atom_id, bonds))

    io = StringIO()
    writer = PDBIO()
    writer.set_structure(builder.get_structure())
    writer.save(io)

    if connect:
        for atom_id, bonds in connections:
            connected = " ".join(str(bonded).ljust(4) for bonded in bonds)
            io.write(_CONNECT_LINE_TPL.format(str(atom_id).ljust(4), connected))

    io.write("END\n")

    return io.getvalue()




def distance_on_sphere_old(u, v, centroid):
    r = norm(u - centroid)
    theta = np.arccos(np.dot(u,v))
    dist = r * theta
    return dist


def distance_on_sphere(u, v, centroid):
    if np.allclose(u,v):  # Special case u==v
        return 0.
    cu = u - centroid
    cv = v - centroid
    r = norm(cu)
    n1 = unit(cu)
    n2 = unit(cv)
    sigma = np.arctan2(norm(np.cross(n1, n2)), np.dot(n1,n2))
    dist = r * sigma
    return dist


def kmeans_on_sphere(points, k, sphere_center=ZERO, sphere_radius=1., 
                                iterations=1000, epsilon=.01):
    points = np.array(points)
    delta, iteration = np.inf, 0
    centroids = points[:k]
    clusters = None
    while delta > epsilon and iteration < iterations:

        # Assignment Phase
        clusters = [[] for i in range(k)]
        for idx, point in enumerate(points):
            min_dist, assigned_cluster = np.inf, None
            for centroid, cluster in zip(centroids, clusters):
                dist = distance_on_sphere(point, centroid, sphere_center)
                if dist < min_dist:
                    min_dist = dist
                    assigned_cluster = cluster
            assigned_cluster.append(idx)

        # Update Phase
        old_centroids = np.array(centroids)
        for idx, cluster in enumerate(clusters):
            cluster_points = points[cluster]
            average = cluster_points.mean(axis=0)
            centroid = project_point_to_sphere(average, sphere_center, sphere_radius)
            centroids[idx] = centroid
        delta = abs(centroids - old_centroids).sum()

    return centroids, clusters



def create_icosahedron_old(radius=1.0, center=ZERO):
    vertexes = np.zeros((12, 3), dtype=np.float)

    theta = 26.56505117707799 * np.pi / 180

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    phi = np.pi / 5
    phis = phi * np.arange(1, 10, 2)

    x_coords = radius * cos_theta * np.cos(phis)
    y_coords = radius * cos_theta * np.sin(phis)
    z_offset = radius * sin_theta * np.ones(5)

    bottom_coords = np.array([x_coords, y_coords, -z_offset]).transpose()
    top_coords = np.array([x_coords, y_coords, z_offset]).transpose()

    vertexes[0] = (0, 0, -radius)
    vertexes[1:6]  = bottom_coords
    vertexes[6:11] = top_coords
    vertexes[11] = (0, 0, radius)

    vertexes += center

    triangles = np.array([
        # Top Triangles forming top petagon
        (0, 2, 1),
        (0, 3, 2),
        (0, 4, 3),
        (0, 5, 4),
        (0, 1, 5), 
 
        # Mid triangles, top to bottom
        (1, 2, 7),
        (2, 3, 8),
        (3, 4, 9),
        (4, 5, 10),
        (5, 1, 6),

        # Mid triangles, bottom to top
        (1, 7, 6),
        (2, 8, 7),
        (3, 9, 8),
        (4, 10, 9),
        (5, 6, 10),

        # Bottom triangles, forming bottom pentagon
        (6, 7, 11),
        (7, 8, 11),
        (8, 9, 11),
        (9, 10, 11),
        (10, 6, 11)
    ])

    return vertexes, triangles
    