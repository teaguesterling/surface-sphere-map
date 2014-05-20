from __future__ import division, print_function

import itertools

import numpy as np
from numpy.linalg import norm

import snake
import sphere

from spatial import unit
from sphere import pointdict


def half_circumference(radius):
    return np.pi * radius


def distance_on_sphere(u, v, centroid, radius=None):
    if np.allclose(u,v):  # Special case u~=v
        return 0.
    cu = u - centroid
    cv = v - centroid
    if radius is None:
        radius = norm(cu)
    n1 = unit(cu)
    n2 = unit(cv)
    sigma = np.arctan2(norm(np.cross(n1, n2)), np.dot(n1,n2))
    dist = radius * sigma
    return dist


def build_histogram(values, num_bins=100, lower_bound=0, upper_bound=100):
    bin_boundaries = np.linspace(lower_bound, upper_bound, num_bins)
    bins = [[] for _ in range(num_bins)]
    for idx, value in enumerate(values):
        bin_idx = np.searchsorted(bin_boundaries, value)
        current_bin = bins[bin_idx]
        current_bin.append(idx)
    return bins


def histogram_counts(histogram):
    return np.array(map(len, histogram))


def build_property_histogram(values, num_bins=100):
    property_min, property_max = values.min(), values.max()
    bins = build_histogram(values, num_bins, property_min, property_max)
    return bins


def cluster_property_histograms(points, bins, sphere_center=sphere.ZERO, 
                                              sphere_radius=1.):
    bin_values = []
    for bin_members in bins:
        bin_values.append(points[bin_members])
    all_centroids = []
    all_clusteres = []
    for bin_idx, values in enumerate(bin_values):
        # Cluster each property bin and add to result
        centroids, clusters = select_best_kmeans(values, max_k=5, min_k=2, # Arbitrarily picked
                                                         sphere_center=sphere_center, 
                                                         sphere_radius=sphere_radius)
        all_centroids.append(centroids)
        all_clusteres.append(clusters)
    return all_centroids, all_clusteres


def build_distance_histogram(centroids, clusters, sphere_center=sphere.ZERO,
                                                  sphere_radius=1., 
                                                  resolution=1.):
    centroid_distances = pairwise_centroid_distances(centroids, sphere_center=sphere_center,
                                                                sphere_radius=sphere_radius)
    pairs = centroid_distances.keys()
    distances = centroid_distances.values()
    distance_bins = quantize_distances_on_sphere(distances, sphere_radius=sphere_radius,
                                                            resolution=resolution)
    distance_histogram = np.zeros(len(distance_bins))
    for bin_num, distance_bin in enumerate(distance_bins):
        membership = set()
        for idx in distance_bin:
            pair = pairs[idx]
            membership.update(pair)
        membership_items = map(lambda idx: clusters[idx], membership)
        membership_size = map(len, membership_items)
        bin_count = sum(membership_size)
        distance_histogram[bin_num] = bin_count
    return distance_histogram


def build_distance_histograms(all_centroids, all_clusters, sphere_center=sphere.ZERO,
                                                           sphere_radius=1., 
                                                           resolution=1.):
    distance_histograms = []
    for idx, (centroids, clusters) in enumerate(zip(all_centroids, all_clusters)):
        distance_histogram = build_distance_histogram(centroids, clusters, sphere_center=sphere_center,
                                                                           sphere_radius=sphere_radius,
                                                                           resolution=resolution)
        distance_histograms.append(distance_histogram)
    return distance_histograms


def pairwise_centroid_distances(centroids, sphere_center=sphere.ZERO,
                                           sphere_radius=1.):
    distances = pointdict()
    # Using replacement to force zero distances between a centroid and itself to show up
    for centroid_pair in itertools.combinations_with_replacement(enumerate(centroids), 2):
        idx1, centroid1 = centroid_pair[0]
        idx2, centroid2 = centroid_pair[1]
        pair = idx1, idx2
        distance = distance_on_sphere(centroid1, centroid2, sphere_center, radius=sphere_radius)
        distances[pair] = distance
    return distances


def quantize_distances_on_sphere(distances, sphere_radius=1., resolution=1.):
    max_distance = half_circumference(sphere_radius)
    num_bins = int(np.ceil(max_distance / resolution))
    bins = build_histogram(distances, num_bins, 0, max_distance)
    return bins


def select_best_kmeans(points, max_k, min_k=2,
                                      sphere_center=sphere.ZERO, sphere_radius=1., 
                                      iterations=1000, epsilon=.01):
    num_points = len(points)
    if num_points == 0:
        return [], []
    if min_k > num_points:
        min_k = num_points
    if max_k > num_points:
        max_k = num_points

    min_inter_cluster_distance, best_run = np.inf, None
    for k in range(min_k, max_k+1):
        centroids, clusters = kmeans_on_sphere(points, k, sphere_center=sphere_center,
                                                          sphere_radius=sphere_radius,
                                                          iterations=iterations,
                                                          epsilon=epsilon)
        dist_sum, num_dist = 0, 0
        for centroid, cluster in zip(centroids, clusters):
            for idx in cluster:
                point = points[idx]
                dist_sum += distance_on_sphere(centroid, point, sphere_center)
                num_dist += 1
        inter_cluster_distance = safe_divide(dist_sum, num_dist, 0)
        if inter_cluster_distance < min_inter_cluster_distance:
            min_inter_cluster_distance = inter_cluster_distance
            best_run = centroids, clusters
    return best_run


def kmeans_on_sphere(points, k, sphere_center=sphere.ZERO, sphere_radius=1., 
                                iterations=100, epsilon=.01):
    points = np.array(points)
    delta, iteration = np.inf, 0
    centroids = points[:k]
    clusters = None
    while delta > epsilon and iteration < iterations:
        
        # Assignment Phase
        clusters = [[] for _ in range(k)]
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
            if len(cluster_points) > 0:
                average = cluster_points.mean(axis=0)
                centroid = sphere.project_point_to_sphere(average, sphere_center, sphere_radius)
                centroids[idx] = centroid
            else:
                centroids[idx] = old_centroids[idx]
        iteration += 1
        delta = abs(centroids - old_centroids).mean()

    return centroids, clusters


def compute_surface_fingerprint(sphmap):
    mappings, radius, center = sphmap
    shape = radius - mappings[:,0]
    sphere_coords = mappings[:,1:4]
    properties = [shape]
    property_histograms = []

    for prop in properties:
        property_bins = build_property_histogram(shape, num_bins=100)
        bin_centroids, bin_clusters = cluster_property_histograms(sphere_coords, property_bins, 
                                                                  sphere_center=center,
                                                                  sphere_radius=radius)
        distance_histograms = build_distance_histograms(bin_centroids, bin_clusters, 
                                                        sphere_center=center,
                                                        sphere_radius=radius,
                                                        resolution=1)
        propety_counts = histogram_counts(property_bins)
        property_fingerprint = (propety_counts, distance_histograms)
        property_histograms.append(property_fingerprint)
    return property_histograms


def safe_divide(numerator, denominator, default=0.):
    if denominator == 0:
        return default
    else:
        return numerator / denominator


def interset_histogram(histogram1, histogram2, bin_weights=None, symmetric=True):
    size1 = len(histogram1)
    size2 = len(histogram2)

    # Extend histograms if needed
    size = max(size1, size2)
    comparision = np.zeros((2, size))
    comparision[0,:size1] = histogram1
    comparision[1,:size2] = histogram2

    if bin_weights is None:
        bin_weights = np.ones(size)
    
    minimums = comparision.min(axis=0)
    weighted = minimums * bin_weights
    numerator = weighted.sum()
    denominator = comparision[1].sum()
    similairty1 = safe_divide(numerator, denominator, 0)

    if symmetric:
        denominator = comparision[0].sum()
        similairty2 = safe_divide(numerator, denominator, 0)
        similairty = (similairty1 + similairty2) / 2
    else:
        similairty = similairty1

    return similairty


def compare_property_fingerprints(fingerprint1, fingerprint2):
    property_bins1, distance_bins1 = fingerprint1
    property_bins2, distance_bins2 = fingerprint2

    num_props = min(len(property_bins1), len(property_bins2))
    weights = np.zeros(num_props)
    for idx in range(num_props):
        bin_dists_1 = distance_bins1[idx]
        bin_dists_2 = distance_bins2[idx]
        dist_similarity = interset_histogram(bin_dists_1, bin_dists_2)
        weights[idx] = dist_similarity

    prop_similarity = interset_histogram(property_bins1, property_bins2, weights)
    return prop_similarity


def load_sphmap(path):
    with open(path) as f:
        mappings, radius, center = snake.load_sphmap(f)
    return mappings, radius, center


def compare_fingerprints(fp1, fp2):
    num_props = min(map(len, (fp1, fp2)))
    prop_similarities = np.zeros(num_props)
    for idx in range(num_props):
        prop_fp1, prop_fp2 = fp1[idx], fp2[idx]
        similarity = compare_property_fingerprints(prop_fp1, prop_fp2)
        prop_similarities[idx] = similarity
    overall_similarity = prop_similarities.mean()
    return overall_similarity



def main(args, stdout=None):
    if stdout is None:
        import sys
        stdout = sys.stdout
    sphmap1, sphmap2 = args[0:2]
    surface1 = load_sphmap(sphmap1)
    surface2 = load_sphmap(sphmap2)

    fp1 = compute_surface_fingerprint(surface1)
    fp2 = compute_surface_fingerprint(surface2)

    similarity = compare_fingerprints(fp1, fp2)

    print("{0:.4f}".format(similarity), file=stdout)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:], stdout=sys.stdout)

