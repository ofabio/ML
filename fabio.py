import time
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, range_x, range_y):
        plt.ion()
        plt.axis([-range_x, range_x, -range_y, range_y])
        self.drawn = []

    def drawn_point_vector(self, points, colour='wo'):
        colours_map = {'blue': 'bo', 'red': 'ro', 'yellow': 'yo'}
        if colour in colours_map:
            colour = colours_map[colour]
        drawn_points, = plt.plot(points[:, 0], points[:, 1], colour)
        plt.pause(0.0001)
        self.drawn.append(drawn_points)
        return drawn_points

    def clear(self):
        for d in self.drawn:
            d.remove()
        plt.pause(0.0001)




def create_dataset(space_size, n_clusters, n_points_per_cluster):
    statistic_space = 0.7 * space_size
    distribution_centroids = np.random.rand(n_clusters, 2) * statistic_space
    points = None
    for c in distribution_centroids:
        cluster_points = np.random.randn(n_points_per_cluster, 2) + c
        if points is None:
            points = cluster_points
        else:
            points = np.concatenate((points, cluster_points), axis=0)
    return points

def get_clusters(points):
    init_centroid()
    for point 
    return clusters


plotter = Plotter(100, 100)

points = create_dataset(space_size=100, n_clusters=3, n_points_per_cluster=10)

drawn_points = plotter.drawn_point_vector(points, 'red')
time.sleep(2)
plotter.clear()
drawn_points = plotter.drawn_point_vector(points, 'yellow')
time.sleep(2)





# TODO
# 1. assign points to clusters (colour the points)
# 2. train the cluster to the network
# 3. evaluate ability to predict belonging of new points