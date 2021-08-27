#!/usr/bin/env python

"""

"""


def map_classes(class_tensor):
    return 'building' if class_tensor[0, 0] == 0 else 'car'


# Function to check if a pixel is a wall of the maze
def is_wall(point, path_pixels):
    x, y = point
    pixel = path_pixels[x, y]
    if any(i < 225 for i in pixel):
        return True


# Function to find neighboring points
def moore_neighbors(point, path_pixels):
    x, y = point
    neighbors = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1), (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y),
                 (x + 1, y + 1)]
    return [point for point in neighbors if not is_wall(point, path_pixels)]


def squared_euclidean(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def AStar(start, end, path_pixels):
    """
    https://en.wikipedia.org/wiki/A*_search_algorithm
    :param start:
    :param goal:
    :param path_pixels:
    :return:
    """

    def construct_path(came_from, current_node):
        path = []
        while current_node is not None:
            path.append(current_node)
            current_node = came_from[current_node]
        return list(reversed(path))

    g = {start: 0}
    f = {start: g[start] + squared_euclidean(start, end)}
    openset = {start}
    closedset = set()
    came_from = {start: None}

    while openset:
        current = min(openset, key=lambda x: f[x])
        if current == end:
            return construct_path(came_from, end)
        openset.remove(current)
        closedset.add(current)
        for neighbor in moore_neighbors(current, path_pixels):
            if neighbor in closedset:
                continue
            if neighbor not in openset:
                openset.add(neighbor)
            tentative_gscore = g[current] + squared_euclidean(current, neighbor)
            if tentative_gscore >= g.get(neighbor, float('inf')):
                continue
            came_from[neighbor] = current
            g[neighbor] = tentative_gscore
            f[neighbor] = tentative_gscore + squared_euclidean(neighbor, end)

    return []

