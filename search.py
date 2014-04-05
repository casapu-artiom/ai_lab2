__author__ = 'Artiom.Casapu'

import logging
import heapq
from collections import deque
from random import random
import sys

"""

    Traveling Salesman throught DFS vs BFS

"""

DEBUG = False

logger = logging.getLogger('Lab1')

logger.addHandler(logging.StreamHandler(stream=sys.stdout))

if (DEBUG):
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

class Graph:

    def __init__(self, cities=None, distances=None):
        if not cities: cities = []
        if not distances: distances = {}
        self.distances = distances
        self.cities = cities

    def getNeighbouringCities(self, city):
        return self.distances[city]

    def getDistance(self, city1, city2):

        if (city1 not in self.distances):
            return None

        for city, dist in self.distances[city1]:
            if (city == city2):
                return dist

        return None

    def dfs(self):
        result = None

        def _dfs(initial_node):

            result = None

            q = deque()

            q.appendleft(initial_node)

            while len(q) > 0:
                node = q.popleft()

                if (node.path_len == len(self.cities)):
                    if (result is None) or (result.distance > node.distance):
                        result = node

                else:
                    logger.debug("Popping out %s", node)

                    neighbours = self.getNeighbouringCities(node.city)

                    for neighbour in neighbours:

                        if (neighbour[0] in node.visited):
                            continue

                        tmp = Graph.Node()
                        tmp.city = neighbour[0]
                        tmp.distance = node.distance + self.getDistance(node.city, tmp.city)
                        tmp.prevNode = node
                        tmp.path_len = node.path_len + 1
                        tmp.visited = set(node.visited)
                        tmp.visited.add(tmp.city)

                        logger.debug("Appending %s", tmp)

                        q.appendleft(tmp)

            return result

        for city in self.cities:

            logger.debug("Starting dfs with %s", city)

            tmp = Graph.Node()

            tmp.city = city
            tmp.distance = 0
            tmp.prevNode = None
            tmp.path_len = 1
            tmp.visited.add(city)

            tmpresult = _dfs(tmp)

            if (tmpresult is None):
                continue

            if (result is None) or (tmpresult.distance < result.distance):
                result = tmpresult

        return self.generate_path(result)

    def gbfs(self):
        result = None

        def _gbfs(initial_node):

            result = None

            q = []

            heapq.heappush(q, (initial_node.distance, initial_node))

            while len(q) > 0:
                node = heapq.heappop(q)[1]

                if (node.path_len == len(self.cities)):
                    if (result is None) or (result.distance > node.distance):
                        result = node

                else:
                    logger.debug("Popping out %s" % node)

                    neighbours = self.getNeighbouringCities(node.city)

                    for neighbour in neighbours:

                        if (neighbour[0] in node.visited):
                            continue

                        tmp = Graph.Node()
                        tmp.city = neighbour[0]
                        tmp.distance = node.distance + self.getDistance(node.city, tmp.city)
                        tmp.prevNode = node
                        tmp.path_len = node.path_len + 1
                        tmp.visited = set(node.visited)
                        tmp.visited.add(tmp.city)

                        logger.debug("Appending %s", tmp)

                        heapq.heappush(q, (tmp.distance, tmp))

            return result

        for city in self.cities:

            logger.debug("Starting dfs with %s", city)

            tmp = Graph.Node()

            tmp.city = city
            tmp.distance = 0
            tmp.prevNode = None
            tmp.path_len = 1
            tmp.visited.add(city)

            tmpresult = _gbfs(tmp)

            if (tmpresult is None):
                continue

            if (result is None) or (tmpresult.distance < result.distance):
                result = tmpresult

        return self.generate_path(result)


    def generate_path(self, node):

        if (node is None):
            return None

        path = deque()
        tmp = node

        while (tmp != None):
            path.appendleft(tmp.city)
            tmp = tmp.prevNode

        return (path, node.distance)

    class Node:

        def __init__(self):
            self.id = random()
            self.city = None
            self.distance = None
            self.prevNode = None
            self.path_len = None
            self.visited = set()

        def __str__(self):
            return self.city + " " + str(self.distance)

    class Path:
        pass

def generate_graph(filename="cities.txt"):
    cities = []
    distances = {}

    parse_distances = False

    for line in open(filename):
        if (line.startswith("---------")):
            parse_distances = True
            continue
        if (parse_distances):
            city1, city2,dist = line.split()

            if (not city1 in distances):
                distances[city1] = []

            if (not city2 in distances):
                distances[city2] = []

            distances[city1].append((city2, int(dist)))
            distances[city2].append((city1, int(dist)))
        else:
            cities.append(line.strip())

    return Graph(cities, distances)

if __name__ == "__main__":
    print generate_graph().dfs()
    print generate_graph().gbfs()