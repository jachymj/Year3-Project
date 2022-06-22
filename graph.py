import networkx as nx
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os
import tikzplotlib
from vertex import *
from PIL import Image
from edge import *
from skimage.morphology import skeletonize


class Graph:

    edges = []
    vertices = []
    single_vertex = None
    no_vertex = None
    adj = []
    nx_graph = None

    def __init__(self, image: np.ndarray, single_image, filled):
        # Initialise object attributes
        self.original_image = image
        self.original_single = single_image

        self.single = np.empty(single_image.shape)
        self.img = np.empty(image.shape)
        self.filled = filled

        # Process images
        self.process_images()

        # Find vertices
        self.find_single_vertex()

        self.find_vertices()
        self.prune_vertices()
        rgb = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        for vertex in self.vertices:
            c = vertex.show(self.img, rgb=rgb)

        cv2.imshow("Vertices", rgb)

        # Find Edges
        self.remove_vertices()
        self.find_edges()

        self.construct_adj_matrix()
        self.adj_to_nx()

    def process_images(self):
        img = self.original_image.copy()
        img[img < 220] = 0
        img[img >= 220] = 255
        # img = cv2.dilate(img, np.ones((3, 3)))
        # img = cv2.erode(img, np.ones((3, 3)))

        single_img = self.original_single.copy()
        single_img[single_img < 220] = 0
        single_img[single_img >= 220] = 255
        # single_img = cv2.dilate(single_img, np.ones((3, 3)))
        # single_img = cv2.erode(single_img, np.ones((3, 3)))

        self.img = img
        self.single = single_img
        self.no_vertex = img.copy()

    def find_single_vertex(self):
        w, h = self.single.shape
        max_r = round(min(w / 2, h / 2))
        min_dist = max(w, h)

        c = cv2.HoughCircles(self.single, method=cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=90, param2=3.1,
                             minRadius=1, maxRadius=max_r)

        c = c.astype(int)

        x, y, r = c[0, 0]

        if self.filled:
            self.single_vertex = FilledVertex(x, y, r)
        else:
            self.single_vertex = EmptyVertex(x, y, r)

    def find_vertices(self):
        rad = self.single_vertex.r
        max_r = rad + (1 + round(0.05 * rad))
        min_r = rad - (1 + round(0.05 * rad))

        d = 2.1 * rad

        c = cv2.HoughCircles(self.img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=d, param1=90, param2=3.1, minRadius=min_r,
                             maxRadius=max_r)
        c = c.astype(int)

        for (x, y, r) in c[0]:
            if self.filled:
                vertex = FilledVertex(x, y, r)
            else:
                vertex = EmptyVertex(x, y, r)
            self.vertices.append(vertex)

    def prune_vertices(self):
        pruned = []
        for vertex in self.vertices:
            if vertex.is_valid(self.img):
                pruned.append(vertex)
        self.vertices = pruned

    def remove_vertices(self):
        for vertex in self.vertices:
            x, y, r = vertex.get_data()
            self.no_vertex = cv2.circle(self.no_vertex, (x, y), round(r*1.3), 255, -1)

    def find_edges(self):
        img = self.no_vertex.copy()
        img[self.no_vertex > 0] = 0
        img[self.no_vertex == 0] = 1

        skel = skeletonize(img)
        skel = skel.astype(int)

        img[skel == 0] = 0
        img[skel > 0] = 255

        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            coordinates = c[:, 0, :]
            edge = Edge(coordinates)
            self.edges.append(edge)

    def add_edge(self, source: int, target: int):
        source_vertex = self.vertices[source]
        target_vertex = self.vertices[target]

        # Create new edge and add it to the edges array
        edge = Edge([[source_vertex.x, source_vertex.y], [target_vertex.x, target_vertex.y]])
        # edge.recompute_coordinates(source_vertex, target_vertex)
        self.edges.append(edge)

        # Add new edge to the graph
        self.adj[source, target] += 1
        self.adj[target, source] += 1
        self.nx_graph.add_edge(source, target)

    def construct_adj_matrix(self):
        accurate_edges = []
        num_vertices = len(self.vertices)
        adj = np.zeros((num_vertices, num_vertices))
        for edge in self.edges:
            closeness = edge.closeness(self.vertices)

            if (closeness < 2).sum() == 2:
                v1, v2 = np.argsort(closeness)[:2]

                adj[v1, v2] += 1
                adj[v2, v1] += 1

                accurate_edges.append(edge)
            else:
                subset = np.where(closeness < 2)
                print(len(subset[0]))
                temp, new_edges = self.overlapping_edges(edge)
                adj = adj + temp
                for e in new_edges:
                    accurate_edges.append(e)

        self.edges = accurate_edges
        self.adj = adj

    def overlapping_edges(self, edge):
        new_edges_coordinates = edge.disjoin(self.img)
        adj_temp = adj = np.zeros((len(self.vertices), len(self.vertices)))
        non_duplicate_edges = []
        for new_edge_coordinates in new_edges_coordinates:
            new_edge = Edge(new_edge_coordinates)
            closeness = new_edge.closeness(self.vertices)
            v1, v2 = np.argsort(closeness)[:2]

            if adj_temp[v1, v2] == 0:
                adj[v1, v2] = 1
                adj[v2, v1] = 1
                new_edge.recompute_coordinates(self.vertices[v1], self.vertices[v2])
                non_duplicate_edges.append(new_edge)

        return adj_temp, non_duplicate_edges

    def construct_nx(self):
        graph = nx.MultiGraph()
        # Add vertices

        for i in range(len(self.vertices)):
            v = self.vertices[i]
            graph.add_node(i, pos=(v.x, v.y))
        for edge in self.edges:
            closeness = edge.closeness(self.vertices)
            v1, v2 = np.argsort(closeness)[:2]

            graph.add_edge(v1, v2)

        self.nx_graph = graph

    def adj_to_nx(self):
        graph = nx.MultiGraph()
        # Add vertices
        for i in range(len(self.vertices)):
            graph.add_node(i)

        for i in range(len(self.adj)):
            for j in range(i, len(self.adj[i])):
                num_edges = int(self.adj[i, j])
                for _ in range(0, num_edges):
                    graph.add_edge(i, j)

        self.nx_graph = graph

    def add_vertex(self):
        # Random x coordinate
        possible_x = np.arange(0, self.img.shape[1])
        possible_y = np.arange(0, self.img.shape[0])
        for v in self.vertices:
            x, y, r = v.get_data()
            for i in range(x-r-5, x+r+5):
                try:
                    possible_x = np.delete(possible_x, i)
                except IndexError:
                    pass
            for i in range(y-r-5, y+r+5):
                try:
                    possible_y = np.delete(possible_y, i)
                except IndexError:
                    pass

        # Choose random position
        try:
            x = np.random.choice(possible_x)
            y = np.random.choice(possible_y)
        except ValueError:
            return False
        vertex = FilledVertex(x, y, self.single_vertex.r)
        self.vertices = np.append(self.vertices, [vertex], axis=0)
        # self.nx_graph.add_node(len(self.vertices))
        self.update_adj()
        self.adj_to_nx()
        return True

    def update_adj(self):
        new_matrix = np.zeros((len(self.vertices), len(self.vertices)), dtype=int)
        for i in range(len(self.adj)):
            for j in range(len(self.adj[i])):
                new_matrix[i, j] = self.adj[i, j]

        self.adj = new_matrix

    def show_graph(self):
        rgb = cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2RGB)

        for i in range(len(self.edges)):
            edge = self.edges[i]
            rgb = edge.plot_edge_rgb(rgb)

        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            cv2.circle(rgb, (vertex.x, vertex.y), vertex.r, (255, 0, 0), -1)
            cv2.putText(rgb, str(i), (vertex.x - 10, vertex.y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)

        cv2.imshow("Graph Plot", rgb)

    def show_vertices(self):
        rgb = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        for vertex in self.vertices:
            cv2.circle(rgb, (vertex.x, vertex.y), vertex.r, (255, 0, 0), 3)

        cv2.imshow("Vertices", rgb)

    def show_nx(self):
        G = self.nx_graph
        pos = {}
        for i in range(len(self.vertices)):
            v = self.vertices[i]
            pos[i] = (v.x, v.y * -1)

        nx.draw_networkx_nodes(G, pos, node_color='r', node_size=400, alpha=1)
        nx.draw_networkx_labels(G, pos)
        ax = plt.gca()
        for e in G.edges:
            ax.annotate("",
                        xy=pos[e[0]], xycoords='data',
                        xytext=pos[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="-", color="0.5",
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])),
                                        ),
                        )
        plt.axis('off')
        try:
            os.mkdir("exports", 0o666)
        except FileExistsError:
            pass
        tikzplotlib.save("exports/graph.tex")
        plt.show()

    def show_edges(self):
        empty = cv2.cvtColor(self.img.copy(), cv2.COLOR_GRAY2BGR)
        for i in range(len(self.edges)):
            edge = self.edges[i]
            empty = edge.plot_edge_rgb(empty)
        cv2.imshow("Edges", empty)

    def export(self):
        df = pd.DataFrame(self.adj)
        try:
            os.mkdir("exports", 0o666)
        except FileExistsError:
            pass
        df.to_csv("exports/adjacency.csv")

    def remove_vertex(self, vertexID):

        if vertexID >= len(self.vertices):
            return None

        if vertexID < 0:
            return None

        new_vertices = np.delete(self.vertices, vertexID, axis=0)
        self.vertices = new_vertices

        new_adj = np.delete(self.adj, vertexID, axis=1)
        new_adj = np.delete(new_adj, vertexID, axis=0)
        self.adj = new_adj

        self.adj_to_nx()

    def remove_edge(self, source, target):
        if source > len(self.vertices) or target > len(self.vertices):
            return None
        if source < 0 or target < 0:
            return None

        if self.adj[source, target] > 0:
            self.adj[source, target] -= 1

        if self.adj[target, source] > 0:
            self.adj[target, source] -= 1

        self.adj_to_nx()


def load_images(path_to_image):
    img = Image.open(path_to_image).convert("L")
    img_array = np.asarray(img)

    return img, img_array