import cv2
import numpy as np


def show_lines(image, lines):
    w, h = image.shape
    max_dim = max(w, h)
    max_dim = w * h

    img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    count = 0
    for r_theta in lines:
        # print(f'r_theta: {r_theta[0], r_theta[1]}')
        r, theta = r_theta[0]
        count = count + 1

        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + max_dim * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + max_dim * a)

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - max_dim * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - max_dim * a)

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Edges", img)


def get_line_coordinates(r, theta, max_dim=1000):
    line_coordinates = []
    for i in range(-max_dim, max_dim):
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x = int(x0 + i * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y = int(y0 + i * a)

        line_coordinates.append([x, y])
    return line_coordinates


class Edge:

    def __init__(self, coordinates):
        self.coordinates = coordinates

    def closeness(self, vertices):
        c = np.empty(len(vertices))
        for i in range(len(vertices)):
            vertex = vertices[i]
            v = min([vertex.distance(x, y) for x, y in self.coordinates])
            c[i] = v

        return c

    def disjoin(self, image):
        # plot the edge to be disjoined
        edge_plot = self.plot_edge(np.zeros(image.shape))
        e = edge_plot.astype(np.uint8)
        new_edges = []

        # This returns an array of r and theta values
        lines = cv2.HoughLines(e, 1, np.pi / 180, 40)
        # show_lines(image, lines)

        for r_theta in lines:
            r, theta = r_theta[0]
            coordinates = get_line_coordinates(r, theta, max(image.shape))
            new_edges.append(coordinates)

        return new_edges

    def plot_edge_rgb(self, canvas):
        for (x, y) in self.coordinates:
            # canvas[y, x] = (0, 0, 255)
            canvas = cv2.rectangle(canvas, (x, y), (x, y), (0, 0, 255), 10)
        return canvas

    def plot_edge(self, canvas):
        for (x, y) in self.coordinates:
            canvas[y, x] = 255
        return canvas

    def recompute_coordinates(self, v1, v2):
        x1, y1 = v1.x, v1.y
        x2, y2 = v2.x, v2.y
        new_coordinates = [[v1.x, v1.y], [v2.x, v2.y]]
        # gradient = (y1-y2)/(x1-x2)
        # c = (x1*y2 - x2*y1)/(x1-x2)
        #
        # step = 1 if x1 < x2 else -1
        #
        # for i in range(x1, x2, step):
        #     y_coordinate = gradient * i + c
        #     new_coordinates.append([i, round(y_coordinate)])

        self.coordinates = new_coordinates
