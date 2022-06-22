import numpy as np
from abc import ABC, abstractmethod
import cv2
import math


class Vertex(ABC):

    filled = None

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    @abstractmethod
    def calculate_accuracy(self, image):
        pass

    def get_data(self):
        return self.x, self.y, self.r

    def contains(self, x, y):
        s = ((x - self.x) ** 2) + ((y - self.y) ** 2)
        # if s <= self.r ** 2:
        #     return True
        # else:
        #     return False

        # print(self.get_data())
        return s - (round(self.r * 1.4) ** 2)

    def distance(self, x, y):
        dist = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        return dist/self.r

    def show(self, img, rgb=None):
        if rgb is None:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        cv2.circle(rgb, (self.x, self.y), self.r, (255, 0, 0), 3)

        return rgb


class FilledVertex(Vertex):

    def __init_subclass__(cls, **kwargs):
        cls.filled = True

    def calculate_accuracy(self, image):
        y = self.y
        x = self.x
        r = self.r

        h, w = image.shape

        min_x = max(0, x-r)
        max_x = min(w, x+r)

        min_y = max(0, y-r)
        max_y = min(h, y+r)

        count = 0
        total_count = 0

        for i in range(min_y, max_y):
            for j in range(min_x, max_x):

                if image[i, j] == 0:
                    count += 1

                total_count += 1
        if total_count == 0:
            return 0
        return count/total_count

    def is_valid(self, image):
        if self.calculate_accuracy(image) > 0.7:
            return True
        else:
            return False


class EmptyVertex(Vertex):

    def __init_subclass__(cls, **kwargs):
        cls.filled = False

    def calculate_accuracy(self, img):
        # Assuming the center of a "correct" circle is within the vertex
        # Recompute the center of the circle and then check that diameters match

        up = self.distance_to_pixel(img, (0, -1))
        down = self.distance_to_pixel(img, (0, 1))

        left = self.distance_to_pixel(img, (-1, 0))
        right = self.distance_to_pixel(img, (1, 0))

        diameter_x = left + right
        diameter_y = up + down

        rad = round(max(diameter_y, diameter_x) / 2)

        # If radius is too big ignore it
        if rad > 2 * self.r:
            rad = -1

        diff = abs(diameter_x - diameter_y) / (2 * rad) if rad > 0 else 1000

        return diff

    def distance_to_pixel(self, img: np.ndarray, vector, c=255):
        i, j = vector
        x, y = self.x, self.y

        dist = 0

        try:
            while img[y][x] == c:
                y += j
                x += i

                dist += 1
        except IndexError:
            return dist

        return dist

    def is_valid(self, image):
        print(self.calculate_accuracy(image))
        if self.calculate_accuracy(image) < 0.05:
            return True
        else:
            return False
