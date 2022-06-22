import argparse

import cv2
import numpy as np

from graph import *
from edge import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-i", dest="image", type=str, help="File path to image of graph")
parser.add_argument("-s", dest="single", type=str, help="File path to image of single vertex of graph")
parser.add_argument("-f", dest="filled", help="True if vertices in the graph are filled, False otherwise.\
                                              Default: False", default=False, action="store_true")


if __name__ == '__main__':
    args = parser.parse_args()
    print(args.image, args.filled)

    image, image_array = load_images(args.image)
    single, single_array = load_images(args.single)

    g1 = Graph(image_array, single_array, args.filled)

    # while True:
    #     inp = input('Add edge: ')
    #     if inp == "-1":
    #         cv2.destroyAllWindows()
    #         break
    #     else:
    #         try:
    #             v1, v2 = inp.split(" ")
    #             g1.add_edge(int(v1), int(v2))
    #             g1.show_edges()
    #         except ValueError:
    #             print("use format *source* *space* *target*")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
