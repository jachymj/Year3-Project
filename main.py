import numpy as np
import os
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.draw import line as get_line_pixels
import networkx

import matplotlib.pyplot as plt
from skimage.transform import probabilistic_hough_line


def load_images(path_to_image):
    img = Image.open(path_to_image).convert("L")
    img_array = np.asarray(img)

    return img, img_array


def process_image(img: np.ndarray):
    img[img < 220] = 0
    img[img >= 220] = 255
    img = cv2.dilate(img, np.ones((3, 3)))
    img = cv2.erode(img, np.ones((3, 3)))

    return img


def find_vertices(img: np.ndarray, r):
    max_r = r + (1 + round(0.05 * r))
    min_r = r - (1 + round(0.05 * r))

    d = 2.1 * r

    c = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=d, param1=90, param2=3.1, minRadius=min_r,
                         maxRadius=max_r)
    c = c.astype(int)

    return c


def find_single_vertex(img: np.ndarray):
    w = img.shape[0]
    h = img.shape[1]

    max_r = round(min(w / 2, h / 2))
    min_dist = round(min(w, h))

    c = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=90, param2=3.1, minRadius=1,
                         maxRadius=max_r)

    c = c.astype(int)

    return c


def find_edges(img: np.ndarray):
    edged = cv2.Canny(img, 30, 200)
    c = img.copy()

    c[img > 0] = 0
    c[img == 0] = 1

    skel = skeletonize(c)

    plt.figure()

    plt.imshow(skel)

    plt.show()

    poi = []
    for i in range(skel.shape[0]):
        for j in range(skel.shape[1]):

            if skel[i, j] == 1:
                # check how many filled neighbours pixel the pixel has
                count = 0
                for x in range(-2, 2, 1):
                    for y in range(-2, 2, 1):
                        if skel[i+x, j+y] == 1:
                            count += 1

                if count > 4:
                    print(f'{j, i} has {count} neighbours')
                    c[i-5:i+5, j-5:j+5] = 0

    cv2.imshow("edged", edged)
    empty = np.zeros(img.shape)

    contours, hierarchy = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = probabilistic_hough_line(skel, line_length=10)

    print(lines)

    """
    cleaned = np.zeros_like(img)
    for ((r0, c0), (r1, c1)) in lines:
        rr, cc = get_line_pixels(r0, c0, r1, c1)
        cleaned[rr, cc] = 1

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Raw')
    axes[1].imshow(skel, cmap='gray')
    axes[1].set_title('Skeleton')
    axes[2].imshow(cleaned, cmap='gray')
    axes[2].set_title('Hough lines')
    plt.show()
    """

    print(f'Number of contours found: {len(contours)}')

    cv2.drawContours(empty, contours, -1, 255, 3)

    cv2.imshow('Contours', empty)

    return 0


def plot_circles(img: np.ndarray, c, t=2):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for (x, y, r) in c[0]:
        cv2.circle(rgb, (x, y), r, (255, 0, 0), t)
        cv2.circle(rgb, (x, y), 1, (255, 0, 0), t)

    return rgb


def delete_false_positives(img: np.ndarray, c, filled=True):
    c_helper = []

    if filled:
        for (x, y, r) in c[0]:
            if circle_accuracy_filled(img, (x, y), r) > 0.8:
                c_helper.append([x, y, r])
    else:
        for (x, y, r) in c[0]:
            diff = circle_accuracy_empty(img, (x, y), r)
            if diff < 0.1:
                c_helper.append([x, y, r])

    return np.array([c_helper])


# Circle accuracy function for "filled" vertices
def circle_accuracy_filled(img: np.ndarray, pos: tuple, r):
    x, y = pos

    count_pixels = 0
    count_black = 0

    for j in range(y - r, y + r):
        for i in range(x - r, x + r):
            # Check if pixel is within circle
            if ((j - y) ** 2) + ((i - x) ** 2) <= r ** 2:
                count_pixels += 1

                try:
                    if img[j][i] == 0:
                        count_black += 1

                except IndexError:
                    count_black += 0

    return count_black / count_pixels if count_pixels > 0 else 0


def circle_accuracy_empty(img: np.ndarray, pos: tuple, r):
    # Assuming the center of a "correct" circle is within the vertex
    # Recompute the center of the circle and then check that diameters match

    up = distance_to_pixel(img, pos, (0, -1))
    down = distance_to_pixel(img, pos, (0, 1))

    left = distance_to_pixel(img, pos, (-1, 0))
    right = distance_to_pixel(img, pos, (1, 0))

    diameter_x = left + right
    diameter_y = up + down

    rad = round(max(diameter_y, diameter_x) / 2)

    # If radius is too big ignore it
    if rad > 2 * r:
        rad = -1

    diff = abs(diameter_x - diameter_y) / (2 * rad) if rad > 0 else 1000

    return diff


def distance_to_pixel(img: np.ndarray, pos, vector, c=255):
    i, j = vector
    x, y = pos

    dist = 0

    try:
        while img[y][x] == c:
            y += j
            x += i

            dist += 1
    except IndexError:
        return dist

    return dist


def filled_distances(img: np.ndarray, pos, vector):
    i, j = vector
    x, y = pos

    distance_array = []

    try:
        while img[y][x] == 0:
            orthogonal = np.array([j, i])

            d1 = distance_to_pixel(img, (x, y), orthogonal, c=0)
            d2 = distance_to_pixel(img, (x, y), orthogonal * -1, c=0)

            distance_array.append(d1 + d2)

            y += j
            x += i

    except IndexError:
        return distance_array

    return distance_array


def recenter_single_circle(img, pos, filled=False):
    x, y = pos

    c = 255

    if filled:

        if img[y, x] != 0:
            return -1, -1

        h1 = np.array(filled_distances(img, pos, [0, 1]))
        h2 = np.array(filled_distances(img, pos, [0, -1]))
        v1 = np.array(filled_distances(img, pos, [1, 0]))
        v2 = np.array(filled_distances(img, pos, [-1, 0]))

        horizontal_distances = np.concatenate([h2[::-1], h1])
        vertical_distances = np.concatenate([v2[::-1], v1])

        original_h_index = len(h2)
        original_v_index = len(v2)

        v_improved, v_rem = improve_distance_array(vertical_distances, original_v_index)
        h_improved, h_rem = improve_distance_array(horizontal_distances, original_h_index)

        if v_rem == original_v_index or h_rem == original_h_index:
            return -1, -1

        v_offset = round(len(v_improved) / 2) - (original_v_index - v_rem)
        h_offset = round(len(h_improved) / 2) - (original_h_index - h_rem)

        return x+v_offset, y+h_offset

    else:

        up = distance_to_pixel(img, pos, (0, -1), c)
        down = distance_to_pixel(img, pos, (0, 1), c)

        left = distance_to_pixel(img, pos, (-1, 0), c)
        right = distance_to_pixel(img, pos, (1, 0), c)

        new_x = x + round((right - left) / 2)
        new_y = y + round((down - up) / 2)

        return new_x, new_y


def recenter_circles(img, c, filled=False):
    rec = []

    for (x, y, r) in c[0]:
        new_x, new_y = recenter_single_circle(img, (x, y), filled)

        if new_x != -1:
            if filled:
                r = recalculate_radius(img, (new_x, new_y), r)

            rec.append([new_x, new_y, r])

    return np.array([rec])


def remove_circles(img: np.ndarray, circles):
    no_circles = np.copy(img)

    for (x, y, r) in circles[0]:
        r += round(0.3 * r)
        no_circles = cv2.circle(no_circles, (x, y), r, 255, -1)

    return no_circles


def get_corners(img: np.ndarray):
    c = np.copy(img)

    dst = cv2.cornerHarris(c, 2, 5, 0.07)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    c[dst > 0.01 * dst.max()] = 155

    indexes = np.where(c == 155)

    print(indexes)

    cv2.imshow("Corners", c)

    return 0


def remove_endpoints(img, corners, circles):
    for corner in corners:
        print(corner)

    return 0


def improve_distance_array(arr, index):
    a = arr.copy()

    improved, b1 = remove_tails(a, index)

    return improved, b1


def remove_tails(arr: np.ndarray, index):

    minimum = sorted(set(arr))[:3]

    # Loop through both sides from index and remove the parts of the array that follow minimum

    id_right = len(arr) - 1
    for i in range(index, len(arr)):

        if arr[i] in minimum:
            id_right = i
            break

    id_left = 0
    for i in range(index, 0, -1):

        if arr[i] in minimum:
            id_left = i
            break

    return arr[id_left:id_right], id_left


def recalculate_radius(img: np.ndarray, pos, r):

    # find accuracies of circles with radius 1 larger and 1 smaller than current
    current = circle_accuracy_filled(img, pos, r)

    if current == 0:
        return 0

    # if current doesnt satisfy 90% recalculate smaller radii
    if current < 0.9:
        r = r-1
        while circle_accuracy_filled(img, pos, r) < 0.8:
            r = r-1

            if r == 1:
                break

        return r

    else:
        larger = circle_accuracy_filled(img, pos, r)
        while larger > 0.95:
            r = r+1
            larger = circle_accuracy_filled(img, pos, r)

        return r-1


if __name__ == '__main__':
    path_to_img = "imgs/custom/custom_rect.png"
    path_to_single = "imgs/custom/custom_rect_single.png"
    f = True

    image, image_array = load_images(path_to_img)
    single, single_array = load_images(path_to_single)

    # circle = [[[404, 179, 18]]]
    #
    # cv2.imshow("Single", plot_circles(process_image(image_array), circle))
    #
    # recentered = recenter_circles(process_image(image_array), circle, filled=f)
    #
    # cv2.imshow("Rec", plot_circles(process_image(image_array), recentered))

    single_circle = find_single_vertex(process_image(single_array))

    # cv2.imshow("Single", plot_circles(single_array, single_circle))

    rec_single = recenter_circles(single_array, single_circle, filled=f)

    # cv2.imshow("Single Rec", plot_circles(single_array, rec_single))

    radius = rec_single[0, 0, 2]

    all_circles = find_vertices(process_image(image_array), radius)

    # cv2.imshow("All", plot_circles(image_array, all_circles))

    rec_centres = recenter_circles(image_array, all_circles, filled=f)

    better_circles = delete_false_positives(process_image(image_array), all_circles, filled=f)
    # cv2.imshow("Better", plot_circles(image_array, better_circles))

    image_no_circles = remove_circles(image_array, better_circles)

    cv2.imshow("No circles", image_no_circles)
    edges = find_edges(image_no_circles)

    # corners = get_corners(image_no_circles)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
