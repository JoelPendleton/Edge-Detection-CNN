import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

data = {}  # create 'data' dictonary to store json data
data['diamonds'] = []

x_image = cv2.imread('Training_Output/output_13.png')
img = cv2.bitwise_not(x_image)
x = edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

IMAGE_WIDTH = x.shape[1]
IMAGE_HEIGHT = x.shape[0]

V_SD_max = 0.1
V_G_min = 0.0
V_G_max = 0.5

V_G_per_pixel = V_G_max / IMAGE_WIDTH
V_SD_per_pixel = 2 * V_SD_max / IMAGE_HEIGHT
gradient_scaling = V_SD_per_pixel / V_G_per_pixel

# Get regions x-range where current doesn't flow when V_SD = 0 ( y =0)
y_axis_index = int((x.shape[0] / 2 - 1))  # index middle of image (vertically)
V_G_y_axis = (x[y_axis_index, :] / 255) == 1  # 0 for region where current doesn't flow


def get_intersect(a1, a2, b1, b2):
    """
    Function to find the point of intersection of two lines.
    :param a1: [x, y] a point on the first line
    :param a2: [x, y] another point on the first line
    :param b1: [x, y] a point on the second line
    :param b2: [x, y] another point on the second line
    :return: the point of intersection of the lines passing through a2, a1 and b2, b1.
    """

    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    else:
        return (x/z, y/z)


def gradient(a1, a2):
    """
    Function to calculate the gradient of a line.
    :param a1: [x, y] a point on the line
    :param a2: [x, y] another point on the  line
    :return: gradient of the line
    """

    return (a2[1] - a1[1]) / (a2[0] - a1[0])


def find_x_intercept(a1, a2):
    """
    Function to find where the line intersects the x-axis.
    :param a1: [x, y] a point on the line
    :param a2: [x, y] another point on the  line
    :return: x-intercept of line
    """
    x_1 = a1[0]
    y_1 = a1[1]

    y_mid = y_axis_index
    gradient_line = gradient(a1, a2)
    x_intercept = (y_mid - y_1) / gradient_line + x_1
    return x_intercept


def distance_between_x_intercepts(a1, a2, b1, b2):

    """
    Function to calculate the distance between the x-intercepts of 2 lines
    :param a1: [x, y] a point on the 1st line
    :param a2: [x, y] another point on the 1st line
    :param b1: [x, y] a point on the 2nd line
    :param b2: [x, y] another point on the 2nd line
    :return: the distance between the x-intercepts of the lines
    """

    x_intercept_previous = find_x_intercept(b1, b2)
    x_intercept_current = find_x_intercept(a1, a2)
    difference = abs(x_intercept_current - x_intercept_previous)

    return difference


def line_coordinates(lines, index):

    """
    Function that returns 2 points on a line found via the Hough Transform
    :param lines: numpy array of lines found from Hough Transform
    :param index: index of line to plot that matches condition
    :return: [x1, y1], [x2, y2] coordinates of plotted lines
    """

    # Iterate through rho and theta for negative lines found
    for rho, theta in lines[index]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b)) + indices[i]
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b)) + indices[i]
        y2 = int(y0 - 1000 * a)

    return [x1, y1], [x2, y2]


def line_plotter(a1, a2, b1, b2):
    """
    Function / procedure to plot a line if various conditions are met.
    :param a1: [x, y] a point on the 1st line
    :param a2: [x, y] another point on the 1st line
    :param b1: [x, y] a point on the 2nd line
    :param b2: [x, y] another point on the 2nd line
    :return: True if line is plotted
    """

    if i == 0:  # if first diamond add line
        cv2.line(img, (a1[0], a1[1]), (a2[0], a2[1]), (0, 0, 255), 2)  # draw line
        return True
    elif i >= 1:  # if not 1st diamond, need to check if line for edge has already been drawn
        difference_in_x_intercept = distance_between_x_intercepts(a1, a2, b1, b2)
        # ensures that the line you're trying to plot is a least 5 px to the right of previously drawn line
        if difference_in_x_intercept > 10:
            cv2.line(img, (a1[0], a1[1]), (a2[0], a2[1]), (0, 0, 255), 2)  # draw line
            return True


indices = []  # indices of start of each diamond / pixel start of each diamond (along y = 0)
distance_checker = int(0.025 * x.shape[0])  # distance used to distinguish between start of diamonds
line1 = []
line2 = []

for i in range(V_G_y_axis.shape[0]):

    condition1 = (V_G_y_axis[i] != V_G_y_axis[i-1])  # check pixel is different than pixel 1 to the left
    condition2 = (V_G_y_axis[i] != V_G_y_axis[i-2])  # check pixel is different than pixel 2 to the left
    # want the start diamond to be where there is at least 2 black pixels to left of current pixel
    total_condition = condition1 & condition2

    # if no indices added then 1st pixel to match this condition is start of 1st diamond
    if total_condition & (len(indices) == 0):
        indices.append(i)
    elif total_condition:  # if not first diamond

        # if matches initial condition and is a few pixels from previous index so an odd pixel is not added.
        if (i - indices[-1]) > distance_checker:
            indices.append(i)

# Initialise the coordinates of the previously positive gradient drawn line

b1_pos = [0, 0]  # the first point on the previously plotted positive grad. line
b2_pos = [0, 0]  # the second point on the previously plotted positive grad. line

# Initialise the coordinates of the previously negative gradient drawn line

b1_neg = [0, 0]  # the first point on the previously plotted negative grad. line
b2_neg = [0, 0]  # the second point on the previously plotted negative grad. line

for i in range(len(indices)-1):

    start_of_diamond = indices[i]
    end_of_diamond = indices[i+1]

    diamond = {
        'diamond number': i+1,
        'width of diamond': (end_of_diamond - start_of_diamond) * V_G_per_pixel,
        'positive line slopes': [],
        'negative line slopes': [],
        'positive line x-intercepts': [],
        'negative line x-intercepts': []
    }

    current_diamond = edges[:, start_of_diamond:end_of_diamond]  # current diamond that we're performing Hough transform on
    lines = cv2.HoughLines(current_diamond, 2, np.pi/180, 10)

    # Create conditions for negative gradient lines (theta is in range for -ve gradient)

    condition1 = lines[:, 0, 1] > 1.5 * np.pi
    condition2 = lines[:, 0, 1] < 2 * np.pi
    condition3 = lines[:, 0, 1] < np.pi
    condition4 = lines[:, 0, 1] > np.pi / 2
    condition_1_2 = condition1 & condition2
    condition_3_4 = condition3 & condition4
    total_condition = condition_1_2 | condition_3_4
    negative_line_indexes = np.argwhere(total_condition)[:, 0]  # Get indexes of lines that satisfy these conditions

    line_plotted = False

    for index in negative_line_indexes:   # for each of the possible negative slope lines
        a1, a2 = line_coordinates(lines, index)  # find two sets of coordinates on the line
        x_intercept = find_x_intercept(a1, a2)  # using coordinates find where line intersects x-axis
        if abs(x_intercept - start_of_diamond) < 10:  # if line is close to start of diamond
            line_plotted = line_plotter(a1, a2, b1_neg, b2_neg)  # plot line
            if line_plotted:  # break loop and don't plot anymore lines
                diamond['negative line slopes'].append(-gradient(a1, a2) * gradient_scaling)  # need -ve gradient since indexing of pixels starts in top-left
                diamond['negative line x-intercepts'].append(x_intercept * V_G_per_pixel)
                break

        if i >= 1:
            '''b1_pos, b2_pos describes previous positive grad. line that intersects
             the start of previous diamond. a1, a2 describes the negative slope of the line that intersects 
              the start of the current diamond'''
            intersection = get_intersect(a1, a2, b1_pos, b2_pos)
            diamond['height'] = abs(int(intersection[1] - y_axis_index)) * V_SD_per_pixel

    if i == (len(indices) - 2): # if dealing with last diamond
        for index in negative_line_indexes:  # for each of the negative slope positive lines
            a1, a2 = line_coordinates(lines, index)  # find two sets of coordinates on the line
            x_intercept = find_x_intercept(a1, a2)  # using coordinates find where line intersects x-axis
            if abs(x_intercept - end_of_diamond) < 10:  # if line is close to end of diamond
                line_plotted = line_plotter(a1, a2, b1_neg, b2_neg)  # plot line
                if line_plotted:  # break loop and don't plot anymore lines
                    diamond['negative line slopes'].append(-gradient(a1, a2) * gradient_scaling)  # need -ve gradient since indexing of pixels starts in top-left
                    diamond['negative line x-intercepts'].append(x_intercept * V_G_per_pixel)
                    c1_neg = a1 # last negative line 1st coordinate
                    c2_neg = a2
                    break

    # Update previous negative line's coordinates
    b1_neg = a1
    b2_neg = a2

    # Create conditions for positive gradient lines (theta is in range for +ve gradient)
    condition1 = lines[:, 0, 1] < np.pi / 2
    condition2 = lines[:, 0, 1] > 0.1
    condition3 = lines[:, 0, 1] < 2 * np.pi
    condition4 = lines[:, 0, 1] > np.pi
    condition_1_2 = condition1 & condition2
    condition_3_4 = condition3 & condition4
    total_condition = condition_1_2 | condition_3_4
    positive_line_indexes = np.argwhere(total_condition)[:, 0]  # Get indexes of lines that satisfy these conditions

    line_plotted = False  # True when line has been plotted.

    for index in positive_line_indexes:   # for each of the possible positive slope lines
        a1, a2 = line_coordinates(lines, index)  # find two sets of coordinates on the line
        x_intercept = find_x_intercept(a1, a2)  # using coordinates find where line intersects x-axis
        if abs(x_intercept - start_of_diamond) < 10:  # if line is close to start of diamond
            line_plotted = line_plotter(a1, a2, b1_pos, b2_pos)  # plot line
            if line_plotted:  # break loop and don't plot anymore lines
                diamond['positive line slopes'].append(-gradient(a1, a2) * gradient_scaling)  # need -ve gradient since indexing of pixels starts in top-left
                diamond['positive line x-intercepts'].append(x_intercept * V_G_per_pixel)
                break

    if i == (len(indices) - 2): # if dealing with last diamond
        intersection = get_intersect(c1_neg, c2_neg, a1, a2)
        for index in positive_line_indexes:  # for each of the positive slope  lines
            a1, a2 = line_coordinates(lines, index)  # find two sets of coordinates on the line
            x_intercept = find_x_intercept(a1, a2)  # using coordinates find where line intersects x-axis
            if abs(x_intercept - end_of_diamond) < 10:  # if line is close to end of diamond
                line_plotted = line_plotter(a1, a2, b1_pos, b2_pos)  # plot line
                if line_plotted:  # break loop and don't plot anymore lines
                    diamond['positive line slopes'].append(-gradient(a1, a2) * gradient_scaling)
                    diamond['positive line x-intercepts'].append(x_intercept * V_G_per_pixel)
                    diamond['height'] = abs(int(intersection[1] - y_axis_index)) * V_SD_per_pixel

                    break

    # Update previous positive line's coordinates
    b1_pos = a1
    b2_pos = a2
    data['diamonds'].append(diamond)

plt.imshow(img)
plt.show()

# save parameters in JSON format to txt file
with open('data.txt', 'w') as outfile:
    json.dump(data, outfile, indent=4)





