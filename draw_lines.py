import numpy as np
import matplotlib.pyplot as plt
import cv2

x_image = cv2.imread('Training_Output/output_10.png')
img = cv2.bitwise_not(x_image)
x = edges = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

IMAGE_WIDTH = x.shape[1]
IMAGE_HEIGHT = x.shape[0]

# Get regions x-range where current doesn't flow when V_SD = 0 ( y =0)
y_axis_index = int((x.shape[0] / 2 - 1))  # index middle of image (vertically)
V_G_y_axis = (x[y_axis_index, :] / 255) == 1  # 0 for region where current doesn't flow


def gradient(x_1, y_1, x_2, y_2):
    """
    Function to calculate the gradient of a line
    :param x_1: 1st x coordinate on 1st line
    :param y_1:
    :param x_2:
    :param y_2:
    :return: gradient of the line
    """
    return (y_2 - y_1) / (x_2 - x_1)


def distance_between_x_intercepts(line1_x1, line1_x2, line1_y1, line1_y2, line2_x1, line2_x2, line2_y1, line2_y2):
    """
    Function to calculate the distance between the x-intercepts of 2 lines
    :param line1_x1: x coordinate of 1st point on 1st line
    :param line1_x2: x coordinate of 2nd point on 1st line
    :param line1_y1: y coordinate of 1st point on 1st line
    :param line1_y2: y coordinate of 2nd point on 1st line
    :param line2_x1: x coordinate of 1st point on 2nd line
    :param line2_x2: x coordinate of 2nd point on 2nd line
    :param line2_y1: y coordinate of 1st point on 2nd line
    :param line2_y2: y coordinate of 2nd point on 2nd line
    :return: the distance between the x-intercepts of the lines
    """

    y_mid = y_axis_index
    gradient_previous = gradient(line2_x1, line2_y1, line2_x2, line2_y2)
    x_intercept_previous = (y_mid - line2_y1) / gradient_previous + line2_x1
    gradient_current = gradient(line1_x1, line1_y1, line1_x2, line1_y2)
    x_intercept_current = (y_mid - line1_y1) / gradient_current + line1_x1
    difference = abs(x_intercept_current - x_intercept_previous)

    return difference


def line_plotter(lines, index, x1_prev , x2_prev, y1_prev, y2_prev):
    """
    Procedure and function to plot lines for given index that was found from some conditions
    :param lines: numpy array of lines found from Hough Transform
    :param index: index of line to plot that matches condition
    :return: x1, y1, x2, y2 coordinates of plotted lines
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

    if i == 0:  # if first diamond add line
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # draw line
        print(lines[index])
    elif i >= 1:  # if not 1st diamond, need to check if line for edge has already been drawn
        difference_in_x_intercept = distance_between_x_intercepts(x1, x2, y1, y2, x1_prev,
                                                                  x2_prev, y1_prev, y2_prev)
        # ensures that the line you're trying to plot is a least 5 px to the right of previously drawn line
        if difference_in_x_intercept > 10:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # draw line

    return x1, y1, x2, y2


indices = []  # indices of start of each diamond / pixel start of each diamond (along y = 0)
distance_checker = int(0.025 * x.shape[0])  # distance used to distinguish between start of diamonds

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

#indices.append(IMAGE_WIDTH-1)  # so it includes all visible diamonds in for loop used further down

print("Start pixels of diamonds along x-axis:", indices)

# Initialise the coordinates of the previously positive gradient drawn line
x1_pos_prev = y1_pos_prev = x2_pos_prev = y2_pos_prev = 0

# Initialise the coordinates of the previously negative gradient drawn line
x1_neg_prev = y1_neg_prev = x2_neg_prev = y2_neg_prev = 0

for i in range(len(indices)-1):
    current_diamond = edges[:, indices[i]:indices[i+1]]  # current diamond that we're performing Hough transform on
    lines = cv2.HoughLines(current_diamond, 2, np.pi/180, 20)

    # Create conditions for negative gradient lines (theta is in range for -ve gradient)

    condition1 = lines[:, 0, 1] > 1.5 * np.pi
    condition2 = lines[:, 0, 1] < 2 * np.pi
    condition3 = lines[:, 0, 1] < np.pi
    condition4 = lines[:, 0, 1] > np.pi / 2
    condition_1_2 = condition1 & condition2
    condition_3_4 = condition3 & condition4
    total_condition = condition_1_2 | condition_3_4
    negative_line_index = np.argwhere(total_condition)[0, 0]  # Get index of 1st line that satisfies these conditions

    # plot -ve gradient lines
    x1, y1, x2, y2 = line_plotter(lines, negative_line_index, x1_neg_prev, x2_neg_prev, y1_neg_prev,  y2_neg_prev)

    # Update previous positive line's coordinates
    x1_neg_prev = x1
    y1_neg_prev = y1
    x2_neg_prev = x2
    y2_neg_prev = y2

    # Create conditions for positive gradient lines (theta is in range for +ve gradient)
    condition1 = lines[:, 0, 1] < np.pi/2
    condition2 = lines[:, 0, 1] > 0.1
    condition3 = lines[:, 0, 1] < 2 * np.pi
    condition4 = lines[:, 0, 1] > np.pi
    condition_1_2 = condition1 & condition2
    condition_3_4 = condition3 & condition4
    total_condition = condition_1_2 | condition_3_4
    positive_line_index = np.argwhere(total_condition)[0,0] # Get index of 1st line that satisfies these conditions

    # plot +ve gradient lines
    x1, y1, x2, y2 = line_plotter(lines, positive_line_index, x1_pos_prev, x2_pos_prev, y1_pos_prev,  y2_pos_prev)

    # Update previous positive line's coordinates
    x1_pos_prev = x1
    y1_pos_prev = y1
    x2_pos_prev = x2
    y2_pos_prev = y2

plt.imshow(img)
plt.show()
