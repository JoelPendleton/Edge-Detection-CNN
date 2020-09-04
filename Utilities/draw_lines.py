import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
img = cv2.imread('../Data/Real_Data/Input/output_1.png')

y_axis_index = int((img.shape[0] / 2 - 1))  # index middle of image (vertically)


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

V_SD_max = 0.2
V_G_min = 0.005
V_G_max = 1.2

line_data = {}  # create 'data' dictonary to store json data
line_data['positive lines'] = []
line_data['negative lines'] = []


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert image to grayscale
edges = cv2.Canny(gray,50,150,apertureSize = 3) # detect the edges of the image

lines = cv2.HoughLines(edges,1,np.pi/180,40) # find the lines to fit the data using Hough Transform


positive_x_intercepts = []
negative_x_intercepts = []


# Function to insert element so it's in a sorted position
def insert(list, n):
    """
    Function to insert item into numerically ascending list.
    :param list: list to insert into
    :param n: the item to be inserted
    :return: the sorted list, and the index i where the item was inserted.
    """
    # Searching for the position
    if len(list) > 0:
        for i in range(len(list)):
            if list[i] > n:
                index = i
                break
        list = list[:i] + [n] + list[i:]
    else:
        list.append(n)
        i = 0

    # Inserting n in the list

    return list, i

def position_checker(x_intercept, x_intercepts_list):
    """
    Function to check whether next line to be plotted is too close to existing lines.
    :param x_intercept: current x-intercept to be tested
    :param x_intercepts_list: positive or negative sloped lines to check against
    :return: False if too close to line. If not too close to existing lines return True
    """
    for intercept in x_intercepts_list: # for each of the line in list
        if abs(x_intercept - intercept) < 20: # is the new line within 20px of this line's x-intercept
            return False
        else:
            continue
    return True


for line in lines:  # for each of the lines found through the hough transform
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # Get two coordinates on the line
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if x1 != x2: # ensures gradient is not undefined
            x_intercept = find_x_intercept([x1,y1], [x2,y2])

            slope = -gradient([x1,y1], [x2,y2])

            line = {
                "x-intercept": x_intercept,
                "gradient": slope,
                "coordinate 1": [x1,y1],
                "coordinate 2": [x2,y2]
            }

            if slope > 0:  # positive gradient

                if len(positive_x_intercepts) == 0:  # first positive line to be drawn

                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

                    # insert coordinate into sorted position
                    positive_x_intercepts, i = insert(positive_x_intercepts, x_intercept)

                    # insert line data into sorted position
                    line_data['positive lines'] = line_data['positive lines'][:i] + [line] + line_data['positive lines'][i:]

                # not first line so check if close to any other lines plotted
                elif position_checker(x_intercept, positive_x_intercepts):

                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

                    # insert coordinate into sorted position
                    positive_x_intercepts, i = insert(positive_x_intercepts, x_intercept)

                    # insert line data into sorted position
                    line_data['positive lines'] = line_data['positive lines'][:i] + [line] + line_data['positive lines'][i:]

            elif slope < 0: # negative gradient

                if len(negative_x_intercepts) == 0: # first negative line to be drawn

                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # insert coordinate into sorted position
                    negative_x_intercepts, i = insert(negative_x_intercepts, x_intercept)

                    # insert line data into sorted position
                    line_data['negative lines'] = line_data['negative lines'][:i] + [line] + line_data['negative lines'][i:]

                # not first line so check if close to any other lines plotted
                elif position_checker(x_intercept, negative_x_intercepts):

                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # insert coordinate into sorted position
                    negative_x_intercepts, i = insert(negative_x_intercepts, x_intercept)

                    # insert line data into sorted position
                    line_data['negative lines'] = line_data['negative lines'][:i] + [line] + line_data['negative lines'][i:]

    cv2.imwrite('houghlines.jpg',img)


# Find the heights of the diamonds by looking at intersection of positive and negative sloped lines
diamond_data = {}

# Find widths of diamonds by looking at line intercepts

# save parameters in JSON format to txt file
with open('line-data.txt', 'w') as outfile:
    json.dump(line_data, outfile, indent=4)
