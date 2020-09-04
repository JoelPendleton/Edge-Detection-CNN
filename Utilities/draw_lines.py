import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
img = cv2.imread('../Data/Real_Data/Output/output_1.png')

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


V_SD_max = 0.2
V_G_min = 0.005
V_G_max = 1.2

data = {}  # create 'data' dictonary to store json data
data['diamonds'] = []

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,50)

positive_x_intercepts = []
negative_x_intercepts = []

def position_checker(x_intercept, x_intercepts_list):
    for intercept in x_intercepts_list:
        if abs(x_intercept - intercept) < 20:
            return False
        else:
            continue
    return True

for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        x_intercept = find_x_intercept([x1,y1], [x2,y2])
        slope = -gradient([x1,y1], [x2,y2])
        if slope > 0:

            if len(positive_x_intercepts) == 0:
                positive_x_intercepts.append(x_intercept)
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

            elif position_checker(x_intercept, positive_x_intercepts):
                positive_x_intercepts.append(x_intercept)

                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        elif slope < 0:

            if len(negative_x_intercepts) == 0:
                negative_x_intercepts.append(x_intercept)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            elif position_checker(x_intercept, negative_x_intercepts):
                negative_x_intercepts.append(x_intercept)

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('houghlines3.jpg',img)

