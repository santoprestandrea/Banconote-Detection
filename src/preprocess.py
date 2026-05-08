"""
This function handles image preprocessing, which means all the steps
needed before extracting features.

It takes four points that represent the corners of the banknote,
but it does not know their order. Its job is to reorder them in a
consistent way: "top-left", "top-right", "bottom-right", "bottom-left".

This step is important because perspective transformation needs the
points to be in the same order every time. We also calculate the width
and height of the banknote using its sides.

Then, we define a function that finds and redefines the most likely
contour of the image. Finally, the preprocessed image is returned.

We apply these steps to both the color image and the grayscale image,
because some features work only on color while others work only on
grayscale.
"""

import cv2
import numpy as np
from config import IMAGE_SIZE

"""
    This function is used to straighten a banknote, so it requires the vertices
    of the banknote to be ordered correctly.

    To achieve this, we create a 4 x 2 matrix where each row corresponds to a vertex.
    The vertices are placed in specific positions based on their coordinates:

    - The point with the minimum sum of coordinates (x + y) will be placed
      at the top-left of the matrix.
    - The point with the maximum sum of coordinates (x + y) will be placed
      at the bottom-right, since it has the largest combined values.
    - The remaining two points are determined using the difference of coordinates (x - y):
        * The point with the minimum difference will be placed at the top-right.
        * The point with the maximum difference will be placed at the bottom-left.

    Finally, the function returns the 4 x 2 matrix containing the ordered vertices.
    """
def order_points(pts):
    matrix = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis= 1)
    matrix[0] = pts[np.argmin(s)]
    matrix[2]=pts[np.argmax(s)]

    difference = np.diff(pts,axis = 1)

    matrix[1]= pts[np.argmin(difference)]
    matrix[3]= pts[np.argmax(difference)]

    return matrix

"""
    This function performs a perspective transformation whose purpose is to
    straighten the banknote as if it were viewed from above with a scanner.

    We first compute the new height and width, and then we create a perfect rectangle
    that will be used to map the transformed image.
    """

def trasform_points(img, points):

    matrix = order_points(points)
    (tl, tr, br, bl) = matrix

    width_sup = np.linalg.norm(tr -tl)
    width_bott = np.linalg.norm(br-bl)

    width_max = max( int(width_sup),int( width_bott))

    height_left = np.linalg.norm(tl -bl)
    height_right = np.linalg.norm(tr -br)

    height_max = max(int(height_left), int(height_right))

    rectangular = np.array([
        [0,0],
        [width_max -1, 0],
        [width_max -1, height_max-1],
        [0 ,height_max-1 ]

    ], dtype="float32")

    matrix_result = cv2.getPerspectiveTransform(matrix, rectangular)
    img_trasformd = cv2.warpPerspective(img, matrix_result, (width_max, height_max))

    return img_trasformd

"""
    This function searches for the contour of the banknote.
    The goal is to find a quadrilateral, which corresponds to the banknote.
"""

def find_contur(img):

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_image,(5,5), 0)
    border = cv2.Canny(blur,50,150)
    holes_operation = np.ones((5,5), np.uint8)
    border = cv2.morphologyEx(border,cv2.MORPH_CLOSE, holes_operation)
    contours, _ = cv2.findContours(border.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea, reverse=True)

    for contour in contours:
        perimetro = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour,0.02 * perimetro, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

'''
let's build a pipeline that reads the image, 
detects the banknote via its contour, straightens it, 
and resizes it.
'''

def pre_process(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError("imagine not found")
    
    contour = find_contur(img)

    if contour is not None:
        streghten = trasform_points(img, contour)
    else:
        streghten = img.copy()
    
    resized = cv2.resize(streghten, IMAGE_SIZE)

    return resized

"""
this function handles the preprocessing of the features, 
where the output will be a normalized color image with predefined dimensions, and a grayscale image.
"""
def pre_process_features(img):

    color = cv2.resize(img, IMAGE_SIZE)
    img_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    
    return color, img_gray





