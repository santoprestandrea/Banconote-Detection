"""
Code needed to modify the images contained within the 'train genuine' folder.
First, we will apply a blur using Gaussian blur, and then add Gaussian noise
through a standard deviation. We will change the brightness of the image using
the HSV algorithm, and then we will compress the image to simulate data loss.
To do this, we will need to process the dataset and iterate through all the
files inside the folder.
"""

import numpy as np
import cv2 
import os

input_dir = "data/train/genuine"
output_dir = "data/train/fake"

os.makedirs(output_dir, exist_ok=True)

def add_noise(img):
    """
We generate noise with a normal distribution using the mean and standard deviation parameters.
Subsequently, we convert everything to uint8, which is simply the standard unit used to represent pixel values.
"""
    noise = np.random.normal(0,40,img.shape).astype(np.uint8)
    return cv2.add(img,noise)   
"""
We apply a blur filter: the larger the kernel size, the more blurred the image will be.
"""
def blur(img):
    return cv2.GaussianBlur(img,(7,7),0)

"""
We change the brightness of the image by converting it from BGR to HSV.
HSV separates the color, saturation, and brightness of the image.
We will only modify the brightness component.
"""
def change_brightness(img):
    value = np.random.randint(-70, 70)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.add(hsv[:,:,2], value)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

"""
We simulate compression and data loss using the imencode function,
which compresses the image in memory, and the imdecode function,
which reconstructs the image. This is useful for simulating saved
or compressed images.
"""

def compress(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),35] 
    result, enc_img = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(enc_img,1)


"""
Dataset processing: we iterate through all the files in the image folder
and apply the functions we have implemented.
"""

for file_name in os.listdir(input_dir):
    path = os.path.join(input_dir, file_name)
    img = cv2.imread(path)
    
    if img is None:
        continue
    name = file_name.split(".")[0]

    cv2.imwrite(f"{output_dir}/{name}_noise.jpg", add_noise(img))
    cv2.imwrite(f"{output_dir}/{name}_blur.jpg", blur(img))
    cv2.imwrite(f"{output_dir}/{name}_change_brightness.jpg", change_brightness(img))
    cv2.imwrite(f"{output_dir}/{name}_compress.jpg", compress(img))

