"""
We need to transform the images into a numerical vector, which is a list
of numbers that will describe our banknote.

We extract a 3D color histogram in the BGR space. The main idea is that,
instead of saving every single pixel, we count how many pixels fall into
specific color ranges.

We also extract statistics from the grayscale image. In particular, we
compute the mean, minimum, maximum, median, and standard deviation.
These values give a general description of the brightness and contrast
of each image.

We extract the LBP feature, which is a texture descriptor that analyzes
the grain of the image. This is important because real and fake banknotes
may have different micro-patterns and pixels.

Then, we extract the HOG feature, which describes the distribution of
edges in the image and the shapes of the objects inside it.

Finally, we combine all the features into a single vector, which will be
used for training.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog

def extract_color_histogram(img, bins=(8,8,8)):
    hist = cv2.calcHist([img],[0,1,2], None,bins,[0,256, 0,256, 0,256] )
    hist = cv2.normalize(hist,hist).flatten()
    
    return hist.astype(np.float32)

def extract_gray_stats(img_gray):
    mean = np.mean(img_gray)
    std = np.std(img_gray)
    min = np.min(img_gray)
    max = np.max(img_gray)
    median = np.median(img_gray)

    return np.array([mean,std,min,max,median ], dtype=np.float32)

def extract_lbp(img_gray, radius=2,n_points=16):
    lbp = local_binary_pattern(img_gray,n_points,radius,method="uniform")
    pattern_number = n_points+2
    hist, _ = np.histogram(
        # convert lbp to vector 
        lbp.ravel(),
        bins = pattern_number,range=(0, pattern_number)  
    )
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)

    return hist

def extract_hog(img_gray):
    features = hog(img_gray,orientations= 9, pixels_per_cell=(8,8),cells_per_block=(2,2), block_norm="L2-Hys",visualize=False, feature_vector=True)
    return features.astype(np.float32)

def extract_sharpness(img_gray):
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return np.array([laplacian_var], dtype= np.float32)



def dark_light_pixel(img_gray):
    dark_pixel = np.sum(img_gray < 50) / img_gray.size
    light_pixel = np.sum(img_gray >200) / img_gray.size
    return np.array([dark_pixel, light_pixel],dtype= np.float32)

def extract_contrast(img_gray):
    blurred = cv2.GaussianBlur(img_gray,(9,9), 0)
    contrast = np.mean(np.abs(img_gray.astype(np.float32)-blurred.astype(np.float32)))
    return np.array([contrast],dtype=np.float32)

def extract_edge_density(img_gray):
    edges = cv2.Canny(img_gray,50,150)
    density = np.sum(edges > 0) / edges.size
    return np.array([density], dtype=np.float32)

def extract_entropy(img_gray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0,256]).flatten()
    hist = hist / (hist.sum()+ 1e-7)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return np.array([entropy], dtype= np.float32)

def extract_local_contrast(img_gray):
    blured = cv2.GaussianBlur(img_gray,(9,9), 0)
    contrast = np.mean(np.abs(img_gray.astype(np.float32)-blured.astype(np.float32)))
    return np.array([contrast], dtype=np.float32)

def extract_quality_features(img_gray):
    sharpness = extract_sharpness(img_gray)
    edge_density = extract_edge_density(img_gray)
    entropy = extract_entropy(img_gray)
    dark_light = dark_light_pixel(img_gray)
    local_contrast = extract_local_contrast(img_gray)
    return np.hstack([sharpness,edge_density,entropy,dark_light,local_contrast]).astype(np.float32)

def split_into_regions(img_color, img_gray):
    
    h,w = img_gray.shape
    h2 = h // 2
    w2 = w // 2
    regions = [
        (img_color[0:h2,0:w2], img_gray[0:h2, 0:w2]),
        (img_color[0:h2,w2:w],img_gray[0:h2, w2:w]),
        (img_color[h2:h, 0:w2], img_gray[h2:h, 0:w2]),
        (img_color[h2:h, w2:w], img_gray[h2:h, w2:w])
    ]
    
    return regions 

def extract_regional_feautures(img_color, img_gray):
    regions = split_into_regions(img_color, img_gray)
    regional_features = []
    for region_color, region_gray in regions:
        color_hist = extract_color_histogram(region_color, bins=(4,4,4))
        gray_state = extract_gray_stats(img_gray)
        quality_features = extract_quality_features(img_gray)
        region_vector = np.hstack([
            color_hist,gray_state,quality_features
        ]).astype(np.float32)
        
        regional_features.append(region_vector)
    return np.hstack(regional_features).astype(np.float32)



def all_feautures(img_color, img_gray):
    hist_color = extract_color_histogram(img_color)
    gray_statists = extract_gray_stats(img_gray)
    lbp = extract_lbp(img_gray)
    hog = extract_hog(img_gray)
    quality_feature = extract_quality_features(img_gray)
    regional_features = extract_regional_feautures(img_color, img_gray)
    features_vector = np.hstack([hist_color,gray_statists,lbp,hog,quality_feature,regional_features]).astype(np.float32)
    return features_vector

    

    
    




    


