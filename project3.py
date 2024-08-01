import cv2
import numpy as np
import math
import sklearn
from PIL import Image
import os 
import random
from sklearn.neighbors import KNeighborsClassifier
from collections import deque
from skimage.segmentation import slic
from skimage.measure import regionprops, label


def load_img(file_name):
    try: 
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.uint8)
        
        
        return image
    except Exception as e:
        print("Error loading image:", e)
        return None

def display_img(image):
    
    if image is not None:
        windowName = 'image'
        cv2.imshow(windowName, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image")

def generate_vocabulary(train_data_file):
    training_data = []
    label_data = []

    all_descriptor = []
    
    
    with open(train_data_file, 'r') as f:
        
        for line in f:
           
            line = line.strip()
            image_path, label = line.split()
            label = int(label)
            label_data.append(label)
            training_data.append(image_path)
    
    for file_path in training_data:
        absolute_path = os.path.abspath(file_path)   
        image = load_img(absolute_path)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image,None)
        all_descriptor.append(descriptors)
    
    combined_descriptors = np.concatenate(all_descriptor,axis=0)
    results =  kmeans_clustering(combined_descriptors,k=10)
    return results
        
    


def kmeans_clustering(data,k):
    centroids = []

    centroids = data[np.random.choice(data.shape[0],k,replace=False)]
    
    for index in range(100):
        distance = np.linalg.norm(data[:, np.newaxis] - centroids,axis=2)
        labels = np.argmin(distance,axis=1)

        new_centroids = np.array([np.nanmean(data[labels == i], axis=0) for i in range(k)])
        centroids = new_centroids
    return centroids
                   

def extract_features(image,vocabulary):
    training_data = []
    label_data = []
    label_photo = []
    histogram = np.zeros(10)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image,None)
    for i in range(len(descriptors)):
            min = 10000
            index = 0
            for j in range(len(vocabulary)):
                descrip = descriptors[i].flatten()
                vocab = vocabulary[j].flatten()
                length = np.dot(descrip,vocab)
                length = np.sqrt(length)
                if length < min:
                   min = length
                   index = j
                
                histogram[index] = histogram[index] + 1
        
                    
         

    return histogram

def train_classifier(train_data_file,vocab):
    training_data = []
    label_data = []
    neigh = KNeighborsClassifier(n_neighbors=3)
    dataset = []
    with open(train_data_file, 'r') as f:
        
        for line in f:
           
            line = line.strip()
            image_path, label = line.split()
            label = int(label)
            label_data.append(label)
            
            training_data.append(image_path)
    
    for index,file_path in enumerate(training_data):
        absolute_path = os.path.abspath(file_path)   
        image = load_img(absolute_path)
        
        feat_vector = extract_features(image,vocab)
        
        dataset.append(feat_vector)
        
        

    dataset = np.copy(dataset)
    neigh.fit(dataset,label_data)

    return neigh

def classify_image(classifier, test_img, vocabulary):
    feat_vector = extract_features(test_img, vocabulary)
    feat = np.copy(feat_vector)
    feat = np.reshape(feat, (1, -1))
    output = classifier.predict(feat)
    if output == 1: 
        return print("cat")
    elif output == 0:
        return print("dog")
    else: 
        return print("No Results")


def get_neighborhood(image, x, y):
    neighborhood = np.zeros((3,3))
    for i in range(max(0, x - 1), min(len(image), x + 2)):
        for j in range(max(0, y - 1), min(len(image[0]), y + 2)):
            neighborhood[i-(x-1)][j-(y-1)] = image[i][j]
    return neighborhood
        
   

    

def threshold_image(image, low_thresh, high_thresh):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] < low_thresh:
                image[i][j] = 0
            elif low_thresh < image[i][j] < high_thresh:
                image[i][j] = 128
            else:
                image[i][j] = 255
    for  x_cord in range(image.shape[0]):
        for  y_cord in range(image.shape[1]):
            if image[x_cord][y_cord] == 128:
                local_neighborhood = get_neighborhood(image,x_cord,y_cord)
                count = 0
                for i in range(3):
                    for j in range(3):
                        if local_neighborhood[i][j] == 128 or local_neighborhood[i][j] == 0:
                            count = count + 1
                if count >= 4:
                    image[x_cord][y_cord] = 0 
    return image


def grow_regions(image):
    height, width = image.shape
    map = np.zeros((height, width))  
    final_image = np.zeros((height, width))
    x_cord_center = 147
    y_cord_center = 166

    stack = []
    stack.append((x_cord_center, y_cord_center))
    
    while stack:
        x, y = stack.pop()
        if map[y, x] == 0: 
            map[y, x] = 1  
            distance = np.linalg.norm(image[y, x].astype(np.float64) - image[y_cord_center, x_cord_center].astype(np.float64) )  
            if distance <= 30: 
                final_image[y, x] = 255  
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if 0 <= x + dx < width and 0 <= y + dy < height:  
                            stack.append((x + dx, y + dy))  
    
    return final_image


                          
def split_regions(image):
    
    height, width = image.shape
    
    final_image = np.zeros((height, width), dtype=np.uint8) 
    


    
    regions = [ ]
    regions.append((0, 0, width, height))

    
    while regions:
        
        x, y, region_width, region_height = regions.pop()
       
        region_pixels = image[y:y + region_height, x:x + region_width]
        
        variance = np.var(region_pixels)
        

       
        if abs(variance) > 6:
            if region_width > region_height:
                
                half_width = region_width // 2
                regions.append((x, y, half_width, region_height))
                regions.append((x + half_width, y, region_width - half_width, region_height))
            else:
                
                half_height = region_height // 2
                regions.append((x, y, region_width, half_height))
                regions.append((x, y + half_height, region_width, region_height - half_height))
        else:
            
            if region_width == 1 and region_height == 1:
                
                final_image[y, x] = 255
            else:
                
                final_image[y:y + region_height, x:x + region_width] = 0

    return final_image









def get_neighbor_cord(row, col, size):
        half_size = size // 2
        neighbors = []
        for i in range(row - half_size, row + half_size + 1):
            for j in range(col - half_size, col + half_size + 1):
                if i >= 0 and i < row and j >= 0 and j < col and (i != row or j != col):
                    neighbors.append((i, j))
        return neighbors


def merge_regions(image):
    merge_threshold = 0.1
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labels = slic(image, n_segments=300, compactness=10, start_label=1)

    props = regionprops(labels, intensity_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
   
    intensity = np.array([prop.mean_intensity for prop in props])
    
   
    label_map = np.arange(len(props) + 1)
    for i, prop1 in enumerate(props):
        for j, prop2 in enumerate(props):
            if i != j and abs(intensity[i] - intensity[j]) / 255 < merge_threshold:
                min_label, max_label = sorted([label_map[i + 1], label_map[j + 1]])
                label_map[max_label] = min_label
    
 
  

    binary_image = (label_map[labels] > 1).astype(np.uint8) * 255
    return binary_image



    
def segment_image(image):
    image1 = np.copy(image)
    image2 = np.copy(image)
    image3 = np.copy(image)


    ###############################################################
    image1 = threshold_image(image1,10,100)
    print("Image 1 is thresholding")
    ################################################################
    image2 = split_regions(image2)
    image2 = merge_regions(image2)
    print("Image 2 is merge and split regions")
    ################################################################
    image3 = grow_regions(image3)
    print("Image 3 is grow regions")
    ################################################################
    

    return image1, image2, image3












# img1,img2,img3 = segment_image(image)

# print("Image 1")
# display_img(img1)
# print("Image 2")
# display_img(img2)
# print("Image 3")
# display_img(img3)







