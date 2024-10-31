# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: [Connor Rullo]
# Last Modified: 2024-09-22

from torchvision.transforms import ToTensor
from collections import Counter
from torch import nn
import numpy as np
import torch
import math
import cv2
import sys
import os

# returns the directory for relative imports
def get_main_dir_loc() -> str:
    script_loc: str = os.path.dirname(__file__)
    return os.path.dirname(script_loc) # getting the directory of the playground folder

class Image_Classify(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x):
        return self.model(x)      

def predict_img_val(model, test_img, device):
    # unsqueeze adds batch size to tensor
    img_tensor = ToTensor()(test_img).unsqueeze(0).to(device) 
    
    # passing the tensor through the trained model to make predictions
    model_raw_vals = model(img_tensor) # running the tensor through the model

    # returns the probabilities of each value and the item (number) itself
    output_probabilities = torch.nn.functional.softmax(model_raw_vals, dim=1) 
    return output_probabilities, torch.argmax(model(img_tensor)).item()

# returns number that is predicted --> -1 if not enough confidence (presumed not number)
def predict_number_from_28x28_img(square_img):
    # init basic vars
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loading the test img and model for predictions
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w') # discard the print of the future warning
    loaded_model = Image_Classify().to(device)
    loaded_states = torch.load(os.path.join("data", "model_100_epochs.pt"), map_location=torch.device(device))
    loaded_model.load_state_dict(loaded_states)
    sys.stderr.flush() # ensuring all bs text is written to devnull
    sys.stderr = stderr # redirecting stdout back to the default stdout

    # predicting the output values
    output_probs_tensor, output_val = predict_img_val(model=loaded_model, test_img=square_img, device=device)
    output_probs_list = output_probs_tensor.detach().tolist()[0]
    prediction_confidence = output_probs_list[output_val]

    # need to analyse probabilties to identify if there is a number in the blob img or not
    confidence_threshold = 0.5
    if prediction_confidence > confidence_threshold:
        return output_val, prediction_confidence
    else:
        return -1, prediction_confidence

# RGB --> Grayscale
def grayscale_rgb_img(base_rgb_img):
    return cv2.cvtColor(base_rgb_img, cv2.COLOR_BGR2GRAY)

def canny_edge_detection(gray_img, t1, t2):
    return cv2.Canny(gray_img, threshold1=t1, threshold2=t2)

def hough_line_detection(canny_edges, t1, min_line_len):
    return cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=t1, minLineLength=min_line_len) 

def grayscale_harris_corner_detection(gray_img):
    gray_np = np.float32(gray_img)
    return cv2.cornerHarris(gray_np, blockSize=2, ksize=3, k=0.04)

def update_img_to_show_hough_lines(lined_img, hough_lines):
    if hough_lines is not None:
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # draw line on prev img

# finding the angle of the most dominant lines
def determine_dominant_line_orientation_counter_impl(hough_lines):
    # iterating over all lines found in the image --> grabbing angles
    angles = []
    if hough_lines is None:
        return None, None

    # should not be reached if there are no detected lines
    for line in hough_lines:
        line_angle = calc_angle(line[0][2], line[0][3], line[0][0], line[0][1])
        angles.append(line_angle)

    # angle_histogram = Counter(angles)
    # most_common_angle_deg = angle_histogram.most_common(1)[0][0]

    ################################
    ### NEW CODE ###################
    ################################
    angle_histogram = {}
    for angle in angles:
        if angle in angle_histogram:
            angle_histogram[angle] += 1
        else:
            angle_histogram[angle] = 1
    
    if len(angle_histogram) == 0 or angle_histogram == {}:
        return None, None

    most_common_angle_deg = max(angle_histogram, key=angle_histogram.get)
    ################################
    ### NEW CODE ###################
    ################################

    return most_common_angle_deg, angle_histogram # Get the most common angle

def calc_angle(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def conv_polar_coords_to_cartesian(rho, theta, img_shape):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    
    # ensure that the calculated values are within the image boundaries
    x1 = max(0, min(img_shape[1] - 1, x1))
    y1 = max(0, min(img_shape[0] - 1, y1))
    x2 = max(0, min(img_shape[1] - 1, x2))
    y2 = max(0, min(img_shape[0] - 1, y2))

    return x1, y1, x2, y2

def keep_only_dominant_lines(hough_lines, dominant_angle, tolerance_deg, img_shape):
    # Filter lines based on angle
    filtered_lines = []
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        angle = calc_angle(x1, y1, x2, y2)
        angle = angle % 180 # normalize the angle to be within [0, 180]
        if abs(angle - dominant_angle) <= tolerance_deg:
            filtered_lines.append(line)
    return filtered_lines

def detect_grayscale_contours(gray_img):
    _, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU) # apply adaptive thresholding to get binary image
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def calc_contour_angle(contour):
    # Compute the minimum area rectangle
    rect = cv2.minAreaRect(contour)
    _, (width, height), angle = rect
    if width < height:
        angle = 90 + angle # adjust angle to be within the range of -90 to 0 degrees
    
    return angle

# creating a binary image (black or white) from a grayscale img
def conv_gray_img_to_bin_img(gray_img):
    gaussian_blur = cv2.GaussianBlur(gray_img, (7, 7), 0) # 5x5 kernel filter
    equalised_img = cv2.equalizeHist(gaussian_blur)
    _, threshold_img = cv2.threshold(equalised_img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    return threshold_img

# applying affine transform (rotating whole image)
def rotate_img(base_img, rotation_angle):
    (img_height, img_width) = base_img.shape[:2] # getting the resolution of the image
    img_centre = (img_width // 2, img_height // 2) # calc image centre
    rotation_matrix = cv2.getRotationMatrix2D(img_centre, rotation_angle, 1.0) # get the rotation matrix from centre point
    
    # Calculate the bounding box of the rotated image
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    new_img_width = int((img_height * sin_theta) + (img_width * cos_theta))
    new_img_height = int((img_height * cos_theta) + (img_width * sin_theta))
    
    # adjusting rotation matrix --> consider new translation
    rotation_matrix[0, 2] += (new_img_width / 2) - img_centre[0]
    rotation_matrix[1, 2] += (new_img_height / 2) - img_centre[1]
    
    # apply the affine transform (rotation)
    rotated_image = cv2.warpAffine(base_img, 
                                   rotation_matrix, 
                                   (new_img_width, new_img_height), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0, 0, 0))
    return rotated_image

# returns a "zoomed in", horizontally reorientated image
def perform_perspective_transform(base_rgb_image, tl_pt, tr_pt, bl_pt, br_pt):
    # calculating distance btwn pts (height and width) using Pythagorous --> all pts have (x, y) format
    height_left = np.sqrt(((tl_pt[0] - bl_pt[0]) ** 2) + ((tl_pt[1] - bl_pt[1]) ** 2))
    height_right = np.sqrt(((tr_pt[0] - br_pt[0]) ** 2) + ((tr_pt[1] - br_pt[1]) ** 2))
    width_top = np.sqrt(((tl_pt[0] - tr_pt[0]) ** 2) + ((tl_pt[1] - tr_pt[1]) ** 2))
    width_bottom = np.sqrt(((bl_pt[0] - br_pt[0]) ** 2) + ((bl_pt[1] - br_pt[1]) ** 2))

    # calculating maximum height and width for 
    max_height = max(int(height_left), int(height_right))
    max_width = max(int(width_top), int(width_bottom))

    # defining the location of the corners we wish to base the transform around
    input_pts=np.float32([tl_pt, bl_pt, tr_pt, br_pt])
    output_pts = np.float32([[0, 0], [0, max_width],
                            [max_height , 0], [max_height , max_width]])


    # Computing the perspective transform
    perspective_transform_image = cv2.getPerspectiveTransform(input_pts, output_pts)
    perspective_warp_image = cv2.warpPerspective(base_rgb_image,
                                       perspective_transform_image,
                                       (max_height, max_width),
                                       flags=cv2.INTER_LINEAR)  
    return perspective_warp_image

def merge_groups_by_prox(conn_labels, prox_max_x, prox_max_y):
    # sorting connected labels into groups --> based on a proximity threshold
    groups = []
    for i in range(len(conn_labels)):
        label = conn_labels[i]
        x, y, w, h = label
        found_group = False
        for j in range(len(groups)):
            group = groups[j]
            last_x, last_y, last_w, last_h = group[-1]
            min_x_dist = max(x, last_x) - min(x + w, last_x + last_w)
            min_y_dist = max(y, last_y) - min(y + h, last_y + last_h)
            if min_x_dist <= prox_max_x and min_y_dist <= prox_max_y:
                group.append(label)
                found_group = True
                break
        if not found_group:
            groups.append([label])

    return groups

def calc_larger_rect_from_two_small(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    min_x = min(x1, x2)
    min_y = min(y1-h1, y2-h2)
    max_x = max(x1 + w1, x2 + w2)
    max_y = max(y1, y2)

    width = max_x - min_x
    height = max_y - min_y

    return min_x, max_y, width, height

def merging_simplified_prox_groups(group_bounds, prox_max_x, prox_max_y):
    # NOTE: x-dir sorted still (this presumption is made in the following code)
    simpli_merged_groups = []

    for bounds in group_bounds:
        x, y, w, h = bounds
        if not simpli_merged_groups: # appending if nothing exists yet
            simpli_merged_groups.append(bounds)
            continue
        else:
            new_rect_flag: bool = True
            # for merged_bounds in simpli_merged_groups: # comparing against all currently merged bounds
            for merge_i in range(len(simpli_merged_groups)): # comparing against all currently merged bounds
                x_merge, y_merge, w_merge, h_merge = simpli_merged_groups[merge_i] # considers the edges of the current blob group
                
                # checking cartesian distances between points
                prev_to_curr_x_dist = abs(x - (x_merge + w_merge)) # sorted list means that left comparisons are not required

                prev_to_curr_y_below_dist = abs((y + h) - y_merge)
                prev_to_curr_y_above_dist = abs(y - (y_merge + h_merge))
                prev_to_curr_y_abs_dist = max([prev_to_curr_y_above_dist, prev_to_curr_y_below_dist])
                curr_group_x_threshold_reached: bool = prev_to_curr_x_dist > prox_max_x # returns a bool
                curr_group_y_threshold_reached: bool = prev_to_curr_y_abs_dist > prox_max_y # returns a bool
                
                if curr_group_x_threshold_reached or curr_group_y_threshold_reached: # blob is outside of previous prox
                    continue
                else:
                    # finding max bounds of the new rect --> assigning in list
                    new_rect = calc_larger_rect_from_two_small(bounds, simpli_merged_groups[merge_i])
                    simpli_merged_groups[merge_i] = new_rect # replacing the previously merged rect with a new, larger rect group
                    new_rect_flag = False
                    break

            if new_rect_flag: # if the bound does not reside within any other, current groups 
                simpli_merged_groups.append(bounds) # creating a new group if necessary (otherwise replace last)
    return simpli_merged_groups

def get_max_min_edges_conn_labels(groups):
    simplified_groups = []
    for curr_label in groups:
        min_x = min(x for x, _, _, _ in curr_label) 
        min_y = min(y for _, y, _, _ in curr_label)
        max_x = max(x for x, _, _, _ in curr_label) 
        max_y = max(y for _, y, _, _ in curr_label)
        min_w = min(w for _, _, w, _ in curr_label)
        max_h = max(h for _, _, _, h in curr_label)

        simply_group_tuple = (min_x, min_y, (max_x - min_x + min_w), (max_y - min_y + max_h))
        simplified_groups.append(simply_group_tuple) # extra brackets for appending a tuple

    return simplified_groups

def get_bounding_box_from_rect_tuples(rect_tuples):
    min_x = min(x for x, _, _, _ in rect_tuples) 
    min_y = min(y for _, y, _, _ in rect_tuples)
    max_x = max(x for x, _, _, _ in rect_tuples) 
    max_y = max(y for _, y, _, _ in rect_tuples)
    min_w = min(w for _, _, w, _ in rect_tuples)
    max_h = max(h for _, _, _, h in rect_tuples) 

    group_tuple = (min_x, min_y, (max_x - min_x + min_w), (max_y - min_y + max_h))
    return group_tuple

def get_most_common_blob_group(conn_labels, simply_merged_groups):
    blob_group_count_dict: tuple[int] = {}
    for group_bound in simply_merged_groups:
        x_group_min, y_group_min, w_group, h_group = group_bound # unravelling tuple
        x_group_max = x_group_min + w_group
        y_group_max = y_group_min + h_group

        for labelled_bound in conn_labels:
            x_label_min, y_label_min, w_label, h_label = labelled_bound # unravelling tuple
            x_label_max = x_label_min + w_label
            y_label_max = y_label_min + h_label
            
            # check if centre of blob is within the group
            blob_centre_x = (x_label_max + x_label_min) // 2
            blob_centre_y = (y_label_max + y_label_min) // 2
            if (x_group_min <= blob_centre_x <= x_group_max) and (y_group_min <= blob_centre_y <= y_group_max):
                if blob_group_count_dict.get(group_bound) is not None:
                    blob_group_count_dict[group_bound] += 1
                else:
                    blob_group_count_dict[group_bound] = 1

    # getting the group with the highest num of blobs (presumed barcode)
    if not blob_group_count_dict:
        raise ValueError("empty dict")
    barcode_group = max(blob_group_count_dict, key=blob_group_count_dict.get) 

    return barcode_group

def calc_loc_barcode_nums_from_prox_barcode_lines(local_img, barcode_stats, x_threshold, y_threshold):
    # unravelling tuple
    barcode_x_min, barcode_y_min, barcode_w, barcode_h = barcode_stats 
    filtered_num_img = np.zeros_like(local_img) # creating image to be tinkered with

    # approximate the size of the nums based on the height and width of the bars --> found by visible inspection of existing barcodes
    # NOTE: first digit is always placed away from barcode to the left
    edge_num_width_addon = (83/590) * barcode_w
    edge_num_height_addon = ((1/6) * barcode_h)

    # finding the points of the rectangle around which objects
    num_rect_xl = int(barcode_x_min - x_threshold - edge_num_width_addon)
    num_rect_xr = int((barcode_x_min + barcode_w) + x_threshold) # dont need the width addon on the right (number only reside outside barcode lines on left side)
    num_rect_yt = int((barcode_y_min + barcode_h)- y_threshold - edge_num_height_addon)
    num_rect_yb = int((barcode_y_min + barcode_h) + y_threshold + edge_num_height_addon)

    filtered_num_img[num_rect_yt:num_rect_yb, num_rect_xl:num_rect_xr] = local_img[num_rect_yt:num_rect_yb, num_rect_xl:num_rect_xr] # zeroing everything but barcode lines

    return filtered_num_img

def calc_edges_barcode_from_prox_conn(local_img, prox_max_x, prox_max_y):
    num_labs, labels, stats, _ = cv2.connectedComponentsWithStats(local_img, connectivity=4) # must share an edge to be "connected"
    img_rm_blobs = np.zeros_like(local_img) # creating an image with all zeros
    
    # iterate through each connected label (x direction)
    connected_labels = []
    for conn_label in range(1, num_labs): 
        x = stats[conn_label, cv2.CC_STAT_LEFT]
        y = stats[conn_label, cv2.CC_STAT_TOP]
        w = stats[conn_label, cv2.CC_STAT_WIDTH]
        h = stats[conn_label, cv2.CC_STAT_HEIGHT]
        connected_labels.append((x, y, w, h)) # appending a new connected label (blob)

    # sort labels from left to right (x-axis)
    connected_labels.sort(key=lambda label: label[0])  # label[0] is the x-coordinate

    # finding blob groups in the x-dir
    groups_x = merge_groups_by_prox(conn_labels=connected_labels, prox_max_x=prox_max_x, prox_max_y=prox_max_y)
    simplified_groups_x = get_max_min_edges_conn_labels(groups=groups_x) # reducing complexity such that each group has a rectangle around it
    
    # merging rectangle groups that are invalidly separate because of noise in middle of x-dir blobs
    simply_merged_groups = merging_simplified_prox_groups(group_bounds=simplified_groups_x, prox_max_x=prox_max_x, prox_max_y=prox_max_y)

    # checking how many blobs reside within each group --> group with most blobs is presumed the barcode
    barcode_group = get_most_common_blob_group(conn_labels=connected_labels, simply_merged_groups=simply_merged_groups)
    x_min_barcode = barcode_group[0]
    y_min_barcode= barcode_group[1]
    x_max_barcode = barcode_group[0] + barcode_group[2]
    y_max_barcode = barcode_group[1] + barcode_group[3]

    img_rm_blobs[y_min_barcode:y_max_barcode, x_min_barcode:x_max_barcode] = local_img[y_min_barcode:y_max_barcode, x_min_barcode:x_max_barcode] # zeroing everything but barcode lines
    return img_rm_blobs, barcode_group

def rm_blobs_using_area(local_img, area_min, area_max):
    num_labs, labels, stats, _ = cv2.connectedComponentsWithStats(local_img, connectivity=4) # must share an edge to be "connected"
    img_rm_blobs = np.zeros_like(local_img) # Create an output image to draw the filtered components
    for label in range(1, num_labs): # Iterate through each component (excluding the background label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area_min <= area <= area_max:
            img_rm_blobs[labels == label] = 255 # If the area is within the desired range, add the component to the output image
    return img_rm_blobs

def rm_blobs_using_hw_ratio(local_img, ratio_min, ratio_max):
    num_labs, labels, stats, _ = cv2.connectedComponentsWithStats(local_img, connectivity=4) # must share an edge to be "connected"
    img_rm_blobs = np.zeros_like(local_img) # Create an output image to draw the filtered components
    for label in range(1, num_labs): # Iterate through each component (excluding the background label 0)
        ratio = stats[label, cv2.CC_STAT_HEIGHT]/stats[label, cv2.CC_STAT_WIDTH]
        if ratio_min <= ratio <= ratio_max:
            img_rm_blobs[labels == label] = 255 # checking if within desired range
    return img_rm_blobs

def rm_blobs_using_extent(local_img, extent_min, extent_max):
    img_rm_blobs = np.zeros_like(local_img) # creating img for return (starting pixels zeroed)
    contours, _ = cv2.findContours(local_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour) # compute the size of the rectangle around the blob
        contour_area = cv2.contourArea(contour) # finds area of the blob itself (within boundary)
        bounding_rect_area = w * h # finds the area if the boundary rectangle (block)
        if bounding_rect_area == 0: # avoiding division by zero errors
            continue

        # filtering based on filling ratio
        extent = contour_area / bounding_rect_area # finding the difference between blob and bounding box
        if extent_min <= extent <= extent_max:
            cv2.drawContours(img_rm_blobs, [contour], -1, (255), thickness=cv2.FILLED)  # draw the relevant blobs
       
    return img_rm_blobs

def merge_two_binary_images(img1, img2):
    img1_np = np.uint8(img1)
    img2_np = np.uint8(img2)
    return cv2.bitwise_or(img1_np, img2_np) # returning a merged image

def check_barcode_exists_in_bin_img(local_img, t1, t2, min_line_len):
    barcode_canny_edges = canny_edge_detection(local_img, t1=t1, t2=t2) # thresholds
    barcode_hough_lines = hough_line_detection(barcode_canny_edges, t1=t1, min_line_len=min_line_len)
    
    if barcode_hough_lines is not None:
        if len(barcode_hough_lines) > 5:
            return True, barcode_hough_lines
    return False, None

def get_connected_labels_in_img_sorted_x(local_img):
    num_labs, labels, stats, _ = cv2.connectedComponentsWithStats(local_img, connectivity=4) # must share an edge to be "connected"
    img_rm_blobs = np.zeros_like(local_img) # creating an image with all zeros
    
    # iterate through each connected label (x direction)
    connected_labels = []
    for conn_label in range(1, num_labs): 
        x = stats[conn_label, cv2.CC_STAT_LEFT]
        y = stats[conn_label, cv2.CC_STAT_TOP]
        w = stats[conn_label, cv2.CC_STAT_WIDTH]
        h = stats[conn_label, cv2.CC_STAT_HEIGHT]
        connected_labels.append((x, y, w, h)) # appending a new connected label (blob)
    connected_labels.sort(key=lambda label: label[0])  # label[0] is the x-coordinate

    return connected_labels

def resize_num_img_to_mnist_scale_w_stretch(num_img):
    resized_img = num_img.copy()
    resized_img = cv2.resize(resized_img, (28, 28))
    return resized_img 

def resize_num_img_to_square_img_max_res(num_img):
    x_res, y_res = num_img.shape
    square_size = max([x_res, y_res])

    # creating a larger image to encompass the num_img
    square_img_mnist_size = np.zeros((square_size, square_size), dtype=np.uint8) # Single channel for binary image
    y_offset = (square_size- x_res) // 2
    x_offset = (square_size - y_res) // 2
    square_img_mnist_size[y_offset:(y_offset + x_res), x_offset:(x_offset + y_res)] = num_img 
    square_img_mnist_size = cv2.resize(square_img_mnist_size, (28, 28)) # downsize num images to 28x28 for processing in OCR model (MNIST training set)

    return square_img_mnist_size

def resize_num_img_to_mnist_scale_ratiod(num_img):
    x_res, y_res = num_img.shape

    # finding the smallest MNIST capable size to hold the num_img
    x_y_current_size = 28 # only one variable used as 28x28 x==y
    while x_res > x_y_current_size or y_res > x_y_current_size: # until the image fits within the larger image
        x_y_current_size *= 2

    # creating a larger image to encompass the num_img
    new_num_img = np.zeros((x_y_current_size, x_y_current_size), dtype=np.uint8) # Single channel for binary image
    y_offset = (x_y_current_size - x_res) // 2
    x_offset = (x_y_current_size - y_res) // 2
    new_num_img[y_offset:(y_offset + x_res), x_offset:(x_offset + y_res)] = num_img 
    new_num_img = cv2.resize(new_num_img, (28, 28)) # downsize num images to 28x28 for processing in OCR model (MNIST training set)

    return new_num_img

def group_blobs_by_y_val(number_labels_x_sorted, y_tolerance):
    y_grouped_num_blobs = []
    for num_index in range(len(number_labels_x_sorted)):
        num_lab = number_labels_x_sorted[num_index]
        x, y, w, h = num_lab
        curr_centre = y + (h // 2)
        found_flag = False

        if not y_grouped_num_blobs: # appending first number label without checking values
            y_grouped_num_blobs.append([num_lab])
        else:
            for j in range(len(y_grouped_num_blobs)):
                group = y_grouped_num_blobs[j]
                _, last_y, _, last_h = group[-1] # grabbing last val in the group
                last_centre = last_y + (last_h // 2)
                if abs(curr_centre - last_centre) <= y_tolerance:
                    y_grouped_num_blobs[j].append(num_lab)
                    found_flag = True
                    break
            if not found_flag:
                y_grouped_num_blobs.append([num_lab])

    return y_grouped_num_blobs

# returns a string that holds all digits within the image --> empty string if error occurs (or barcode dne)
def task4(input_img_loc):
    # gathering the image files as cv2 objects (imgs)
    input_img_bgr = cv2.imread(input_img_loc)
    input_img_bin = conv_gray_img_to_bin_img(cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2GRAY))

    # morphology methods
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    open_img = cv2.morphologyEx(input_img_bin, cv2.MORPH_OPEN, open_kernel)
    closed_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, closed_kernel)
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion_img = cv2.morphologyEx(closed_img, cv2.MORPH_ERODE, erosion_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilate_img = cv2.morphologyEx(erosion_img, cv2.MORPH_DILATE, dilate_kernel)
    last_kernel_img = dilate_img

    # removing numbers before looking for bars
    bar_numbers_img = rm_blobs_using_hw_ratio(local_img=last_kernel_img, ratio_min=0.5, ratio_max=5.0)
    bar_numbers_img = rm_blobs_using_area(local_img=bar_numbers_img, area_min=50, area_max=3000)
    bar_number_mask = bar_numbers_img == 255 # mask that removes nums from image
    img_no_nums = last_kernel_img.copy()
    img_no_nums[bar_number_mask] = 0 # removing all number blobs from img (for barcode scanning)

    # barcode filtering
    barcode_img = rm_blobs_using_area(local_img=img_no_nums, area_min=500, area_max=100000)
    barcode_img = rm_blobs_using_hw_ratio(local_img=barcode_img, ratio_min=0.7, ratio_max=500.0)

    # performing feature detection
    canny_edges = canny_edge_detection(barcode_img, t1=1, t2=500) # thresholds
    hough_lines = hough_line_detection(canny_edges, t1=200, min_line_len=20) # min line length set to avoid weak lines
    dominant_orientation, orientation_counter = determine_dominant_line_orientation_counter_impl(hough_lines)
    if dominant_orientation is None or orientation_counter is None: # means that there are no detected barcode lines
        return ""
    if len(hough_lines) < 10: # not enough "hard lines" to be considered a barcode
        return ""

    # applying perspective transformation for correct orientation --> 0deg and 180 grabbed to avoid missing nums as a result of incorrect flip
    expected_rot_a1 = dominant_orientation - 90 # perpendicular barcode to horizontal axis
    expected_rot_a2 = expected_rot_a1 + 180 # flipping on head

    # rotating imgs are per described lines --> both angles used to avoid discrepancies due to text on top of barcode (and failed rotation)
    rotate_bin_barcode_lines_a1 = rotate_img(barcode_img, expected_rot_a1)
    rotate_bin_barcode_nums_a1 = rotate_img(bar_numbers_img, expected_rot_a1)
    rotate_bin_barcode_lines_a2 = rotate_img(barcode_img, expected_rot_a2)
    rotate_bin_barcode_nums_a2 = rotate_img(bar_numbers_img, expected_rot_a2)

    # filter barcode lines from bin image based on proximity between lines
    _, barcode_stats_a1 = calc_edges_barcode_from_prox_conn(local_img=rotate_bin_barcode_lines_a1, prox_max_x=250, prox_max_y=250)
    _, barcode_stats_a2 = calc_edges_barcode_from_prox_conn(local_img=rotate_bin_barcode_lines_a2, prox_max_x=250, prox_max_y=250)

    # filter barcode numbers based on proximity to barcode lines
    barcode_nums_img_a1 = calc_loc_barcode_nums_from_prox_barcode_lines(local_img=rotate_bin_barcode_nums_a1, barcode_stats=barcode_stats_a1, x_threshold=100, y_threshold=150)
    barcode_nums_img_a2 = calc_loc_barcode_nums_from_prox_barcode_lines(local_img=rotate_bin_barcode_nums_a2, barcode_stats=barcode_stats_a2, x_threshold=100, y_threshold=150)
   
    # removing noises after refiltering
    barcode_nums_img_a1 = rm_blobs_using_hw_ratio(local_img=barcode_nums_img_a1, ratio_min=0.7, ratio_max=3.0)
    barcode_nums_img_a1 = rm_blobs_using_area(local_img=barcode_nums_img_a1, area_min=100, area_max=3000)
    barcode_nums_img_a2 = rm_blobs_using_hw_ratio(local_img=barcode_nums_img_a2, ratio_min=0.7, ratio_max=3.0)
    barcode_nums_img_a2 = rm_blobs_using_area(local_img=barcode_nums_img_a2, area_min=100, area_max=3000)

    # extracting all connected labels from num img into singular list
    number_labels_x_sorted_a1 = get_connected_labels_in_img_sorted_x(local_img=barcode_nums_img_a1)
    number_labels_x_sorted_a2 = get_connected_labels_in_img_sorted_x(local_img=barcode_nums_img_a2)

    # grouping remaining blobs (nums + noise) based on their y value
    y_grouped_num_blobs_a1 = group_blobs_by_y_val(number_labels_x_sorted=number_labels_x_sorted_a1, y_tolerance=25)
    y_grouped_num_blobs_a2 = group_blobs_by_y_val(number_labels_x_sorted=number_labels_x_sorted_a2, y_tolerance=25)
                        
    # again filtering the img to select only the area that has barcode digits
    excl_barcode_nums_a1 = max(y_grouped_num_blobs_a1, key=len) # presuming that there are more blobs left in the image for valid numbers
    excl_barcode_nums_a2 = max(y_grouped_num_blobs_a2, key=len) # presuming that there are more blobs left in the image for valid numbers
    if len(excl_barcode_nums_a1) < 12: # EAN-13 barcodes have 12 digits. This will only say that the barcode is valid if more than 10 blobs are visible (incase of missed digits)
        excl_barcode_nums_a1 = None
    elif len(excl_barcode_nums_a2) < 12:
        excl_barcode_nums_a2 = None

    # using angle 1 by default --> deciding which is the most suitable angle based on blob numbers
    excl_barcode_nums = excl_barcode_nums_a1
    if excl_barcode_nums_a1 is not None and excl_barcode_nums_a2 is not None: # presuming selected orientation is the correct one
        pass # leaving excl_barcode_nums as is
    elif excl_barcode_nums_a1 is None and excl_barcode_nums_a2 is not None:
        excl_barcode_nums = excl_barcode_nums_a2
    elif excl_barcode_nums_a1 is not None and excl_barcode_nums_a2 is None:
        excl_barcode_nums = excl_barcode_nums_a1
    else:
        return "" # neither of the "barcode nums" adhere to expected sizing (probably not numbers)

    # collecting all numbers as list of cropped images --> resizing to 28x28 for MNIST training
    cropped_nums = [] # stores the individual number images
    for number_label in excl_barcode_nums:
        x, y, w, h = number_label
        interest_region = barcode_nums_img_a1[y:y+h, x:x+w] # grabbing box around digit/blob
        resized_roi = resize_num_img_to_mnist_scale_ratiod(interest_region) # creating square img before resizing to MNIST scale
        cropped_nums.append(resized_roi)

    # analysing each collected digit and concatenating them into one string to return
    output_number_string = ""
    for square_img in cropped_nums:
        predicted_num, predict_conf = predict_number_from_28x28_img(square_img)
    
        if predicted_num != -1 and (len(output_number_string) <= 12): # don't count anything after 13 digits
            output_number_string += str(predicted_num)

    return output_number_string

def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        if type(content) == np.ndarray:
            cv2.imwrite(output_path, content)
        else:
            content.save(output_path)

        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")


def run_task4(sub_dir, config=None):
    for image_path in os.listdir(sub_dir):
        # setting the filename based on the current image letter/name
        abs_img_path = os.path.join(sub_dir, image_path)
        filename = os.path.basename(image_path)
        output_filename = filename.replace(".jpg", "")

        # calculating the barcode number string (if available)
        output_num_string = task4(input_img_loc=abs_img_path)
        if output_num_string == "": # don't print anything to file if output number dne
            print("no barcode detected")
            continue # skipping when no barcode
        
        print(f"Barcode String: {output_num_string}")

        # defining the save location and output
        output_path = f"output/task4/{output_filename}.txt"
        save_output(output_path, output_num_string, output_type='txt')
