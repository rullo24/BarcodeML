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

# from collections import Counter
import numpy as np
import math
import cv2
import os

def get_main_dir():
    return os.path.dirname(__file__)

def conv_gray_img_to_bin_img(gray_img):
    gaussian_blur = cv2.GaussianBlur(gray_img, (7, 7), 0) # 7x7 kernel filter
    equalised_img = cv2.equalizeHist(gaussian_blur)
    _, threshold_img = cv2.threshold(equalised_img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    return threshold_img

def rm_blobs_using_hw_ratio(local_img, ratio_min, ratio_max):
    num_labs, labels, stats, _ = cv2.connectedComponentsWithStats(local_img, connectivity=4) # must share an edge to be "connected"
    img_rm_blobs = np.zeros_like(local_img) # Create an output image to draw the filtered components
    for label in range(1, num_labs): # Iterate through each component (excluding the background label 0)
        ratio = stats[label, cv2.CC_STAT_HEIGHT]/stats[label, cv2.CC_STAT_WIDTH]
        if ratio_min <= ratio <= ratio_max:
            img_rm_blobs[labels == label] = 255 # checking if within desired range
    return img_rm_blobs

def rm_blobs_using_area(local_img, area_min, area_max):
    num_labs, labels, stats, _ = cv2.connectedComponentsWithStats(local_img, connectivity=4) # must share an edge to be "connected"
    img_rm_blobs = np.zeros_like(local_img) # Create an output image to draw the filtered components
    for label in range(1, num_labs): # Iterate through each component (excluding the background label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area_min <= area <= area_max:
            img_rm_blobs[labels == label] = 255 # If the area is within the desired range, add the component to the output image
    return img_rm_blobs

def canny_edge_detection(gray_img, t1, t2):
    return cv2.Canny(gray_img, threshold1=t1, threshold2=t2)

def hough_line_detection(canny_edges, t1, min_line_len):
    return cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=t1, minLineLength=min_line_len) 

def calc_angle(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

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
    return rotated_image, rotation_matrix

# applying an inverse rotation algo to backprop the loc of bbox in rotate img --> original img
def identify_corners_inv_rotate(rect_corners, rot_matrix):
    inv_rot_matrix = cv2.invertAffineTransform(rot_matrix) # create the inverse of the affine transformation matrix
    rect_corners = np.array(rect_corners, dtype=np.float32) # convert the corners to a np arr with coords
    
    # add column of 1's for homogeneous transformation
    ones = np.ones((rect_corners.shape[0], 1), dtype=np.float32)
    homogeneous_corners = np.hstack([rect_corners, ones])
    
    # apply inv affine transformation to each corner
    original_corners = np.dot(homogeneous_corners, inv_rot_matrix.T)
    return original_corners

# returns the boundary box 4x corners from the label format (x, y, w, h) 
def get_corners_from_rect_label(rect_label) -> list: # top-left > top-right > bottom-right > bottom-left > top-left
    x, y, w, h = rect_label
    x1, y1 = x, y
    x2, y2 = (x+w), y
    x3, y3 = (x+w), (y+h) 
    x4, y4 = x, (y+h)
    return [(x1 , y1), (x2, y2), (x3, y3), (x4, y4)]

# merging blob groups based on their relative proximity
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

# getting the maximum and minimum edges of the larger bbox, encompassing all smaller bboxs
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

def update_img_to_show_hough_lines(lined_img, hough_lines):
    if hough_lines is not None:
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lined_img, (x1, y1), (x2, y2), (0, 255, 0), 7)  # draw line on prev img
    return None

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

def get_connected_labels_in_img_sorted_x(local_img):
    num_labs, _, stats, _ = cv2.connectedComponentsWithStats(local_img, connectivity=4) # must share an edge to be "connected"
    
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

def get_bounding_box_from_rect_tuples(rect_tuples):
    min_x = min(x for x, _, _, _ in rect_tuples) 
    min_y = min(y for _, y, _, _ in rect_tuples)
    max_x = max(x for x, _, _, _ in rect_tuples) 
    max_y = max(y for _, y, _, _ in rect_tuples)
    min_w = min(w for _, _, w, _ in rect_tuples)
    max_h = max(h for _, _, _, h in rect_tuples) 
    group_tuple = (min_x, min_y, (max_x - min_x + min_w), (max_y - min_y + max_h))
    return group_tuple

# returns (default_img_w_rect, number_cropped_img)
def task1(input_img_loc): # -> tuple[np.ndarray, np.ndarray] || tuple[None, None]
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
    bar_numbers_img = rm_blobs_using_hw_ratio(local_img=last_kernel_img, ratio_min=0.5, ratio_max=3.0)
    bar_numbers_img = rm_blobs_using_area(local_img=bar_numbers_img, area_min=100, area_max=3000)
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
        return None, None
    if len(hough_lines) < 10: # not enough "hard lines" to be considered a barcode
        return None, None

    # applying perspective transformation for correct orientation --> 0deg and 180 grabbed to avoid missing nums as a result of incorrect flip
    expected_rot_a1 = dominant_orientation - 90 # perpendicular barcode to horizontal axis
    expected_rot_a2 = expected_rot_a1 + 180 # flipping on head

    # rotating imgs are per described lines --> both angles used to avoid discrepancies due to text on top of barcode (and failed rotation)
    rotate_bin_barcode_lines_a1, _ = rotate_img(barcode_img, expected_rot_a1)
    rotate_bin_barcode_nums_a1, barcode_nums_rot_matrix_a1 = rotate_img(bar_numbers_img, expected_rot_a1)
    rotate_bgr_barcode_a1, _ = rotate_img(input_img_bgr, expected_rot_a1)
    rotate_bin_barcode_lines_a2, _ = rotate_img(barcode_img, expected_rot_a2)
    rotate_bin_barcode_nums_a2, barcode_nums_rot_matrix_a2 = rotate_img(bar_numbers_img, expected_rot_a2)
    rotate_bgr_barcode_a2, _ = rotate_img(input_img_bgr, expected_rot_a2)

    # filter barcode lines from bin image based on proximity between lines
    _, barcode_stats_a1 = calc_edges_barcode_from_prox_conn(local_img=rotate_bin_barcode_lines_a1, prox_max_x=250, prox_max_y=250)
    _, barcode_stats_a2 = calc_edges_barcode_from_prox_conn(local_img=rotate_bin_barcode_lines_a2, prox_max_x=250, prox_max_y=250)
    
    # filter barcode numbers based on proximity to barcode lines
    barcode_nums_img_a1 = calc_loc_barcode_nums_from_prox_barcode_lines(local_img=rotate_bin_barcode_nums_a1, barcode_stats=barcode_stats_a1, 
                                                                        x_threshold=100, y_threshold=150)
    barcode_nums_img_a2 = calc_loc_barcode_nums_from_prox_barcode_lines(local_img=rotate_bin_barcode_nums_a2, barcode_stats=barcode_stats_a2, 
                                                                        x_threshold=100, y_threshold=150)
    
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
    barcode_img_type = "a1" 
    if excl_barcode_nums_a1 is not None and excl_barcode_nums_a2 is not None: # presuming that the first rotation is correct
        pass
    elif excl_barcode_nums_a1 is None and excl_barcode_nums_a2 is not None:
        excl_barcode_nums = excl_barcode_nums_a2
        barcode_img_type = "a2"
    elif excl_barcode_nums_a1 is not None and excl_barcode_nums_a2 is None:
        excl_barcode_nums = excl_barcode_nums_a1
        barcode_img_type = "a1"
    else:
        return None, None # neither of the "barcode nums" adhere to expected sizing (probably not numbers)

    # getting the barcode number area (ONLY) as an image
    x_nums_tolerance: int = 50
    barcode_nums_bbox = get_bounding_box_from_rect_tuples(excl_barcode_nums)

    # adding x tolerances (for getting clean edge of barcode)
    nums_x, nums_y, nums_w, nums_h = barcode_nums_bbox
    barcode_nums_bbox = (nums_x - x_nums_tolerance), nums_y, (nums_w + 2*x_nums_tolerance), nums_h 
    nums_x, nums_y, nums_w, nums_h = barcode_nums_bbox

    # inverse rotated img bbox to find the coords of the bbox in original img
    barcode_nums_bbox_as_coords = get_corners_from_rect_label(barcode_nums_bbox)
    barcode_nums_bbox_img_bgr = None
    if barcode_img_type == "a1":
        barcode_nums_bbox_img_bgr = rotate_bgr_barcode_a1[nums_y:(nums_y + nums_h), (nums_x):(nums_x + nums_w)]
        original_img_bbox_corners = identify_corners_inv_rotate(rect_corners=barcode_nums_bbox_as_coords, rot_matrix=barcode_nums_rot_matrix_a1)
    elif barcode_img_type == "a2":
        barcode_nums_bbox_img_bgr = rotate_bgr_barcode_a2[nums_y:(nums_y + nums_h), (nums_x):(nums_x + nums_w)]
        original_img_bbox_corners = identify_corners_inv_rotate(rect_corners=barcode_nums_bbox_as_coords, rot_matrix=barcode_nums_rot_matrix_a2)
    else:
        return None, None

    # drawing a bbox around the nums on the original img --> not required for automation process
    # bbox_arr = np.array(original_img_bbox_corners, dtype=np.int32).reshape((-1, 1, 2)) # converting to np arr for polylines func
    # cv2.polylines(input_img_bgr, [bbox_arr], isClosed=True, color=(0, 0, 255), thickness=5) # drawing red bbox on original img

    # converting the bbox coords to a singular-lined string for printing to file
    original_bbox_coords_str = ', '.join(map(str, original_img_bbox_corners.flatten()))

    return barcode_nums_bbox_img_bgr, original_bbox_coords_str

# part of the EXT (COMP3007) framework for marking
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

# part of the EXT (COMP3007) framework for marking
def run_task1(sub_dir, config=None):
    for image_path in os.listdir(sub_dir):
        # removing the file extension from the image path for reformatting w/ same name i.e hey.jpg --> hey.txt
        abs_img_path = os.path.join(sub_dir, image_path)
        test_img_basename_w_ext = os.path.basename(image_path)
        img_num = test_img_basename_w_ext[3] # considers the image is in form "imgX.jpg"

        # performing all calcs required for task 1 completion
        barcode_nums_bbox_img, original_img_bbox_corners = task1(input_img_loc=abs_img_path) 
        if barcode_nums_bbox_img is None or original_img_bbox_corners is None:
            print("no barcode found")
            continue

        # carrying through with COMP3007 provided framework
        output_path_txt = f"output/task1/img{img_num}.txt"
        save_output(output_path_txt, original_img_bbox_corners, output_type='txt')
        output_path_png = f"output/task1/barcode{img_num}.png"
        save_output(output_path_png, barcode_nums_bbox_img, output_type='image')