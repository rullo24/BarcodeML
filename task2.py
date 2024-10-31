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

import numpy as np
import cv2
import os

def conv_gray_img_to_bin_img(gray_img):
    _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
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

def task2(input_img_loc):
    # NOTE: morphology methods not required in task2 because most pieces are already removed (will be required in task4)

    # gathering the image files as cv2 objects (imgs)
    input_img_bgr = cv2.imread(input_img_loc)
    input_img_bin = conv_gray_img_to_bin_img(cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2GRAY))

    # removing false nums based on number characteristics
    barcode_numbers_img = rm_blobs_using_area(local_img=input_img_bin, area_min=10, area_max=1000000)
    barcode_numbers_img = rm_blobs_using_hw_ratio(local_img=barcode_numbers_img, ratio_min=0.1, ratio_max=4.0)

    # grab all connected labels in the image
    barcode_conn_labels = get_connected_labels_in_img_sorted_x(local_img=barcode_numbers_img)

    # remove all connected labels that don't start from 1/2 down the page
    centre_y = input_img_bin.shape[0] // 2 # calculating centre y-val of the provided number image
    barcode_conn_labels_below_centre = []
    for index in range(len(barcode_conn_labels)):
        x, y, w, h = barcode_conn_labels[index]
        if (y+h) >= centre_y:
            barcode_conn_labels_below_centre.append((x, y, w, h))

    # get box around each remaining label and save to an img in the local dir
    output_list = [] # stores (img, str(x1, y1, x2, y2))
    for label in barcode_conn_labels_below_centre:
        x, y, w, h = label
        interest_region = input_img_bgr[y:y+h, x:x+w] # grabbing box around digit/blob from original img
        tl_coord = x, y
        br_coord = (x+w), (y+h)
        bbox_rect_str = f"{tl_coord[0]}, {tl_coord[1]}, {br_coord[0]}, {br_coord[1]}"
        output_list.append((interest_region, bbox_rect_str))

    return output_list # stores all of the nums (images) and their relavent bbox coords

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

def run_task2(sub_dir, config=None):
    for image_path in os.listdir(sub_dir): # iterating over every image in the sub directory
        # getting the save letter from the image name
        abs_img_path = os.path.join(sub_dir, image_path)
        file_dir_name = os.path.basename(image_path)
        curr_letter = file_dir_name[7] # considering dir form "barcodeX.png"

        # returns all single number imgs and bbox coords in a list
        output_data = task2(abs_img_path) 
        if len(output_data) == 0:
            print("no numbers available")
            return
        
        # iterating over all numbers so that they can all be saved as per the framework guidelines
        save_index = 1
        for val in output_data:
            save_index_str = str(save_index) if len(str(save_index)) > 1 else f"0{str(save_index)}" # adding a zero to the start of singular digits

            # saving the img to its required location
            img_output_path = f"output/task2/barcode{curr_letter}/d{save_index_str}.png"
            save_output(img_output_path, val[0], output_type='image') 

            # saving the digit coordinates
            txt_output_path = f"output/task2/barcode{curr_letter}/d{save_index_str}.txt"
            save_output(txt_output_path, val[1], output_type='txt') 

            # incrementing the counter by one (for next blob save name)
            save_index += 1