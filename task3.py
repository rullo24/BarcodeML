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
from datetime import datetime
from torch import nn
import numpy as np
import torch
import cv2
import sys
import os  

class Image_Classify(nn.Module):
    def __init__(self):
        super(Image_Classify, self).__init__()
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

def get_main_dir():
    return os.path.dirname(__file__)

def conv_gray_img_to_bin_img(gray_img):
    _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    return threshold_img

def train_nn(training_dataset, model, loss_fn, opt, scheduler, num_epochs, device, save_freq):
    main_dir_loc: str = get_main_dir()

    for epoch in range(num_epochs):
        for batch in training_dataset:
            x, y = batch
            x, y = x.to(device), y.to(device)

            # predicting value w/ current model for error calculations
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            # applying the back propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

        scheduler.step() # moving the scheduler down 1x epoch
        print(f"Epoch: {epoch} | Loss: {loss.item()}")
   
        if epoch % save_freq == 0:
            curr_dt_str = datetime.now().strftime("%d%%%m%%%y_%H:%M:%S")
            save_name: str = f"model_{curr_dt_str}.pt"
            save_loc: str = os.path.join(main_dir_loc, save_name)
            with open(save_loc, "wb") as save_file:
                torch.save(model, save_file)
    return model

def predict_img_val(model, test_img, device):
    # unsqueeze adds batch size to tensor
    img_tensor = ToTensor()(test_img).unsqueeze(0).to(device) 
    
    # passing the tensor through the trained model to make predictions
    model_raw_vals = model(img_tensor) # running the tensor through the model

    # returns the probabilities of each value and the item (number) itself
    output_probabilities = torch.nn.functional.softmax(model_raw_vals, dim=1) 
    return output_probabilities, torch.argmax(model(img_tensor)).item()

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

# returns number that is predicted --> -1 if not enough confidence (presumed not number)
def task3(image_path):
    # init basic vars
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loading the test img and model for predictions
    input_img_bgr = cv2.imread(image_path)
    input_img_bin = conv_gray_img_to_bin_img(cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2GRAY))

    # loading the pretrained pytorch model
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w') # discard the print of the "FutureWarning" for weights_only
    loaded_model = Image_Classify().to(device)
    loaded_states = torch.load(os.path.join("data", "model_100_epochs.pt"), map_location=torch.device(device))
    loaded_model.load_state_dict(loaded_states) 
    sys.stderr.flush() # ensuring all bs "FutureWarning" text is written to devnull (not displayed to console)
    sys.stderr = stderr # redirecting stdout back to the default stdout

    # creating predictions of number in image using trained model
    resized_roi = resize_num_img_to_mnist_scale_ratiod(input_img_bin) # creating square img before resizing to MNIST scale
    output_probs_tensor, output_val = predict_img_val(model=loaded_model, test_img=resized_roi, device=device) # predicting
    output_probs_list = output_probs_tensor.detach().tolist()[0] # converting tensor to usable Python List
    prediction_confidence = output_probs_list[output_val] # getting confidence values for each possible num

    # returning number and suggested confidence
    return output_val, prediction_confidence
    
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

def run_task3(sub_dir, config=None):
# NOTE: don't check if values are actually existent/correct in task3 ONLY --> numbers always provided
    for barcode_folder_loc in os.listdir(sub_dir): # getting each barcode folder for iterating over
        abs_barcode_loc = os.path.join(sub_dir, barcode_folder_loc)
        dir_name = os.path.basename(barcode_folder_loc)
        letter = dir_name[-1]

        for file in os.listdir(abs_barcode_loc): # getting each number snip in the barcode folder
            file_abs_path = os.path.join(abs_barcode_loc, file)
            output_filename = file.replace(".png", "") # changing the extension
            
            # running image through prediction model --> predicting what num
            pred_output_val, pred_conf = task3(image_path=file_abs_path)
            print(f"Digit Prediction: {pred_output_val}")

            # saving the output to the path as explained
            output_path = f"output/task3/barcode{letter}/{output_filename}.txt"
            save_output(output_path, str(pred_output_val), output_type='txt')