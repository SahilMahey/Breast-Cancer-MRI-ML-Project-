
import os
import cv2
import random
import pydicom
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from data import prepare_data
import torch
from torch import nn, optim
from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication
from train import Trainer
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



target_images_dir = '/users/fs2/smahey/Project/All_256_images'
label_no_cancer = '0'
label_cancer = '1'
num_augmented_images_per_sample = 50

device = "cpu"

def random_rotate(image):
    temp_num = np.random.random()
    if temp_num > 0.5:
        if temp_num > 0.75:
            return np.rot90(image) # Rotate 90 degrees counterclockwise
        else:
            return np.rot90(image, k=-1)
            
    else:
        return image

def collect_image_paths_in_array(source_dir, target_array):
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.png'):  
            file_path = os.path.join(source_dir, file_name)
            target_array.append(file_path)

def _get_collected_test_train_val_img_array(dir):
    cancer_imgs = []
    collect_image_paths_in_array(os.path.join(target_images_dir,dir), cancer_imgs)
    return cancer_imgs



def process_images_into_array(image_paths, label, output_list, size=(256, 256)):
    for img in image_paths:
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        rotated_image  = random_rotate(image)
#         resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        output_list.append([rotated_image, label])

def preprocess_image(image_path, size=(256, 256)):
    # Read the image from the given path
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Randomly choose between +90 and -90 degrees rotation
    rotation_angle = random.choice([90, -90])
    # Rotate the image by the chosen angle
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Resize the rotated image to the desired size
    resized_image = cv2.resize(rotated_image, size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def reduce_train_array_to_min(train_0_array,train_1_array):
    min_len = min(len(train_0_array), len(train_1_array))
    return min_len  


def get_processed_images(array,label):
    temp_cancer_array = []
    process_images_into_array(array,label,temp_cancer_array)
    return temp_cancer_array



def shuffle_array(temp_array_0,temp_array_1):
    breast_img_arr = temp_array_0 + temp_array_1
    random.shuffle(breast_img_arr)
    return breast_img_arr

def split_data(random_shuffled_array):
    X = []
    y = []

    for feature, label in random_shuffled_array:
        X.append(feature)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    
    return {'X':X,'y':y}

def make_vit_model(size=256,patch_size=16):
    
    config = {
    "patch_size": patch_size,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072, # 4 * hidden_size
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": size,
    "num_classes": 2, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
    
    return config

# Define the directories
dir_0 = '/users/fs2/smahey/Project/Validation/0'
dir_1 = '/users/fs2/smahey/Project/Validation/1'

# Function to get file paths and labels
def get_image_paths_and_labels(directory, label):
    # Get the list of all files in the directory
    file_names = os.listdir(directory)
    # Create the full path for each file and pair with the label
    full_paths = [os.path.join(directory, file_name) for file_name in file_names]
    return full_paths, [label] * len(full_paths)

# Get paths and labels for both directories
paths_0, labels_0 = get_image_paths_and_labels(dir_0, 0)
paths_1, labels_1 = get_image_paths_and_labels(dir_1, 1)


# Combine the paths and labels into a DataFrame
data = {
    'image_full_path': paths_0 + paths_1,
    'label': labels_0 + labels_1
}
df = pd.DataFrame(data)

# Separate the DataFrame into two groups based on labels
df_0 = df[df['label'] == 0]
df_1 = df[df['label'] == 1]

# Find the smaller group size
min_size = min(len(df_0), len(df_1))

# Sample the same number of rows from each group
df_0_balanced = df_0.sample(min_size, random_state=42)
df_1_balanced = df_1.sample(min_size, random_state=42)

# Combine the balanced groups into a new DataFrame
df_balanced = pd.concat([df_0_balanced, df_1_balanced]).reset_index(drop=True)

# Shuffle the combined DataFrame (optional)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X_val = df_balanced['image_full_path']
y_val = df_balanced['label']
X_val_processed = np.array([preprocess_image(path) for path in X_val])
y_val = np.array(y_val)

def sixty_four_model_non_augmented():
    train_0_cancer = _get_collected_test_train_val_img_array('train_by_256/0')
    train_1_cancer = _get_collected_test_train_val_img_array('train_by_256/1')
    test_0_cancer  = _get_collected_test_train_val_img_array('test_by_256/0')
    test_1_cancer  = _get_collected_test_train_val_img_array('test_by_256/1')
    
    min_len_train = reduce_train_array_to_min(train_0_cancer,train_1_cancer)
  
    min_len_test = reduce_train_array_to_min(test_0_cancer,test_1_cancer)
    
    train_0_cancer = train_0_cancer[:min_len_train]
    train_1_cancer = train_1_cancer[:min_len_train]
    test_0_cancer = test_0_cancer[:min_len_test]
    test_1_cancer = test_1_cancer[:min_len_test]
    
    train_processed_0 = get_processed_images(train_0_cancer,0)
    train_processed_1 = get_processed_images(train_1_cancer,1)
    test_processed_0  = get_processed_images(test_0_cancer,0)
    test_processed_1  = get_processed_images(test_1_cancer,1)

    
    total_len_train_array = len(train_0_cancer) + len (train_1_cancer)
    
    train_shuffled_array = shuffle_array(train_processed_0,train_processed_1)
    test_shuffled_array  = shuffle_array(test_processed_0,test_processed_1)
   
    train_array = split_data(train_shuffled_array)
    test_array  = split_data(test_shuffled_array)
    
    print("\nData Processed\n")
    
    trainloader, testloader, valloader = prepare_data(batch_size=64,x_train=train_array['X'],
        y_train=train_array['y'],
        x_test=test_array['X'],
        y_test=test_array['y'],
        x_val=X_val,
        y_val=y_val)
    
    print("\nData Loaded")
    
    steps_per_epoch = int(np.ceil(num_augmented_images_per_sample * (total_len_train_array) / 64))
    model_256_naug = make_vit_model(256,16)
    model = ViTForClassfication(model_256_naug)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, exp_name = "model-256-non-aug", device=device)
    trainer.train(trainloader, testloader, epochs=100, config = model_256_naug, steps_per_epoch = steps_per_epoch, augment = False,  save_model_every_n_epochs = 0)
    
    print("Validating on validation data...")
    val_predictions = trainer.predict(valloader)

    # Calculate accuracy on validation data
    val_labels = y_val
    val_accuracy = np.mean(val_predictions == val_labels)
    print(f"Validation Accuracy: {val_accuracy:.4f}")


def sixty_four_model_augmented():
    train_0_cancer = _get_collected_test_train_val_img_array('train_by_256/0')
    train_1_cancer = _get_collected_test_train_val_img_array('train_by_256/1')
    test_0_cancer  = _get_collected_test_train_val_img_array('test_by_256/0')
    test_1_cancer  = _get_collected_test_train_val_img_array('test_by_256/1')
  
    
    min_len_train = reduce_train_array_to_min(train_0_cancer,train_1_cancer)
  
    min_len_test = reduce_train_array_to_min(test_0_cancer,test_1_cancer)
    
    train_0_cancer = train_0_cancer[:min_len_train]
    train_1_cancer = train_1_cancer[:min_len_train]
    test_0_cancer = test_0_cancer[:min_len_test]
    test_1_cancer = test_1_cancer[:min_len_test]
    
    train_processed_0 = get_processed_images(train_0_cancer,0)
    train_processed_1 = get_processed_images(train_1_cancer,1)
    test_processed_0  = get_processed_images(test_0_cancer,0)
    test_processed_1  = get_processed_images(test_1_cancer,1)
        
    train_shuffled_array = shuffle_array(train_processed_0,train_processed_1)
    test_shuffled_array  = shuffle_array(test_processed_0,test_processed_1)

    train_array = split_data(train_shuffled_array)
    test_array  = split_data(test_shuffled_array)
    
    print("\nData Processed\n")
    
    trainloader, testloader, valloader = prepare_data(batch_size=64,x_train=train_array['X'],
        y_train=train_array['y'],
        x_test=test_array['X'],
        y_test=test_array['y'],
        x_val=X_val,
        y_val=y_val)
    
    print("\nData Loaded")
    steps_per_epoch = int(np.ceil(num_augmented_images_per_sample * (total_len_train_array) / 64))
    model_256_aug = make_vit_model(256,16)
    model = ViTForClassfication(model_256_aug)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, exp_name = "model-256-aug", device=device)
    trainer.train(trainloader, testloader, epochs=100, config = model_256_aug, steps_per_epoch = steps_per_epoch, augment = True, save_model_every_n_epochs = 0)
    
    print("Validating on validation data...")
    val_predictions = trainer.predict(valloader)

    # Calculate accuracy on validation data
    val_labels = y_val
    val_accuracy = np.mean(val_predictions == val_labels)
    print(f"Validation Accuracy: {val_accuracy:.4f}")


def main():
    sixty_four_model_non_augmented() 
    print("\n Entering the Augmented Function\n")
    sixty_four_model_augmented()

if __name__ == "__main__":
    main() 