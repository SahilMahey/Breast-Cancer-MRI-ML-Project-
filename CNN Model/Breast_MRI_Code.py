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
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator



target_images_dir = '/users/fs2/smahey/Project/All_256_images'
label_no_cancer = '0'
label_cancer = '1'

def count_png_images(dir):
    count = 0
    for file in os.listdir(dir):
        if file.lower().endswith('.png'):
            count += 1
    return count

def collect_image_paths_in_array(source_dir, target_array):
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.png'):  
            file_path = os.path.join(source_dir, file_name)
            target_array.append(file_path)

def _get_collected_test_train_val_img_array(dir):
    cancer_imgs = []
    collect_image_paths_in_array(os.path.join(target_images_dir,dir), cancer_imgs)
    return cancer_imgs

def random_rotate(image):
    temp_num = np.random.random()
    if temp_num > 0.5:
        if temp_num > 0.75:
            return np.rot90(image) # Rotate 90 degrees counterclockwise
        else:
            return np.rot90(image, k=-1)
            
    else:
        return image

def process_images_into_array(image_paths, label, output_list, size=(256, 256)):
    for img in image_paths:
        image = cv2.imread(img, cv2.IMREAD_COLOR)
        rotated_image  = random_rotate(image)
#         resized_image = cv2.resize(rotated_image, size, interpolation=cv2.INTER_LINEAR)
        
#         resized_image = resized_image / 255.0
        output_list.append([image, label])
    
def preprocess_image(image_path, size=(256, 256)):
    # Read the image from the given path
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Randomly choose between +90 and -90 degrees rotation
    rotated_image  = random_rotate(image)
    resized_image = cv2.resize(rotated_image, size, interpolation=cv2.INTER_LINEAR)
#     resized_image = resized_image / 255.0  # Normalization between 0 and 1
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
    y = to_categorical(y, 2)
    
    return {'X':X,'y':y}



def data_generator_with_augmentation(X_train, y_train, batch_size=32, num_augmented_images_per_sample=10):
    

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest'
    )

    generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    steps_per_epoch = int(np.ceil(num_augmented_images_per_sample * len(X_train) / batch_size))

    while True:
        for _ in range(steps_per_epoch):
            augmented_batch = next(generator)
            yield augmented_batch


def plot_image(array):
    num_images_to_plot = 10
    plt.figure(figsize=(10, 10))
    for i in range(num_images_to_plot):
        plt.subplot(1, num_images_to_plot, i + 1)
        plt.imshow(array[i][0].astype('uint8'))
        plt.title('label',array[i][1])
        plt.axis('off')
plt.show()



from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_loss = ModelCheckpoint('best_model_loss.h5', monitor='val_loss', save_best_only=True, verbose=1)
checkpoint_acc = ModelCheckpoint('best_model_acc.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)


from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.regularizers import l2


def make_cnn_model(size):
    tf.random.set_seed(42)

    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(size, size, 3))

    # Freeze the base model layers
    base_model.trainable = True
    
    for layer in base_model.layers[:-20]:  # Freeze all layers except the last 20
        layer.trainable = False

    model_cnn = tf.keras.Sequential([
    base_model,
    
#     tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#     tf.keras.layers.Dropout(0.3),
    
#     tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#     tf.keras.layers.Dropout(0.3),
    
#     tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Dropout(0.3),
    
#     tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Dropout(0.3),
        
    
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(1024, activation='relu',kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

    model_cnn.summary()

    model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=CategoricalCrossentropy(),
              metrics=['accuracy',tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='AUC')])
    return model_cnn


def fit_model(temp_model,train_data,train_y,test_data,test_y):
    history = temp_model.fit(
    train_data,
    train_y,
    validation_data = (test_data, test_y),epochs=25 ,callbacks=[reduce_lr, early_stopping,checkpoint_loss,checkpoint_acc],verbose=2)
    return history

def fit_model_with_generator(temp_model, X_train, y_train, X_test, y_test, batch_size=64, epochs=25, num_augmented_images_per_sample=10):
    train_generator = data_generator_with_augmentation(X_train, y_train, batch_size, num_augmented_images_per_sample)
    steps_per_epoch = int(np.ceil(num_augmented_images_per_sample * len(X_train) / batch_size))
    
    history = temp_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_test),
        callbacks=[reduce_lr, early_stopping,checkpoint_loss,checkpoint_acc],
        epochs=epochs,
        verbose=2
    )
    
    return history

# In[24]:


def evaluate_model(temp_model,val_img,val_label):
    evaluation = temp_model.evaluate(val_img,val_label)
    return evaluation


# In[25]:


import matplotlib.pyplot as plt

def plot_metrices_graph(history,image_size,text):
    fig, ax = plt.subplots(3, 2, figsize=(15, 30))
    ax = ax.ravel()

    metrics = ['accuracy', 'loss', 'precision', 'recall', 'AUC']

    for i, met in enumerate(metrics):
        ax[i].plot(history.history[met], label='Train')
        ax[i].plot(history.history['val_' + met], label='Rest')
        ax[i].set_title(f'Model {met} for {text} Images ({image_size}px)')
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(met)
        ax[i].legend()
    
    fig.delaxes(ax[-1])

    # Save the plots to a file with a descriptive filename
    plt.tight_layout()
    plt.savefig(f'{text}_cnn_cancer_metrices_{image_size}.png')
    plt.close()  # Close the plot to free up memory



# In[26]:

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

def sixty_four_model_non_augmented():
    print ("\nsixty_four_model_non_augmented\n")
    train_0_cancer = _get_collected_test_train_val_img_array('train_by_256/0')
    train_1_cancer = _get_collected_test_train_val_img_array('train_by_256/1')
    test_0_cancer  = _get_collected_test_train_val_img_array('test_by_256/0')
    test_1_cancer  = _get_collected_test_train_val_img_array('test_by_256/1')
   
    min_len = reduce_train_array_to_min(train_0_cancer,train_1_cancer)
    
    print(min_len)
    
    train_0_cancer = train_0_cancer[:min_len]
    train_1_cancer = train_1_cancer[:min_len]
     
    min_len = reduce_train_array_to_min(test_0_cancer,test_1_cancer)
    
    print("\ntest: ",min_len)
    
    test_0_cancer = test_0_cancer[:min_len]
    test_1_cancer = test_1_cancer[:min_len]
 
    train_processed_0 = get_processed_images(train_0_cancer,0)
    train_processed_1 = get_processed_images(train_1_cancer,1)
    test_processed_0  = get_processed_images(test_0_cancer,0)
    test_processed_1  = get_processed_images(test_1_cancer,1)
    
    
    train_shuffled_array = shuffle_array(train_processed_0,train_processed_1)
    test_shuffled_array  = shuffle_array(test_processed_0,test_processed_1)
    
    train_array = split_data(train_shuffled_array)
    test_array  = split_data(test_shuffled_array)
    
    model_256 = make_cnn_model(256)
    history  = fit_model(model_256,train_array['X'],train_array['y'],test_array['X'],test_array['y'])
    predictions = model_256.predict(X_val_processed)
    report = classification_report(y_val, np.argmax(predictions, axis=1))
    print(report)
    conf_matrix = confusion_matrix(y_val,np.argmax(predictions, axis=1) )
    print('Confusion Matrix:\n', conf_matrix)
    plot_metrices_graph(history,256,"non_augment")
    model_256.save('model_256_non_augment_cancer_cnn.keras')
    
def sixty_four_model_augmented():
    print ("\nsixty_four_model_augmented\n")
    train_0_cancer = _get_collected_test_train_val_img_array('train_by_256/0')
    train_1_cancer = _get_collected_test_train_val_img_array('train_by_256/1')
    test_0_cancer  = _get_collected_test_train_val_img_array('test_by_256/0')
    test_1_cancer  = _get_collected_test_train_val_img_array('test_by_256/1')
    
        
    
    min_len = reduce_train_array_to_min(train_0_cancer,train_1_cancer)
    
    print(min_len)
    
    train_0_cancer = train_0_cancer[:min_len]
    train_1_cancer = train_1_cancer[:min_len]
    
    min_len = reduce_train_array_to_min(test_0_cancer,test_1_cancer)
    
    print("\ntest: ",min_len)
    
    test_0_cancer = test_0_cancer[:min_len]
    test_1_cancer = test_1_cancer[:min_len]
    
    train_processed_0 = get_processed_images(train_0_cancer,0)
    train_processed_1 = get_processed_images(train_1_cancer,1)
    test_processed_0  = get_processed_images(test_0_cancer,0)
    test_processed_1  = get_processed_images(test_1_cancer,1)
    
    
    train_shuffled_array = shuffle_array(train_processed_0,train_processed_1)
    test_shuffled_array  = shuffle_array(test_processed_0,test_processed_1)
   
    
    train_array = split_data(train_shuffled_array)
    test_array  = split_data(test_shuffled_array)
    
    
    print("\nEntered the augmented function\n")
    model_256 = make_cnn_model(256)
    history  = fit_model_with_generator(model_256, train_array['X'], train_array['y'], test_array['X'], test_array['y'])
    predictions = model_256.predict(X_val_processed)
    report = classification_report(y_val, np.argmax(predictions, axis=1))
    print(report)
    conf_matrix = confusion_matrix(y_val,np.argmax(predictions, axis=1) )
    print('Confusion Matrix:\n', conf_matrix)
    plot_metrices_graph(history,256,"augment")
    model_256.save('model_256_augment_cancer_cnn.keras')



sixty_four_model_augmented()