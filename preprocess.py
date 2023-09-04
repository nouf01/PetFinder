import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from PIL import Image
import cv2
import seaborn as sns


#delete images of the same pet "only keep the first one"
rootdir = "/Users/lena/Desktop/MLproject/train_images"
regex_img = re.compile('[a-zA-Z0-9]*-[1]\.jpg')
for root, dirs, files in os.walk(rootdir):
  for file in files:
    if regex_img.match(file):
       continue
    else:
       os.remove(rootdir+'/'+file)

files=os.listdir(path=rootdir)
files.sort()

#delete rows(pets) from csv that doesn't have images 
df=pd.read_csv("/Users/lena/Desktop/MLproject/train.csv")
i=df[df['PhotoAmt'] == 0].index
df=df.drop(i)
df.to_csv('train.csv', index=False)
df.sort_values(["PetID"],axis=0, ascending=True,inplace=True,na_position='first')

 

def load_data(path_to_data):
    # Load the dataset into a DataFrame
    df = pd.read_csv(path_to_data)
    df.drop('Name', axis=1, inplace=True)
    df.drop('Description', axis=1, inplace=True)
    df.drop('RescuerID', axis=1, inplace=True)
    # Display the head records from the dataset
    print("Head Records:")
    print(df.head())
    # Display basic statistics for each feature
    print("Basic Statistics:")
    print(df.describe())
    # Display label distribution
    print("Label Distribution:")
    print(df['AdoptionSpeed'].value_counts())
    # Visualize correlation between each feature and the label
    print("Correlation Matrix:")
    corr = df.corr().iloc[:-1]
   #  sns.heatmap(corr, annot=True, cmap="YlGnBu")
   #  plt.show()
    # Visualize distribution for all continuous value features using histograms
    print("Histograms:")
    df.hist(bins=20, figsize=(20,15))
    plt.show()
    # Return the loaded data
    return df 


def split_data(data):
    # Separate the features from the labels
    X = data.drop('AdoptionSpeed', axis=1)
    y = data['AdoptionSpeed']
    # Split the data into training and testing sets with an 80/20 partition
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    return train_X, test_X, train_y, test_y


def preprocess_data(train_X, test_X, train_y, test_y):
    # One-hot encode categorical features
    categorical_cols = train_X.select_dtypes(include='object').columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    train_X_categorical = encoder.fit_transform(train_X[categorical_cols])
    test_X_categorical = encoder.transform(test_X[categorical_cols])
    # Scale continuous features to be in range [0, 1]
    continuous_cols = train_X.select_dtypes(include=np.number).columns
    scaler = MinMaxScaler()
    train_X_continuous = scaler.fit_transform(train_X[continuous_cols])
    test_X_continuous = scaler.transform(test_X[continuous_cols])
    # Concatenate one-hot encoded categorical features and scaled continuous features
    pr_train_X = np.concatenate([train_X_categorical.toarray(), train_X_continuous], axis=1)
    pr_test_X = np.concatenate([test_X_categorical.toarray(), test_X_continuous], axis=1)
    # One-hot encode labels
    encoder = OneHotEncoder()
    pr_train_y = encoder.fit_transform(train_y.values.reshape(-1, 1)).toarray()
    pr_test_y = encoder.transform(test_y.values.reshape(-1, 1)).toarray()
    return pr_train_X, pr_test_X, pr_train_y, pr_test_y 
 

def preprocess_images(train_X, test_X, path_img):
    # Define the target size for the resized images
    target_size = (32, 32)
    # Create empty arrays to store the preprocessed images
    train_imgs = np.empty((len(train_X), *target_size , 3), dtype=np.uint8)
    test_imgs = np.empty((len(test_X), *target_size ,3 ), dtype=np.uint8)
    # Loop through the training image indices and preprocess each image
    for filename in files:
        i=0
        # Load the image using PIL
        image_path = path_img +filename 
        image = Image.open(image_path)
        # Resize the image to the target size
        image = image.resize(target_size)
        # Convert the PIL image to a NumPy array
        image_array = np.array(image)
        # Normalize the pixel values to be in the range [0, 255]
        image_array = (image_array / 255).astype(np.float32)
        # Add the preprocessed image to the training image array
        train_imgs[i] = image_array
        i+=1
    # Loop through the test image indices and preprocess each image
    for i, idx in enumerate(test_X):
        # Load the image using PIL
        image_path = path_img + str(idx) + '.jpg'
        image = Image.open(image_path)
        # Resize the image to the target size
        image = image.resize(target_size)
        # Convert the PIL image to a NumPy array
        image_array = np.array(image)
        # Normalize the pixel values to be in the range [0, 255]
        image_array = (image_array / 255).astype(np.float32)
        # Add the preprocessed image to the test image array
        test_imgs[i] = image_array
    return train_imgs, test_imgs

