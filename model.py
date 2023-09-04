import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import *
 

def create_MLP(dim):
    # Define the input layer for the MLP model
    inputs = Input(shape=(dim,))
    # Add two fully connected layers with ReLU activation and dropout
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Add the output layer with softmax activation for multi-class classification
    outputs = Dense(5, activation='softmax')(x)
    # Define the model with the input and output layers
    model = Model(inputs=inputs, outputs=outputs)
    return model

 

def create_CNN(width, height, depth, filters=(16, 32, 64)):
    # Define the input layer for the CNN model
    inputs = Input(shape=(height, width, depth))

    # Add a convolutional layer with ReLU activation and max pooling
    x = Conv2D(filters[0], (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Add a second convolutional layer with ReLU activation and max pooling
    x = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Add a third convolutional layer with ReLU activation and max pooling
    x = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output from the convolutional layers
    x = Flatten()(x)

    # Add a fully connected layer with ReLU activation and dropout
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Add the output layer with softmax activation for multi-class classification
    outputs = Dense(5, activation='softmax')(x)

    # Define the model with the input and output layers
    model = Model(inputs=inputs, outputs=outputs)

    return model

 

def combine_mlp_cnn(mlp_model, cnn_model):
    # Get the output from the MLP model
    mlp_output = mlp_model.output

    # Get the output from the CNN model
    cnn_output = cnn_model.output

    # Concatenate the outputs from the MLP and CNN models
    combined_output = Concatenate()([mlp_output, cnn_output])

    # Add a fully connected layer with ReLU activation and dropout
    x = Dense(64, activation='relu')(combined_output)
    x = Dropout(0.5)(x)

    # Add the output layer with softmax activation for multi-class classification
    outputs = Dense(5, activation='softmax')(x)

    # Define the combined model with the input layers from both the MLP and CNN models and the output layer
    model = Model(inputs=[mlp_model.input, cnn_model.input], outputs=outputs)

    return model

 

def train_model(train_X, train_y, model):
    # Compile the model with the Adam optimizer and categorical crossentropy loss function
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Split the training data into training and validation sets
    num_samples = train_X.shape[0]
    split_idx = int(0.8 * num_samples)
    train_indices = np.arange(num_samples)
    np.random.shuffle(train_indices)
    train_indices_train = train_indices[:split_idx]
    train_indices_val = train_indices[split_idx:]
    train_X_train, train_y_train = train_X[train_indices_train], train_y[train_indices_train]
    train_X_val, train_y_val = train_X[train_indices_val], train_y[train_indices_val]

    # Train the model with the training data and validation set
    history = model.fit([train_X_train[:, :-1], train_X_train[:, -1]], train_y_train,
                        validation_data=([train_X_val[:, :-1], train_X_val[:, -1]], train_y_val),
                        epochs=50, batch_size=32)

    return model, history

 

def evaluate_model(test_X, test_y, model):
    # Use the model to make predictions on the test data
    y_pred = model.predict([test_X[:, :-1], test_X[:, -1]])
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_y, axis=1)

    # Calculate the accuracy, precision, recall, and f1-score using macro averaging
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1

 

def main():
    # Load the pet adoption dataset

    dataset=load_data('/Users/lena/Desktop/MLproject/train.csv')


    # Split the data into train and testing sets
    # train_df = df.sample(frac=0.8, random_state=42)
    # test_df = df.drop(train_df.index)
    train_X, test_X, train_y, test_y=split_data(dataset)


    # Resize the images to be 32 x 32 pixels
    # train_X_imgs = np.load('train_X_imgs.npy')
    # test_X_imgs = np.load('test_X_imgs.npy')
    # train_X_imgs = tf.image.resize(train_X_imgs, size=(32, 32)).numpy()
    # test_X_imgs = tf.image.resize(test_X_imgs, size=(32, 32)).numpy()
    train_imgs, test_imgs=preprocess_images(train_X, test_X, '/Users/lena/Desktop/MLproject/train_images/')

    # Preprocess the categorical and numerical data
    scaler = MinMaxScaler()
    train_X_num = scaler.fit_transform(train_df[['Age', 'Fee']])
    test_X_num = scaler.transform(test_df[['Age', 'Fee']])

    # encoder = OneHotEncoder(sparse=False)
    train_X_cat = encoder.fit_transform(train_df[['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'Color3']])
    test_X_cat = encoder.transform(test_df[['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'Color3']])

    train_y = encoder.fit_transform(train_df[['AdoptionSpeed']])
    test_y = encoder.transform(test_df[['AdoptionSpeed']])
    pr_train_X, pr_test_X, pr_train_y, pr_test_y=preprocess_data(train_X, test_X, train_y, test_y)

    # Combine the numerical and categorical data for the MLP model
    train_X_mlp = np.hstack((train_X_num, train_X_cat))
    test_X_mlp = np.hstack((test_X_num, test_X_cat))

    # Define the MLP and CNN models
    mlp_model = create_MLP(dim=train_X_mlp.shape[1])
    cnn_model = create_CNN(width=32, height=32, depth=3, filters=(16, 32, 64))

    # Combine the MLP and CNN models into one model
    combined_model = combine_mlp_cnn(mlp_model, cnn_model)

    # Train the combined model
    trained_model, history = train_model(np.hstack((train_X_mlp, train_X_imgs.reshape(-1, 32*32*3))),
                                          train_y, combined_model)

    # Evaluate the trained model
    accuracy, precision, recall, f1 = evaluate_model(np.hstack((test_X_mlp, test_X_imgs.reshape(-1, 32*32*3))),
                                                     test_y, trained_model)

    print("Accuracy: {:.2f}%".format(accuracy*100))
    print("Precision: {:.2f}%".format(precision*100))
    print("Recall: {:.2f}%".format(recall*100))
    print("F1 Score: {:.2f}%".format(f1*100))


main()