from .architectures import unet_model
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow as tf

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import re

from tensorflow.keras.models import load_model

# Create the U-Net model
unet = unet_model()
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# def segment_image(image_tensor):
#     segmented_image = unet.predict(tf.expand_dims(image_tensor, axis=0))
#     segmented_image = tf.squeeze(segmented_image).numpy()
#     fig, ax = plt.subplots()
#     ax.imshow(segmented_image, cmap='gray')
#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     buffer.close()
#     return base64.b64encode(image_png).decode('utf-8')



from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate


def unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # First DownConvolution / Encoder Leg will begin, so start with Conv2D
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
    bn1 = Activation("relu")(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation("relu")(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(pool1)
    bn2 = Activation("relu")(conv2)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation("relu")(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(pool2)
    bn3 = Activation("relu")(conv3)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation("relu")(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(pool3)
    bn4 = Activation("relu")(conv4)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation("relu")(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(pool4)
    bn5 = Activation("relu")(conv5)
    conv5 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation("relu")(bn5)

    """ Now UpConvolution / Decoder Leg will begin, so start with Conv2DTranspose
    The gray arrows (in the above image) indicate the skip connections that concatenate the encoder feature map with the decoder, which helps the backward flow of gradients for improved training. """
    """ After every concatenation we again apply two consecutive regular convolutions so that the model can learn to assemble a more precise output """

    up6 = concatenate([Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn5), conv4], axis=3)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(up6)
    bn6 = Activation("relu")(conv6)
    conv6 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation("relu")(bn6)

    up7 = concatenate([Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn6), conv3], axis=3)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(up7)
    bn7 = Activation("relu")(conv7)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation("relu")(bn7)

    up8 = concatenate([Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn7), conv2], axis=3)
    conv8 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(up8)
    bn8 = Activation("relu")(conv8)
    conv8 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation("relu")(bn8)

    up9 = concatenate([Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn8), conv1], axis=3)
    conv9 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(up9)
    bn9 = Activation("relu")(conv9)
    conv9 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation("relu")(bn9)

    conv10 = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(bn9)

    return Model(inputs=[inputs], outputs=[conv10])
# Load the architecture and weights
imaging_model = unet(input_size=(256, 256, 3))
imaging_model.load_weights('models/imaging/segmentation_model.h5')

# Custom functions if needed
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = tf.keras.backend.flatten(y_true)
    y_pred_flatten = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_flatten * y_pred_flatten)
    union = tf.keras.backend.sum(y_true_flatten) + tf.keras.backend.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def iou_coef(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    sum = tf.keras.backend.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou



""" ///////////////////////////////////////////////////// """

clinical_imputer = joblib.load('models/clinical/imputer.pkl')
clinical_scaler = joblib.load('models/clinical/scaler.pkl')
clinical_label_encoders = joblib.load('models/clinical/label_encoders.pkl')
clinical_model = load_model('models/clinical/clinical_model.h5', compile=False)

clinical_models = {"clinical_imputer": clinical_imputer, "clinical_scaler": clinical_scaler, "clinical_label_encoders": clinical_label_encoders, "clinical_model": clinical_model}

def preprocess_clinical_data(df, label_encoders, imputer, scaler):
    df.iloc[:, 4:] = imputer.transform(df.iloc[:, 4:])
    
    for column in ['Gender', 'Primary Site']:
        if column in df.columns:
            df[column] = label_encoders[column].transform(df[column])
    
    X_new = scaler.transform(df.drop(columns=['Case ID', 'Project']))
    
    return X_new


""" ///////////////////////////////////////////////////// """

gene_model_path = 'models/gene/gene_model.h5'
gene_scaler_path = 'models/gene/scaler.pkl'
gene_label_encoder_path = 'models/gene/label_encoder.pkl'

gene_model = load_model(gene_model_path, compile=False)
gene_scaler = joblib.load(gene_scaler_path)

gene_models = [gene_model, gene_scaler, gene_label_encoder_path]

def preprocess_gene_expression_data(new_data):
    def extract_numerical_value(column_data):
        return column_data.apply(lambda x: int(re.search(r'\d+', str(x).replace(',', '')).group()) if pd.notnull(x) else 0)
    
    new_data['SSM_Affected_Cases_in_Cohort'] = extract_numerical_value(new_data['# SSM Affected Cases in Cohort'])
    new_data['SSM_Affected_Cases_Across_GDC'] = extract_numerical_value(new_data['# SSM Affected Cases Across the GDC'])
    new_data['CNV_Gain'] = extract_numerical_value(new_data['# CNV Gain'])
    new_data['CNV_Loss'] = extract_numerical_value(new_data['# CNV Loss'])
    new_data['Mutations'] = new_data['# Mutations'].astype(int)
    
    features = ['SSM_Affected_Cases_in_Cohort', 'SSM_Affected_Cases_Across_GDC', 'CNV_Gain', 'CNV_Loss', 'Mutations']
    
    if not all(col in new_data.columns for col in features):
        raise ValueError(f"Input data must contain the following columns: {', '.join(features)}")
    
    X_new = new_data[features]
    X_new.fillna(X_new.median(), inplace=True)
    X_new_scaled = gene_models[1].transform(X_new)
    
    return X_new_scaled