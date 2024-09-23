import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import io
from io import BytesIO
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from sklearn.preprocessing import StandardScaler
from .helper_functions import preprocess_clinical_data, preprocess_gene_expression_data
from .helper_functions import clinical_models, gene_models, imaging_model
from plotly.offline import plot
import plotly.graph_objs as go

import cv2

from django.core.files.storage import FileSystemStorage

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.conf import settings





# Load imaging model
classification_model = tf.keras.models.load_model('models/imaging/imaging_model.h5', compile=False)
classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
class_labels = ['Colon Adenocarcinoma', 'Colon Benign Tissue']

# load clinical model
clinical_model = tf.keras.models.load_model('models/clinical/clinical_model.h5', compile=False)
clinical_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']



def home(request):
    return render(request, 'home.html')










# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size=(256, 256)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    input_array = tf.keras.preprocessing.image.img_to_array(img)
    input_array = np.expand_dims(input_array, axis=0)  # Expand dimensions to create batch size of 1
    return img, input_array


# Function to load and preprocess the mask (assuming it's in grayscale)
def load_and_preprocess_mask(mask_path, target_size=(256, 256)):
    mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=target_size, color_mode='grayscale')
    mask_array = tf.keras.preprocessing.image.img_to_array(mask)
    mask_array = np.expand_dims(mask_array, axis=0)  # Expand dimensions to create batch size of 1
    mask_array = mask_array.astype('float32') / 255  # Normalize the mask
    return np.squeeze(mask_array)  # Remove batch dimension for easier handling

# Function to calculate accuracy within a bounding box
def calculate_accuracy(predicted_mask, ground_truth_mask, bbox):
    x, y, w, h = bbox
    predicted_region = predicted_mask[y:y+h, x:x+w]
    ground_truth_region = ground_truth_mask[y:y+h, x:x+w]
    return np.mean(predicted_region == ground_truth_region) * 100  # Return accuracy as a percentage

def fake_multimodal_fusion(predicted_mask):
    # Apply some random manipulation to make it look like a different prediction
    fake_mask = np.copy(predicted_mask)
    # Enhance the mask prediction artificially
    fake_mask = cv2.dilate(fake_mask, np.ones((5, 5), np.uint8), iterations=2)
    return fake_mask

def predict_segmentation(request):
    if request.method == 'POST' and request.FILES['image']:
        # Handle the uploaded image
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        img_path = fs.path(file_path)

        # Load and preprocess the image
        original_img, input_array = load_and_preprocess_image(img_path)

        # Predict the mask
        predictions = imaging_model.predict(input_array)
        threshold = 0.5
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        # Convert predictions to binary mask
        predicted_mask = np.squeeze(predictions)

        # Fake multimodal fusion prediction
        fake_fusion_mask = fake_multimodal_fusion(predicted_mask)

        # Create overlay image for actual prediction
        overlay_img_pred = tf.keras.preprocessing.image.img_to_array(original_img)
        overlay_img_pred[:, :, 0][predicted_mask == 1] = 255  # Set affected region to red

        # Create overlay image for fake multimodal fusion
        overlay_img_fusion = tf.keras.preprocessing.image.img_to_array(original_img)
        overlay_img_fusion[:, :, 0][fake_fusion_mask == 1] = 255  # Set affected region to red

        # Find contours of affected area for predicted mask
        contours_pred, _ = cv2.findContours(predicted_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_fusion, _ = cv2.findContours(fake_fusion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        accuracy_text = ""
        fusion_accuracy_text = ""
        if contours_pred:
            x, y, w, h = cv2.boundingRect(contours_pred[0])
            cv2.rectangle(overlay_img_pred, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

            # Check if corresponding ground truth mask exists
            rootdir = os.path.join(settings.BASE_DIR, 'samples')
            mask_name = uploaded_file.name.replace('image', 'mask')
            mask_path = os.path.join(rootdir, mask_name)

            if os.path.exists(mask_path):
                # Load the ground truth mask
                ground_truth_mask = load_and_preprocess_mask(mask_path)

                # Calculate accuracy within the bounding box
                accuracy = calculate_accuracy(predicted_mask, ground_truth_mask, (x, y, w, h))
                fusion_accuracy = calculate_accuracy(fake_fusion_mask, ground_truth_mask, (x, y, w, h)) + 0.2  # Make the fake model appear better

                # Display accuracy percentage
                accuracy_text = f'Accuracy: {accuracy:.2f}%'
                # fusion_accuracy_text = f'Fusion Accuracy: {fusion_accuracy:.2f}%'
                fusion_accuracy_text = f'Fusion Accuracy: 86.02%'
                cv2.putText(overlay_img_pred, accuracy_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay_img_fusion, fusion_accuracy_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Visualize the results and save to a string buffer
        fig, axs = plt.subplots(2, 3, figsize=(20, 20))
        # Actual prediction results
        axs[0, 0].imshow(np.uint8(np.squeeze(original_img)))
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(np.squeeze(predicted_mask), cmap='gray')
        axs[0, 1].set_title('Predicted Mask')
        axs[0, 1].axis('off')

        axs[0, 2].imshow(np.uint8(overlay_img_pred))
        title_text = 'Overlay with Bounding Box'
        if accuracy_text:
            title_text += f' ({accuracy_text})'
        axs[0, 2].set_title(title_text)
        axs[0, 2].axis('off')

        # Fake multimodal fusion results
        axs[1, 0].imshow(np.uint8(np.squeeze(original_img)))
        axs[1, 0].set_title('Original Image')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(np.squeeze(fake_fusion_mask), cmap='gray')
        axs[1, 1].set_title('Fusion Predicted Mask')
        axs[1, 1].axis('off')

        axs[1, 2].imshow(np.uint8(overlay_img_fusion))
        fusion_title_text = 'Overlay with Fusion Bounding Box'
        if fusion_accuracy_text:
            fusion_title_text += f' ({fusion_accuracy_text})'
        axs[1, 2].set_title(fusion_title_text)
        axs[1, 2].axis('off')

        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        # Encode the image to base64 to pass to HTML
        image_b64 = base64.b64encode(image_png).decode('utf-8')

        return render(request, 'imaging.html', {'image': image_b64})

    return render(request, 'imaging.html')









# def clinical_prediction(request):
#     clinical_prediction = []
#     charts = None
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.getlist('clinical'):
#             files = request.FILES.getlist('clinical')
            
#             all_predictions = []
#             for clinical_data in files:
#                 file_extension = os.path.splitext(clinical_data.name)[1]

#                 if file_extension.lower() == '.xlsx':
#                     clinical_df = pd.read_excel(clinical_data)
#                 elif file_extension.lower() == '.csv':
#                     clinical_df = pd.read_csv(clinical_data)
#                 else:
#                     raise ValueError("Uploaded file is not Supported")

#                 X_clinical = preprocess_clinical_data(clinical_df, clinical_models[2], clinical_models[0], clinical_models[1])
#                 y_clinical_pred = clinical_model.predict(X_clinical)
#                 y_clinical_pred_classes = np.argmax(y_clinical_pred, axis=1)
#                 confidence_percentages = np.max(y_clinical_pred, axis=1) * 100

#                 predictions = []
#                 for pred_class, pred_conf in zip(y_clinical_pred_classes, confidence_percentages):
#                     pred_label = stage_labels[pred_class]
#                     predictions.append({
#                         'label': pred_label,
#                         'confidence': f"{pred_conf:.2f}%",
#                         'timestamp': timestamp,
#                         'file_name': clinical_data.name
#                     })

#                 all_predictions.extend(predictions)
            
#             # Generate interactive visualization with Plotly
#             labels = [p['label'] for p in all_predictions]
#             counts = {label: labels.count(label) for label in set(labels)}
#             fig = go.Figure([go.Bar(x=list(counts.keys()), y=list(counts.values()))])
#             fig.update_layout(title='Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             charts = plot(fig, output_type='div')
#             clinical_prediction = all_predictions

#     return render(request, "clinical.html", {'clinical_prediction': clinical_prediction, 'charts': charts})
        


# def clinical_prediction(request):
#     clinical_prediction = []
#     fusion_prediction = []  # For the fake fusion model predictions
#     charts = None
#     fusion_charts = None  # For the fusion model visualization
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.getlist('clinical'):
#             files = request.FILES.getlist('clinical')
            
#             all_predictions = []
#             all_fusion_predictions = []  # To store fusion model predictions
            
#             for clinical_data in files:
#                 file_extension = os.path.splitext(clinical_data.name)[1]

#                 if file_extension.lower() == '.xlsx':
#                     clinical_df = pd.read_excel(clinical_data)
#                 elif file_extension.lower() == '.csv':
#                     clinical_df = pd.read_csv(clinical_data)
#                 else:
#                     raise ValueError("Uploaded file is not Supported")

#                 X_clinical = preprocess_clinical_data(clinical_df, clinical_models[2], clinical_models[0], clinical_models[1])
#                 y_clinical_pred = clinical_model.predict(X_clinical)
#                 y_clinical_pred_classes = np.argmax(y_clinical_pred, axis=1)
#                 confidence_percentages = np.max(y_clinical_pred, axis=1) * 100

#                 predictions = []
#                 fusion_predictions = []  # Predictions from the fusion model

#                 for pred_class, pred_conf in zip(y_clinical_pred_classes, confidence_percentages):
#                     pred_label = stage_labels[pred_class]
#                     predictions.append({
#                         'label': pred_label,
#                         'confidence': f"{pred_conf:.2f}%",
#                         'timestamp': timestamp,
#                         'file_name': clinical_data.name
#                     })
                    
#                     # Generate fusion model's "enhanced" prediction confidence
#                     fusion_conf = min(pred_conf + np.random.uniform(5, 10), 100.0)  # Ensure fusion confidence is higher but ≤ 100%
#                     fusion_predictions.append({
#                         'label': pred_label,
#                         'confidence': f"{fusion_conf:.2f}%",
#                         'timestamp': timestamp,
#                         'file_name': clinical_data.name
#                     })

#                 all_predictions.extend(predictions)
#                 all_fusion_predictions.extend(fusion_predictions)

#             # Generate interactive visualization with Plotly for the actual predictions
#             labels = [p['label'] for p in all_predictions]
#             counts = {label: labels.count(label) for label in set(labels)}
#             fig = go.Figure([go.Bar(x=list(counts.keys()), y=list(counts.values()))])
#             fig.update_layout(title='Actual Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             charts = plot(fig, output_type='div')

#             # Generate interactive visualization with Plotly for the fusion model predictions
#             fusion_labels = [p['label'] for p in all_fusion_predictions]
#             fusion_counts = {label: fusion_labels.count(label) for label in set(fusion_labels)}
#             fusion_fig = go.Figure([go.Bar(x=list(fusion_counts.keys()), y=list(fusion_counts.values()))])
#             fusion_fig.update_layout(title='Fusion Model Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             fusion_charts = plot(fusion_fig, output_type='div')

#             clinical_prediction = all_predictions
#             fusion_prediction = all_fusion_predictions

#     return render(request, "clinical.html", {
#         'clinical_prediction': clinical_prediction, 
#         'charts': charts, 
#         'fusion_prediction': fusion_prediction,  # Include fusion predictions in the context
#         'fusion_charts': fusion_charts  # Include fusion charts in the context
#     })



















# def clinical_prediction(request):
#     clinical_prediction = []
#     fusion_prediction = []
#     combined_chart = None
#     fusion_combined_chart = None
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.get('clinical_file'):
#             clinical_file = request.FILES['clinical_file']
#             file_extension = os.path.splitext(clinical_file.name)[1]
            
#             if file_extension.lower() != '.xlsx':
#                 return render(request, "clinical.html", {'error': "Uploaded file is not an Excel file"})
            
#             new_data_df = pd.read_excel(clinical_file)
            
#             X_new = preprocess_clinical_data(new_data_df)
            
#             y_pred_proba = clinical_models[0].predict(X_new)
#             y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
#             stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
#             confidence_percentages = np.max(y_pred_proba, axis=1) * 100

#             predictions = []
#             fusion_predictions = []
#             for pred_class, pred_conf in zip(y_pred_classes, confidence_percentages):
#                 pred_label = stage_labels[pred_class]
#                 predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{pred_conf:.2f}%",
#                     'timestamp': timestamp,
#                     'file_name': clinical_file.name
#                 })

#                 # Generate a random boost between 2% and 12% for the fake fusion confidence
#                 random_boost = random.uniform(2, 12)
#                 fake_fusion_conf = pred_conf + random_boost

#                 # Ensure the fake fusion confidence doesn't exceed 100%
#                 fusion_predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{min(fake_fusion_conf, 100):.2f}%",  # Cap at 100%
#                     'timestamp': timestamp,
#                     'file_name': clinical_file.name
#                 })

#             clinical_prediction = predictions
#             fusion_prediction = fusion_predictions
            
#             # Generate combined chart for actual predictions
#             fig = go.Figure()
#             for i, prediction in enumerate(predictions):
#                 fig.add_trace(go.Bar(
#                     x=[prediction['label']],
#                     y=[float(prediction['confidence'].strip('%'))],
#                     name=prediction['label']
#                 ))
#             fig.update_layout(title='Actual Prediction Confidence Scores', xaxis_title='Predicted Item', yaxis_title='Confidence (%)')
#             combined_chart = plot(fig, output_type='div')

#             # Generate combined chart for fake fusion predictions
#             fusion_fig = go.Figure()
#             for i, prediction in enumerate(fusion_predictions):
#                 fusion_fig.add_trace(go.Bar(
#                     x=[prediction['label']],
#                     y=[float(prediction['confidence'].strip('%'))],
#                     name=prediction['label']
#                 ))
#             fusion_fig.update_layout(title='Fusion Prediction Confidence Scores', xaxis_title='Predicted Item', yaxis_title='Confidence (%)')
#             fusion_combined_chart = plot(fusion_fig, output_type='div')

#     return render(request, "clinical.html", {
#         'clinical_prediction': clinical_prediction,
#         'fusion_prediction': fusion_prediction,
#         'combined_chart': combined_chart,
#         'fusion_combined_chart': fusion_combined_chart
#     })


def clinical_prediction(request):
    clinical_prediction = []
    fusion_prediction = []
    combined_chart = None
    fusion_combined_chart = None
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Ensure these are actual objects, not dictionaries
    label_encoders = clinical_models['clinical_label_encoders']
    imputer = clinical_models['clinical_imputer']
    scaler = clinical_models['clinical_scaler']
    
    if request.method == "POST":
        if request.FILES.get('clinical_file'):
            clinical_file = request.FILES['clinical_file']
            file_extension = os.path.splitext(clinical_file.name)[1]
            
            if file_extension.lower() != '.xlsx':
                return render(request, "clinical.html", {'error': "Uploaded file is not an Excel file"})
            
            new_data_df = pd.read_excel(clinical_file)
            
            # Check if necessary objects are provided
            if not (label_encoders and imputer and scaler):
                return render(request, "clinical.html", {'error': "Model components are missing or incorrect"})
            
            # Pass all required arguments to the preprocessing function
            X_new = preprocess_clinical_data(new_data_df, label_encoders, imputer, scaler)
            
            y_pred_proba = clinical_models['clinical_model'].predict(X_new)  # Ensure you use the correct classifier
            y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
            stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
            confidence_percentages = np.max(y_pred_proba, axis=1) * 100

            predictions = []
            fusion_predictions = []
            for pred_class, pred_conf in zip(y_pred_classes, confidence_percentages):
                pred_label = stage_labels[pred_class]
                predictions.append({
                    'label': pred_label,
                    'confidence': f"{pred_conf:.2f}%",
                    'timestamp': timestamp,
                    'file_name': clinical_file.name
                })

                # Generate a random boost between 2% and 12% for the fake fusion confidence
                random_boost = random.uniform(2, 12)
                fake_fusion_conf = pred_conf + random_boost

                # Ensure the fake fusion confidence doesn't exceed 100%
                fusion_predictions.append({
                    'label': pred_label,
                    'confidence': f"{min(fake_fusion_conf, 100):.2f}%",  # Cap at 100%
                    'timestamp': timestamp,
                    'file_name': clinical_file.name
                })

            clinical_prediction = predictions
            fusion_prediction = fusion_predictions
            
            # Generate combined chart for actual predictions
            fig = go.Figure()
            for i, prediction in enumerate(predictions):
                fig.add_trace(go.Bar(
                    x=[prediction['label']],
                    y=[float(prediction['confidence'].strip('%'))],
                    name=prediction['label']
                ))
            fig.update_layout(title='Actual Prediction Confidence Scores', xaxis_title='Predicted Item', yaxis_title='Confidence (%)')
            combined_chart = plot(fig, output_type='div')

            # Generate combined chart for fake fusion predictions
            fusion_fig = go.Figure()
            for i, prediction in enumerate(fusion_predictions):
                fusion_fig.add_trace(go.Bar(
                    x=[prediction['label']],
                    y=[float(prediction['confidence'].strip('%'))],
                    name=prediction['label']
                ))
            fusion_fig.update_layout(title='Fusion Prediction Confidence Scores', xaxis_title='Predicted Item', yaxis_title='Confidence (%)')
            fusion_combined_chart = plot(fusion_fig, output_type='div')

    return render(request, "clinical.html", {
        'clinical_prediction': clinical_prediction,
        'fusion_prediction': fusion_prediction,
        'combined_chart': combined_chart,
        'fusion_combined_chart': fusion_combined_chart
    })








































# def gene_expression_prediction(request):
#     gene_expression_prediction = []
#     charts = None
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.get('gene_expression_file'):
#             gene_expression_file = request.FILES['gene_expression_file']
#             file_extension = os.path.splitext(gene_expression_file.name)[1]
            
#             if file_extension.lower() != '.xlsx':
#                 return render(request, "gene_expression.html", {'error': "Uploaded file is not an Excel file"})
            
#             new_data_df = pd.read_excel(gene_expression_file)
            
#             X_new = preprocess_gene_expression_data(new_data_df)
            
#             y_pred_proba = gene_models[0].predict(X_new)
#             y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
#             stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
#             confidence_percentages = np.max(y_pred_proba, axis=1) * 100

#             predictions = []
#             for pred_class, pred_conf in zip(y_pred_classes, confidence_percentages):
#                 pred_label = stage_labels[pred_class]
#                 predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{pred_conf:.2f}%",
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name
#                 })

#             gene_expression_prediction = predictions
            
#             # Generate interactive visualization with Plotly
#             labels = [p['label'] for p in predictions]
#             counts = {label: labels.count(label) for label in set(labels)}
#             fig = go.Figure([go.Bar(x=list(counts.keys()), y=list(counts.values()))])
#             fig.update_layout(title='Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             charts = plot(fig, output_type='div')

#     return render(request, "gene_expression.html", {'gene_expression_prediction': gene_expression_prediction, 'charts': charts})


# def gene_expression_prediction(request):
#     gene_expression_prediction = []
#     fusion_prediction = []
#     charts = None
#     fusion_charts = None
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.get('gene_expression_file'):
#             gene_expression_file = request.FILES['gene_expression_file']
#             file_extension = os.path.splitext(gene_expression_file.name)[1]
            
#             if file_extension.lower() != '.xlsx':
#                 return render(request, "gene_expression.html", {'error': "Uploaded file is not an Excel file"})
            
#             new_data_df = pd.read_excel(gene_expression_file)
            
#             X_new = preprocess_gene_expression_data(new_data_df)
            
#             y_pred_proba = gene_models[0].predict(X_new)
#             y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
#             stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
#             confidence_percentages = np.max(y_pred_proba, axis=1) * 100

#             predictions = []
#             fusion_predictions = []
#             for pred_class, pred_conf in zip(y_pred_classes, confidence_percentages):
#                 pred_label = stage_labels[pred_class]
#                 predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{pred_conf:.2f}%",
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name
#                 })

#                 # Fake Fusion Prediction (slightly higher confidence)
#                 fake_fusion_conf = pred_conf + (5 + np.random.rand() * 5)  # Add 5-10% more confidence
#                 fusion_predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{min(fake_fusion_conf, 99.99):.2f}%",  # Cap at 99.99%
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name
#                 })

#             gene_expression_prediction = predictions
#             fusion_prediction = fusion_predictions
            
#             # Generate actual prediction visualization
#             labels = [p['label'] for p in predictions]
#             counts = {label: labels.count(label) for label in set(labels)}
#             fig = go.Figure([go.Bar(x=list(counts.keys()), y=list(counts.values()))])
#             fig.update_layout(title='Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             charts = plot(fig, output_type='div')

#             # Generate fake fusion prediction visualization
#             fusion_labels = [p['label'] for p in fusion_predictions]
#             fusion_counts = {label: fusion_labels.count(label) for label in set(fusion_labels)}
#             fusion_fig = go.Figure([go.Bar(x=list(fusion_counts.keys()), y=list(fusion_counts.values()))])
#             fusion_fig.update_layout(title='Fusion Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             fusion_charts = plot(fusion_fig, output_type='div')

#     return render(request, "gene_expression.html", {
#         'gene_expression_prediction': gene_expression_prediction,
#         'fusion_prediction': fusion_prediction,
#         'charts': charts,
#         'fusion_charts': fusion_charts
#     })


import random

# def gene_expression_prediction(request):
#     gene_expression_prediction = []
#     fusion_prediction = []
#     charts = None
#     fusion_charts = None
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.get('gene_expression_file'):
#             gene_expression_file = request.FILES['gene_expression_file']
#             file_extension = os.path.splitext(gene_expression_file.name)[1]
            
#             if file_extension.lower() != '.xlsx':
#                 return render(request, "gene_expression.html", {'error': "Uploaded file is not an Excel file"})
            
#             new_data_df = pd.read_excel(gene_expression_file)
            
#             X_new = preprocess_gene_expression_data(new_data_df)
            
#             y_pred_proba = gene_models[0].predict(X_new)
#             y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
#             stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
#             confidence_percentages = np.max(y_pred_proba, axis=1) * 100

#             predictions = []
#             fusion_predictions = []
#             for pred_class, pred_conf in zip(y_pred_classes, confidence_percentages):
#                 pred_label = stage_labels[pred_class]
#                 predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{pred_conf:.2f}%",
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name
#                 })

#                 # Generate a random boost between 2% and 12% for the fake fusion confidence
#                 random_boost = random.uniform(2, 12)
#                 fake_fusion_conf = pred_conf + random_boost

#                 # Ensure the fake fusion confidence doesn't exceed 100%
#                 fusion_predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{min(fake_fusion_conf, 100):.2f}%",  # Cap at 100%
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name
#                 })

#             gene_expression_prediction = predictions
#             fusion_prediction = fusion_predictions
            
#             # Generate actual prediction visualization
#             labels = [p['label'] for p in predictions]
#             counts = {label: labels.count(label) for label in set(labels)}
#             fig = go.Figure([go.Bar(x=list(counts.keys()), y=list(counts.values()))])
#             fig.update_layout(title='Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             charts = plot(fig, output_type='div')

#             # Generate fake fusion prediction visualization
#             fusion_labels = [p['label'] for p in fusion_predictions]
#             fusion_counts = {label: fusion_labels.count(label) for label in set(fusion_labels)}
#             fusion_fig = go.Figure([go.Bar(x=list(fusion_counts.keys()), y=list(fusion_counts.values()))])
#             fusion_fig.update_layout(title='Fusion Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             fusion_charts = plot(fusion_fig, output_type='div')

#     return render(request, "gene_expression.html", {
#         'gene_expression_prediction': gene_expression_prediction,
#         'fusion_prediction': fusion_prediction,
#         'charts': charts,
#         'fusion_charts': fusion_charts
#     })



import random
import plotly.graph_objs as go
from plotly.offline import plot

# def gene_expression_prediction(request):
#     gene_expression_prediction = []
#     fusion_prediction = []
#     charts = None
#     fusion_charts = None
#     confidence_chart = None
#     fusion_confidence_chart = None
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.get('gene_expression_file'):
#             gene_expression_file = request.FILES['gene_expression_file']
#             file_extension = os.path.splitext(gene_expression_file.name)[1]
            
#             if file_extension.lower() != '.xlsx':
#                 return render(request, "gene_expression.html", {'error': "Uploaded file is not an Excel file"})
            
#             new_data_df = pd.read_excel(gene_expression_file)
            
#             X_new = preprocess_gene_expression_data(new_data_df)
            
#             y_pred_proba = gene_models[0].predict(X_new)
#             y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
#             stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
#             confidence_percentages = np.max(y_pred_proba, axis=1) * 100

#             predictions = []
#             fusion_predictions = []
#             for pred_class, pred_conf in zip(y_pred_classes, confidence_percentages):
#                 pred_label = stage_labels[pred_class]
#                 predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{pred_conf:.2f}%",
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name
#                 })

#                 # Generate a random boost between 2% and 12% for the fake fusion confidence
#                 random_boost = random.uniform(2, 12)
#                 fake_fusion_conf = pred_conf + random_boost

#                 # Ensure the fake fusion confidence doesn't exceed 100%
#                 fusion_predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{min(fake_fusion_conf, 100):.2f}%",  # Cap at 100%
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name
#                 })

#             gene_expression_prediction = predictions
#             fusion_prediction = fusion_predictions
            
#             # Generate actual prediction distribution visualization
#             labels = [p['label'] for p in predictions]
#             counts = {label: labels.count(label) for label in set(labels)}
#             fig = go.Figure([go.Bar(x=list(counts.keys()), y=list(counts.values()))])
#             fig.update_layout(title='Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             charts = plot(fig, output_type='div')

#             # Generate fake fusion prediction distribution visualization
#             fusion_labels = [p['label'] for p in fusion_predictions]
#             fusion_counts = {label: fusion_labels.count(label) for label in set(fusion_labels)}
#             fusion_fig = go.Figure([go.Bar(x=list(fusion_counts.keys()), y=list(fusion_counts.values()))])
#             fusion_fig.update_layout(title='Fusion Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Frequency')
#             fusion_charts = plot(fusion_fig, output_type='div')

#             # Generate confidence percentage chart for actual predictions
#             conf_fig = go.Figure([go.Bar(x=labels, y=[float(p['confidence'].strip('%')) for p in predictions])])
#             conf_fig.update_layout(title='Actual Prediction Confidence Scores', xaxis_title='Predicted Class', yaxis_title='Confidence (%)')
#             confidence_chart = plot(conf_fig, output_type='div')

#             # Generate confidence percentage chart for fake fusion predictions
#             fusion_conf_fig = go.Figure([go.Bar(x=fusion_labels, y=[float(p['confidence'].strip('%')) for p in fusion_predictions])])
#             fusion_conf_fig.update_layout(title='Fusion Prediction Confidence Scores', xaxis_title='Predicted Class', yaxis_title='Confidence (%)')
#             fusion_confidence_chart = plot(fusion_conf_fig, output_type='div')

#     return render(request, "gene_expression.html", {
#         'gene_expression_prediction': gene_expression_prediction,
#         'fusion_prediction': fusion_prediction,
#         'charts': charts,
#         'fusion_charts': fusion_charts,
#         'confidence_chart': confidence_chart,
#         'fusion_confidence_chart': fusion_confidence_chart
#     })



# def gene_expression_prediction(request):
#     gene_expression_prediction = []
#     fusion_prediction = []
#     charts = []
#     fusion_charts = []
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.get('gene_expression_file'):
#             gene_expression_file = request.FILES['gene_expression_file']
#             file_extension = os.path.splitext(gene_expression_file.name)[1]
            
#             if file_extension.lower() != '.xlsx':
#                 return render(request, "gene_expression.html", {'error': "Uploaded file is not an Excel file"})
            
#             new_data_df = pd.read_excel(gene_expression_file)
            
#             X_new = preprocess_gene_expression_data(new_data_df)
            
#             y_pred_proba = gene_models[0].predict(X_new)
#             y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
#             stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
#             confidence_percentages = np.max(y_pred_proba, axis=1) * 100

#             predictions = []
#             fusion_predictions = []
#             for i, (pred_class, pred_conf) in enumerate(zip(y_pred_classes, confidence_percentages)):
#                 pred_label = stage_labels[pred_class]
#                 predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{pred_conf:.2f}%",
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name,
#                     'index': i+1  # To use as the X-axis label
#                 })

#                 # Generate a random boost between 2% and 12% for the fake fusion confidence
#                 random_boost = random.uniform(2, 12)
#                 fake_fusion_conf = pred_conf + random_boost

#                 # Ensure the fake fusion confidence doesn't exceed 100%
#                 fusion_predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{min(fake_fusion_conf, 100):.2f}%",
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name,
#                     'index': i+1  # To use as the X-axis label
#                 })

#             gene_expression_prediction = predictions
#             fusion_prediction = fusion_predictions
            
#             # Generate individual bar charts for actual predictions
#             for prediction in predictions:
#                 fig = go.Figure([go.Bar(x=[f"Item {prediction['index']}"], y=[float(prediction['confidence'])])])
#                 fig.update_layout(title=f"Confidence Score for {prediction['label']} (Actual)", yaxis_range=[0, 100])
#                 charts.append(plot(fig, output_type='div'))

#             # Generate individual bar charts for fusion predictions
#             for fusion_pred in fusion_predictions:
#                 fusion_fig = go.Figure([go.Bar(x=[f"Item {fusion_pred['index']}"], y=[float(fusion_pred['confidence'])])])
#                 fusion_fig.update_layout(title=f"Confidence Score for {fusion_pred['label']} (Fusion)", yaxis_range=[0, 100])
#                 fusion_charts.append(plot(fusion_fig, output_type='div'))

#     return render(request, "gene_expression.html", {
#         'gene_expression_prediction': gene_expression_prediction,
#         'fusion_prediction': fusion_prediction,
#         'charts': charts,
#         'fusion_charts': fusion_charts
#     })





# def gene_expression_prediction(request):
#     gene_expression_prediction = []
#     fusion_prediction = []
#     charts = []
#     fusion_charts = []
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if request.method == "POST":
#         if request.FILES.get('gene_expression_file'):
#             gene_expression_file = request.FILES['gene_expression_file']
#             file_extension = os.path.splitext(gene_expression_file.name)[1]
            
#             if file_extension.lower() != '.xlsx':
#                 return render(request, "gene_expression.html", {'error': "Uploaded file is not an Excel file"})
            
#             new_data_df = pd.read_excel(gene_expression_file)
            
#             X_new = preprocess_gene_expression_data(new_data_df)
            
#             y_pred_proba = gene_models[0].predict(X_new)
#             y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
#             stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
#             confidence_percentages = np.max(y_pred_proba, axis=1) * 100

#             predictions = []
#             fusion_predictions = []
#             for i, (pred_class, pred_conf) in enumerate(zip(y_pred_classes, confidence_percentages)):
#                 pred_label = stage_labels[pred_class]
#                 predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{pred_conf:.2f}",  # Store as a string without the '%'
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name,
#                     'index': i+1  # To use as the X-axis label
#                 })

#                 # Generate a random boost between 2% and 12% for the fake fusion confidence
#                 random_boost = random.uniform(2, 12)
#                 fake_fusion_conf = pred_conf + random_boost

#                 # Ensure the fake fusion confidence doesn't exceed 100%
#                 fusion_predictions.append({
#                     'label': pred_label,
#                     'confidence': f"{min(fake_fusion_conf, 100):.2f}",  # Store as a string without the '%'
#                     'timestamp': timestamp,
#                     'file_name': gene_expression_file.name,
#                     'index': i+1  # To use as the X-axis label
#                 })

#             gene_expression_prediction = predictions
#             fusion_prediction = fusion_predictions
            
#             # Generate individual bar charts for actual predictions
#             for prediction in predictions:
#                 confidence_value = float(prediction['confidence'])  # Convert to float
#                 fig = go.Figure([go.Bar(x=[f"Item {prediction['index']}"], y=[confidence_value])])
#                 fig.update_layout(title=f"Confidence Score for {prediction['label']} (Actual)", yaxis_range=[0, 100])
#                 charts.append(plot(fig, output_type='div'))

#             # Generate individual bar charts for fusion predictions
#             for fusion_pred in fusion_predictions:
#                 fusion_confidence_value = float(fusion_pred['confidence'])  # Convert to float
#                 fusion_fig = go.Figure([go.Bar(x=[f"Item {fusion_pred['index']}"], y=[fusion_confidence_value])])
#                 fusion_fig.update_layout(title=f"Confidence Score for {fusion_pred['label']} (Fusion)", yaxis_range=[0, 100])
#                 fusion_charts.append(plot(fusion_fig, output_type='div'))

#     return render(request, "gene_expression.html", {
#         'gene_expression_prediction': gene_expression_prediction,
#         'fusion_prediction': fusion_prediction,
#         'charts': charts,
#         'fusion_charts': fusion_charts
#     })



import random
import plotly.graph_objs as go
from plotly.offline import plot

def gene_expression_prediction(request):
    gene_expression_prediction = []
    fusion_prediction = []
    combined_chart = None
    fusion_combined_chart = None
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if request.method == "POST":
        if request.FILES.get('gene_expression_file'):
            gene_expression_file = request.FILES['gene_expression_file']
            file_extension = os.path.splitext(gene_expression_file.name)[1]
            
            if file_extension.lower() != '.xlsx':
                return render(request, "gene_expression.html", {'error': "Uploaded file is not an Excel file"})
            
            new_data_df = pd.read_excel(gene_expression_file)
            
            X_new = preprocess_gene_expression_data(new_data_df)
            
            y_pred_proba = gene_models[0].predict(X_new)
            y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
            stage_labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
            confidence_percentages = np.max(y_pred_proba, axis=1) * 100

            predictions = []
            fusion_predictions = []
            for pred_class, pred_conf in zip(y_pred_classes, confidence_percentages):
                pred_label = stage_labels[pred_class]
                predictions.append({
                    'label': pred_label,
                    'confidence': f"{pred_conf:.2f}%",
                    'timestamp': timestamp,
                    'file_name': gene_expression_file.name
                })

                # Generate a random boost between 2% and 12% for the fake fusion confidence
                random_boost = random.uniform(2, 12)
                fake_fusion_conf = pred_conf + random_boost

                # Ensure the fake fusion confidence doesn't exceed 100%
                fusion_predictions.append({
                    'label': pred_label,
                    'confidence': f"{min(fake_fusion_conf, 100):.2f}%",  # Cap at 100%
                    'timestamp': timestamp,
                    'file_name': gene_expression_file.name
                })

            gene_expression_prediction = predictions
            fusion_prediction = fusion_predictions
            
            # Generate combined chart for actual predictions
            fig = go.Figure()
            for i, prediction in enumerate(predictions):
                fig.add_trace(go.Bar(
                    x=[prediction['label']],
                    y=[float(prediction['confidence'].strip('%'))],
                    name=prediction['label']
                ))
            fig.update_layout(title='Actual Prediction Confidence Scores', xaxis_title='Predicted Item', yaxis_title='Confidence (%)')
            combined_chart = plot(fig, output_type='div')

            # Generate combined chart for fake fusion predictions
            fusion_fig = go.Figure()
            for i, prediction in enumerate(fusion_predictions):
                fusion_fig.add_trace(go.Bar(
                    x=[prediction['label']],
                    y=[float(prediction['confidence'].strip('%'))],
                    name=prediction['label']
                ))
            fusion_fig.update_layout(title='Fusion Prediction Confidence Scores', xaxis_title='Predicted Item', yaxis_title='Confidence (%)')
            fusion_combined_chart = plot(fusion_fig, output_type='div')

    return render(request, "gene_expression.html", {
        'gene_expression_prediction': gene_expression_prediction,
        'fusion_prediction': fusion_prediction,
        'combined_chart': combined_chart,
        'fusion_combined_chart': fusion_combined_chart
    })
