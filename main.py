from flask import Flask, request, jsonify, send_from_directory
from keras.models import load_model
# In Keras 3, utils are directly in keras.utils
from keras.utils import img_to_array
from keras.preprocessing.image import load_img  # or from PIL import Image

import numpy as np
import os
from rag_chatbot import query_gemini  # Import the chatbot function

# Initialize Flask app
app = Flask(__name__)

# Load the trained tumor classification model
model = load_model('model.keras')

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 299
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Route for text-based medical chatbot
@app.route('/query', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    answer = query_gemini(question)
    return jsonify({"answer": answer})

# Route for tumor classification (image upload)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_location)

        result, confidence = predict_tumor(file_location)

        return jsonify({"result": result, "confidence": f"{confidence*100:.2f}%"}), 200

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
