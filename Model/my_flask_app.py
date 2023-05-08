
from flask_cors import CORS
import torch
from flask import Flask , request , jsonify
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import cv2
import os

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

@app.route('/predict', methods=['POST'])
def predict():
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []
    image_path = request.files['image']
    img = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (img_h, img_w))
    imgs.append(img)
    imgs = np.stack(imgs, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    means.reverse()  # BGR --> RGB
    stdevs.reverse()
    norm_mean = [0.7630392, 0.5456477, 0.57004845]
    norm_std = [0.1409286, 0.15261266, 0.16997074]
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    # Forward pass the preprocessed image through the ResNet model to get the predicted output
    with torch.no_grad():
       
        model.eval()
        output = model(image_tensor)

        # Convert the predicted output to human-readable form
        predicted_class = torch.argmax(output).item()
        if predicted_class == 2:
            predicted_class = 'Basal Cell Carcinoma'
        elif predicted_class == 3:
            predicted_class = 'Benign Keratosis Lesion'
        elif predicted_class == 6:
            predicted_class = 'Melanocytic Nevi'
        elif predicted_class == 8:
            predicted_class = 'Melanoma'
        elif predicted_class == 4:
            predicted_class = 'Dermatofibroma'
        elif predicted_class == 7:
            predicted_class = 'Vascular Lesion'
        elif predicted_class == 1:
            predicted_class = 'Actinic keratosis and Intrapithelial carcinoma'
        elif predicted_class == 0:
            predicted_class = 'Acne'
        elif predicted_class == 5:
            predicted_class = 'Eczema'
        print(predicted_class)
    return jsonify(predicted_class)

if __name__ == '__main__':
    app.run(port=8000, debug=True)

