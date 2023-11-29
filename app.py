# app.py
from flask import Flask, request, jsonify
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

app = Flask(__name__)

# Load pre-trained model
car_detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Endpoint for car detection
@app.route('/detect_car', methods=['POST'])
def detect_car():
    try:
        # Get the image from the request
        image = request.files.get('image')

        # Process the image
        inputs = processor(images=image, return_tensors="pt")

        # Perform car detection
        with torch.no_grad():
            outputs = car_detection_model(**inputs)

        # Post-process the results
        results = processor.post_process_object_detection(outputs, inputs)

        # Extract information about detected cars
        detected_cars = []
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            if processor.id2label[label] == 'car':
                detected_cars.append({
                    'label': processor.id2label[label],
                    'score': score.item(),
                    'box': box.tolist()
                })

        # Return the detected cars as JSON
        return jsonify({'detected_cars': detected_cars})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)

