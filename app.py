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

        # Assuming class index for "car" is 3
        car_class_index = 3

        car_indices = torch.where(results['labels'] == 3)[0]
        car_scores = results['scores'][car_indices]
        car_labels = results['labels'][car_indices]
        car_boxes = results['boxes'][car_indices]



        # Create a list to store detected car information
        detected_cars = []

        # Iterate through the filtered car results
        for score, label, box in zip(car_scores, car_labels, car_boxes):
            # Check if the label corresponds to the "car" class
            if label == car_class_index:
                detected_cars.append({
                    'score': int(score.item()),
                    'label': int(car_class_index),
                    'box': box.tolist()
                })

        # Return the detected cars as JSON
        return jsonify({'detected_cars': detected_cars})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

