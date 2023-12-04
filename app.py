from flask import Flask, request, jsonify
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import mpld3
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load pre-trained model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    fig = plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    fig.savefig("temp.png")
    file = open("temp.png", mode="rb")
    image_binary_data = file.read()
    file.close()
    return image_binary_data
    
# Endpoint for car detection
@app.route('/detect_car', methods=['POST'])
def detect_car():
    try:
        # Get the image from the request
        image = Image.open(request.files.get('image'))
        encoding = processor(image, return_tensors="pt")
        # Perform car detection
        with torch.no_grad():
          outputs = model(**encoding)

        # postprocess model outputs
        width, height = image.size

        postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                        target_sizes=[(height, width)],
                                                                        threshold=0.9)
        results = postprocessed_outputs[0]
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
        detected_cars
        image_binary_data = plot_results(image, car_scores, car_labels, car_boxes)
          encoded_image = base64.b64encode(image_binary_data).decode('utf-8')
        detected_cars.append({
            'image': encoded_image
        })

        # Return the detected cars as JSON
        return jsonify({'detected_cars': detected_cars})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
