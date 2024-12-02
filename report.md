# Food Segmentation and Area Calculation Project

In this project, I built a food segmentation and area calculation application using the YOLOv11 model. The process involved training the model, optimizing it for fast inference, and deploying it using modern tools like ONNX Runtime, FastAPI, Docker, and Kubernetes. Additionally, I used a dynamic batching technique to improve efficiency.


## Training the Model

I started with the YOLOv11 model pre-trained on the COCO dataset. It already had some food-related classes, so I used those as a starting point for food detection and segmentation.

To make the model more accurate for food segmentation, I fine-tuned it using the [Food V18 dataset](https://universe.roboflow.com/lawrence-hair-wpavf/food-v18). Since I didn’t have access to NVIDIA GPUs over the weekend, I couldn’t train the model for a long time. As a result, the accuracy is not perfect but good enough for this project. I used both models in this project.

After training, I exported the model using ONNX Runtime. ONNX is great for optimizing models to run efficiently on different hardware. However, I had to write custom code for the preprocessing and postprocessing steps to make it all work smoothly.


## Preprocessing and Postprocessing

### Preprocessing

Before feeding images into the model, they need to be prepared in a specific way:
1. Convert the images to RGB format.
2. Resize them to match YOLO’s input size.
3. Normalize the pixel values to a range of 0 to 1.
4. Move the color channels to the beginning.

This part was straightforward.

### Postprocessing

Handling the model's output was the tricky part. YOLO gives two outputs: one for detection and another for segmentation.
1. Removed low-confidence boxes.
2. Used Non-Maximum Suppression (NMS) to eliminate overlapping boxes and keep the most accurate ones.
3. I improved accuracy by combining the detection results with the segmentation masks. Finally, I calculated the number of pixels for each food class and scaled them based on the original and resized image sizes.

My first implementation was inefficient because it removed low-confidence boxes at the end. However, I discovered that over 90% of the boxes were low-confidence, so I modified the process to filter them out at the beginning. This made the code less clean but much more efficient.


## Deployment

### FastAPI Integration

The model was deployed using FastAPI, chosen for its asynchronous design. Two main functions were implemented:
1. **`predict`**: Acts as the user-facing endpoint, queuing incoming requests.
2. **`process_batches`**: Processes the queue in batches, triggered either when a minimum batch size is reached or after a 0.5-second timeout. This dynamic batching approach maximizes resource efficiency.

The `process_batches` and `predict` functions work asynchronously and are connected via request and response queues. This design made dynamic batching possible by waiting 0.5 seconds for more requests to batch together or until the minimum batch size was reached.

### Containerization and Kubernetes Deployment

1. **Docker**: The application, along with models trained on COCO and Food V18 datasets, was containerized. Instructions for usage were documented in the README file.
2. **Kubernetes**: Separate pods were created for the two models. A load balancer service was configured to distribute requests between the pods, enabling A/B testing to analyze user responses to different models.


## Future Work

Although dynamic batching improved efficiency, the system currently lacks autoscaling capabilities. Adding an autoscaling service would allow automatic scaling of pods based on demand, whereas the current setup requires manual scaling during peak times.
