# Food Segmentation and Area Calculation

This project is a FastAPI-based application for performing food segmentation and calculating the area of each food. The application uses ONNX Runtime for inference and supports dynamic batching for efficient processing of image requests. It also includes Kubernetes deployments for horizontal scaling. The project comes with a client script for sending concurrent requests to the API, a Dockerfile for containerization, and Kubernetes manifests for deployment.

## Features

- **Food Detection and Segmentation**: Uses a pre-trained YOLO model to detect and segment food items from uploaded images.
- **ONNX Runtime**: The model is deployed using ONNX Runtime, supporting CPU inference.
- **Dynamic Batching**: Supports batching of requests to optimize throughput.
- **FastAPI**: Provides an easy-to-use API interface to upload images and receive detection results.
- **Client for Testing**: A `client.py` script for sending multiple concurrent image requests to the server.
- **Containerization**: Dockerfile for easy containerization of the application.
- **Kubernetes Deployment**: YAML files for deploying the service and the application in Kubernetes.

## Project Structure

```
.
├── .gitignore
├── app.py                # FastAPI application for food detection and segmentation
├── client.py             # Client script for sending requests to the API
├── Dockerfile            # Dockerfile to build the container image
├── images/               # Folder containing example food images
├── k8s/                  # Kubernetes manifests for deployment
│   ├── deployment.yaml   # Deployment definition for Kubernetes
│   └── service.yaml      # Service definition for Kubernetes
├── requirements.txt      # Python dependencies
├── yolo11m-seg-food.onnx # ONNX model fine-tuned on Food v18 dataset for food segmentation
└── yolo11m-seg.onnx      # ONNX model pre-trained on COCO dataset for general object detection
```

## Setup and Installation

There are three independent methods to deploy this project.

### Source Code Installation

Requires Python 3.12.

1. Clone the repository:

   ```bash
   git clone https://github.com/mirzaim/yolo-food-segmentation.git
   cd yolo-food-segmentation
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:

   ```bash
   fastapi run app --port 8000
   ```

4. (Optional) Run the client script to test the API:

   ```bash
   python client.py --num_requests 50
   ```

### Docker

Requires Docker.

1. Build the Docker image:

   ```bash
   docker build -t yolo-food-segmentation:latest .
   ```

2. Run the container:

   ```bash
   docker run -p 8000:8000 yolo-food-segmentation:latest
   ```

Alternatively, you can use prebuilt Docker images instead of building the project yourself:

- For the COCO dataset:

   ```bash
   docker run -p 8000:8000 mirzaim/fastapi-food-segmentation:0.6.coco
   ```

- For the Food v18 dataset:

   ```bash
   docker run -p 8000:8000 mirzaim/fastapi-food-segmentation:0.6.food
   ```

### Kubernetes Deployment

Requires Docker and Kubernetes.

1. Deploy the application to Kubernetes:

   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   ```

2. If using Minikube, assign an endpoint to the service:

   ```bash
   minikube service fastapi-yolo-service --url
   ```

3. Update the `--api_url` parameter with the output of the previous step:

   ```bash
   python client.py --num_requests 200 --api_url <output of "minikube service fastapi-yolo-service --url">/predict/
   ```

## API Endpoints

- `POST /predict/`: Upload an image for food detection and segmentation.
  - **Request**: Multipart/form-data with an image file.
  - **Response**: JSON with detected food items and their respective areas.

## Client Script (`client.py`)

The `client.py` script sends concurrent requests to the FastAPI server using images from a specified folder. This is useful for load testing and benchmarking the server.

Usage:

```bash
python client.py --image_folder images --num_requests 100 --api_url http://0.0.0.0:8000/predict/
```

If using Kubernetes with Minikube, update the `--api_url` parameter with the output of:

```bash
minikube service fastapi-yolo-service --url
```

and append `/predict/`.

## Model Information

- **YOLO Models**:
  - `yolo11m-seg.onnx`: A model pre-trained on the COCO dataset, focusing on subclasses related to food items.
  - `yolo11m-seg-food.onnx`: A model fine-tuned on the [Food v18 dataset](https://universe.roboflow.com/lawrence-hair-wpavf/food-v18) for more accurate food detection and segmentation.

## Deployment Details

- **Docker**: The Dockerfile sets up the environment and installs dependencies for running the FastAPI server.
- **Kubernetes**: The Kubernetes manifests (`deployment.yaml` and `service.yaml`) configure the application to deploy in a Kubernetes cluster, with dedicated deployments for food-specific and general object segmentation models.
