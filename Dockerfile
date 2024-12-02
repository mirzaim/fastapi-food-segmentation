# Use an official Python image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (including libGL for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app files into the container
# COPY yolo11m-seg.onnx yolo11m-seg.onnx
COPY yolo11m-seg-food.onnx yolo11m-seg-food.onnx
COPY app.py app.py

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the default command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
