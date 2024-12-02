import asyncio
from collections import defaultdict


import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, HTTPException


# COCO
# MODEL = "yolo11m-seg.onnx"
# yolo_classes = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
#     "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
#     "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
#     "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
#     "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
#     "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
#     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
#     "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
# ]
# food_classes = set([
#     "bear", "tie", "bottle", "wine glass", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
#     "cake",
# ])

# FOOD DATASET
MODEL = "yolo11m-seg-food.onnx"
yolo_classes = [
    'French beans', 'almond', 'apple', 'apricot', 'asparagus', 'avocado', 'bamboo shoots', 'banana', 'bean sprouts', 'biscuit', 'blueberry',
    'bread', 'broccoli', 'cabbage', 'cake', 'candy', 'carrot', 'cashew', 'cauliflower', 'celery stick', 'cheese butter', 'cherry',
    'chicken duck', 'chocolate', 'cilantro mint', 'coffee', 'corn', 'crab', 'cucumber', 'date', 'dried cranberries', 'egg', 'egg tart',
    'eggplant', 'enoki mushroom', 'fig', 'fish', 'french fries', 'fried meat', 'garlic', 'ginger', 'grape', 'green beans', 'hamburg',
    'hanamaki baozi', 'ice cream', 'juice', 'kelp', 'king oyster mushroom', 'kiwi', 'lamb', 'lemon', 'lettuce', 'mango', 'melon', 'milk',
    'milkshake', 'noodles', 'okra', 'olives', 'onion', 'orange', 'other ingredients', 'oyster mushroom', 'pasta', 'peach', 'peanut', 'pear',
    'pepper', 'pie', 'pineapple', 'pizza', 'popcorn', 'pork', 'potato', 'pudding', 'pumpkin', 'rape', 'raspberry', 'red beans', 'rice', 'salad',
    'sauce', 'sausage', 'seaweed', 'shellfish', 'shiitake', 'shrimp', 'snow peas', 'soup', 'soy', 'spring onion', 'steak', 'strawberry', 'tea',
    'tofu', 'tomato', 'walnut', 'watermelon', 'white button mushroom', 'white radish', 'wine', 'wonton dumplings'
]
food_classes = set(yolo_classes)


nc = len(yolo_classes)


# Load the ONNX model
session = ort.InferenceSession(
    MODEL,
    providers=['CPUExecutionProvider',
               #    'CUDAExecutionProvider',
               ]
)

app = FastAPI()

# Parameters for dynamic batching
BATCH_SIZE = 10  # Maximum batch size
BATCH_TIMEOUT = 0.5  # Maximum wait time for batch processing in seconds

request_queue = asyncio.Queue()
response_queue = {}


def preprocess_image(image: np.ndarray):
    input_size = (640, 480)
    original_shape = image.shape[:2]
    resized_image = cv2.resize(image, input_size)
    image_data = resized_image.astype('float32') / 255.0
    image_data = np.transpose(image_data, [2, 0, 1])  # HWC to CHW
    return image_data, original_shape


def decode_boxes(output, input_shape=(1, 1)):
    center_x, center_y, width, height = output[0], output[1], output[2], output[3]
    x_min = (center_x - width / 2) * input_shape[0]
    y_min = (center_y - height / 2) * input_shape[1]
    x_max = (center_x + width / 2) * input_shape[0]
    y_max = (center_y + height / 2) * input_shape[1]
    return np.stack([x_min, y_min, x_max, y_max]).round().astype(int)


def non_maximum_suppression(bboxes, scores, iou_threshold=0.7):
    x_min = bboxes[0]
    y_min = bboxes[1]
    x_max = bboxes[2]
    y_max = bboxes[3]

    areas = (x_max - x_min + 1) * (y_max - y_min + 1)
    order = np.argsort(scores)[::-1]

    keep_indices = []

    while len(order) > 0:
        i = order[0]
        keep_indices.append(i)

        xx_min = np.maximum(x_min[i], x_min[order[1:]])
        yy_min = np.maximum(y_min[i], y_min[order[1:]])
        xx_max = np.minimum(x_max[i], x_max[order[1:]])
        yy_max = np.minimum(y_max[i], y_max[order[1:]])

        inter_area = np.maximum(0, xx_max - xx_min + 1) * \
            np.maximum(0, yy_max - yy_min + 1)
        union_area = areas[i] + areas[order[1:]] - inter_area
        iou = inter_area / union_area

        # Keep boxes with IoU less than the threshold
        remaining_indices = np.where(iou <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return np.sort(np.array(keep_indices))


@app.post("/predict/")
async def predict(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image.")
    processed_image, original_shape = preprocess_image(image)

    request_id = id(processed_image)

    future = asyncio.get_event_loop().create_future()
    response_queue[request_id] = future

    await request_queue.put((request_id, (processed_image, original_shape)))

    result = await future
    del response_queue[request_id]
    return result


async def process_batches():
    while True:
        batch = []
        original_shapes = []
        request_ids = []
        start_time = asyncio.get_event_loop().time()

        while len(batch) < BATCH_SIZE:
            try:
                timeout = max(0, BATCH_TIMEOUT -
                              (asyncio.get_event_loop().time() - start_time))
                request_id, (image, original_shape) = await asyncio.wait_for(request_queue.get(), timeout)
                await asyncio.sleep(0.01)


                batch.append(image)
                original_shapes.append(original_shape)
                request_ids.append(request_id)
            except asyncio.TimeoutError:
                break

        if batch:
            batch = np.stack(batch)
            outputs = session.run(None, {session.get_inputs()[0].name: batch})

            results = []
            for original_shape, bbox_output, masks in zip(original_shapes, outputs[0], outputs[1]):
                class_areas = defaultdict(float)
                class_scores = bbox_output[4:4+nc]
                predicted_classes = np.argmax(class_scores, axis=0)
                class_confidences = class_scores[predicted_classes, np.arange(
                    class_scores.shape[1])]
                valid_indices = class_confidences >= 0.7
                valid_classes = predicted_classes[valid_indices]

                bboxes = decode_boxes(
                    bbox_output[:4, valid_indices], (0.25, 0.25))
                mask_coefficients = bbox_output[4+nc:, valid_indices]

                nms_indices = non_maximum_suppression(
                    bboxes, class_confidences[valid_indices], iou_threshold=0.7)
                if len(nms_indices) == 0:
                    results.append({
                        "class_areas": {},
                        "model": MODEL,
                    })
                    continue

                mask_coefficients = mask_coefficients[:, nms_indices]
                masks = mask_coefficients.T @ masks.reshape(32, -1)
                masks = 1 / (1 + np.exp(-masks))
                masks = (masks > 0.5).astype(np.uint8)

                area_coefficients = (
                    original_shape[0] / 160) * (original_shape[1] / 120)
                for cls_idx, mask, bbox in zip(valid_classes[nms_indices], masks, bboxes[:, nms_indices].T):
                    cropped_mask = mask.reshape(
                        120, 160)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    class_areas[int(cls_idx)] += (np.sum(cropped_mask)
                                                  * area_coefficients).round().astype(int)

                results.append({
                    "class_areas": {yolo_classes[cls_idx]: area for cls_idx, area in class_areas.items() if yolo_classes[cls_idx] in food_classes},
                    "model": MODEL,
                })

            for req_id, result in zip(request_ids, results):
                if req_id in response_queue:
                    response_queue[req_id].set_result(result)


# Background task for processing batches
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_batches())
