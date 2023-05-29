from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from scipy.spatial.distance import cdist
import math
import time


class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(centroids), 2), dtype="int")
        for (i, (cX, cY)) in enumerate(centroids):
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

    def get_active_objects(self):
        return self.objects



cap = cv2.VideoCapture("./Videos/video1.mp4")  # For Video

model = YOLO("./Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

vehicle_count = 0  # Counter for the number of vehicles
counted_vehicles = set()  # Set to store the IDs of counted vehicles
tracker = CentroidTracker()  # Object tracker instance

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # List to store the centroids of the tracked objects
    tracked_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                # Draw dot on vehicle
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
                
                # Add centroid to the list of tracked objects
                tracked_objects.append((cx, cy))

    # Update the object tracker with the detected centroids
    tracker.update(tracked_objects)

    # Get the active tracked objects and their IDs
    active_objects = tracker.get_active_objects()
    active_object_ids = set(active_objects.keys())

    # Count the vehicles that haven't been counted yet
    new_vehicles = active_object_ids - counted_vehicles
    vehicle_count += len(new_vehicles)
    counted_vehicles.update(new_vehicles)

    # Display vehicle count
    cv2.putText(img, f"Vehicle count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    density = vehicle_count / 20
    print(f"Density: {density}")
    cv2.imshow("Image", img)
    cv2.waitKey(1)

