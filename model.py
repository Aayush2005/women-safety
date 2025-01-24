from ultralytics import YOLO
import cv2
import numpy as np
from utils.centroidtracker import CentroidTracker
from utils.object_trackable import TrackableObject
from deepface import DeepFace
from datetime import datetime
import json

# Initializing Parameters
confThreshold = 0.6
inpWidth, inpHeight = 640, 640

# Load YOLOv8 Model
print("Loading model...")
model = YOLO('yolov8n.pt')

# Centroid Tracker Initialization
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackableObjects = {}
totalDown = 0
totalUp = 0

# Initialize JSON data dictionary
json_data = {
    "people_count": 0,
    "men": 0,
    "women": 0,
    "lone_women": 0
}

# Function to Compute IoU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

# Function to Merge Overlapping Boxes
def merge_boxes(boxes, iou_threshold=0.3):
    merged_boxes = []
    while boxes:
        box = boxes.pop(0)
        to_merge = []
        for other_box in boxes:
            if compute_iou(box, other_box) > iou_threshold:
                to_merge.append(other_box)
        for merge_box in to_merge:
            boxes.remove(merge_box)
            box[0] = min(box[0], merge_box[0])
            box[1] = min(box[1], merge_box[1])
            box[2] = max(box[2], merge_box[2])
            box[3] = max(box[3], merge_box[3])
        merged_boxes.append(box)
    return merged_boxes

# Gender Prediction using DeepFace
def predict_gender(cropped_person):
    try:
        # Convert OpenCV image (BGR) to RGB
        rgb_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
        
        # Perform gender analysis
        results = DeepFace.analyze(img_path=rgb_person, actions=['gender'], enforce_detection=False)
        
        # Ensure results are a dictionary
        if isinstance(results, list):
            results = results[0] 
        
        # Get dominant gender
        gender = max(results['gender'], key=results['gender'].get)
        print(f"Predicted Gender: {gender}")
        
        # Return the gender
        return gender

    except Exception as e:
        print(f"Error during gender prediction: {e}")
        return "Unknown"


def counting(objects, frame):
    global totalDown, totalUp
    frameHeight = frame.shape[0]

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
                if direction < 0 and frameHeight // 2 - 30 <= centroid[1] <= frameHeight // 2 + 30:
                    totalUp += 1
                    to.counted = True
                elif direction > 0 and frameHeight // 2 - 30 <= centroid[1] <= frameHeight // 2 + 30:
                    totalDown += 1
                    to.counted = True

        trackableObjects[objectID] = to

def is_night_time():
    current_hour = datetime.now().hour
    return current_hour >= 18 or current_hour < 6

def is_lone_person(objects):
    return len(objects) == 1

def process_frame(frame):
    # Initialize default JSON data structure
    json_data = {
        "people_count": 0,
        "men": 0,
        "women": 0,
        "lone_women": 0,
        "timestamp": datetime.now().isoformat(),
        "error": None
    }

    try:
        # YOLOv8 Inference
        results = model(frame)
        detections = results[0].boxes
        rects = []

        # Process detections
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0]
            class_id = int(box.cls[0])
            
            if class_id == 0 and conf > confThreshold:
                rects.append([x1, y1, x2, y2])

        # Merge overlapping boxes
        rects = merge_boxes(rects.copy()) if rects else []

        # Gender detection
        male_count = 0
        female_count = 0
        for rect in rects:
            x1, y1, x2, y2 = rect
            try:
                cropped_person = frame[y1:y2, x1:x2]
                if cropped_person.size == 0:
                    continue

                gender = predict_gender(cropped_person)
                if gender == 'Man':
                    male_count += 1
                elif gender == 'Woman':
                    female_count += 1
            except Exception as e:
                print(f"Gender detection error: {e}")

        # Update tracker
        objects = ct.update([(x1, y1, x2, y2) for x1, y1, x2, y2 in rects])
        counting(objects, frame)

        # Lone woman check
        lone_women_count = 0
        try:
            if (is_night_time() and 
                is_lone_person(objects) and 
                female_count == 1):
                cv2.imwrite(f"lone_woman_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
                lone_women_count = 1
        except Exception as e:
            print(f"Lone woman check failed: {e}")

        # Update JSON data
        json_data.update({
            "people_count": len(objects),
            "men": male_count,
            "women": female_count,
            "lone_women": lone_women_count
        })

        # # Annotate frame
        # cv2.putText(frame, f"People: {len(objects)}", (10, 35), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # cv2.putText(frame, f"M: {male_count} | F: {female_count}", (10, 55),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        json_data["error"] = str(e)
        print(f"Frame processing error: {e}")

    finally:
        # Always save JSON data regardless of errors
        try:
            with open('people_count.json', 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
        except Exception as e:
            print(f"Failed to save JSON: {e}")

    return frame, json_data