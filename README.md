# AIM
This project aims to enhance **women's safety** using OpenCV by detecting lone women in public spaces during nighttime for timely interventions.

# Project Overview
This project is a **Real-Time Human Detection and Gender Prediction System** utilizing YOLOv8 and DeepFace. The system tracks and counts people in a video feed, predicts their gender, and identifies specific conditions such as lone women at night for enhanced surveillance applications. It employs a centroid tracking algorithm to monitor objects across frames and integrates gender prediction using DeepFace for precise analysis.

# Domain
**Computer Vision and Artificial Intelligence**
This project belongs to the domain of computer vision, with a focus on human detection, tracking, and behavior analysis using state-of-the-art machine learning and AI techniques.

# Idea
The core idea is to develop an intelligent surveillance system that:
- Detects and tracks people in real-time using YOLOv8.
- Predicts the gender of detected individuals using DeepFace.
- Identifies scenarios such as lone women at night for potential safety interventions.
- Provides robust insights by counting the total number of people, segregating them by gender, and exporting data in JSON format for further processing.

The system aims to enhance security and monitoring in public places, offices, and other environments, ensuring efficient human behavior tracking with minimal manual intervention.

# Achievements Thus Far
- **Integration of YOLOv8:** Successfully implemented YOLOv8 for accurate real-time human detection.
- **Gender Prediction:** Integrated DeepFace for reliable gender analysis with minimal computational overhead.
- **Custom Tracker:** Developed a centroid tracker to monitor detected objects across frames efficiently.
- **Real-Time Alerts:** Identified specific conditions such as lone women at night and generated alerts.
- **Data Persistence:** Implemented JSON export to store information like total people count, gender count, and lone women detection.
- **Scalability:** Designed the system to work with video files, live feeds, and webcams.

# How to Execute the Code Sample

## Prerequisites
Ensure the following dependencies are installed:
- Python 3.12.4
- Required Python libraries:
  ```bash
  pip install ultralytics opencv-python-headless numpy deepface argparse
  ```
  ```pip install -r requirements.txt```
- A pre-trained YOLOv8 model file (`yolov8n.pt`) in the project directory.

## Steps to Run the Code
1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Prepare the Environment:**
   Ensure that you have installed all the dependencies mentioned in the prerequisites.

3. **Run the Script:**
   To execute the script, use the following command:
   ```bash
   python model.py --video <video_source>
   ```
   Replace `<video_source>` with:
   - A video file path (e.g., `video.mp4`).
   - A webcam index (e.g., `0` or `1`).

4. **Output:**
   - The system displays the video feed with bounding boxes, gender labels, and people count.
   - JSON data is saved in `people_count.json`.
   - Images of lone women detected at night are saved in the project directory.

5. **Stop the Program:**
   Press `q` to quit the video processing loop.

# Team Details
- **Team Name:** Syntax Error
- **Members:**
 1. Aayush Kumar Singh - AI/ML and Computer vision
 2. Adesh Dutta
 3. Aman Kumar

We are a team of enthusiastic engineers and researchers passionate about leveraging AI to solve real-world problems.

