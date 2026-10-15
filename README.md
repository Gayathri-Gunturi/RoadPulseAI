# RoadPulse AI — Road Damage Intelligence & Repair Prioritization

An AI-powered road inspection prototype that detects potholes in road images and helps organize inspection records and repair progress through an interactive dashboard.

## Overview

RoadPulse AI combines computer vision with a lightweight inspection management system. It uses a pretrained YOLOv8 model to identify potholes in uploaded road images, displays the detected regions, and stores inspection summaries for later review.

The application also includes rule-based priority estimates, inspection analytics, and repair status tracking to demonstrate how road damage detection can be connected to a practical maintenance workflow.

## Images
<img width="1117" height="684" alt="image" src="https://github.com/user-attachments/assets/0a700219-9d41-412f-a730-7770cba809a4" />
<img width="1190" height="494" alt="image" src="https://github.com/user-attachments/assets/6a857920-6d87-4df9-bd35-d3775963d164" />
<img width="1495" height="438" alt="image" src="https://github.com/user-attachments/assets/7b600a4c-e1dc-4d95-b129-700fb75f78df" />
<img width="1503" height="666" alt="image" src="https://github.com/user-attachments/assets/8cbefaba-00da-443f-aab8-345e2df1ebf6" />
<img width="1531" height="680" alt="image" src="https://github.com/user-attachments/assets/90664372-3e19-49da-bd79-aa8d18d67de8" />

## Features

- **Pothole Detection:** Detect potholes in uploaded road images using a pretrained YOLOv8 model.
- **Visual Results:** Display annotated images with detected bounding boxes.
- **Detection Summary:** Show the number of detections and the highest detection confidence.
- **Priority Estimation:** Assign preliminary priority levels using detection count and bounding-box area.
- **Inspection History:** Save and review inspection records using SQLite.
- **Repair Status Tracking:** Manage inspection progress through `Reported`, `Under Review`, `Repair Scheduled`, and `Resolved`.
- **Analytics Dashboard:** Visualize inspection records, estimated priorities, and repair statuses.
- **CSV Export:** Download inspection history and detection reports for further analysis.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| YOLOv8 (Ultralytics) | Pothole detection |
| OpenCV | Image processing and annotation |
| Streamlit | Interactive web dashboard |
| SQLite | Persistent local inspection records |
| Pandas | Data handling and analytics |
| Matplotlib | Visualization of detection results |

## How It Works

1. Upload a road image through the dashboard.
2. The pretrained YOLOv8 model analyzes the image.
3. The application displays detected potholes and their bounding boxes.
4. A preliminary priority level is assigned using simple, rule-based criteria.
5. The inspection summary is saved in a local SQLite database.
6. Review previous inspections, explore analytics, and update repair statuses.

## Project Structure

```text
RoadPulseAI/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── road_test.jpg
```

`road_test.jpg` is an optional sample image for testing. Add it only if you have permission to share it.

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Git (optional, for cloning the repository)
- An internet connection for downloading the pretrained model

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/RoadPulseAI.git
cd RoadPulseAI
```

Replace `YOUR-USERNAME` with your GitHub username.

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Model and Evaluation

The application loads a pretrained pothole detection model from Hugging Face. Detection results depend on the model, image quality, and selected confidence threshold.

The priority classification is a prototype heuristic based on the number of detections and the relative area of detected bounding boxes. It is not a validated road-severity assessment and should not be used as the sole basis for real maintenance decisions.

Model accuracy and real-world performance have not been established by this project description. They should be evaluated using a representative, labeled dataset before making quantitative performance claims.

## Future Improvements

- Add location and geographic mapping for road inspections.
- Evaluate precision, recall, and other detection metrics on labeled road images.
- Improve priority estimation using validated severity criteria.
- Add inspection images and detailed repair notes to historical records.
- Deploy the application for broader access.
- Add authentication and role-based access for different users.

## Limitations

- Detection quality depends on the pretrained model and input image conditions.
- Priority levels are rule-based estimates rather than verified severity measurements.
- Inspection records are stored in a local SQLite database.
- The application is a prototype and has not been validated for operational road maintenance.


## Author

Gunturi Gayathri
---

*RoadPulse AI is an educational portfolio project exploring computer vision and inspection workflow management.*
