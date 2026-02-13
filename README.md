
<p align="center">
  <h1 align="center">Construction & Demolition Waste Intelligence System</h1>
  <p align="center">
    AI-powered Waste Detection • Classification • Sustainability Analytics
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange" />
  <img src="https://img.shields.io/badge/TensorFlow-MobileNetV2-FF6F00?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-WebApp-ff4b4b?logo=streamlit" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" />
</p>


An end-to-end, explainable AI system for detecting, classifying, and analyzing construction and demolition (C&D) waste materials using computer vision. This project goes beyond typical academic work by providing actionable insights for recycling and sustainability decisions.

## Features
- **Object Detection**: Identifies waste objects (bricks, concrete, etc.) in images using YOLOv8.
- **Material Classification**: Classifies detected objects into 6 material types using MobileNetV2, fused with multi-dataset training (CODD + TrashNet).
- **Explainability**: Visual overlays with bounding boxes, class labels, and confidence scores.
- **Actionable Analytics**: Provides material breakdowns and recyclability estimates.
- **Advanced Analytics**: Graphs for accuracy/loss curves, confusion matrix, and class metrics.
- **Video Processing**: Real-time analysis of videos for dynamic waste monitoring.
- **User Feedback Loop**: Active learning with correction and retraining for model improvement.
- **Sustainability Calculator**: CO2 savings estimates based on recyclability (e.g., recycling saves emissions from new production).
- **Deployment**: Streamlit web app with tabs for upload, results, analytics, feedback, and video.

## Dataset
- Primary: CODD (Construction and Demolition Waste Object Detection Dataset, 6,254 images).
- Fused: Integrated with TrashNet (additional plastics for robustness).
- Classes: brick, concrete, foam, plastic, stone, wood.
- Preprocessing: Converted to YOLO format, augmented for balance.

## Model Performance
- Detection (YOLOv8): mAP50=0.802, precision=89.4%, recall=69.3%.
- Classification (MobileNetV2, Fused): ~96% train accuracy, ~92% val accuracy.
- Fusion: Improved generalization via multi-dataset training.

## Setup Instructions
1. Clone/download this repository.
2. Install dependencies: `pip install ultralytics tensorflow streamlit opencv-python seaborn sklearn pyngrok pandas`.
3. Download models from `final_models/` folder.
4. Run `streamlit run app.py` to launch the app (use ngrok for public access if needed).

## Usage
- Launch the Streamlit app.
- Upload an image/video in respective tabs.
- View results, provide feedback, and check analytics/CO2 savings.

## Results & Insights
- Successfully detects and classifies waste in real images/videos.
- Multi-dataset fusion boosts performance on diverse inputs.
- Analytics graphs highlight model strengths (e.g., high precision for brick/concrete).
- CO2 calculator provides environmental impact (e.g., "500kg CO2 saved by recycling").
- Provides decision-oriented outputs for industry use.

## Future Improvements
- Add more datasets (e.g., metal/glass).
- Integrate video processing for real-time analysis.
- Deploy to cloud (e.g., AWS) for scalability.
- Address edge cases (e.g., improve stone detection).

## Technologies
- Python, PyTorch (YOLO), TensorFlow (MobileNet), Streamlit, Seaborn.
- Trained on Google Colab with GPU.

## License
Open-source for educational and industry use.

Contact: dharshrabon2004@gmail.com for questions.

## Project Structure

```
WasteDetectionAI/
│
├── final_models/               # Trained YOLO and MobileNet models
│   ├── yolo_detection_model.pt
│   ├── mobilenet_classification_model.h5
│   └── mobilenet_fused_model.h5
│
├── datasets/                   # Processed datasets (not uploaded to GitHub)
│
├── app.py                      # Streamlit web application
├── pipeline.py                 # Detection + Classification pipeline
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
└── .gitignore                  # Ignored files
```



