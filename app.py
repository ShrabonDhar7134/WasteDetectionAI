
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load models (adjust paths as needed)
from ultralytics import YOLO
import tensorflow as tf

detection_model = YOLO('final_models/yolo_detection_model.pt')
classification_model = tf.keras.models.load_model('final_models/mobilenet_fused_model.h5')

class_names = ['brick', 'concrete', 'foam', 'plastic', 'stone', 'wood']

def fused_pipeline_with_explainability(image):
    image_np = np.array(image)
    temp_path = '/tmp/temp_image.jpg'
    cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    results = detection_model(temp_path, conf=0.5)
    image_rgb = cv2.cvtColor(cv2.imread(temp_path), cv2.COLOR_BGR2RGB)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            crop = image_rgb[int(y1):int(y2), int(x1):int(x2)]
            detections.append((crop, class_id, x1, y1, x2, y2))

    material_counts = {name: 0 for name in class_names}
    for crop, det_class, x1, y1, x2, y2 in detections:
        if crop.size > 0:
            crop_resized = cv2.resize(crop, (224, 224)) / 255.0
            pred = classification_model.predict(np.expand_dims(crop_resized, axis=0), verbose=0)
            pred_class = class_names[np.argmax(pred)]
            pred_conf = np.max(pred)
            material_counts[pred_class] += 1
            cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_rgb, f"{pred_class} ({pred_conf:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    total_objects = sum(material_counts.values())
    recyclable = material_counts['brick'] + material_counts['concrete'] + material_counts['wood']
    summary = f"Total objects: {total_objects}\nMaterial breakdown: {material_counts}\nRecyclable estimate: {recyclable}/{total_objects} ({recyclable/total_objects*100:.1f}%)"
    return Image.fromarray(image_rgb), summary

# Streamlit App
st.title("🚀 Construction & Demolition Waste Intelligence System")
st.sidebar.header("Navigation")

tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results", "📈 Analytics"])

with tab1:
    st.header("Upload a Waste Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze"):
            with st.spinner("Processing..."):
                explained_image, summary = fused_pipeline_with_explainability(image)
                st.session_state['explained_image'] = explained_image
                st.session_state['summary'] = summary
            st.success("Analysis complete! Check Results tab.")

with tab2:
    st.header("Analysis Results")
    if 'explained_image' in st.session_state:
        st.image(st.session_state['explained_image'], caption="Explained Detection & Classification", use_column_width=True)
        st.text_area("Summary", st.session_state['summary'], height=100)
    else:
        st.info("Upload and analyze an image first.")

with tab3:
    st.header("Model Analytics")
    analytics_dir = 'analytics'
    if os.path.exists(analytics_dir):
        st.subheader("Accuracy & Loss Curves")
        st.image(os.path.join(analytics_dir, 'accuracy_loss_curves.png'), caption="Training Performance")
        st.subheader("Confusion Matrix")
        st.image(os.path.join(analytics_dir, 'confusion_matrix.png'), caption="Prediction Accuracy")
        st.subheader("Class Metrics")
        st.image(os.path.join(analytics_dir, 'class_metrics.png'), caption="Precision & Recall")
    else:
        st.error("Analytics not found. Run Cell 40 first.")
