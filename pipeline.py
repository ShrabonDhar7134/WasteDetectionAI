
# Fused Pipeline Code (Copy this into a .py file for deployment)
from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np

detection_model = YOLO('yolo_detection_model.pt')
classification_model = tf.keras.models.load_model('mobilenet_fused_model.h5')
class_names = ['brick', 'concrete', 'foam', 'plastic', 'stone', 'wood']

def fused_pipeline_with_explainability(image_path):
    results = detection_model(image_path, conf=0.5)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            detections.append((crop, class_id, x1, y1, x2, y2, conf))

    material_counts = {name: 0 for name in class_names}
    for crop, det_class, x1, y1, x2, y2, conf in detections:
        if crop.size > 0:
            crop_resized = cv2.resize(crop, (224, 224)) / 255.0
            pred = classification_model.predict(np.expand_dims(crop_resized, axis=0), verbose=0)
            pred_class = class_names[np.argmax(pred)]
            pred_conf = np.max(pred)
            material_counts[pred_class] += 1
            cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{pred_class} ({pred_conf:.2f})"
            cv2.putText(image_rgb, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    total_objects = sum(material_counts.values())
    recyclable = material_counts['brick'] + material_counts['concrete'] + material_counts['wood']
    summary = f"Total objects: {total_objects}\nMaterial breakdown: {material_counts}\nRecyclable estimate: {recyclable}/{total_objects} ({recyclable/total_objects*100:.1f}%)"
    return image_rgb, summary

# For Gradio
import gradio as gr
from PIL import Image

def gradio_pipeline(image):
    image_np = np.array(image)
    temp_path = '/tmp/temp_image.jpg'
    cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    explained_image, summary = fused_pipeline_with_explainability(temp_path)
    explained_pil = Image.fromarray(explained_image)
    return explained_pil, summary

interface = gr.Interface(fn=gradio_pipeline, inputs=gr.Image(type="pil"), outputs=[gr.Image(), gr.Textbox()], title="C&D Waste Intelligence System")
interface.launch(share=True)
