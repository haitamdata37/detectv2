import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch

# Load model and processor with no_timm revision to avoid timm dependency
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")

# Define labels of interest
objects_to_detect = {"cell phone", "laptop"}

# Streamlit UI
st.title("Student Device Detector")
st.write("Upload an image, and the model will detect if any phones or laptops are present.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and run model inference
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Process outputs to keep only high-confidence detections
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    # Initialize a flag to track if any devices are detected
    device_detected = False
    detected_objects = []  # List to store all detected objects and their scores

    # Draw bounding boxes on the image and check for phones/laptops
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        detected_objects.append((label_name, round(score.item(), 3)))
        
        if label_name in objects_to_detect:
            device_detected = True
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label_name}: {round(score.item(), 3)}", fill="red")
    
    # Display the labeled image with bounding boxes
    st.image(image, caption="Image with Detected Objects", use_column_width=True)

    # Display detected objects with confidence scores
    st.write("### Detected Objects:")
    for obj_name, score in detected_objects:
        st.write(f"- {obj_name}: {score}")

    # Show warning if a phone or laptop is detected
    if device_detected:
        st.warning("Warning: Phone or Laptop detected in the image!")
    else:
        st.success("No phones or laptops detected.")
