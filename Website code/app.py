import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Load a Region of Interest YOLOv8 model
model = YOLO('best.pt')

def detect_plate(image_path):
    # Detect objects in the image
    results = model(image_path)
    
    # Extract the first result (assuming only one detection)
    result = results[0]
    
    # Convert the tensor to a NumPy array and access the first element
    bounding_boxes = result.boxes.xyxy[0].cpu().numpy()

    # Extract the first bounding box coordinates
    x_min = int(bounding_boxes[0])
    y_min = int(bounding_boxes[1])
    x_max = int(bounding_boxes[2])
    y_max = int(bounding_boxes[3])
    
    # Load the original image
    original_image = cv2.imread(image_path)

    # Draw the bounding box on the image
    cv2.rectangle(original_image, (x_min, y_max), (x_max, y_min), (0, 255, 0), 2)

    # Extract the cropped region from the original image
    cropped_image = original_image[y_min:y_max, x_min:x_max]

    return original_image, cropped_image

# Streamlit UI
st.title('Vehicle Number Plate Detection')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to PNG format
    image_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    _, temp_png = cv2.imencode('.png', img)
    uploaded_image_path = 'temp_image.png'
    with open(uploaded_image_path, 'wb') as f:
        f.write(temp_png)

    # Display the original image
    original_image = cv2.imread(uploaded_image_path)
    st.image(original_image, caption='Original Image', use_column_width=True)

    if st.button('Predict'):
        # Perform detection
        original_image, cropped_image = detect_plate(uploaded_image_path)
        
        # Display the cropped image
        st.image(cropped_image, caption='Number Plate', use_column_width=True)
