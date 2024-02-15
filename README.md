# Vehicle Detection System with YOLOv8

This repository contains code for a Vehicle Detection System using YOLOv8. The system is capable of detecting vehicle number plates in images. The model is trained using the YOLOv8 framework, and the dataset is provided in the repository.

## Model Training using Google Colab

To train the model using Google Colab, follow these steps:

1. Open the `Vehicle_NumberPlate.ipynb` notebook in Google Colab.
2. Mount your Google Drive to access the dataset and save the trained model.
3. Execute the cells in the notebook to train the YOLOv8 model using the provided dataset.
4. The trained model will be saved as `best.pt`.

## Vehicle Detection System Usage

## Important Files

- `model_training_colab.ipynb`: Google Colab notebook for training the YOLOv8 model.
- `app.py`: Streamlit app for using the trained model to detect vehicle number plates.
- `best.pt`: The trained YOLOv8 model weights.

## Dependencies

- `streamlit`: for creating the web application.
- `cv2`: OpenCV for image processing.
- `numpy`: NumPy for array manipulation.
- `ultralytics`: YOLOv8's training and inference framework.

## Streamlit App

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Run Streamlit app:
   ```bash
   streamlit run app.py
3. Open your web browser and navigate to the provided Streamlit URL (usually http://localhost:8501).

4. Use the file uploader to upload an image.

5. Click the "Predict" button to detect the vehicle number plate in the uploaded image.

## Demo

![ezgif-4-3b12535a90-ezgif com-video-to-gif-converter](https://github.com/Ibrahim-Naseef/Vehicle-NumberPlate-Detection-YOLOV8/assets/156147657/18fe1abd-0def-4d6f-9c85-3b843e263585)




