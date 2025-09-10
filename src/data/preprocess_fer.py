import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import os
from tqdm import tqdm

#Configuration
CSV_PATH = 'data/raw/fer2013/fer2013.csv' 
# Saves the processed, cropped images to the data/processed/ directory
SAVE_DIR = 'data/processed/fer2013'
IMG_SIZE = 112

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

#Emotion Mapping
emotion_map = {0: '0_angry', 1: '1_disgust', 2: '2_fear', 3: '3_happy', 4: '4_sad', 5: '5_surprise', 6: '6_neutral'}

def preprocess_and_save(df, usage_type):
    """Processes rows from the dataframe and saves cropped faces."""
    print(f"Processing {usage_type} data...")
    usage_dir = os.path.join(SAVE_DIR, usage_type)
    os.makedirs(usage_dir, exist_ok=True)
    for emotion_code, emotion_name in emotion_map.items():
        os.makedirs(os.path.join(usage_dir, emotion_name), exist_ok=True)

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Saving {usage_type} images"):
        emotion_code = row['emotion']
        pixels = np.array(row['pixels'].split(), 'uint8').reshape(48, 48)

        image_rgb = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
        results = face_detector.process(image_rgb)

        cropped_face = pixels 
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw = pixels.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            x, y = max(0, x), max(0, y)
            cropped_face = pixels[y:y+h, x:x+w]

        if cropped_face.size > 0:
            resized_face = cv2.resize(cropped_face, (IMG_SIZE, IMG_SIZE))
            save_path = os.path.join(usage_dir, emotion_map[emotion_code], f'{usage_type}_{index}.png')
            cv2.imwrite(save_path, resized_face)

if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please make sure you have downloaded and extracted the fer2013.csv file to data/raw/fer2013/")
    else:
        df = pd.read_csv(CSV_PATH)
        
        df_train = df[df['Usage'] == 'Training']
        df_val = df[df['Usage'] == 'PublicTest']
        df_test = df[df['Usage'] == 'PrivateTest']
        
        preprocess_and_save(df_train, 'train')
        preprocess_and_save(df_val, 'val')
        preprocess_and_save(df_test, 'test')
        
        print(f"\nPreprocessing complete! Images are saved in '{SAVE_DIR}'")
