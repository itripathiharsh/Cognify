import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
import mediapipe as mp
import os

MODEL_PATH = 'models/fer_mobilenet_v2.pth' 
IMG_SIZE = 112
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_model():
    """Loads the pre-trained emotion detection model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run the training script (src/fer/train.py) first.")
        return None

    model = models.mobilenet_v2()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_emotion_probs(image: np.ndarray):
    """
    Predicts emotion probabilities from a single image frame.
    Args:
        image: A numpy array representing the image (in BGR format from OpenCV).
    Returns:
        A dictionary containing label probabilities, the top prediction, and its probability.
        Returns None if no face is detected.
    """
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    if not results.detections:
        return None

    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                 int(bboxC.width * iw), int(bboxC.height * ih)

    x, y = max(0, x), max(0, y)
    face_img = image_rgb[y:y+h, x:x+w]

    if face_img.size == 0:
        return None

    
    face_tensor = inference_transforms(face_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(face_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        
    probs_cpu = probabilities.cpu().numpy()
    
    label_probs = {EMOTION_LABELS[i]: float(probs_cpu[i]) for i in range(len(EMOTION_LABELS))}
    top1_prob, top1_idx = torch.max(probabilities, 0)
    top1_label = EMOTION_LABELS[top1_idx.item()]

    return {
        'label_probs': label_probs,
        'top1': top1_label,
        'top1_prob': float(top1_prob.item())
    }

if __name__ == '__main__':
    if model is None:
        exit()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        prediction = predict_emotion_probs(frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        if prediction:
            text = f"{prediction['top1']} ({prediction['top1_prob']:.2f})"
            cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Webcam Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
