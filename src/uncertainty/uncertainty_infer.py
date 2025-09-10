import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import os
from torchvision import models, transforms
import mediapipe as mp
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy

class UncertaintyInference:
    """
    A class to handle emotion inference with MC Dropout for uncertainty estimation.
    """
    def __init__(self, model_path='models/fer_mobilenet_v2_dropout.pth', temp_path='models/temperature.json'):
        self.IMG_SIZE = 112
        self.NUM_CLASSES = 7
        self.N_PASSES = 20  
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        self.model = models.mobilenet_v2()
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.last_channel, self.NUM_CLASSES),
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        self.model = self.model.to(self.DEVICE)
        print("Uncertainty model loaded.")

        try:
            with open(temp_path, 'r') as f:
                self.temperature = json.load(f)['temperature']
        except FileNotFoundError:
            print("Warning: temperature.json not found. Using T=1.0 (no calibration).")
            self.temperature = 1.0

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        self.inference_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_with_uncertainty(self, image: np.ndarray):
        """
        Predicts emotion probabilities from a single image frame with uncertainty.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image_rgb)

        if not results.detections:
            return None

        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        x, y = max(0, x), max(0, y)
        face_img = image_rgb[y:y+h, x:x+w]

        if face_img.size == 0:
            return None

        face_tensor = self.inference_transforms(face_img).unsqueeze(0).to(self.DEVICE)

        self.model.train()
        
        stochastic_predictions = []
        with torch.no_grad():
            for _ in range(self.N_PASSES):
                logits = self.model(face_tensor)
                stochastic_predictions.append(logits)
        
        stochastic_predictions = torch.stack(stochastic_predictions).squeeze(1).cpu().numpy()

        mean_probs = softmax(np.mean(stochastic_predictions, axis=0) / self.temperature, axis=-1)
        predictive_entropy = scipy_entropy(mean_probs, base=2)

        all_probs = softmax(stochastic_predictions / self.temperature, axis=-1)
        individual_entropies = np.apply_along_axis(scipy_entropy, axis=1, arr=all_probs, base=2)
        expected_entropy = np.mean(individual_entropies)
        mutual_information = predictive_entropy - expected_entropy
      
        normalized_entropy = predictive_entropy / np.log2(self.NUM_CLASSES)
        confidence = 1 - normalized_entropy
        
       
        top1_idx = np.argmax(mean_probs)
        top1_label = self.EMOTION_LABELS[top1_idx]
        top1_prob = float(mean_probs[top1_idx])
        label_probs = {self.EMOTION_LABELS[i]: float(mean_probs[i]) for i in range(len(self.EMOTION_LABELS))}

        return {
            'label_probs': label_probs,
            'top1': top1_label,
            'top1_prob': top1_prob,
            'confidence': float(confidence),
            'entropy': float(predictive_entropy),
            'mutual_info': float(mutual_information)
        }

def main_test():
    """Main function to test the inference class with a webcam."""
    inference_engine = UncertaintyInference()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = inference_engine.predict_with_uncertainty(frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0) # Green for high confidence
        if prediction:
            if prediction['confidence'] < 0.6:
                color = (0, 0, 255) # Red for low confidence
            
            text = f"{prediction['top1']} ({prediction['top1_prob']:.2f})"
            conf_text = f"Conf: {prediction['confidence']:.2f}"
            
            cv2.putText(frame, text, (10, 30), font, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, conf_text, (10, 70), font, 1, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Uncertainty Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_test()

