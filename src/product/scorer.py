import numpy as np

EMOTION_VA_MAP = {
    'angry':    {'v': -0.6, 'a': 0.8},
    'disgust':  {'v': -0.8, 'a': 0.3},
    'fear':     {'v': -0.4, 'a': 0.7},
    'happy':    {'v': 0.8,  'a': 0.6},
    'sad':      {'v': -0.7, 'a': -0.5},
    'surprise': {'v': 0.4,  'a': 0.9},
    'neutral':  {'v': 0.0,  'a': -0.2}
}

EMOTION_PRODUCTIVITY_MAP = {
    'angry':    -0.8,
    'disgust':  -0.7,
    'fear':     -0.5,
    'happy':     0.6,
    'sad':      -0.9,
    'surprise':  0.2,
    'neutral':   0.7
}

def calculate_scores(emotion_data, fatigue_index):
    """
    Calculates final scores for Productivity, Valence, and Arousal.

    Args:
        emotion_data (dict): Output from uncertainty_infer module.
            Required keys: 'label_probs', optional: 'confidence'.
        fatigue_index (float): From fatigue_calculator, in [0, 1].

    Returns:
        dict: Keys: 'productivity', 'valence', 'arousal' (all floats in [-1, 1])
    """
    if not emotion_data:
        return {'productivity': 0.0, 'valence': 0.0, 'arousal': 0.0}

    emotion_probs = emotion_data['label_probs']
    confidence = emotion_data.get('confidence', 1.0)

    valence = sum(prob * EMOTION_VA_MAP[emo]['v'] for emo, prob in emotion_probs.items())
    arousal = sum(prob * EMOTION_VA_MAP[emo]['a'] for emo, prob in emotion_probs.items())

    emotion_productivity = sum(prob * EMOTION_PRODUCTIVITY_MAP[emo] for emo, prob in emotion_probs.items())


    fatigue_penalty = np.sqrt(fatigue_index)  
    final_productivity = emotion_productivity * (1 - fatigue_penalty) * confidence

    final_productivity = np.clip(final_productivity, -1.0, 1.0)

    return {
        'productivity': round(final_productivity, 4),
        'valence': round(valence, 4),
        'arousal': round(arousal, 4)
    }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧪 PRODUCTIVITY SCORER - TEST OUTPUTS")
    print("="*60)

    sample_emotion_data_1 = {
        'label_probs': {'happy': 0.8, 'neutral': 0.15, 'surprise': 0.05},
        'confidence': 0.95
    }
    sample_fatigue_1 = 0.1
    scores_1 = calculate_scores(sample_emotion_data_1, sample_fatigue_1)
    print(f"😊 Happy + Low Fatigue (0.1): {scores_1}")

    sample_fatigue_2 = 0.5
    scores_2 = calculate_scores(sample_emotion_data_1, sample_fatigue_2)
    print(f"😐 Happy + Medium Fatigue (0.5): {scores_2}")

    sample_fatigue_3 = 0.9
    scores_3 = calculate_scores(sample_emotion_data_1, sample_fatigue_3)
    print(f"😴 Happy + High Fatigue (0.9): {scores_3}")


    sample_emotion_data_4 = {
        'label_probs': {'sad': 0.9, 'neutral': 0.1},
        'confidence': 0.90
    }
    scores_4 = calculate_scores(sample_emotion_data_4, sample_fatigue_2)
    print(f"😞 Sad + Medium Fatigue (0.5): {scores_4}")


    sample_emotion_data_5 = {
        'label_probs': {'angry': 0.2, 'fear': 0.2, 'sad': 0.2, 'happy': 0.1, 'neutral': 0.3},
        'confidence': 0.45
    }
    scores_5 = calculate_scores(sample_emotion_data_5, sample_fatigue_1)
    print(f"❓ Uncertain + Low Fatigue (0.1): {scores_5}")