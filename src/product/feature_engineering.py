import pandas as pd
import numpy as np
import os

def engineer_features(per_frame_log_path, task_summary_log_path):
    """
    Reads raw per-frame data and task summary data to create a model-ready feature set.

    Args:
        per_frame_log_path (str): Path to the directory containing per-frame CSV files.
        task_summary_log_path (str): Path to the task_summary.csv file.

    Returns:
        pandas.DataFrame: A dataframe where each row represents one task session
                          with aggregated features and ground truth labels.
    """
    try:
        task_summary_df = pd.read_csv(task_summary_log_path)
    except FileNotFoundError:
        print(f"Error: Task summary file not found at {task_summary_log_path}")
        return None

    all_features = []

    for index, row in task_summary_df.iterrows():
        user_id = row['user_id']
        session_id = row['session_id']
        task_id = row['task_id']

      
        matching_files = [f for f in os.listdir(per_frame_log_path) if f.startswith(f"{user_id}_{session_id}_{task_id}")]
        if not matching_files:
            print(f"Warning: No per-frame log found for {user_id}/{session_id}/{task_id}")
            continue
        
        frame_log_file = os.path.join(per_frame_log_path, matching_files[0])
        frame_df = pd.read_csv(frame_log_file)
        
        if frame_df.empty:
            continue

        
        features = {
            'user_id': user_id,
            'session_id': session_id,
            'task_id': task_id
        }

        metrics_to_agg = ['confidence', 'entropy', 'valence', 'arousal', 'EAR', 'MAR', 'FI']
        for col in metrics_to_agg:
            features[f'{col}_mean'] = frame_df[col].mean()
            features[f'{col}_std'] = frame_df[col].std()
            features[f'{col}_volatility'] = frame_df[col].diff().std() 
   
        emotion_cols = [col for col in frame_df.columns if col.startswith('probs_')]
        if not emotion_cols and 'top1' in frame_df.columns:
             
             emotion_histogram = frame_df['top1'].value_counts(normalize=True).to_dict()
             for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                 features[f'emotion_hist_{emotion}'] = emotion_histogram.get(emotion, 0)
        
        all_features.append(features)

    if not all_features:
        print("No features could be engineered. Check log files.")
        return pd.DataFrame()

    feature_df = pd.DataFrame(all_features)
    model_ready_df = pd.merge(feature_df, task_summary_df, on=['user_id', 'session_id', 'task_id'])
    
    return model_ready_df

if __name__ == '__main__':
    if not os.path.exists("data_log/per_frame"):
        print("Creating dummy data for demonstration...")
        os.makedirs("data_log/per_frame", exist_ok=True)
        os.makedirs("data_log/per_task", exist_ok=True)
        
       
        dummy_frame_data = {
            'timestamp': pd.to_datetime(pd.date_range(start='1/1/2025', periods=12, freq='5S')),
            'top1': ['neutral']*6 + ['happy']*6, 'confidence': np.linspace(0.8, 0.9, 12),
            'entropy': np.linspace(0.5, 0.4, 12), 'valence': np.linspace(0.1, 0.6, 12),
            'arousal': np.linspace(-0.2, 0.4, 12), 'EAR': 0.25, 'MAR': 0.1, 'FI': np.linspace(0.1, 0.3, 12)
        }
        pd.DataFrame(dummy_frame_data).to_csv("data_log/per_frame/user_01_session_01_typing_test_dummy.csv", index=False)

        dummy_task_data = {
            'user_id': ['user_01'], 'session_id': ['session_01'], 'task_id': ['typing_test'],
            'wpm': [50], 'accuracy': [0.95], 'task_perf_score': [47.5],
            'self_prod_1_5': [4], 'self_fatigue_1_5': [2], 'self_stress_1_5': [2]
        }
        pd.DataFrame(dummy_task_data).to_csv("data_log/per_task/task_summary.csv", index=False)


    final_df = engineer_features(
        per_frame_log_path="data_log/per_frame",
        task_summary_log_path="data_log/per_task/task_summary.csv"
    )

    if final_df is not None and not final_df.empty:
        print("Feature engineering complete!")
        print(final_df.head())
        final_df.to_csv("data_log/model_ready_dataset.csv", index=False)
        print("\nSaved model-ready dataset to 'data_log/model_ready_dataset.csv'")
