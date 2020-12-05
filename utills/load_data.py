import os
import pandas as pd


def load_fk_dataset(data_path):
    """Load Facial Keypoints dataset as df, add path to file column"""

    train_labels_df = pd.read_csv(os.path.join(data_path, "training_frames_keypoints.csv"))
    test_labels_df = pd.read_csv(os.path.join(data_path, "test_frames_keypoints.csv"))
    train_labels_df.rename(columns = {"Unnamed: 0": "file_name"}, inplace=True)
    test_labels_df.rename(columns = {"Unnamed: 0": "file_name"}, inplace=True)
    train_labels_df['file_name'] = train_labels_df['file_name'].apply(
        lambda x: os.path.join(data_path, "training",  x))
    test_labels_df['file_name'] = test_labels_df['file_name'].apply(
        lambda x: os.path.join(data_path, "test",  x))
    return train_labels_df, test_labels_df


def extract_target_fk(labels_df):
    """Extract columns corresponding to face center from FK dataset"""
    labels_df = labels_df.rename(columns = {'58': "x_coord", '59':"y_coord"})
    labels_df = labels_df[['file_name', 'x_coord', 'y_coord']]
    return labels_df