import pandas as pd
from torch.utils.data import Dataset
import torch
import os

class CustomFeatureDataset(Dataset):

    def __init__(self, pkl_file):

        self.df = pd.read_pickle(pkl_file)

        self.feature_vectors = self.df['feature_vector'].tolist()
        self.video_paths = self.df['video_path'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        feature_vector = torch.tensor(self.feature_vectors[idx], dtype=torch.float32)
        video_path = self.video_paths[idx]
        label = self.labels[idx]


        return feature_vector, video_path, label

class CustomFeatureSensorDataset(Dataset):
    def __init__(self, pkl_file):
        self.df = pd.read_pickle(pkl_file)

        self.feature_vectors = self.df['feature_vector'].tolist()
        self.video_paths = self.df['video_path'].tolist()
        self.labels = self.df['label'].tolist()
        self.sensor_data = self.df['sensor_data'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Base image feature vector
        feature_vector = torch.tensor(self.feature_vectors[idx], dtype=torch.float32).squeeze()

        # Handle variable-length or empty sensor data
        sensor_matrix = self.sensor_data[idx]
        if isinstance(sensor_matrix, list) and len(sensor_matrix) > 0:
            sensor_tensor = torch.tensor(sensor_matrix, dtype=torch.float32)

            # Flatten sensor matrix (e.g., NxM → 1x(N*M))
            sensor_tensor = sensor_tensor.flatten()
        else:
            sensor_tensor = torch.empty((0,), dtype=torch.float32)  # or some fallback

        # Concatenate image features and sensor features
        final_feature_vector = torch.cat((feature_vector, sensor_tensor), dim=0)

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        return final_feature_vector, video_path, label
    
class CustomFeatureSensorSceneDataset(Dataset):
    def __init__(self, feature_pkl_file, sgg_pkl_file):
        self.df = pd.read_pickle(feature_pkl_file)
        self.sgg_df = pd.read_pickle(sgg_pkl_file)

        self.sgg_map = {clip: vec for clip, vec in zip(self.sgg_df["Clip"], self.sgg_df["Matrix"])}

        self.feature_vectors = self.df['feature_vector'].tolist()
        self.video_paths = self.df['video_path'].tolist()
        self.labels = self.df['label'].tolist()
        self.sensor_data = self.df['sensor_data'].tolist()

        self.scene_graphs = []
        for video_path in self.video_paths:
            clip_name = os.path.basename(video_path)  # assumes Clip == filename
            if clip_name in self.sgg_map:
                self.scene_graphs.append(self.sgg_map[clip_name])
            else:
                # fallback: vector of zeros if missing
                self.scene_graphs.append(torch.zeros(len(next(iter(self.sgg_map.values())))))
                print(f"Warning: No scene graph found for {clip_name}, filling with zeros")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Base image feature vector
        feature_vector = torch.tensor(self.feature_vectors[idx], dtype=torch.float32).squeeze()

        # Handle variable-length or empty sensor data
        sensor_matrix = self.sensor_data[idx]
        if isinstance(sensor_matrix, list) and len(sensor_matrix) > 0:
            sensor_tensor = torch.tensor(sensor_matrix, dtype=torch.float32)
            sensor_tensor = sensor_tensor.flatten()
        else:
            sensor_tensor = torch.empty((0,), dtype=torch.float32)

        # Scene graph vector
        scene_vector = torch.tensor(self.scene_graphs[idx], dtype=torch.float32).squeeze()

        # Combine all modalities
        final_feature_vector = torch.cat((feature_vector, sensor_tensor, scene_vector), dim=0)

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        return final_feature_vector, video_path, label
    
class CustomFeatureSceneDataset(Dataset):
    def __init__(self, feature_pkl_file, sgg_pkl_file):
        # Load video features
        self.df = pd.read_pickle(feature_pkl_file)

        # Load scene graph features
        self.sgg_df = pd.read_pickle(sgg_pkl_file)
        self.sgg_map = {clip: vec for clip, vec in zip(self.sgg_df["Clip"], self.sgg_df["Matrix"])}

        # Core info
        self.feature_vectors = self.df['feature_vector'].tolist()
        self.video_paths = self.df['video_path'].tolist()
        self.labels = self.df['label'].tolist()

        # Match scene graph vectors to videos
        self.scene_graphs = []
        example_vec = next(iter(self.sgg_map.values()))  # grab one to get vector length
        for video_path in self.video_paths:
            clip_name = os.path.basename(video_path)  # assumes Clip == filename
            if clip_name in self.sgg_map:
                self.scene_graphs.append(self.sgg_map[clip_name])
            else:
                # fallback: zeros if missing
                self.scene_graphs.append(torch.zeros(len(example_vec)))
                print(f"Warning: No scene graph found for {clip_name}, filling with zeros")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Video feature vector
        feature_vector = torch.tensor(self.feature_vectors[idx], dtype=torch.float32).squeeze()

        # Scene graph vector
        scene_vector = torch.tensor(self.scene_graphs[idx], dtype=torch.float32).squeeze()

        # Concatenate video features + scene graph
        final_feature_vector = torch.cat((feature_vector, scene_vector), dim=0)

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        return final_feature_vector, video_path, label