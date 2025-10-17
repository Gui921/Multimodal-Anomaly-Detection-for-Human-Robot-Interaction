import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomKineticsDataset(Dataset):
    def __init__(self, root_dir, frames_per_clip=16, transform=None):
        """
        Args:
            root_dir (str): Root dataset directory where each subfolder is a class.
            frames_per_clip (int): Number of frames to sample per video.
            transform (callable, optional): Transformations for the frames.
        """

        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform

        # Scan all class folders and collect video paths
        self.video_paths, self.labels, self.class_to_idx = self._scan_videos()
        print(self.class_to_idx)
        

    def _scan_videos(self):
        """Scans the dataset directory to get video paths and class labels."""
        class_to_idx = {}  # Maps class name to label index
        video_paths = []
        labels = []

        # Iterate through subdirectories
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                class_to_idx[class_name] = idx  # Assign index to class

                # Collect video files inside the class folder
                for video_file in os.listdir(class_path):
                    if video_file.endswith((".mp4", ".avi", ".mov")):
                        video_paths.append(os.path.join(class_path, video_file))
                        labels.append(idx)  # Assign class index

        return video_paths, labels, class_to_idx

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract frames
        frames = self._extract_frames(video_path)

        # Apply transformations
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        frames = frames.permute(1, 0, 2, 3)

        return frames, torch.tensor(label)  # Return video clip + label

    def _extract_frames(self, video_path):
        """Extracts `frames_per_clip` frames from a video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:  # Handle case where video is empty
            cap.release()
            return [torch.zeros((3, 112, 112), dtype=torch.float32)] * self.frames_per_clip

        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
        frames = []

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor
                frames.append(frame)
            else:
                break  # Stop if video ends early

        cap.release()

        if len(frames) == 0:  # If no frames were read, return blank frames
            print(f"Warning: No frames extracted from {video_path}. Returning blank frames.")
            return [torch.zeros((3, 112, 112), dtype=torch.float32)] * self.frames_per_clip

        # If not enough frames, duplicate the last one
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])

        return frames

    
    def get_labels_per_index(self):
        return self.class_to_idx

class CustomKineticsDataset_UNet(Dataset):
    def __init__(self, root_dir, frames_per_clip=16, transform=None):
        """
        Args:
            root_dir (str): Root dataset directory where each subfolder is a class.
            frames_per_clip (int): Number of frames to sample per video.
            transform (callable, optional): Transformations for the frames.
        """

        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform

        # Scan all class folders and collect video paths
        self.video_paths, self.labels, self.class_to_idx = self._scan_videos()
        print(self.class_to_idx)

    def _scan_videos(self):
        """Scans the dataset directory to get video paths and class labels."""
        class_to_idx = {}  # Maps class name to label index
        video_paths = []
        labels = []

        # Iterate through subdirectories
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                class_to_idx[class_name] = idx  # Assign index to class

                # Collect video files inside the class folder
                for video_file in os.listdir(class_path):
                    if video_file.endswith((".mp4", ".avi", ".mov")):
                        video_paths.append(os.path.join(class_path, video_file))
                        labels.append(idx)  # Assign class index

        return video_paths, labels, class_to_idx

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract frames (list of PIL Images)
        frames = self._extract_frames(video_path)

        # Apply transformations (convert each PIL to tensor, etc.)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        else:
            frames = [transforms.ToTensor()(frame) for frame in frames]

        # Stack frames -> shape: (T, C, H, W)
        frames = torch.stack(frames)

        # Rearrange -> (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)

        return frames, torch.tensor(label)

    def _extract_frames(self, video_path):
        """Extracts `frames_per_clip` frames from a video as PIL Images."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:  # Handle empty video
            cap.release()
            blank = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8))
            return [blank] * self.frames_per_clip

        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
        frames = []

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR→RGB
                frame = Image.fromarray(frame)  # Convert to PIL Image
                frames.append(frame)
            else:
                break

        cap.release()

        if len(frames) == 0:  # If no frames were read, return blank frames
            print(f"Warning: No frames extracted from {video_path}. Returning blank frames.")
            blank = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8))
            return [blank] * self.frames_per_clip

        # If not enough frames, duplicate the last one
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])

        return frames

    def get_labels_per_index(self):
        return self.class_to_idx