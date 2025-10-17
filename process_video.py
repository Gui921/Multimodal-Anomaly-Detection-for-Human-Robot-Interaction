import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

def preprocess_frames(frames):
    transform = transforms.Compose([
        transforms.Resize((128, 171)),  # Resize the frames
        transforms.CenterCrop(112),     # Crop the frames to 112x112
        transforms.ToTensor(),          # Convert frames to tensor (values scaled to [0, 1])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        transforms.ConvertImageDtype(torch.float32)  # Convert to float32
    ])

    processed_frames = []
    for frame in frames:
        pil_frame = Image.fromarray(frame)
        
        # Apply the transformation (which includes scaling to [0, 1], normalizing, and converting to float32)
        processed_frame = transform(pil_frame)
        
        # Clip values to [0, 1] after normalization (if necessary)
        processed_frame = torch.clamp(processed_frame, 0, 1)

        processed_frames.append(processed_frame)

    # Stack frames → Shape: (T, C, H, W) → (16, 3, 112, 112)
    video_tensor = torch.stack(processed_frames)

    # Rearrange dimensions to (B, C, T, H, W) → (1, 3, 16, 112, 112)
    return video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

def load_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        frame_idx = int(i * (frame_count / num_frames))  # Sample evenly
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)

    cap.release()
    return frames  # List of (H, W, C) numpy arrays


# CHANGE THIS SO THAT IT SAVES THE IMAGE INSTEAD OF SHOWING
def show_frames_from_list(frame_list):

    num_frames = len(frame_list)
    
    # Calculate number of rows and columns for subplots (assuming square layout)
    rows = int(num_frames**0.5)
    cols = (num_frames + rows - 1) // rows  # Round up to fill the grid
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    
    axes = axes.flatten()

    for i in range(num_frames):
        frame = frame_list[i]  # Shape: (480, 640, 3)
        
        axes[i].imshow(frame)
        axes[i].axis("off")  # Hide axes for better visualization
        axes[i].set_title(f"Frame {i+1}")
    
    for i in range(num_frames, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

# THE SAME HERE
def show_frames(input_frames):
    """
    Displays the video frames from the input tensor.

    Args:
    - input_frames: torch.Tensor of shape [1, 3, 16, 112, 112] (video frames)
    """
    # Move tensor to CPU and convert to numpy
    frames = input_frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # Shape: [16, 112, 112, 3]

    # Normalize frames to [0, 1] if they are in range [0, 255]
    if frames.max() > 1:
        frames = frames / 255.0

    # Create a figure
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()  # Ensure correct subplot indexing

    for i in range(16):
        frame = frames[i]  # Shape: [112, 112, 3]
        
        # Display result
        #frame = frame[..., ::-1]
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()