import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import multiprocessing
from torchvision import transforms
import cv2
from PIL import Image
from optuna.samplers import TPESampler

import os

from FRM import AutoEncoder, TransformerAutoencoder
from feature_dataset import CustomFeatureDataset
from Swin3d import Swin3D_fine_tune, Swin3D

def train_model(frm_model, optimizer, criterion, num_epochs = 100):

    dataset = CustomFeatureDataset('annotated_tecnico_dataset_3.pkl')
    train_size = int(0.85 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=multiprocessing.cpu_count() // 2,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() // 2,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() // 2,
        pin_memory=True,
        persistent_workers=True
    )

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        # -------------- Train --------------------
        frm_model.train()
        train_loss = 0.0
        
        for feature_vector, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            
            feature_vector = feature_vector.squeeze(dim=1)

            feature_vector = feature_vector.cuda()
            optimizer.zero_grad()

            output = frm_model(feature_vector)
            loss = criterion(feature_vector,output)

            loss.backward()
            optimizer.step()

            
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ------------------- Val --------------------------------
        frm_model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for feature_vector, _, _ in val_loader:
                
                feature_vector = feature_vector.squeeze(dim=1)
                feature_vector = feature_vector.cuda()

                output = frm_model(feature_vector)
                loss = criterion(feature_vector, output)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

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

def preprocess_frames_2(frames):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the frames
        transforms.CenterCrop(224),     # Crop the frames to 112x112
        transforms.ToTensor(),          # Convert frames to tensor (values scaled to [0, 1])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        transforms.ConvertImageDtype(torch.float32)  # Convert to float32
    ])

    processed_frames = []
    for frame in frames:
        pil_frame = Image.fromarray(frame)
        
        processed_frame = transform(pil_frame)
        
        # Clip values to [0, 1] after normalization (if necessary)
        processed_frame = torch.clamp(processed_frame, 0, 1)

        processed_frames.append(processed_frame)

    # Stack frames → Shape: (T, C, H, W) → (16, 3, 112, 112)
    video_tensor = torch.stack(processed_frames)

    # Rearrange dimensions to (B, C, T, H, W) → (1, 3, 16, 112, 112)
    return video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

def anomaly_inference(folder_path, model, frm_model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model = model.to(device)

    #original_features = []
    #reconstructed_features = []
    labels = [] 
    reconstruction_errors = []

    for video in range(len(os.listdir(folder_path))):

        video_path = os.path.join(folder_path,f'clip_{video + 1}.mp4')

        frames = load_video_frames(video_path, num_frames=32)
        if len(frames) > 0:

            inputs = preprocess_frames_2(frames).to(device)

            with torch.no_grad():
                output = model(inputs)  # Original R2+1D features
                reconstruction_output = frm_model(output)  # Reconstructed features
                reconstruction_errors.append(criterion(reconstruction_output, output).cpu().numpy())
            #original_features.append(output.cpu().numpy().squeeze())
            #reconstructed_features.append(reconstruction_output.cpu().numpy().squeeze())
            
            #labels.append(0 if (video + 1) < 6 else 1)

    reconstruction_errors = np.array(reconstruction_errors)

    percent = 0.25 
    num_normals = int(len(reconstruction_errors) * percent)

    lowest_errors = np.sort(reconstruction_errors)[:num_normals]

    threshold = np.mean(lowest_errors) + 1.5 * np.std(reconstruction_errors)

    for error in reconstruction_errors:
        if error > threshold:
            labels.append(1)
        else:
            labels.append(0)

    return reconstruction_errors, labels, threshold

def objective(trial):
    # Hyperparameters to tune
    model_dim = trial.suggest_int('model_dim', 4, 256,step=4)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    loss_name = trial.suggest_categorical('loss_fn', ['l1', 'mse'])
    num_epochs = trial.suggest_int('num_epochs',1,100,step=1)
    #model_name = trial.suggest_categorical('model', ['Transformer', 'Linear'])
    model_name = 'Linear'

    # Loss function
    criterion = nn.L1Loss() if loss_name == 'l1' else nn.MSELoss()

    # Define model
    if model_name == 'Transformer':
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 6)

        frm_model = TransformerAutoencoder(
            input_dim=768,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).cuda()

    else:
        num_encoder_layers = trial.suggest_int('num_encoder_layers',1,6)
        num_decoder_layers = trial.suggest_int('num_decoder_layers',1,6)
        dropout_prob = trial.suggest_float('dropout_prob',0,1,step=0.02)

        frm_model = AutoEncoder(num_encoder_layers = num_encoder_layers,
                                num_decoder_layers = num_decoder_layers,
                                bottleneck_dim = model_dim,
                                dropout_prob = dropout_prob).cuda()

    #model = Swin3D_fine_tune(n_classes=6)
    #torch.load('checkpoints/Swin3d_fine_tune/checkpoint_1.pt',weights_only=False)
    model = Swin3D(n_classes=6)

    checkpoint = torch.load('tecnico_6classes_v2_frozen.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.classifier = nn.Identity()

    optimizer = optim.AdamW(frm_model.parameters(), lr=lr)


    print("------------------------------------------- TRAINING MODEL !!! --------------------------------------------")
    train_model(frm_model, optimizer, criterion, num_epochs)
    print("------------------------------------------- EVALUATION !!! --------------------------------------------")

    all_labels = []

    labels_per_video = [[0,0,0,0,0,0,1,1,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,0,1,1,1,1,1,1,1],
                    #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    #[0,0,0,0,0,0,0,0,0,0,0,0,0],
                    #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    #[1,1,1,1,1,1,1,1,1,1],
                    [0,0,0,0,0,1,1,1,1,0,0,0,0,0]
                    ]
    
    real_labels = [
    0,0,0,0,0,0,1,1,0,0, #Drop Cup
    0,0,0,0,1,1,0,0,0,0, #Fail to Grasp
    0,0,0,0,0,1,1,1,1,1,1,1, #Goes Up
    #0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, #Good
    #0,0,0,0,0,0,0,0,0,0,0,0,0, #Good 2
    #0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, #Good 3
    #1,1,1,1,1,1,1,1,1,1, #No Cup
    0,0,0,0,0,1,1,1,1,0,0,0,0,0, #Wrong Side
    ]

    idx = 0

    for video in sorted(os.listdir('Fails v2 - clips')):
        video_path = os.path.join('Fails v2 - clips', video)
        _, pred_labels, _ = anomaly_inference(video_path, model, frm_model, criterion)

        #real = labels_per_video[idx]
        all_labels.extend(pred_labels)
        #all_preds.extend(pred_labels)
        idx += 1

    #f1 = f1_score(real_labels, all_labels)
    recall = recall_score(real_labels,all_labels)
    #acc = accuracy_score(real_labels, all_labels)
    return recall

# -------- Run Optuna ------------

study = optuna.create_study(
    sampler=TPESampler(),
    direction='maximize',
    study_name="FRM_tuning",
    storage="sqlite:///optuna_study_8.db",
    load_if_exists=True
)

study.set_metric_names(['Recall'])
study.optimize(objective, n_trials=100)

print('Best trial:')
print(study.best_trials)
#trial = study.best_trials
#print(f'  Value: {trial.value}')
#print(f'  Params: {trial.params}')