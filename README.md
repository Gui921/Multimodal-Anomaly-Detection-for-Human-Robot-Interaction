# MADRI : Multimodal Anomaly Detection for Human-Robot Interaction

![Overall Framework](images/framework.png)

## Abstract

Ensuring safety and reliability in human-robot interaction (HRI) requires the timely detection of unexpected events that could lead to system failures or unsafe behaviours. 
Anomaly detection thus plays a critical role in enabling robots to recognize and respond to deviations from normal operation during collaborative tasks. 
While reconstruction models have been actively explored in HRI, approaches that operate directly on feature vectors remain largely unexplored. 
In this work, we propose MADRI, a framework that first transforms video streams into semantically meaningful feature vectors before performing reconstruction-based anomaly detection. 
Additionally, we augment these visual feature vectors with the robot’s internal sensors' readings and a Scene Graph, enabling the model to capture both external anomalies in the visual environment and 
internal failures within the robot itself. To evaluate our approach, we collected a custom dataset consisting of a simple pick-and-place robotic task under normal and anomalous conditions. 
Experimental results demonstrate that reconstruction on vision-based feature vectors alone is effective for detecting anomalies, while incorporating other modalities further improves detection performance, 
highlighting the benefits of multimodal feature reconstruction for robust anomaly detection in human-robot collaboration.

## Methods

### Train Action Recognition for Human-Robot Interaction

![Action Recognition Confusion Matrix](images/Action%20Recognition%20CM.png)

### Train Baseline - Full Clip Reconstruction

![Baseline Reconstruction Output Example](images/Reconstruction.png)

### Define Threshold

![Threshold Definition Plot](images/video+sensor.png)

## Results:

![ROC Curves](images/Final%20ROC.png)

## Baseline vs All Modalities

| KDE for the Baseline | KDE for the MADRI (all modalities) |
| -------------------- | -------------------------------- |
| ![KDE for the Baseline](images/KDE_baseline.png) | ![KDE for the all modalities model](images/KDE_all.png) |

### Reconstruction Plots:

| Failure Type         | Baseline | Video Only | Video + Sensor | Video + SG | All Modalities |
|----------------------|----------|------------|----------------|------------|----------------|
| Drop Cup             |          |            |                |            |                |
| Robot Torque Failure |          |            |                |            |                |
| Collision            |          |            |                |            |                |
| Extra Person         |          |            |                |            |                |
| Person Disappears    |          |            |                |            |                |



## Model Weights available at: 

https://drive.google.com/drive/folders/1VdQBex_KjnO9MbtvfjZtfBq3zJHJHwMw?usp=sharing
