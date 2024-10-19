#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import tifffile
import pandas as pd
from PIL import Image
from common import get_autoencoder, get_pdn_small, get_pdn_medium
from PIL import Image

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing transformation (resize, normalize)
image_size = 256
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    """Loads and preprocesses an image."""
    image = Image.open(image_path).convert('RGB')
    image = default_transform(image)  # Preprocess image
    image = image.unsqueeze(0)  # Add batch dimension (batch_size=1)
    return image.to(device)

def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std):
    """Generates anomaly maps and combines them."""
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    
    # Compute anomaly maps
    map_st = torch.mean((teacher_output - student_output[:, :384])**2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, 384:])**2, dim=1, keepdim=True)
    map_combined = 0.5 * map_st + 0.5 * map_ae

    return map_combined, map_st, map_ae

def load_model(checkpoint_path, model_size, out_channels=384):
    """Loads the teacher, student, and autoencoder models."""
    # Load models
    if model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise ValueError("Invalid model size specified.")

    autoencoder = get_autoencoder(out_channels)

    # Load the full model object (not just the state dict)
    teacher = torch.load(os.path.join(checkpoint_path, 'teacher_final.pth'), map_location=device)
    student = torch.load(os.path.join(checkpoint_path, 'student_final.pth'), map_location=device)
    autoencoder = torch.load(os.path.join(checkpoint_path, 'autoencoder_final.pth'), map_location=device)

    # Move models to device
    teacher = teacher.to(device).eval()
    student = student.to(device).eval()
    autoencoder = autoencoder.to(device).eval()

    return teacher, student, autoencoder

def teacher_normalization(teacher, train_loader):
    """Compute mean and std for teacher's output (use training loader if available)."""
    mean_outputs = []
    for image in train_loader:
        image = image.to(device)
        teacher_output = teacher(image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    
    teacher_mean = torch.mean(torch.stack(mean_outputs), dim=0).to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    mean_distances = []
    for image in train_loader:
        image = image.to(device)
        teacher_output = teacher(image)
        distance = (teacher_output - teacher_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    
    teacher_std = torch.sqrt(torch.mean(torch.stack(mean_distances), dim=0)).to(device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    return teacher_mean, teacher_std

def run_inference(image_dir, model_checkpoint, output_dir, model_size='small', map_format='tiff'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    
    # Load models
    teacher, student, autoencoder = load_model(model_checkpoint, model_size)

    # Assume teacher normalization is precomputed
    # In practice, you may want to compute it using teacher_normalization and a training set
    teacher_mean = torch.zeros(1, 384, 1, 1).to(device)  # Placeholder: Use precomputed values if available
    teacher_std = torch.ones(1, 384, 1, 1).to(device)  # Placeholder: Use precomputed values if available
    
    # Prepare to store results
    results = []

    # Iterate over images in the directory
    for image_name in tqdm(os.listdir(image_dir), desc="Processing images"):
        image_path = os.path.join(image_dir, image_name)
        if not image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            continue  # Skip non-image files

        # Load and preprocess the image
        image = load_image(image_path)

        # Run the model and get anomaly maps
        map_combined, _, _ = predict(image, teacher, student, autoencoder, teacher_mean, teacher_std)
        
        # Convert anomaly map to numpy and scale to [0, 255]
        map_combined = map_combined[0, 0].detach().cpu().numpy()  # Detach the tensor before converting to NumPy
        map_combined = map_combined * 255
        map_combined = map_combined.astype(np.int16)  # Convert to uint8 for JPG format

        if map_format == 'tiff':
            # Save the anomaly map as a TIFF file
            anomaly_map_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_anomaly_map.tiff")
            tifffile.imwrite(anomaly_map_path, map_combined) # Save as TIFF
        elif map_format == 'jpg':
            # Convert input image to a PIL Image
            original_image = Image.open(image_path).convert('RGB')
            # original_image = original_image.resize((map_combined.shape[1], map_combined.shape[0]))  # Resize to match anomaly map size
            
            # Convert anomaly map to PIL image
            anomaly_map_image = Image.fromarray(map_combined)
            anomaly_map_image = anomaly_map_image.resize((original_image.width, original_image.height))  # Resize to match original image size
            
            # Combine the original image and the anomaly map side by side
            combined_image_width = original_image.width + anomaly_map_image.width
            combined_image_height = original_image.height
            combined_image = Image.new('RGB', (combined_image_width, combined_image_height))
            combined_image.paste(original_image, (0, 0))
            combined_image.paste(anomaly_map_image.convert('RGB'), (original_image.width, 0))

            # Save the combined image as JPG
            combined_image_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_combined.jpg")
            combined_image.save(combined_image_path)  # Save as JPG
        else:
            raise ValueError("Invalid map format specified. Use 'tiff' or 'jpg'.")

        # Calculate anomaly score (maximum value of the combined map)
        anomaly_score = np.max(map_combined)

        # Append result to the list
        results.append({'image': image_name, 'anomaly_score': anomaly_score})

    # Save anomaly scores to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "anomaly_scores.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(f"Anomaly maps and scores saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection inference on a directory of images.")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to directory containing images.")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save anomaly maps and scores.")
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium'], help="Model size (small or medium).")
    parser.add_argument('--map_format', type=str, default='tiff', choices=['tiff', 'jpg'], help="Anomaly map format (tiff or jpg).")
    
    args = parser.parse_args()
    run_inference(args.image_dir, args.model_checkpoint, args.output_dir, args.model_size, args.map_format)