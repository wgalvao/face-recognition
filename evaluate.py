import os
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from models import (
    sphere20,
    sphere36,
    sphere64,
    MobileNetV1,
    MobileNetV2,
    mobilenet_v3_small,
    mobilenet_v3_large,
)


def extract_deep_features(model, image, device):
    """
    Extracts deep features for an image using the model, including both the original and flipped versions.

    Args:
        model (torch.nn.Module): The pre-trained deep learning model used for feature extraction.
        image (PIL.Image): The input image to extract features from.
        device (torch.device): The device (CPU or GPU) on which the computation will be performed.

    Returns:
        torch.Tensor: Combined feature vector of original and flipped images.
    """

    # Define transforms
    original_transform = transforms.Compose([
	transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    flipped_transform = transforms.Compose([
	transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=1.0),  # Always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Apply transforms
    original_image_tensor = original_transform(image).unsqueeze(0).to(device)
    flipped_image_tensor = flipped_transform(image).unsqueeze(0).to(device)

    # Extract features
    original_features = model(original_image_tensor)
    flipped_features = model(flipped_image_tensor)

    # Combine and return features
    combined_features = torch.cat([original_features, flipped_features], dim=1).squeeze()
    return combined_features


def k_fold_split(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    fold_size = n // n_folds

    for idx in range(n_folds):
        test = base[idx * fold_size:(idx + 1) * fold_size]
        train = base[:idx * fold_size] + base[(idx + 1) * fold_size:]
        folds.append([train, test])

    return folds


def eval_accuracy(predictions, threshold):
    y_true = []
    y_pred = []

    for _, _, distance, gt in predictions:
        y_true.append(int(gt))
        pred = 1 if float(distance) > threshold else 0
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    return accuracy


def find_best_threshold(predictions, thresholds):
    best_accuracy = 0
    best_threshold = 0

    for threshold in thresholds:
        accuracy = eval_accuracy(predictions, threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def eval(model, model_path=None, device=None, val_dataset='lfw', val_root='data/lfw/val'):
    """
    Evaluate the model on validation dataset (LFW or CelebA).
    
    Args:
        model: The model to evaluate
        model_path: Path to model weights (optional)
        device: Device to run evaluation on
        val_dataset: Dataset to use for validation ('lfw' or 'celeba')
        val_root: Root directory of validation data
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device).eval()

    root = val_root
    
    # Select annotation file and image path logic based on dataset
    if val_dataset == 'lfw':
        ann_file = os.path.join(root, 'lfw_ann.txt')
        try:
            with open(ann_file) as f:
                pair_lines = f.readlines()[1:]
        except FileNotFoundError:
            print(f"ERROR: Annotation file 'lfw_ann.txt' not found in '{root}'. Check the path.")
            return 0.0, np.array([])
    elif val_dataset == 'celeba':
        ann_file = os.path.join(root, 'celeba_pairs.txt')
        try:
            with open(ann_file) as f:
                pair_lines = f.readlines()[1:]  # Skip header if exists
        except FileNotFoundError:
            print(f"ERROR: Annotation file 'celeba_pairs.txt' not found in '{root}'. Check the path.")
            return 0.0, np.array([])
    else:
        raise ValueError(f"Unsupported validation dataset: {val_dataset}. Choose 'lfw' or 'celeba'.")

    predicts = []
    with torch.no_grad():
        for line in pair_lines:
            parts = line.strip().split()

            if val_dataset == 'lfw':
                if len(parts) == 3:
                    person_name, img_num1, img_num2 = parts[0], parts[1], parts[2]
                    
                    # Format filename as: "Person_Name_0001.jpg"
                    filename1 = f'{person_name}_{int(img_num1):04d}.jpg'
                    filename2 = f'{person_name}_{int(img_num2):04d}.jpg'
                    
                    # Build full path
                    path1 = os.path.join(root, person_name, filename1)
                    path2 = os.path.join(root, person_name, filename2)
                    is_same = '1'
                else:
                    # Skip lines that don't have 3 columns
                    continue
                    
            elif val_dataset == 'celeba':
                if len(parts) == 2:
                    # Format: img1.jpg img2.jpg (both same identity - positive pairs only)
                    filename1, filename2 = parts[0], parts[1]
                    
                    # CelebA images are in img_align_celeba/img_align_celeba folder
                    path1 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', filename1)
                    path2 = os.path.join(root, 'img_align_celeba', 'img_align_celeba', filename2)
                    is_same = '1'
                else:
                    continue

            try:
                img1 = Image.open(path1).convert('RGB')
                img2 = Image.open(path2).convert('RGB')
            except FileNotFoundError:
                print(f"Warning: Image not found, skipping pair: {path1} or {path2}")
                continue

            f1 = extract_deep_features(model, img1, device)
            f2 = extract_deep_features(model, img2, device)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append([path1, path2, distance.item(), is_same])
    
    if len(predicts) == 0:
        print("Warning: No valid pairs were processed in the evaluation.")
        return 0.0, np.array([])

    predicts = np.array(predicts)
    similarities = predicts[:, 2].astype(float)
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)

    dataset_name = val_dataset.upper()
    print(f'{dataset_name} - Simplified Evaluation (Positive Pairs Only):')
    print(f'Mean Similarity: {mean_similarity:.4f} | Standard Deviation: {std_similarity:.4f}')

    accuracy_proxy = mean_similarity 
    
    return accuracy_proxy, predicts

if __name__ == '__main__':
    _, result = eval(sphere20(512).to('cuda'), model_path='weights/sphere20_mcp.pth')
    _, result = eval(sphere36(512).to('cuda'), model_path='weights/sphere36_mcp.pth')
    _, result = eval(MobileNetV1(512).to('cuda'), model_path='weights/mobilenetv1_mcp.pth')
    _, result = eval(MobileNetV2(512).to('cuda'), model_path='weights/mobilenetv2_mcp.pth')
    _, result = eval(mobilenet_v3_small(512).to('cuda'), model_path='weights/mobilenetv3_small_mcp.pth')
    _, result = eval(mobilenet_v3_large(512).to('cuda'), model_path='weights/mobilenetv3_large_mcp.pth')