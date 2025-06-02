import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path, resize=None, grayscale=True):
    """Load and preprocess an image"""
    image = Image.open(image_path)
    
    if grayscale:
        image = image.convert('L')
    
    if resize:
        image = image.resize((resize[1], resize[0]))  # PIL uses (width, height)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]) if grayscale else 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)

def rbd(x):
    """Remove batch dimension"""
    return {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

class DroneMapDataset(Dataset):
    def __init__(self, train_image_paths, train_pos, image_map, target_height, target_width):
        self.train_image_paths = train_image_paths
        self.train_pos = train_pos
        self.image_map = image_map
        self.target_height = target_height
        self.target_width = target_width
    
    def __len__(self):
        return len(self.train_image_paths)
    
    def __getitem__(self, idx):
        img_path = self.train_image_paths[idx]
        img_id = int(img_path.name.split('.')[0])
        
        # Load and resize drone image
        drone_image = load_image(img_path, resize=(self.target_height, self.target_width))
        
        # Get ground truth position
        gt_row = self.train_pos[self.train_pos['id'] == img_id].iloc[0]
        gt_pos = torch.tensor([gt_row['x_pixel'], gt_row['y_pixel']], dtype=torch.float32)
        
        return drone_image, self.image_map, gt_pos, img_id

def pose_prediction_loss(drone_img, map_img, gt_pos, extractor, matcher):
    """Compute pose prediction and euclidean loss"""
    # Extract features
    feats0 = extractor.extract(drone_img.unsqueeze(0))
    feats1 = extractor.extract(map_img.unsqueeze(0))
    
    # Match features
    matches01 = matcher({"image0": feats0, "image1": feats1})
    
    # Remove batch dimension
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    
    if len(matches) == 0:
        # No matches - return high loss and map center as prediction
        pred_pos = torch.tensor([2500.0, 1250.0], device=gt_pos.device)
        loss = torch.norm(pred_pos - gt_pos.to(pred_pos.device))
        return loss, pred_pos
    
    # Get matched keypoints and compute centroid
    m_kpts1 = kpts1[matches[..., 1]]
    pred_pos = m_kpts1.mean(dim=0)  # [x, y]
    
    # Euclidean loss
    loss = torch.norm(pred_pos - gt_pos.to(pred_pos.device))
    
    return loss, pred_pos

def tile_ranking_loss(drone_img, tile_results, gt_pos, margin=100):
    """
    Ranking loss: encourage tiles closer to GT to have more matches
    """
    losses = []
    
    for i, tile_result in enumerate(tile_results):
        tile_pos = tile_result['tile_pos']
        num_matches = tile_result['num_matches']
        
        # Calculate distance from tile center to ground truth
        tile_center_x = (tile_pos['left'] + tile_pos['right']) / 2
        tile_center_y = (tile_pos['top'] + tile_pos['bottom']) / 2
        tile_center = torch.tensor([tile_center_x, tile_center_y])
        
        dist_to_gt = torch.norm(tile_center - gt_pos)
        
        # Ranking loss: closer tiles should have more matches
        for j, other_tile in enumerate(tile_results[i+1:], i+1):
            other_pos = other_tile['tile_pos']
            other_center_x = (other_pos['left'] + other_pos['right']) / 2
            other_center_y = (other_pos['top'] + other_pos['bottom']) / 2
            other_center = torch.tensor([other_center_x, other_center_y])
            other_dist = torch.norm(other_center - gt_pos)
            
            # If tile i is closer to GT than tile j, it should have more matches
            if dist_to_gt < other_dist:
                # Margin ranking loss
                target_diff = num_matches - other_tile['num_matches']
                loss = torch.clamp(margin - target_diff, min=0)
                losses.append(loss)
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)

def weighted_pose_loss(matches01, kpts1, gt_pos):
    """
    Use LightGlue's confidence scores to weight the pose prediction
    """
    matches = matches01["matches"]
    scores = matches01["matching_scores0"]  # Confidence scores
    
    if len(matches) == 0:
        return torch.tensor(1000.0), torch.tensor([2500.0, 1250.0])
    
    # Get matched keypoints and their confidence scores
    m_kpts1 = kpts1[matches[..., 1]]
    m_scores = scores[matches[..., 0]]
    
    # Weighted centroid based on confidence
    weights = torch.softmax(m_scores, dim=0)
    pred_pos = (m_kpts1 * weights.unsqueeze(1)).sum(dim=0)
    
    # Euclidean loss
    loss = torch.norm(pred_pos - gt_pos.to(pred_pos.device))
    
    return loss, pred_pos

def finetune_lightglue(extractor, matcher, train_image_paths, train_pos, image_map, 
                      target_height, target_width, device, num_epochs=10, learning_rate=1e-5):
    """Main finetuning function"""
    # Create dataset and dataloader
    dataset = DroneMapDataset(train_image_paths[:100], train_pos, image_map, target_height, target_width)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Set models to training mode
    matcher.train()
    extractor.eval()  # Keep extractor frozen
    
    # Optimizer - only optimize matcher parameters
    optimizer = optim.Adam(matcher.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_error = 0
        
        for batch_idx, (drone_img, map_img, gt_pos, img_id) in enumerate(dataloader):
            drone_img = drone_img.squeeze(0).to(device)
            map_img = map_img.squeeze(0).to(device)
            gt_pos = gt_pos.squeeze(0)
            
            optimizer.zero_grad()
            
            # Forward pass and loss computation
            loss, pred_pos = pose_prediction_loss(drone_img, map_img, gt_pos, extractor, matcher)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(matcher.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            error = torch.norm(pred_pos.cpu() - gt_pos).item()
            total_error += error
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.1f}, Error={error:.1f}px")
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_error = total_error / len(dataloader)
        print(f"Epoch {epoch}: Avg Loss={avg_loss:.1f}, Avg Error={avg_error:.1f}px, LR={scheduler.get_last_lr()[0]:.2e}")
    
    # Set back to eval mode
    matcher.eval()
    print("Finetuning completed!")
    
    return matcher

def evaluate_model(extractor, matcher, test_image_paths, test_pos, image_map, 
                  target_height, target_width, device):
    """Evaluate the finetuned model"""
    matcher.eval()
    extractor.eval()
    
    total_error = 0
    predictions = []
    
    with torch.no_grad():
        for img_path in test_image_paths:
            img_id = int(img_path.name.split('.')[0])
            
            # Load drone image
            drone_image = load_image(img_path, resize=(target_height, target_width)).to(device)
            
            # Get ground truth position
            gt_row = test_pos[test_pos['id'] == img_id].iloc[0]
            gt_pos = torch.tensor([gt_row['x_pixel'], gt_row['y_pixel']], dtype=torch.float32)
            
            # Predict position
            _, pred_pos = pose_prediction_loss(drone_image, image_map.to(device), gt_pos, extractor, matcher)
            
            error = torch.norm(pred_pos.cpu() - gt_pos).item()
            total_error += error
            
            predictions.append({
                'img_id': img_id,
                'gt_x': gt_pos[0].item(),
                'gt_y': gt_pos[1].item(),
                'pred_x': pred_pos[0].item(),
                'pred_y': pred_pos[1].item(),
                'error': error
            })
    
    avg_error = total_error / len(test_image_paths)
    print(f"Average test error: {avg_error:.1f} pixels")
    
    return predictions, avg_error

def save_model(matcher, save_path):
    """Save the finetuned model"""
    torch.save(matcher.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(matcher, save_path):
    """Load a finetuned model"""
    matcher.load_state_dict(torch.load(save_path))
    print(f"Model loaded from {save_path}")
    return matcher