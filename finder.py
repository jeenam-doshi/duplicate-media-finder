import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DuplicateMediaFinder:
    def __init__(self):
        print("[Loading AI]: Initializing pre-trained ResNet-50 vision model for cross-media feature extraction...")
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(tensor)
        return embedding.squeeze().numpy()

    def get_video_embedding(self, video_path: str, sample_fps: int = 1) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        frame_interval = int(fps / sample_fps) if fps >= sample_fps else 1
        embeddings = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = self.transform(image).unsqueeze(0)
                with torch.no_grad():
                    emb = self.model(tensor).squeeze().numpy()
                    embeddings.append(emb)

            frame_count += 1
        cap.release()

        if not embeddings:
            raise ValueError("Could not extract any valid frames from the video file.")

        return np.mean(embeddings, axis=0)

    def analyze_folder_profile(self, dir_path: str) -> dict:
        """Inspects folder contents first to profile media types and recommend an optimal threshold."""
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
        video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')
        valid_extensions = image_exts + video_exts
        
        media_paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    media_paths.append(os.path.join(root, file))

        img_count = sum(1 for p in media_paths if p.lower().endswith(image_exts))
        vid_count = sum(1 for p in media_paths if p.lower().endswith(video_exts))
        total = len(media_paths)

        # Smart recommendation logic
        if vid_count > 0 and img_count > 0:
            rec_threshold = 0.82
            profile_type = "Mixed Media (Images & Videos)"
            advice = "Recommended lower threshold (0.80 - 0.85) to successfully match cross-format items or varying angles."
        elif vid_count > 1 and img_count == 0:
            rec_threshold = 0.85
            profile_type = "Video Library"
            advice = "Recommended threshold (0.83 - 0.88) to catch similar video scenes or re-encoded clips."
        else:
            rec_threshold = 0.92
            profile_type = "Static Image Collection"
            advice = "Recommended threshold (0.90 - 0.95) for spotting exact duplicates or compressed copies."

        return {
            "total": total,
            "images": img_count,
            "videos": vid_count,
            "profile_type": profile_type,
            "recommended_threshold": rec_threshold,
            "advice": advice
        }

    def scan_directory(self, dir_path: str, threshold: float = 0.92) -> list:
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
        video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v')
        valid_extensions = image_exts + video_exts
        
        media_paths = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    media_paths.append(os.path.join(root, file))

        if len(media_paths) < 2:
            return []

        embeddings = []
        valid_paths = []

        for path in media_paths:
            try:
                if path.lower().endswith(video_exts):
                    emb = self.get_video_embedding(path)
                else:
                    emb = self.get_image_embedding(path)
                
                embeddings.append(emb)
                valid_paths.append(path)
            except Exception as e:
                print(f"[Warning] Skipping '{os.path.basename(path)}': {e}")

        if len(embeddings) < 2:
            return []

        embeddings = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings)

        visited = set()
        duplicate_groups = []

        for i in range(len(valid_paths)):
            if i in visited:
                continue
            
            group = [valid_paths[i]]
            for j in range(i + 1, len(valid_paths)):
                if j not in visited and similarity_matrix[i][j] >= threshold:
                    group.append(valid_paths[j])
                    visited.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                
        return duplicate_groups