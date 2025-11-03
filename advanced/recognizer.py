import os
from typing import Dict, Tuple

import cv2

from .config import LBPH_GRID_X, LBPH_GRID_Y, LBPH_NEIGHBORS, LBPH_RADIUS, MODELS_DIR
from .dataset import load_labels, save_labels
from .utils import ensure_dir


class LBPHRecognizer:
    def __init__(self):
        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError(
                "LBPH recognizer requires opencv-contrib-python. Install it to enable recognition."
            )
        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=LBPH_RADIUS,
            neighbors=LBPH_NEIGHBORS,
            grid_x=LBPH_GRID_X,
            grid_y=LBPH_GRID_Y,
        )
        ensure_dir(MODELS_DIR)
        self.model_path = os.path.join(MODELS_DIR, "lbph.yml")
        self.labels_path = os.path.join(MODELS_DIR, "labels.json")

    def train(self, features, targets, labels: Dict[int, str]) -> None:
        if not features or not targets:
            raise RuntimeError("Training data is empty. Enroll images first.")
        # Convert targets to numpy array
        import numpy as np
        targets_array = np.array(targets, dtype=np.int32)
        # LBPH recognizer only needs features and targets
        self.model.train(features, targets_array)
        self.model.write(self.model_path)
        save_labels(self.labels_path, labels)

    def load(self) -> Dict[int, str]:
        if not os.path.isfile(self.model_path):
            raise RuntimeError("Model file not found. Train first.")
        self.model.read(self.model_path)
        return load_labels(self.labels_path)

    def predict(self, face_gray) -> Tuple[int, float]:
        return self.model.predict(face_gray)


