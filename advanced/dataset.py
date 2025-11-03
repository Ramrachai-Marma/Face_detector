import json
import os
from typing import Dict, List, Tuple

import cv2

from .config import DATA_DIR, CASCADE_PATH
from .utils import list_images


cascade = cv2.CascadeClassifier(CASCADE_PATH)


def person_dir(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def enroll_images(name: str, image_paths: List[str]) -> List[str]:
    os.makedirs(person_dir(name), exist_ok=True)
    saved: List[str] = []
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        crop = gray[y:y+h, x:x+w]
        crop = cv2.resize(crop, (100, 100))
        base = f"{name}_{idx:04d}.jpg"
        out = os.path.join(person_dir(name), base)
        cv2.imwrite(out, crop)
        saved.append(out)
    return saved


def load_dataset() -> Tuple[List, List[int], Dict[int, str]]:
    labels: Dict[int, str] = {}
    features: List = []
    targets: List[int] = []
    current_label = 0
    if not os.path.isdir(DATA_DIR):
        return features, targets, labels
    for name in sorted(os.listdir(DATA_DIR)):
        pdir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(pdir):
            continue
        label = current_label
        labels[label] = name
        current_label += 1
        for img_path in list_images(pdir):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (100, 100):
                faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
                if len(faces) == 0:
                    continue
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                img = cv2.resize(img[y:y+h, x:x+w], (100, 100))
            features.append(img)
            targets.append(label)
    return features, targets, labels


def save_labels(path: str, labels: Dict[int, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def load_labels(path: str) -> Dict[int, str]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


