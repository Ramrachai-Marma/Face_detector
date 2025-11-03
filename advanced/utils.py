import os
from typing import Iterable, List, Tuple

import cv2


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def draw_boxes(image, boxes: Iterable[Tuple[int, int, int, int]], color=(0, 255, 0)):
    output = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    return output


def save_image(path: str, image) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, image)


def list_images(root: str, pattern: str = ".jpg|.jpeg|.png") -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            lower = name.lower()
            if any(lower.endswith(ext) for ext in pattern.split("|")):
                files.append(os.path.join(dirpath, name))
    return files


