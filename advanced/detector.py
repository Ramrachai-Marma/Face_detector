from typing import List, Tuple

import cv2

from .config import CASCADE_PATH, DETECT_MIN_NEIGHBORS, DETECT_MIN_SIZE, DETECT_SCALE_FACTOR


class HaarFaceDetector:
    def __init__(self, cascade_path: str = CASCADE_PATH):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(
                f"Failed to load cascade at '{cascade_path}'. Ensure the file exists."
            )

    def detect(self, image) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=DETECT_SCALE_FACTOR,
            minNeighbors=DETECT_MIN_NEIGHBORS,
            minSize=DETECT_MIN_SIZE,
        )
        return list(faces)


