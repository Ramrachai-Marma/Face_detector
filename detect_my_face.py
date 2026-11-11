#!/usr/bin/env python3
"""
Simple face detection: First detect all faces, then identify which one is you
"""

import os
from rich import print

import cv2

from advanced.detector import HaarFaceDetector
from advanced.recognizer import LBPHRecognizer
from advanced.utils import ensure_dir, save_image
from advanced.dataset import enroll_images, load_dataset
from advanced.identity import find_label_id
from advanced.config import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_DIR

DEFAULT_PERSON = "myself"


def _ensure_trained(person_name: str = DEFAULT_PERSON) -> bool:
    """
    Ensure a trained model exists, training it from the enrolled dataset if needed.
    """
    try:
        recognizer = LBPHRecognizer()
        # Preserve an existing model when it does not contain the requested identity.
        if os.path.exists(os.path.join(MODELS_DIR, "lbph.yml")):
            labels = recognizer.load()
            have = {name.lower().strip() for name in labels.values()}
            if person_name.lower().strip() in have:
                return True
            available = ", ".join(sorted(labels.values())) or "<none>"
            print(
                f"[red]Model does not contain '{person_name}'. "
                f"Available labels: {available}. The existing model was not changed.[/red]"
            )
            return False

        data_root = DATA_DIR
        # Check the data directory and project root for a single source image.
        candidates = [
            os.path.join(data_root, f"{person_name}.jpg"),
            os.path.join(data_root, f"{person_name}.jpeg"),
            os.path.join(data_root, f"{person_name}.png"),
            os.path.join(PROJECT_DIR, f"{person_name}.jpg"),
            os.path.join(PROJECT_DIR, f"{person_name}.jpeg"),
            os.path.join(PROJECT_DIR, f"{person_name}.png"),
            os.path.join(PROJECT_DIR, "Mark.jpeg"),
            os.path.join(PROJECT_DIR, "Mark.jpg"),
            os.path.join(PROJECT_DIR, "Mark.png"),
        ]
        single_image = next((p for p in candidates if os.path.isfile(p)), None)
        if single_image is not None:
            saved = enroll_images(person_name, [single_image])
            # Retain a resized fallback image when enrollment detects no face.
            if not saved:
                try:
                    img = cv2.imread(single_image)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (100, 100))
                        out_dir = os.path.join(data_root, person_name)
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"{person_name}_0000.jpg")
                        save_image(out_path, resized)
                except Exception as e:
                    raise OSError(f"Could not create fallback training image: {e}") from e

        # Augment small enrollment sets with mirrored, brightness-adjusted, and blurred samples.
        try:
            person_dir = os.path.join(data_root, person_name)
            existing = []
            if os.path.isdir(person_dir):
                for root, _, files in os.walk(person_dir):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            existing.append(os.path.join(root, f))
            if len(existing) <= 2 and existing:
                base = cv2.imread(existing[0], cv2.IMREAD_GRAYSCALE)
                if base is not None:
                    variants = []
                    variants.append(cv2.flip(base, 1))
                    brighter = cv2.convertScaleAbs(base, alpha=1.0, beta=20)
                    darker = cv2.convertScaleAbs(base, alpha=1.0, beta=-20)
                    variants.extend([brighter, darker])
                    blurred = cv2.GaussianBlur(base, (3, 3), 0)
                    variants.append(blurred)
                    os.makedirs(person_dir, exist_ok=True)
                    for idx, v in enumerate(variants):
                        out_path = os.path.join(person_dir, f"{person_name}_aug_{idx:02d}.jpg")
                        save_image(out_path, v)
        except Exception as e:
            raise OSError(f"Could not create augmented training image: {e}") from e

        features, targets, labels = load_dataset()
        if not features:
            print("[red]No training data found in 'data/people'. Add images and retry.[/red]")
            return False
        available = {name.lower().strip() for name in labels.values()}
        if person_name.lower().strip() not in available:
            names = ", ".join(sorted(labels.values())) or "<none>"
            print(
                f"[red]No training examples found for '{person_name}'. "
                f"Available labels: {names}. No model was saved.[/red]"
            )
            return False
        recognizer.train(features, targets, labels)
        print("[green]Model trained and saved.[/green]\n")
        return True
    except Exception as e:
        print(f"[red]Training failed: {e}[/red]")
        return False


def detect_my_face(image_path: str, person_name: str = DEFAULT_PERSON):
    """
    Detect all faces first, then identify which one is you
    """
    try:
        # Load the existing model, training from enrolled data when necessary.
        if not os.path.exists(os.path.join(MODELS_DIR, "lbph.yml")):
            if not _ensure_trained(person_name):
                return False
        recognizer = LBPHRecognizer()
        labels = recognizer.load()
        
        your_label_id = find_label_id(labels, person_name)
        if your_label_id is None:
            available = ", ".join(sorted(labels.values())) or "<none>"
            print(
                f"[red]Error: Could not find '{person_name}' in the trained model. "
                f"Available: {available}[/red]"
            )
            return False
        
        # Load the source image.
        img = cv2.imread(image_path)
        if img is None:
            print(f"[red]Error: Could not load image '{image_path}'[/red]")
            return False
        
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Detect faces before evaluating identity matches.
        detector = HaarFaceDetector()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            gray = cv2.equalizeHist(gray)
        faces = detector.cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(24, 24)
        )
        # Retry detection on an enlarged image when the initial pass finds no faces.
        if len(faces) == 0:
            try:
                up = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                faces_up = detector.cascade.detectMultiScale(
                    up,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                if len(faces_up) > 0:
                    # Convert the enlarged-image coordinates to the source-image scale.
                    mapped = []
                    for (x, y, w, h) in faces_up:
                        mapped.append((int(x/1.5), int(y/1.5), int(w/1.5), int(h/1.5)))
                    faces = mapped
            except Exception:
                pass
        
        print(f"Found {len(faces)} face(s) in the image")
        
        if len(faces) == 0:
            print("[yellow]No faces detected![/yellow]")
            return False
        
        # Compare each detected face with the requested identity.
        output_img = img.copy()
        my_face_found = False
        THRESHOLD = 135.0
        best_conf = float('inf')
        best_box = None
        
        def crop_and_score(x, y, w, h):
            """Try multiple slight variations of the crop and return best (min) confidence."""
            H, W = gray.shape[:2]
            tweaks = [
                (0.0, 0.0, 1.00),
                (0.0, 0.0, 1.10),
                (0.0, 0.0, 0.90),
                (-0.05, 0.0, 1.00),
                (0.05, 0.0, 1.00),
                (0.0, -0.05, 1.00),
                (0.0, 0.05, 1.00),
            ]
            local_best = float('inf')
            local_box = (x, y, w, h)
            for dx_ratio, dy_ratio, scale in tweaks:
                cx = x + w / 2.0
                cy = y + h / 2.0
                tw = max(10, int(w * scale))
                th = max(10, int(h * scale))
                tx = int(cx + dx_ratio * w - tw / 2.0)
                ty = int(cy + dy_ratio * h - th / 2.0)
                tx = max(0, min(tx, W - tw))
                ty = max(0, min(ty, H - th))
                roi = gray[ty:ty+th, tx:tx+tw]
                try:
                    face_resized = cv2.resize(roi, (100, 100))
                    lid, conf = recognizer.predict(face_resized)
                    if lid == your_label_id and conf < local_best:
                        local_best = conf
                        local_box = (tx, ty, tw, th)
                except Exception:
                    continue
            return local_best, local_box

        for i, (x, y, w, h) in enumerate(faces):
            # Score the face region using several crop variations.
            confidence, tuned_box = crop_and_score(x, y, w, h)
            label_id = your_label_id if confidence != float('inf') else -1
            
            print(f"  Face {i+1}: label_id={label_id}, confidence={confidence:.1f}")
            
            # Keep the lowest-distance match for the requested identity.
            if label_id == your_label_id:
                if confidence < best_conf:
                    best_conf = confidence
                    best_box = tuned_box
        
        if best_box is not None and best_conf <= THRESHOLD:
            x, y, w, h = best_box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(output_img, f"ME ({best_conf:.1f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            my_face_found = True
        # Save result
        ensure_dir(OUTPUTS_DIR)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUTS_DIR, f"{base_name}_my_face.jpg")
        save_image(output_path, output_img)
        
        print(f"\n[green]Result saved: {output_path}[/green]")
        if my_face_found:
            print("[bold green]✓ Your face was found![/bold green]")
        else:
            print("[bold yellow]⚠ Your face was not found in this image.[/bold yellow]")
        
        return my_face_found
        
    except Exception as e:
        print(f"[red]Error: {e}[/red]")
        return False


def main():
    print("[bold blue]Detect My Face - Simple Approach[/bold blue]")
    print("This script detects all faces first, then identifies which one is you.\n")
    
    # Ensure the model exists or train it from the local enrollment dataset.
    if not _ensure_trained(DEFAULT_PERSON):
        return
    
    # Process the configured group photographs.
    group_files = [
        os.path.join(PROJECT_DIR, "Mark group.jpg"),
        os.path.join(PROJECT_DIR, "Mark group1.jpg"),
    ]
    total_found = 0

    for name in group_files:
        group_file = name if os.path.exists(name) else None
        if group_file is None:
            print(f"⚠ {name} not found, skipping...")
            continue
        
        print(f"\n[yellow]Processing {group_file}...[/yellow]")
        print("-" * 50)
        
        if detect_my_face(group_file, person_name=DEFAULT_PERSON):
            total_found += 1
    
    print(f"\n[bold]Summary: Found your face in {total_found} out of {len(group_files)} images[/bold]")
    print(f"Check '{OUTPUTS_DIR}' for results with your face highlighted in GREEN!")


if __name__ == "__main__":
    main()
