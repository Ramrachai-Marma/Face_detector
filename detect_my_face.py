#!/usr/bin/env python3
"""
Simple face detection: First detect all faces, then identify which one is you
"""

import os
import cv2
from rich import print

from advanced.detector import HaarFaceDetector
from advanced.recognizer import LBPHRecognizer
from advanced.utils import ensure_dir, save_image
from advanced.dataset import enroll_images, load_dataset


def _ensure_trained(person_name: str = "mark") -> bool:
    """
    Ensure a trained model exists. If missing, enroll from a single image
    like data/people/mark.jpg and train the recognizer.
    """
    try:
        recognizer = LBPHRecognizer()
        # If a model exists, ensure it includes the requested person; otherwise retrain
        if os.path.exists(os.path.join(os.getcwd(), "models", "lbph.yml")):
            try:
                labels = recognizer.load()
                have = {v.lower().strip() for v in labels.values()}
                if person_name.lower().strip() in have:
                    return True
                # Otherwise fall through to retrain with current dataset
            except Exception:
                pass

        data_root = os.path.join(os.getcwd(), "data", "people")
        # Look inside data/people and also project root for a single training image
        candidates = [
            os.path.join(data_root, f"{person_name}.jpg"),
            os.path.join(data_root, f"{person_name}.jpeg"),
            os.path.join(data_root, f"{person_name}.png"),
            os.path.join(os.getcwd(), f"{person_name}.jpg"),
            os.path.join(os.getcwd(), f"{person_name}.jpeg"),
            os.path.join(os.getcwd(), f"{person_name}.png"),
            os.path.join(os.getcwd(), "Mark.jpeg"),
            os.path.join(os.getcwd(), "Mark.jpg"),
            os.path.join(os.getcwd(), "Mark.png"),
        ]
        single_image = next((p for p in candidates if os.path.isfile(p)), None)
        if single_image is not None:
            saved = enroll_images(person_name, [single_image])
            # Fallback: if no face detected during enrollment, save a resized full image
            if not saved:
                try:
                    import cv2
                    img = cv2.imread(single_image)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (100, 100))
                        out_dir = os.path.join(data_root, person_name)
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"{person_name}_0000.jpg")
                        cv2.imwrite(out_path, resized)
                except Exception:
                    pass

        # Simple augmentation when there are too few samples (mirror/brightness/blur)
        try:
            person_dir = os.path.join(data_root, person_name)
            existing = []
            if os.path.isdir(person_dir):
                for root, _, files in os.walk(person_dir):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            existing.append(os.path.join(root, f))
            if len(existing) <= 2 and existing:
                import cv2
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
                        cv2.imwrite(out_path, v)
        except Exception:
            pass

        features, targets, labels = load_dataset()
        if not features:
            print("[red]No training data found in 'data/people'. Add images and retry.[/red]")
            return False
        recognizer.train(features, targets, labels)
        print("[green]Model trained and saved.[/green]\n")
        return True
    except Exception as e:
        print(f"[red]Training failed: {e}[/red]")
        return False


def detect_my_face(image_path: str, person_name: str = "mark"):
    """
    Detect all faces first, then identify which one is you
    """
    try:
        # Load the trained model (train if missing)
        if not os.path.exists("models/lbph.yml"):
            if not _ensure_trained(person_name):
                return False
        recognizer = LBPHRecognizer()
        labels = recognizer.load()
        
        # Find your label ID
        your_label_id = None
        target = person_name.lower().strip()
        for label_id, name in labels.items():
            if name.lower().strip() == target:
                your_label_id = int(label_id)
                break
        
        if your_label_id is None:
            available = ", ".join(sorted(labels.values())) or "<none>"
            # Backward-compat: if only one label exists, assume it's the target
            if len(labels) == 1:
                only_id = int(next(iter(labels.keys())))
                print(f"[yellow]Label '{person_name}' not found. Using the only available label: '{labels[str(only_id)]}'.[/yellow]")
                your_label_id = only_id
            else:
                print(f"[red]Error: Could not find '{person_name}' in the trained model. Available: {available}[/red]")
                return False
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[red]Error: Could not load image '{image_path}'[/red]")
            return False
        
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Step 1: Detect ALL faces
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
        # Second pass: if no faces, upscale and try slightly different params
        if len(faces) == 0:
            try:
                import numpy as np
                up = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                faces_up = detector.cascade.detectMultiScale(
                    up,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                if len(faces_up) > 0:
                    # Map back to original scale
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
        
        # Step 2: Check each face to see if it's you
        output_img = img.copy()
        my_face_found = False
        THRESHOLD = 135.0
        best_conf = float('inf')
        best_box = None
        candidate_confs = []
        
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
            # Extract face region
            confidence, tuned_box = crop_and_score(x, y, w, h)
            label_id = your_label_id if confidence != float('inf') else -1
            
            print(f"  Face {i+1}: label_id={label_id}, confidence={confidence:.1f}")
            
            # Track the single best match to 'myself' only
            if label_id == your_label_id:
                candidate_confs.append(confidence)
                if confidence < best_conf:
                    best_conf = confidence
                    best_box = tuned_box
        
        # Draw only the best matching face (if confident enough)
        dyn_ok = False
        if candidate_confs:
            s = sorted(candidate_confs)
            if len(s) % 2 == 1:
                median_conf = s[len(s)//2]
            else:
                median_conf = (s[len(s)//2 - 1] + s[len(s)//2]) / 2.0
            # Accept if clearly better than peers by margin
            if best_conf <= (median_conf - 15.0):
                dyn_ok = True
        if best_box is not None and (best_conf <= THRESHOLD or dyn_ok):
            x, y, w, h = best_box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(output_img, f"ME ({best_conf:.1f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            my_face_found = True
        # Save result
        ensure_dir("outputs")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"outputs/{base_name}_my_face.jpg"
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
    
    # Ensure model exists or try to train from data/people/mark.jpg
    if not _ensure_trained("mark"):
        return
    
    # Process the provided group photos
    group_files = ["Mark group.jpg", "Mark group1.jpg"]
    total_found = 0

    for name in group_files:
        candidates = [name, f"{name}.jpg", f"{name}.JPG", f"{name}.png", f"{name}.PNG"]
        group_file = next((c for c in candidates if os.path.exists(c)), None)
        if group_file is None:
            print(f"⚠ {name} not found, skipping...")
            continue
        
        print(f"\n[yellow]Processing {group_file}...[/yellow]")
        print("-" * 50)
        
        if detect_my_face(group_file, person_name="mark"):
            total_found += 1
    
    print(f"\n[bold]Summary: Found your face in {total_found} out of {len(group_files)} images[/bold]")
    print("Check the 'outputs/' folder for results with your face highlighted in GREEN!")


if __name__ == "__main__":
    main()

