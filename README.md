# Face Detection and Recognition

A Python/OpenCV demonstration that detects faces in group photos and uses an LBPH recognizer to look for one enrolled identity. Annotated results are saved locally.

## How it works

- A Haar Cascade classifier proposes face regions.
- An LBPH (Local Binary Patterns Histograms) model compares detected faces with enrolled training images.
- The script draws a green box and confidence label around a recognized match.
- Annotated images are saved under `outputs/`.

## Requirements

- Python 3.8 or later
- The dependencies in [`requirements.txt`](requirements.txt)

## Setup

From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place permitted training images in `data/people/<person-name>/`. Clear, front-facing images with varied lighting generally provide a more useful enrollment set. The default identity is `myself`, matching the model in this repository. The current repository also contains example training and group photos. `.gitignore` prevents new local training images and models from being added by default, but it does not untrack files already committed. If no model is present, the app trains one from the local training folders. Existing models are not silently replaced when their labels do not match the requested identity.

## Run

```bash
python detect_my_face.py
```

The current batch entry point looks for `Mark group.jpg` and `Mark group1.jpg` in the project root and searches for the `myself` identity. Supply consented input photos with those names, or update the image list and `DEFAULT_PERSON` in `detect_my_face.py`. The example photos are excluded from Git by default because they contain identifiable people. The requested identity must exactly match a label in the trained model (case and surrounding whitespace are ignored); a different single label is never treated as a match.

Output names are based on the input image, for example `outputs/Mark group_my_face.jpg`.

If the model is missing, the script may enroll or augment local training images and create a model under `models/`. Review the training images and back up any model you need to keep.

## Configuration and code map

- `detect_my_face.py` - Batch entry point and recognition flow
- `advanced/config.py` - Detection, model, and data paths
- `advanced/dataset.py` - Enrollment and training-data loading
- `advanced/detector.py` - Haar Cascade face detection
- `advanced/recognizer.py` - LBPH model operations
- `advanced/identity.py` - Exact identity-label matching
- `data/people/` - Training images, grouped by identity (example files are already tracked)
- `models/` - Saved recognizer and label mapping (example files are already tracked)
- `outputs/` - Annotated results
- `tests/` - Offline tests for identity-label matching
- `.github/workflows/tests.yml` - GitHub Actions workflow for offline tests

Detection and recognition parameters can be adjusted in `advanced/config.py` and `detect_my_face.py`. Run the offline tests with:

```bash
python -m unittest discover -s tests -v
```

## Privacy and limitations

Face recognition is probabilistic and can produce false matches or misses. A match is drawn only when the LBPH distance is at most 135; this fixed threshold is illustrative and has not been calibrated for real-world use. Training photos and recognizer models are biometric data. The current repository already contains example photos and model files; publish or retain them only with the subjects' consent and awareness. `.gitignore` does not remove previously committed files. This demonstration is not suitable for consequential identification decisions. Performance depends on image quality, pose, lighting, and the training set.

## License

See [`LICENSE`](LICENSE).
