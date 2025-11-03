# Face Detector - Personal Face Recognition System

A Python-based face detection and recognition system that specializes in finding a specific person's face within group photos. This project uses OpenCV and machine learning to automatically detect all faces in an image and identify which one belongs to you.

## 🎯 Features

- **Automatic Face Detection**: Uses Haar Cascade classifiers to detect all faces in images
- **Personal Face Recognition**: Trains a model to recognize your specific face using LBPH (Local Binary Pattern Histogram)
- **Smart Training**: Automatically trains the model from your photos with data augmentation when needed
- **Robust Detection**: Multiple detection passes with different parameters for better accuracy
- **Visual Output**: Generates annotated images highlighting your detected face with confidence scores
- **Batch Processing**: Processes multiple group photos automatically

## 📁 Project Structure

```
Face_detector/
├── detect_my_face.py              # Main script
├── haarcascade_frontalface_default.xml  # Haar cascade classifier
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── advanced/                     # Core modules
│   ├── __init__.py
│   ├── config.py                # Configuration settings
│   ├── detector.py              # Face detection logic
│   ├── recognizer.py            # Face recognition model
│   ├── dataset.py               # Data loading and preprocessing
│   └── utils.py                 # Helper utilities
├── data/
│   └── people/
│       └── myself/              # Your training photos go here
│           ├── Mark1.jpg
│           ├── Mark2.png
│           └── Mrak.jpeg
├── models/                      # Trained model files
│   ├── lbph.yml                # LBPH face recognition model
│   └── labels.json             # Label mappings
├── outputs/                     # Generated results
│   ├── Mark group_my_face.jpg
│   └── Mark group1_my_face.jpg
└── myenv/                       # Virtual environment (optional)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation


 **Create a virtual environment** (recommended):
   ```bash
   python -m venv myenv
   # On Windows:
   myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   ```

 **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Setup  Training Data

1. **Add  photos**: Place 2-3 clear photos in the `data/people/myself/` directory
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Use clear, front-facing photos for best results

2. **Prepare group photos**: Placed the group photos   to analyze in the project root directory

### Usage

**Run the face detection**:
```bash
python detect_my_face.py
```

The script will:
1. Automatically train a model from your photos (if not already trained)
2. Process the group photos (`Mark group.jpg`, `Mark group1.jpg`)
3. Generate annotated results in the `outputs/` folder

## 🔧 Configuration

Key parameters can be adjusted in `advanced/config.py`:

```python
# Detection parameters
DETECT_SCALE_FACTOR = 1.3    # How much the image size is reduced at each scale
DETECT_MIN_NEIGHBORS = 5     # How many neighbors each face should have to retain it
DETECT_MIN_SIZE = (30, 30)   # Minimum possible face size

# LBPH Recognition parameters
LBPH_RADIUS = 1              # Radius for LBP computation
LBPH_NEIGHBORS = 8           # Number of neighbors for LBP
LBPH_GRID_X = 8             # Grid cells in X direction
LBPH_GRID_Y = 8             # Grid cells in Y direction
```

## 📊 How It Works

### 1. Training Phase
- Loads your photos from `data/people/myself/`
- Detects faces in training images
- Performs data augmentation (mirroring, brightness adjustment, blurring) if needed
- Trains an LBPH face recognizer model
- Saves the model to `models/lbph.yml`

### 2. Detection Phase
- Uses Haar Cascade classifier to detect ALL faces in target images
- For each detected face, uses the trained LBPH model to check if it matches you
- Applies confidence thresholds and multiple cropping variations for accuracy
- Draws green rectangles around faces identified as you

### 3. Output
- Saves annotated images to `outputs/` folder
- Shows confidence scores for detected matches
- Provides summary of results

## 🎨 Example Output

The system will generate images like this:
- **Green rectangles**: Your detected face
- **Labels**: "ME (confidence_score)"
- **File naming**: `original_filename_my_face.jpg`

## 🛠️ Troubleshooting

### Common Issues

1. **No faces detected**:
   - Ensure photos are clear and well-lit
   - Try photos with different angles/lighting
   - Check if faces are large enough in the image

2. **Poor recognition accuracy**:
   - Add more training photos (3-5 recommended)
   - Use photos with varied lighting and angles
   - Ensure training photos are high quality

3. **Model training fails**:
   - Verify `opencv-contrib-python` is installed
   - Check that training photos contain detectable faces
   - Ensure sufficient disk space for model files

### Dependencies

If you encounter issues with OpenCV, try:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.8.1.78
```

## 📋 Requirements

- **opencv-python** >= 4.8: Core OpenCV functionality
- **opencv-contrib-python** >= 4.8: LBPH face recognizer
- **numpy** >= 1.24: Numerical computations
- **rich** >= 13.7: Beautiful terminal output
- **typer** >= 0.12: CLI framework
- **Pillow** >= 10.3: Image processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Acknowledgments

- OpenCV community for computer vision tools
- Haar Cascade classifiers for face detection
- LBPH algorithm for face recognition
- Rich library for beautiful terminal output

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Review the code comments for implementation details

---

**Happy face detecting! 😊**