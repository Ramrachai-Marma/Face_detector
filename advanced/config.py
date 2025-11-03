import os


CASCADE_PATH = os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml")
DATA_DIR = os.path.join(os.getcwd(), "data", "people")
MODELS_DIR = os.path.join(os.getcwd(), "models")
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")

DETECT_SCALE_FACTOR = 1.3
DETECT_MIN_NEIGHBORS = 5
DETECT_MIN_SIZE = (30, 30)

LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8


