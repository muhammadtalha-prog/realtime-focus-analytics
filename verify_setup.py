import sys
import cv2
import numpy as np

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)

try:
    import mediapipe as mp
    print("MediaPipe imported successfully. Version:", mp.__version__)
except Exception as e:
    print("Failed to import MediaPipe:", e)

try:
    from src.landmark_extractor import LandmarkExtractor
    print("Initializing LandmarkExtractor...")
    extractor = LandmarkExtractor()
    print("LandmarkExtractor initialized successfully.")
except Exception as e:
    print("Failed to load LandmarkExtractor:", e)

try:
    from src.feature_engineer import FeatureEngineer
    from src.focus_scorer import FocusScorer
    from src.session_logger import SessionLogger
    print("All core modules (FeatureEngineer, FocusScorer, SessionLogger) imported successfully.")
except Exception as e:
    print("Failed to load core modules:", e)

print("Verification complete!")
