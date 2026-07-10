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
    from object_detector import ObjectDetector
    print("Initializing ObjectDetector wrapper...")
    detector = ObjectDetector()
    detector.start()
    print("ObjectDetector started successfully.")
    detector.stop()
except Exception as e:
    print("Failed to load ObjectDetector wrapper:", e)

print("All dependencies checked successfully!")
