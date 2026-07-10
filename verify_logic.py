import sys
sys.path.insert(0, '.')

from src.feature_engineer import FeatureEngineer
from src.focus_scorer import FocusScorer
from src.session_logger import SessionLogger

print("Checking imports...")
eng = FeatureEngineer()
scr = FocusScorer()
log = SessionLogger()
print("  FeatureEngineer OK")
print("  FocusScorer OK")
print("  SessionLogger OK")

# Synthetic 'focused' features test
feat = {
    'ear': 0.30, 'mar': 0.20,
    'gaze_x': 0.05, 'gaze_y': 0.02,
    'yaw': 3.0, 'pitch': 2.0, 'roll': 1.0, 'posture': 8.0
}
result = scr.update(feat, fps=20)
score = result['score']
state = result['state']
print(f"\nSynthetic 'focused' stream:")
print(f"  Score: {score}  State: {state}")
assert score > 0.5, f"Expected score > 0.5 for focused input, got {score}"
print("  PASS: score > 0.5 for focused input")

# Synthetic 'distracted' features test
scr2 = FocusScorer()
feat_dist = {
    'ear': 0.28, 'mar': 0.25,
    'gaze_x': 0.45, 'gaze_y': 0.30,
    'yaw': 30.0, 'pitch': 5.0, 'roll': 2.0, 'posture': 12.0
}
for _ in range(30):
    result2 = scr2.update(feat_dist, fps=20)
print(f"\nSynthetic 'distracted' stream:")
print(f"  Score: {result2['score']}  State: {result2['state']}")
assert result2['score'] < score, "Distracted score should be lower than focused score"
print("  PASS: distracted score < focused score")

print("\nAll checks PASSED!")
