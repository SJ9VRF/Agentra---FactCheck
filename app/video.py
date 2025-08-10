import cv2, os
from typing import List

def extract_keyframes(video_path: str, out_dir: str, max_frames: int = 5) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    # sample evenly
    steps = max(1, total // max_frames) if total else 50
    idx = 0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % steps == 0:
            fname = os.path.join(out_dir, f"key_{count:02d}.jpg")
            cv2.imwrite(fname, frame)
            frames.append(fname)
            count += 1
            if count >= max_frames:
                break
        idx += 1
    cap.release()
    return frames
