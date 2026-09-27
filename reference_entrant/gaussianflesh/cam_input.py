"""
cam_input.py
Background webcam + MediaPipe Hand Landmarker (Tasks API) thread.

Exposes thread-safe latest landmarks for the GaussianFlesh sim to consume.
The producer (this thread) runs the webcam at ~30 fps; the consumer (main
sim loop) polls get_latest() each frame and uses whatever is current.

MediaPipe model file (hand_landmarker.task) is downloaded once on first run
into ./models/.
"""
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def _ensure_model() -> Path:
    """Download the hand_landmarker.task model on first use."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"[HandTracker] downloading hand model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[HandTracker] model downloaded ({MODEL_PATH.stat().st_size//1024} KiB)")
    return MODEL_PATH


class HandTracker:
    """Run webcam + MediaPipe Hand Landmarker in a background thread.
    Thread-safe snapshot of the latest hand landmarks is available via
    get_latest()."""

    def __init__(self, camera_index: int = 0, max_hands: int = 2):
        self.camera_index = camera_index
        self.max_hands    = max_hands

        self._lock    = threading.Lock()
        self._latest  = {"hands": [], "timestamp": 0.0, "frame_count": 0}
        self._fps     = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        _ensure_model()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="HandTracker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- consumer API --------------------------------------------------

    def get_latest(self) -> dict:
        """Return latest snapshot.  Caller should treat values as read-only."""
        with self._lock:
            return {
                "hands":       list(self._latest["hands"]),
                "timestamp":   self._latest["timestamp"],
                "frame_count": self._latest["frame_count"],
                "fps":         self._fps,
            }

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    # ---- internal ------------------------------------------------------

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[HandTracker] ERROR: could not open camera {self.camera_index}")
            self._running = False
            return
        # Lower res to keep MediaPipe fast on CPU
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = mp_vision.HandLandmarker.create_from_options(options)

        last_t  = time.time()
        frames  = 0
        ema_dt  = 1.0 / 30.0
        t0_ms   = int(time.time() * 1000)

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # VIDEO mode wants monotonically increasing timestamps in ms
                stamp_ms = int(time.time() * 1000) - t0_ms
                result = detector.detect_for_video(mp_image, stamp_ms)

                hand_list = []
                if result.hand_landmarks:
                    for lm_list, hd_list in zip(result.hand_landmarks, result.handedness):
                        pts = np.empty((21, 3), dtype=np.float32)
                        for i, lm in enumerate(lm_list):
                            pts[i, 0] = lm.x
                            pts[i, 1] = lm.y
                            pts[i, 2] = lm.z
                        side  = hd_list[0].category_name if hd_list else "Right"
                        score = float(hd_list[0].score) if hd_list else 1.0
                        hand_list.append({
                            "landmarks":  pts,
                            "handedness": side,
                            "score":      score,
                        })

                frames += 1
                now    = time.time()
                ema_dt = 0.9 * ema_dt + 0.1 * (now - last_t)
                last_t = now

                with self._lock:
                    self._latest = {
                        "hands": hand_list,
                        "timestamp": now,
                        "frame_count": frames,
                    }
                    self._fps = 1.0 / max(ema_dt, 1e-6)
        finally:
            cap.release()
            detector.close()


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    print("Starting HandTracker smoke test (10s)")
    tracker = HandTracker()
    tracker.start()
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < 10.0:
        snap = tracker.get_latest()
        now = time.time()
        if now - last_print >= 0.5:
            last_print = now
            n = len(snap["hands"])
            sides = [h["handedness"] for h in snap["hands"]]
            tips = ""
            if n > 0:
                tip = snap["hands"][0]["landmarks"][8]
                tips = f"  hand0 idx-tip: ({tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:+.2f})"
            print(f"  t={now-t0:5.2f}s  fps={snap['fps']:5.1f}  "
                  f"frames={snap['frame_count']:4d}  hands={n} {sides}{tips}")
        time.sleep(0.1)
    tracker.stop()
    print("Done.")
