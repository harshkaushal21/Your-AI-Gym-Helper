import os
import cv2
import av
import numpy as np
import mediapipe as mp
import threading
import time

from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector

from services.config.workout_config import POSE_CONNECTIONS


class VideoProcessorClass(VideoProcessorBase):

    def __init__(self):

        # Thread safety
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

        # Shared state
        self._latest_metrics = None
        self._exercise_type = "Squats"

        # -------------------------------------------------
        # Performance settings
        # -------------------------------------------------

        # Camera/display resolution
        self._frame_width = 640
        self._frame_height = 480

        # Process pose on every Nth frame
        # Camera can still display more smoothly
        self._frame_count = 0
        self._process_every_n_frames = 3

        # Last pose result
        self._last_result = None

        # Last processed metrics
        self._last_metrics = None

        # -------------------------------------------------
        # Async inference state
        # -------------------------------------------------

        # Background thread that runs detect_for_video()
        self._inference_thread = None

        # Set while a background inference is running, so we
        # don't start a second one on top of it
        self._inference_running = False

        # -------------------------------------------------
        # Load MediaPipe Pose Landmarker
        # -------------------------------------------------

        model_path = os.path.join(
            os.getcwd(),
            "ml_models",
            "pose_landmarker_lite.task"
        )

        base_option = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_option,

            # VIDEO mode
            running_mode=vision.RunningMode.VIDEO,

            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,

            output_segmentation_masks=False
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(
            options
        )

        # -------------------------------------------------
        # Exercise detectors
        # -------------------------------------------------

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        # MediaPipe video timestamp
        self._frame_timestamps_ms = 0

    # =====================================================
    # METRICS
    # =====================================================

    def set_latest_metrics(self, metrics):

        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):

        with self._lock:
            if self._latest_metrics is None:
                return None

            return self._latest_metrics.copy()

    # =====================================================
    # EXERCISE
    # =====================================================

    def set_exercise(self, exercise_type):

        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):

        with self._lock:
            return self._exercise_type

    # =====================================================
    # DRAW SKELETON
    # =====================================================

    def _draw_skeleton(self, img, landmarks):

        h, w = img.shape[:2]

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:

            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if p1.visibility > 0.7 and p2.visibility > 0.7:

                cv2.line(
                    img,
                    (
                        int(p1.x * w),
                        int(p1.y * h)
                    ),
                    (
                        int(p2.x * w),
                        int(p2.y * h)
                    ),
                    (0, 255, 0),
                    4
                )

        # Draw landmarks
        for lm in landmarks:

            if lm.visibility > 0.7:

                cv2.circle(
                    img,
                    (
                        int(lm.x * w),
                        int(lm.y * h)
                    ),
                    5,
                    (255, 0, 0),
                    -1
                )

    # =====================================================
    # NO POSE WARNING
    # =====================================================

    def _draw_no_pose_warnings(self, img):

        cv2.putText(
            img,
            "NO POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # =====================================================
    # OVERLAYS
    # =====================================================

    def _draw_overlays(self, img, metrics, ex_type):

        if ex_type == "Squats":

            self._draw_squats_overlays(
                img,
                metrics
            )

        elif ex_type == "Push-ups":

            self._draw_pushup_overlays(
                img,
                metrics
            )

        elif ex_type == "Biceps Curls (Dumbbell)":

            self._draw_curl_overlays(
                img,
                metrics
            )

        elif ex_type == "Shoulder Press":

            self._draw_press_overlays(
                img,
                metrics
            )

        elif ex_type == "Lunges":

            self._draw_lunge_overlays(
                img,
                metrics
            )

    # =====================================================
    # SQUAT OVERLAY
    # =====================================================

    def _draw_squats_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"DEPTH: {metrics['depth_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =====================================================
    # PUSHUP OVERLAY
    # =====================================================

    def _draw_pushup_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"BODY: {metrics['body_alignment']} | "
            f"HIP: {metrics['hip_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =====================================================
    # CURL OVERLAY
    # =====================================================

    def _draw_curl_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"SWING: {metrics['swing_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =====================================================
    # SHOULDER PRESS OVERLAY
    # =====================================================

    def _draw_press_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"EXT: {metrics['extension_status']} | "
            f"BACK: {metrics['back_arch_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =====================================================
    # LUNGE OVERLAY
    # =====================================================

    def _draw_lunge_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"BALANCE: {metrics['balance_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # =====================================================
    # ASYNC POSE INFERENCE (runs in background thread)
    # =====================================================

    def _run_inference(self, mp_image, timestamp_ms):
        """
        Runs detect_for_video() off the main recv() thread so the
        video pipeline never blocks waiting on pose detection.
        Updates self._last_result / self._last_metrics under a lock
        once inference finishes.
        """

        try:

            result = self._landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                ex_type = self.get_exercise()

                detector = self._detectors.get(ex_type)

                metrics = None

                if detector:

                    metrics = detector.process(landmarks)
                    metrics["pose_detected"] = True

                with self._inference_lock:
                    self._last_result = result
                    if metrics is not None:
                        self._last_metrics = metrics

                if metrics is not None:
                    self.set_latest_metrics(metrics)

            else:

                no_pose_metrics = {"pose_detected": False}

                with self._inference_lock:
                    self._last_result = result
                    self._last_metrics = no_pose_metrics

                with self._lock:

                    if self._latest_metrics is not None:
                        self._latest_metrics["pose_detected"] = False
                    else:
                        self._latest_metrics = no_pose_metrics

        finally:

            self._inference_running = False

    # =====================================================
    # MAIN VIDEO PROCESSING
    # =====================================================

    def recv(self, frame):

        # -------------------------------------------------
        # 1. Get camera frame
        # -------------------------------------------------

        image = frame.to_ndarray(format="bgr24")

        # Mirror webcam
        image = cv2.flip(image, 1)

        # -------------------------------------------------
        # 2. Resize BEFORE AI processing
        # -------------------------------------------------

        image = cv2.resize(
            image,
            (
                self._frame_width,
                self._frame_height
            ),
            interpolation=cv2.INTER_AREA
        )

        # -------------------------------------------------
        # 3. Count frame
        # -------------------------------------------------

        self._frame_count += 1

        # -------------------------------------------------
        # 4. Kick off pose detection on a background thread
        #    every Nth frame -- never blocks this recv() call
        # -------------------------------------------------

        should_process = (
            self._frame_count % self._process_every_n_frames == 0
        )

        if should_process and not self._inference_running:

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )

            # MediaPipe requires increasing timestamps
            current_timestamp = int(time.monotonic() * 1000)

            if current_timestamp <= self._frame_timestamps_ms:
                current_timestamp = self._frame_timestamps_ms + 1

            self._frame_timestamps_ms = current_timestamp

            self._inference_running = True

            self._inference_thread = threading.Thread(
                target=self._run_inference,
                args=(mp_image, self._frame_timestamps_ms),
                daemon=True
            )
            self._inference_thread.start()

        # -------------------------------------------------
        # 5. Draw the most recently available pose result
        #    (may be from a slightly earlier frame -- that's
        #    fine, it keeps video smooth)
        # -------------------------------------------------

        with self._inference_lock:
            last_result = self._last_result
            last_metrics = self._last_metrics

        if last_result is not None and last_result.pose_landmarks:

            landmarks = last_result.pose_landmarks[0]

            # Draw skeleton
            self._draw_skeleton(
                image,
                landmarks
            )

            # Draw latest exercise metrics
            if last_metrics:

                ex_type = self.get_exercise()

                self._draw_overlays(
                    image,
                    last_metrics,
                    ex_type
                )

        else:

            self._draw_no_pose_warnings(
                image
            )

        # -------------------------------------------------
        # 6. Return processed video frame
        # -------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )