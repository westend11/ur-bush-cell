"""RealSense D435 capture with depth aligned to the colour image."""

import os
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

CAPTURE_DIR = "captures"
WARMUP_FRAMES = 15  # let auto-exposure settle


def _intrinsics_to_array(intrinsics):
    return np.array([intrinsics.width, intrinsics.height,
                     intrinsics.ppx, intrinsics.ppy,
                     intrinsics.fx, intrinsics.fy,
                     int(intrinsics.model), *intrinsics.coeffs], dtype=np.float64)


def load_intrinsics(path):
    """Rebuild rs.intrinsics from a saved *_intrinsics.npy."""
    values = np.load(path)
    intrinsics = rs.intrinsics()
    intrinsics.width, intrinsics.height = int(values[0]), int(values[1])
    intrinsics.ppx, intrinsics.ppy = float(values[2]), float(values[3])
    intrinsics.fx, intrinsics.fy = float(values[4]), float(values[5])
    intrinsics.model = rs.distortion(int(values[6]))
    intrinsics.coeffs = [float(c) for c in values[7:12]]
    return intrinsics


class RealSenseCamera:
    # colour at 1080p for detection; the depth module maxes out at 720p and
    # gets resampled onto the colour grid by align()
    def __init__(self, color_size=(1920, 1080), depth_size=(1280, 720), fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, *color_size, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)

        self._use_high_density(profile.get_device().first_depth_sensor())
        for _ in range(WARMUP_FRAMES):
            self.pipeline.wait_for_frames()
        os.makedirs(CAPTURE_DIR, exist_ok=True)

    @staticmethod
    def _use_high_density(sensor):
        # high density preset: ~80% -> 95% valid depth on our table
        if not sensor.supports(rs.option.visual_preset):
            return
        preset_range = sensor.get_option_range(rs.option.visual_preset)
        for value in range(int(preset_range.max) + 1):
            try:
                name = sensor.get_option_value_description(rs.option.visual_preset, value)
            except Exception:
                continue
            if name and "density" in name.lower():
                sensor.set_option(rs.option.visual_preset, value)
                return

    def capture(self):
        """Returns (color_image, depth_m, intrinsics, filename).

        depth_m is float32 metres, same size as the colour image, 0 = no reading.
        The png, depth and intrinsics are saved to CAPTURE_DIR so captures can
        be replayed offline.
        """
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete frameset")

        # copy out of the librealsense buffers, they get recycled
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_m = (np.asanyarray(depth_frame.get_data()).astype(np.float32)
                   * depth_frame.get_units())
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        stem = os.path.join(CAPTURE_DIR, f"{datetime.now():%Y%m%d_%H%M%S_%f}")
        filename = stem + ".png"
        cv2.imwrite(filename, color_image)
        np.save(stem + "_depth.npy", depth_m)
        np.save(stem + "_intrinsics.npy", _intrinsics_to_array(intrinsics))

        return color_image, depth_m, intrinsics, filename

    def close(self):
        self.pipeline.stop()
