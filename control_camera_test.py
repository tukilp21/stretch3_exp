#!/usr/bin/env python3
"""
Stretch 3 Head Camera Live Feed
================================
Pops up a live OpenCV window showing the D435i head camera RGB stream.

Two modes (auto-detected):
  1. Standalone  – uses pyrealsense2 directly (no ROS 2 required).
  2. ROS 2 node  – subscribes to /camera/color/image_raw via rclpy.

Usage
-----
  Standalone (quickest):
      python3 stretch_head_camera_live.py

  Force ROS 2 mode:
      python3 stretch_head_camera_live.py --ros2

Keys while the window is open:
  q / Esc  – quit
  s        – save a snapshot to ./snapshots/

Requirements
------------
  Standalone : pip install pyrealsense2 opencv-python
  ROS 2 mode : ros2 launch stretch_core d435i_low_resolution.launch.py
               (in a separate terminal first)
               packages: rclpy, sensor_msgs, cv_bridge, opencv-python
"""

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Stretch 3 head camera live feed",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "--ros2",
    action="store_true",
    help="Force ROS 2 subscriber mode (default: auto-detect)",
)
parser.add_argument(
    "--depth",
    action="store_true",
    help="[Standalone only] Show depth stream side-by-side with RGB",
)
parser.add_argument(
    "--colormap",
    type=int,
    default=2,       # cv2.COLORMAP_JET
    help="[Standalone only] OpenCV colormap index for depth (default: 2=JET)",
)
parser.add_argument(
    "--topic",
    default="/camera/color/image_raw",
    help="[ROS 2 only] Image topic to subscribe to "
         "(default: /camera/color/image_raw)",
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def save_snapshot(frame):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SNAPSHOT_DIR, f"stretch_head_{ts}.jpg")
    import cv2
    cv2.imwrite(path, frame)
    print(f"[snapshot] saved → {path}")


def draw_hud(frame, fps: float):
    """Overlay FPS counter and key hints onto the frame (in-place)."""
    import cv2
    h, w = frame.shape[:2]
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "q/Esc=quit  s=snapshot", (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Mode 1 – Standalone via pyrealsense2
# ---------------------------------------------------------------------------
def run_standalone(show_depth: bool, colormap_idx: int):
    try:
        import pyrealsense2 as rs
    except ImportError:
        sys.exit(
            "[error] pyrealsense2 not found.\n"
            "Install with:  pip install pyrealsense2\n"
            "Or use ROS 2 mode:  python3 stretch_head_camera_live.py --ros2"
        )
    import cv2
    import time

    # ------------------------------------------------------------------
    # Connect to the D435i
    # ------------------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # The head camera is the D435i (not the wrist D405).
    # We select it by name if multiple cameras are attached.
    ctx = rs.context()
    devices = ctx.query_devices()
    target_serial = None
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  Found camera: {name}  (serial {serial})")
        if "D435" in name:
            target_serial = serial
            print(f"  → Using this as the head D435i camera.")
            break

    if target_serial is None and len(devices) == 1:
        target_serial = devices[0].get_info(rs.camera_info.serial_number)
        print(f"  Only one camera found; using it (serial {target_serial}).")
    elif target_serial is None:
        sys.exit("[error] Could not find a D435 camera. Is the robot on?")

    config.enable_device(target_serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    if show_depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color) if show_depth else None

    print("\n[info] Streaming from head D435i.  Press q or Esc in the window to quit.\n")
    window_title = "Stretch Head Camera (D435i)  –  standalone"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)

            if show_depth:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            if show_depth:
                depth_frame = frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
                # Normalise + apply colormap
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    colormap_idx,
                )
                display = np.hstack((color_image, depth_colormap))
            else:
                display = color_image.copy()

            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            draw_hud(display, fps)
            cv2.imshow(window_title, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q or Esc
                break
            elif key == ord("s"):
                save_snapshot(color_image)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Mode 2 – ROS 2 subscriber
# ---------------------------------------------------------------------------
def run_ros2(topic: str):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge, CvBridgeError
    except ImportError as e:
        sys.exit(
            f"[error] ROS 2 import failed: {e}\n"
            "Make sure you have sourced your ROS 2 workspace:\n"
            "  source ~/ament_ws/install/setup.bash\n"
            "And that rclpy / cv_bridge are installed."
        )
    import cv2
    import threading
    import time

    class HeadCameraViewer(Node):
        def __init__(self):
            super().__init__("stretch_head_camera_live")
            self.bridge = CvBridge()
            self.latest_frame = None
            self.lock = threading.Lock()
            self.sub = self.create_subscription(
                Image, topic, self._image_cb, 10
            )
            self.get_logger().info(
                f"Subscribed to {topic}  |  waiting for frames …"
            )

        def _image_cb(self, msg: "Image"):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().warn(f"CV Bridge error: {e}")
                return
            with self.lock:
                self.latest_frame = frame

    rclpy.init()
    node = HeadCameraViewer()

    # Spin ROS in a background thread so the main thread can own the GUI
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()

    print(
        f"\n[info] ROS 2 mode – listening on '{topic}'.\n"
        "       Press q or Esc in the window to quit.\n"
        "       (Make sure the camera launch file is running!)\n"
    )
    window_title = f"Stretch Head Camera  –  {topic}"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            with node.lock:
                frame = node.latest_frame

            if frame is not None:
                display = frame.copy()
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now
                draw_hud(display, fps)
                cv2.imshow(window_title, display)
            else:
                # Show a waiting screen until the first frame arrives
                import numpy as np
                placeholder = np.zeros((240, 640, 3), dtype="uint8")
                cv2.putText(
                    placeholder,
                    "Waiting for camera frames …",
                    (80, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2,
                )
                cv2.imshow(window_title, placeholder)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s") and frame is not None:
                save_snapshot(frame)

    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point – auto-detect mode unless --ros2 was passed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if args.ros2:
        run_ros2(args.topic)
    else:
        # Try pyrealsense2 first; fall back to ROS 2 if it's not available
        try:
            import pyrealsense2  # noqa: F401 – just checking availability
            import numpy as np   # needed in standalone
            run_standalone(args.depth, args.colormap)
        except ImportError:
            print(
                "[info] pyrealsense2 not found; trying ROS 2 mode instead.\n"
                "       (Tip: install pyrealsense2 for the standalone mode.)"
            )
            run_ros2(args.topic)