#!/usr/bin/env python3
"""
Stretch 3 — Camera + Gamepad/Keyboard Base Controller
======================================================
Runs two things simultaneously in one terminal:

  1. Live head camera feed (reuses run_standalone() from stretch_head_camera_live.py)
  2. Gamepad OR keyboard control of the mobile base via stretch_body

Camera window keys
------------------
  q / Esc   – quit everything
  s         – save snapshot to ./snapshots/

Base controls — KEYBOARD (default)
------------------------------------
  W / ↑     – drive forward
  S / ↓     – drive backward
  A / ←     – rotate left
  D / →     – rotate right
  Space     – full stop (send zero velocity)
  (hold keys for continuous motion, release to stop)

Base controls — GAMEPAD (pass --gamepad flag)
----------------------------------------------
  Left stick Y   – forward / backward
  Left stick X   – rotate left / right
  B / Circle     – emergency stop
  (uses pygame, works with any XInput/DInput USB gamepad)

Usage
-----
  # Keyboard mode (default):
  python3 stretch_cam_drive.py

  # Gamepad mode:
  python3 stretch_cam_drive.py --gamepad

  # If head_cam_feed.py is NOT in the same directory, pass its path:
  python3 stretch_cam_drive.py --cam-script /path/to/stretch_head_camera_live.py

Requirements
------------
  pip install stretch_body pynput opencv-python pyrealsense2 numpy
  pip install pygame   # only needed for --gamepad mode
"""

import argparse
import importlib.util
import os
import sys
import threading
import time

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Stretch 3 camera + base controller")
parser.add_argument(
    "--gamepad", action="store_true",
    help="Use a USB gamepad instead of keyboard for base control",
)
parser.add_argument(
    "--cam-script",
    default=os.path.join(os.path.dirname(__file__), "stretch_head_camera_live.py"),
    metavar="PATH",
    help="Path to stretch_head_camera_live.py (default: same directory as this script)",
)
parser.add_argument(
    "--lin-vel", type=float, default=0.12,
    help="Max linear velocity m/s (default: 0.12)",
)
parser.add_argument(
    "--rot-vel", type=float, default=0.7,
    help="Max rotational velocity rad/s (default: 0.7)",
)
parser.add_argument(
    "--depth", action="store_true",
    help="Show depth stream alongside RGB in the camera window",
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Import camera helpers from stretch_head_camera_live.py
# ---------------------------------------------------------------------------
def _load_cam_module(path: str):
    """Dynamically import stretch_head_camera_live.py regardless of location."""
    if not os.path.isfile(path):
        print(
            f"[error] Cannot find camera script at: {path}\n"
            f"        Place stretch_head_camera_live.py next to this script,\n"
            f"        or pass --cam-script /full/path/to/stretch_head_camera_live.py"
        )
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("head_cam_feed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cam_mod = _load_cam_module(args.cam_script)
# Re-use helpers from the camera module
save_snapshot = cam_mod.save_snapshot
draw_hud      = cam_mod.draw_hud


# ---------------------------------------------------------------------------
# Shared state between threads
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.quit        = False   # set True to stop everything
        self.lin_vel     = 0.0     # m/s  commanded this cycle
        self.rot_vel     = 0.0     # rad/s commanded this cycle
        self.lock        = threading.Lock()

state = SharedState()


# ---------------------------------------------------------------------------
# THREAD 1 — Camera (runs in main thread so OpenCV GUI works on macOS/Linux)
# ---------------------------------------------------------------------------
def run_camera():
    """Open head D435i feed; handles q/Esc to set state.quit."""
    try:
        import pyrealsense2 as rs
        import numpy as np
        import cv2
    except ImportError as e:
        print(f"[cam] Import error: {e}")
        state.quit = True
        return

    ctx = rs.context()
    devices = ctx.query_devices()
    target_serial = None
    for dev in devices:
        name   = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        if "D435" in name:
            target_serial = serial
            print(f"[cam] Using head D435i  (serial {serial})")
            break
    if target_serial is None and len(devices) == 1:
        target_serial = devices[0].get_info(rs.camera_info.serial_number)
    if target_serial is None:
        print("[cam] No D435 camera found — is the robot on?")
        state.quit = True
        return

    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(target_serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    if args.depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color) if args.depth else None

    window = "Stretch Head Camera  |  q=quit  s=snapshot"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Overlay for controller status
    ctrl_mode = "GAMEPAD" if args.gamepad else "KEYBOARD"
    t_prev, fps = time.time(), 0.0

    try:
        while not state.quit:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            if args.depth:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            if args.depth:
                depth_frame = frames.get_depth_frame()
                depth_cm    = cv2.applyColorMap(
                    cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), 2
                )
                display = np.hstack((color_image, depth_cm))
            else:
                display = color_image.copy()

            # FPS
            now  = time.time()
            fps  = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            draw_hud(display, fps)

            # Controller mode + current velocity overlay
            with state.lock:
                lv = state.lin_vel
                rv = state.rot_vel
            h = display.shape[0]
            cv2.putText(
                display,
                f"[{ctrl_mode}]  lin={lv:+.2f} m/s  rot={rv:+.2f} rad/s",
                (10, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                state.quit = True
                break
            elif key == ord("s"):
                save_snapshot(color_image)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# THREAD 2a — Keyboard input (pynput, non-blocking)
# ---------------------------------------------------------------------------
def run_keyboard_listener():
    """Listen to WASD / arrow keys and update state velocities."""
    try:
        from pynput import keyboard
    except ImportError:
        print("[kbd] pynput not found — run:  pip install pynput")
        state.quit = True
        return

    keys = {
        "forward": False, "backward": False,
        "left":    False, "right":    False,
    }

    LIN = args.lin_vel
    ROT = args.rot_vel

    def _update():
        lv = LIN  * int(keys["forward"]) - LIN  * int(keys["backward"])
        rv = ROT  * int(keys["left"])    - ROT  * int(keys["right"])
        with state.lock:
            state.lin_vel = lv
            state.rot_vel = rv

    def on_press(key):
        if state.quit:
            return False   # stop listener
        k = getattr(key, "char", None)
        if k in ("w", "W") or key == keyboard.Key.up:
            keys["forward"]  = True
        elif k in ("s", "S") or key == keyboard.Key.down:
            keys["backward"] = True
        elif k in ("a", "A") or key == keyboard.Key.left:
            keys["left"]     = True
        elif k in ("d", "D") or key == keyboard.Key.right:
            keys["right"]    = True
        elif key == keyboard.Key.space:
            keys.update(forward=False, backward=False, left=False, right=False)
        _update()

    def on_release(key):
        if state.quit:
            return False
        k = getattr(key, "char", None)
        if k in ("w", "W") or key == keyboard.Key.up:
            keys["forward"]  = False
        elif k in ("s", "S") or key == keyboard.Key.down:
            keys["backward"] = False
        elif k in ("a", "A") or key == keyboard.Key.left:
            keys["left"]     = False
        elif k in ("d", "D") or key == keyboard.Key.right:
            keys["right"]    = False
        _update()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not state.quit:
            time.sleep(0.05)
        listener.stop()

    print("[kbd] Keyboard listener stopped.")


# ---------------------------------------------------------------------------
# THREAD 2b — Gamepad input (pygame)
# ---------------------------------------------------------------------------
def run_gamepad_listener():
    """Poll a connected gamepad and update state velocities."""
    try:
        import pygame
    except ImportError:
        print("[pad] pygame not found — run:  pip install pygame")
        state.quit = True
        return

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[pad] No gamepad detected. Plug in a USB gamepad and retry.")
        state.quit = True
        return

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"[pad] Using gamepad: {joy.get_name()}")

    DEADZONE = 0.15
    LIN = args.lin_vel
    ROT = args.rot_vel

    # Button index for "emergency stop" (B on Xbox = 1, Circle on PS = 1)
    STOP_BTN = 1

    while not state.quit:
        pygame.event.pump()

        # Left stick: axis 1 = forward/back (inverted), axis 0 = rotate
        raw_lin = -joy.get_axis(1)   # push up = negative on most pads
        raw_rot = -joy.get_axis(0)   # push right = positive → rotate left

        lv = LIN * raw_lin if abs(raw_lin) > DEADZONE else 0.0
        rv = ROT * raw_rot if abs(raw_rot) > DEADZONE else 0.0

        # Emergency stop button
        if joy.get_button(STOP_BTN):
            lv, rv = 0.0, 0.0
            print("[pad] STOP button pressed — halting base.")

        with state.lock:
            state.lin_vel = lv
            state.rot_vel = rv

        time.sleep(0.033)   # ~30 Hz

    pygame.quit()
    print("[pad] Gamepad listener stopped.")


# ---------------------------------------------------------------------------
# THREAD 3 — stretch_body base driver (sends velocity commands at ~30 Hz)
# ---------------------------------------------------------------------------
def run_base_driver():
    """Translate state.lin_vel / state.rot_vel into stretch_body commands."""
    try:
        import stretch_body.robot as sb_robot
    except ImportError:
        print(
            "[base] stretch_body not found.\n"
            "       Install with:  pip install hello-robot-stretch-body"
        )
        state.quit = True
        return

    robot = sb_robot.Robot()
    print("[base] Connecting to Stretch...")
    if not robot.startup():
        print("[base] Failed to start robot. Is another process using it?")
        print("       Try:  stretch_free_robot_process.py")
        state.quit = True
        return

    if not robot.is_calibrated():
        print("[base] WARNING: Robot is not homed. Run stretch_robot_home.py first.")

    print("[base] Robot ready. Sending base commands.")
    prev_lv, prev_rv = None, None

    try:
        while not state.quit:
            with state.lock:
                lv = state.lin_vel
                rv = state.rot_vel

            # Only push a new command if velocity changed (saves bus bandwidth)
            if (lv, rv) != (prev_lv, prev_rv):
                robot.base.set_velocity(lv, rv)
                robot.push_command()
                prev_lv, prev_rv = lv, rv

            time.sleep(0.033)   # ~30 Hz

    finally:
        # Send a definitive stop before disconnecting
        robot.base.set_velocity(0.0, 0.0)
        robot.push_command()
        time.sleep(0.1)
        robot.stop()
        print("[base] Robot stopped and disconnected.")


# ---------------------------------------------------------------------------
# Entry point — wire up threads
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Stretch 3  |  Camera + Base Controller")
    print(f"  Mode: {'GAMEPAD' if args.gamepad else 'KEYBOARD'}")
    print(f"  Lin vel: {args.lin_vel} m/s   Rot vel: {args.rot_vel} rad/s")
    print("  Press  q  in the camera window to quit.")
    print("=" * 60)

    # Choose input thread
    input_fn = run_gamepad_listener if args.gamepad else run_keyboard_listener

    # Spin up non-camera threads
    threads = [
        threading.Thread(target=input_fn,     daemon=True, name="input"),
        threading.Thread(target=run_base_driver, daemon=True, name="base"),
    ]
    for t in threads:
        t.start()

    # Camera runs on the main thread (OpenCV GUI requirement)
    try:
        run_camera()
    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt — shutting down.")
    finally:
        state.quit = True

    # Wait for background threads to finish
    for t in threads:
        t.join(timeout=3.0)

    print("[main] All done. Bye!")
