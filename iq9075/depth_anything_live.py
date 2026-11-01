#!/usr/bin/env python3
"""
Depth Anything live on IQ-9075 NPU via TFLite + QNN delegate.

Displays: RGB | Model Depth (NPU) | D415 Hardware Depth

Install:
    pip install tflite-runtime pyrealsense2 opencv-python numpy

Usage:
    python3 depth_anything_live.py
    python3 depth_anything_live.py --cpu         # skip QNN, run on CPU
    python3 depth_anything_live.py --rerun       # stream to Rerun
"""

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

# TFLite runtime import (try multiple package names)
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate


INPUT_SIZE = 518
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# Model lives in the repo's models/ folder by default. Override with --model.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "depth_anything", "depth_anything.tflite")
QNN_DELEGATE_PATH = "/usr/lib/libQnnTFLiteDelegate.so"


def preprocess(frame_rgb, size=INPUT_SIZE):
    """Resize and normalize an RGB frame for Depth Anything."""
    img = cv2.resize(frame_rgb, (size, size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> NHWC (TFLite expects NHWC by default, but check model)
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(output, target_h, target_w):
    """Convert model output to a displayable depth map."""
    depth = output.squeeze()
    depth = depth - depth.min()
    if depth.max() > 0:
        depth = depth / depth.max()
    depth = (depth * 255).astype(np.uint8)
    depth = cv2.resize(depth, (target_w, target_h))
    return depth


def colorize_depth(depth_gray):
    return cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)


def create_interpreter(model_path, use_qnn=True):
    """Create a TFLite interpreter, optionally with the QNN delegate for NPU."""
    delegates = []

    if use_qnn and os.path.exists(QNN_DELEGATE_PATH):
        try:
            qnn_delegate = load_delegate(QNN_DELEGATE_PATH, options={
                "backend_type": "htp",
            })
            delegates.append(qnn_delegate)
            print(f"[NPU] QNN delegate loaded from {QNN_DELEGATE_PATH}")
        except Exception as e:
            print(f"[NPU] Failed to load QNN delegate: {e}")
            print("[NPU] Falling back to CPU")

    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=delegates if delegates else None,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input:  {input_details[0]['name']} shape={input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"Output: {output_details[0]['name']} shape={output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    return interpreter, input_details, output_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--cpu", action="store_true", help="Force CPU, skip QNN")
    parser.add_argument("--rerun", action="store_true", help="Stream to Rerun")
    args = parser.parse_args()

    print(f"Model: {args.model}")

    # ── Init TFLite + QNN ──
    interpreter, input_details, output_details = create_interpreter(
        args.model, use_qnn=not args.cpu
    )

    # Check if input is NHWC or NCHW
    input_shape = input_details[0]['shape']
    is_nchw = (input_shape[1] == 3)
    print(f"Input layout: {'NCHW' if is_nchw else 'NHWC'}")

    # ── Init Rerun (optional) ──
    if args.rerun:
        import rerun as rr
        rr.init("depth_anything_live")
        rr.spawn()

    # ── Init RealSense D415 ──
    print("Starting D415...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.rgb8, CAMERA_FPS)
    config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"D415 streaming. Depth scale: {depth_scale:.6f}")

    # Warmup
    print("Warming up...")
    dummy = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32) if is_nchw \
        else np.random.rand(1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    for _ in range(3):
        interpreter.invoke()
    print("Ready. Press 'q' to quit.\n")

    frame_count = 0
    total_infer_time = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            hw_depth = np.asanyarray(depth_frame.get_data())

            # ── Preprocess ──
            img = preprocess(color_img)
            if is_nchw:
                # HWC -> CHW
                img = np.transpose(img, (0, 3, 1, 2))

            # ── Inference ──
            interpreter.set_tensor(input_details[0]['index'], img)

            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()

            output = interpreter.get_tensor(output_details[0]['index'])

            infer_ms = (t1 - t0) * 1000
            total_infer_time += infer_ms
            frame_count += 1
            avg_ms = total_infer_time / frame_count

            # ── Postprocess ──
            model_depth = postprocess(output, CAMERA_HEIGHT, CAMERA_WIDTH)
            model_depth_color = colorize_depth(model_depth)

            hw_depth_norm = cv2.normalize(hw_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            hw_depth_color = colorize_depth(hw_depth_norm)

            # ── Display ──
            color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            label = "NPU" if not args.cpu else "CPU"
            cv2.putText(color_bgr, "RGB", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(model_depth_color, f"{label} Depth ({infer_ms:.1f}ms)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(hw_depth_color, "D415 Depth", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            combined = np.hstack([color_bgr, model_depth_color, hw_depth_color])
            cv2.putText(combined, f"Avg: {avg_ms:.1f}ms ({1000/avg_ms:.1f} fps)",
                        (10, CAMERA_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Depth Anything Live - IQ-9075", combined)

            # ── Rerun ──
            if args.rerun:
                import rerun as rr
                rr.set_time("frame", sequence=frame_count)
                rr.log("cameras/rgb", rr.Image(color_img))
                rr.log("cameras/model_depth",
                       rr.Image(model_depth_color[:, :, ::-1]))
                rr.log("cameras/hw_depth",
                       rr.DepthImage(hw_depth, meter=1.0 / depth_scale))
                rr.log("perf/inference_ms", rr.Scalars(infer_ms))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if frame_count > 0:
            print(f"\n{frame_count} frames, avg inference: {avg_ms:.1f}ms ({1000/avg_ms:.1f} fps)")


if __name__ == "__main__":
    main()
