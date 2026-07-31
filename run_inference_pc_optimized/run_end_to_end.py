import argparse
import os
import sys
import time
import queue
import threading
import numpy as np
import torch
import onnxruntime as ort
import multiprocessing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import (
    add_generate_training_dataset_scripts_to_syspath,
    layout_oas,
    inference_dataset_dir,
    gradcam_results_dir,
    cnn_drc_report_path,
    cnn_violation_mask_gds,
    merged_layout_with_cnn_mask,
    MODEL_WEIGHTS_PTH,
    MODEL_WEIGHTS_ONNX,
)

add_generate_training_dataset_scripts_to_syspath()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from define_cnn_model import NCSU_DRCNN
from GRAD_CAM import GradCAM, generate_gradcam
from generate_inference_dataset import generate_inference_dataset
from run_inference_ONNX import apply_nms
from build_cnn_violation_mask_gds import build_mask_and_merge, write_cnn_drc_report

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
LAYOUT_NAME = "tt_um_cmos_inverter"

INPUT_GDS = layout_oas(LAYOUT_NAME)
OUTPUT_DIR = inference_dataset_dir(LAYOUT_NAME)
GRADCAM_DIR = gradcam_results_dir(LAYOUT_NAME)
REPORT_PATH = cnn_drc_report_path(LAYOUT_NAME)
CONFIDENCE_THRESHOLD = 0.80
BATCH_SIZE = 256
NMS_DISTANCE_THRESHOLD = 1600
# ---------------------------------------------------------------------------


def configure_layout(layout_name):
    global LAYOUT_NAME, INPUT_GDS, OUTPUT_DIR, GRADCAM_DIR, REPORT_PATH
    LAYOUT_NAME = layout_name
    INPUT_GDS = layout_oas(layout_name)
    OUTPUT_DIR = inference_dataset_dir(layout_name)
    GRADCAM_DIR = gradcam_results_dir(layout_name)
    REPORT_PATH = cnn_drc_report_path(layout_name)


def producer_thread(input_gds, batch_queue):
    generate_inference_dataset(
        input_gds=input_gds,
        output_dir=str(OUTPUT_DIR),
        batch_queue=batch_queue,
        batch_size=BATCH_SIZE,
    )


def consumer_thread(onnx_weights, threshold, batch_queue, raw_violations_list):
    print("[Consumer] Initializing ONNX C++ Engine...")
    cpu_cores = multiprocessing.cpu_count()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = cpu_cores

    ort_session = ort.InferenceSession(onnx_weights, sess_options, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    print("[Consumer] Waiting for batches...")

    tiles_processed = 0
    while True:
        batch = batch_queue.get()
        if batch is None:
            break

        batch_matrices, batch_coords = batch

        batch_np = batch_matrices.astype(np.float32)
        batch_inputs = np.expand_dims(batch_np, axis=1)

        logits = ort_session.run(None, {input_name: batch_inputs})[0]

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        for j, prob_arr in enumerate(probs):
            prob = prob_arr[1]
            if prob >= threshold:
                x, y = batch_coords[j]
                raw_violations_list.append((x, y, prob, batch_matrices[j]))

        tiles_processed += len(batch_matrices)
        print(f"[Consumer] Processed {tiles_processed} tiles... Flagged: {len(raw_violations_list)}")

    print("[Consumer] Finished inference.")


def run_end_to_end():
    if not os.path.exists(MODEL_WEIGHTS_ONNX):
        print(f"[!] ONNX model not found at '{MODEL_WEIGHTS_ONNX}'. Please run export_to_onnx.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(GRADCAM_DIR, exist_ok=True)

    batch_queue = queue.Queue(maxsize=10)
    raw_violations = []

    t0 = time.time()

    producer = threading.Thread(
        target=producer_thread,
        args=(str(INPUT_GDS), batch_queue),
    )
    consumer = threading.Thread(
        target=consumer_thread,
        args=(str(MODEL_WEIGHTS_ONNX), CONFIDENCE_THRESHOLD, batch_queue, raw_violations),
    )

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

    phase1_time = time.time() - t0
    print(f"\n[+] Phase 1 (Generation + Inference) complete in {phase1_time:.1f}s — {len(raw_violations)} raw violation(s) found.")

    merged_violations = apply_nms(raw_violations, NMS_DISTANCE_THRESHOLD)
    if len(raw_violations) > 0:
        print(f"[*] NMS reduced {len(raw_violations)} overlapping detections into {len(merged_violations)} unique regions.")

    phase2_start = time.time()

    if merged_violations:
        print("\n" + "=" * 55)
        print(" PHASE 2: Running Grad-CAM on flagged tiles...")
        print("=" * 55)

        device = torch.device("cpu")
        model = NCSU_DRCNN().to(device)
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PTH, map_location=device))
        model.eval()

        grad_cam = GradCAM(model, model.conv4)

        for v_idx, (x, y, prob, matrix) in enumerate(merged_violations, 1):
            matrix_f32 = matrix.astype(np.float32)
            save_path = os.path.join(GRADCAM_DIR, f"gradcam_x{x}_y{y}.png")
            generate_gradcam(matrix_f32, grad_cam, device, save_path)
            print(f"[>] ({v_idx}/{len(merged_violations)}) tile_x{x}_y{y} | conf: {prob:.2%}")

    phase2_time = time.time() - phase2_start
    total_time = phase1_time + phase2_time

    print("\n" + "=" * 55)
    if not merged_violations:
        print("[*] LAYOUT IS CLEAN — No DRC violations found.")
        write_cnn_drc_report(
            REPORT_PATH,
            header_lines=[f"Unique (NMS)  : 0"],
        )
        print(f"[*] Report saved to '{REPORT_PATH}'")
    else:
        print(f"[!] FOUND {len(merged_violations)} POTENTIAL VIOLATION(S)!")

        violation_lines = []
        for x, y, prob, _ in merged_violations:
            line = f"Location: x{x}_y{y} | Confidence: {prob:.2%}"
            print(f"  - {line}")
            violation_lines.append(line)

        write_cnn_drc_report(
            REPORT_PATH,
            violation_lines=violation_lines,
            header_lines=[f"Unique (NMS)  : {len(merged_violations)}"],
        )
        print(f"\n[*] Full report saved to '{REPORT_PATH}'")

        build_mask_and_merge(
            input_layout_path=str(INPUT_GDS),
            report_path=str(REPORT_PATH),
            mask_output_path=str(cnn_violation_mask_gds(LAYOUT_NAME)),
            merged_output_path=str(merged_layout_with_cnn_mask(LAYOUT_NAME)),
        )

    print(f"\n[*] Total Pipeline time: {total_time:.1f}s")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimized CNN DRC inference on a layout.")
    parser.add_argument("--layout", default=LAYOUT_NAME, help="Layout name under real_layouts_tt (without .oas)")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--nms-distance", type=float, default=NMS_DISTANCE_THRESHOLD)
    args = parser.parse_args()
    configure_layout(args.layout)
    CONFIDENCE_THRESHOLD = args.threshold
    BATCH_SIZE = args.batch_size
    NMS_DISTANCE_THRESHOLD = args.nms_distance
    run_end_to_end()
