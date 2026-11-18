"""
Skrip ALPR (Automatic License Plate Recognition) real-time pakai fast-alpr.
Bisa dipakai untuk webcam, file video, atau stream RTSP kamera CCTV.

Install dulu (pilih salah satu sesuai device):
    pip install fast-alpr[onnx]           # CPU
    pip install fast-alpr[onnx-gpu]       # NVIDIA GPU (CUDA)
    pip install fast-alpr[onnx-directml]  # Windows GPU (DirectML)

Jalankan:
    python alpr_camera.py --source 0                     # webcam
    python alpr_camera.py --source video.mp4              # file video
    python alpr_camera.py --source rtsp://user:pass@ip/stream  # kamera CCTV
"""

import argparse
import time

import cv2
from fast_alpr import ALPR


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time ALPR dari kamera/video")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Sumber video: index webcam (0,1,...), path file video, atau URL RTSP/HTTP",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="yolo-v9-s-608-license-plate-end2end",
        help="Model deteksi plat nomor (default: model paling akurat yang tersedia)",
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default="cct-s-v2-global-model",
        help="Model OCR pembaca plat nomor (default: varian lebih besar/akurat)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.5,
        help="Confidence threshold untuk deteksi plat",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Path file video output kalau mau disimpan (mis. output.mp4)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Jangan tampilkan window preview (dipakai kalau jalan headless di server)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # source bisa berupa index webcam (angka) atau path/URL (string)
    source = int(args.source) if args.source.isdigit() else args.source

    print(f"[INFO] Memuat model ALPR: detector={args.detector_model}, ocr={args.ocr_model}")
    alpr = ALPR(
        detector_model=args.detector_model,
        ocr_model=args.ocr_model,
        detector_conf_thresh=args.conf_thres,
    )

    print(f"[INFO] Membuka sumber video: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka sumber video: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_output, fourcc, fps_in, (width, height))
        print(f"[INFO] Output video akan disimpan ke: {args.save_output}")

    seen_plates = {}  # cache biar tidak print/log plat yang sama berulang tiap frame
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream/video selesai atau gagal membaca frame.")
                break

            frame_count += 1

            # Jalankan deteksi + OCR plat nomor, sekaligus gambar hasilnya di frame
            drawn = alpr.draw_predictions(frame)
            annotated_frame = drawn.image
            results = drawn.results

            # Log plat nomor baru yang terdeteksi
            for r in results:
                plate_text = r.ocr.text if r.ocr else None
                if plate_text:
                    now = time.time()
                    last_seen = seen_plates.get(plate_text)
                    # hanya log kalau plat ini belum terlihat 5 detik terakhir
                    if last_seen is None or (now - last_seen) > 5:
                        raw_conf = r.ocr.confidence if r.ocr else 0.0
                        if isinstance(raw_conf, (list, tuple)):
                            conf = sum(raw_conf) / len(raw_conf) if raw_conf else 0.0
                        else:
                            conf = raw_conf
                        print(f"[PLAT TERDETEKSI] {plate_text} (confidence: {conf:.2f})")
                    seen_plates[plate_text] = now

            if writer:
                writer.write(annotated_frame)

            if not args.no_display:
                cv2.imshow("Fast-ALPR - tekan 'q' untuk keluar", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Dihentikan oleh user.")
                    break

    finally:
        elapsed = time.time() - t_start
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"[INFO] Selesai. Total frame: {frame_count}, rata-rata FPS: {avg_fps:.2f}")
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()