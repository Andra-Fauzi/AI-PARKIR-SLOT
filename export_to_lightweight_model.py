"""
Export YOLO (YOLOv8/v9/v10/v11) ke format ringan untuk Raspberry Pi.

Requirement:
    pip install ultralytics onnx onnxruntime onnxsim ncnn

Cara pakai cepat:
    python export_yolo_raspi.py --weights yolo11n.pt --format ncnn
    python export_yolo_raspi.py --weights best.pt --format onnx --int8 --data data.yaml
    python export_yolo_raspi.py --weights best.pt --format tflite --int8 --data data.yaml

Rekomendasi buat Raspberry Pi (CPU only, tanpa GPU/NPU):
    1. NCNN            -> paling cepat inference di ARM CPU, size kecil
    2. ONNX + INT8      -> portable, cepat lewat onnxruntime, akurasi lebih terjaga dari TFLite INT8
    3. TFLite + INT8    -> pilih kalau butuh runtime TFLite (misal edge TPU / Coral)

Catatan soal "tanpa ngorbanin akurasi":
    - Export FP32 -> tidak ada penurunan akurasi sama sekali, tapi paling berat/lambat.
    - Export dengan quantization (INT8) -> ada potensi drop akurasi 0.5-3% mAP,
      TAPI kalau pakai calibration dataset yang representatif (--data), dropnya
      biasanya kecil dan sepadan dengan speedup 2-4x di CPU ARM.
    - Kalau model kamu udah kecil (n/s variant), coba dulu FP16 atau NCNN tanpa
      quant sebelum turun ke INT8, karena speedup NCNN native udah signifikan.
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO model untuk Raspberry Pi")
    p.add_argument("--weights", type=str, required=True,
                    help="Path ke model .pt (misal yolo11n.pt atau best.pt)")
    p.add_argument("--format", type=str, default="ncnn",
                    choices=["ncnn", "onnx", "tflite", "openvino"],
                    help="Format target export (default: ncnn, paling direkomendasikan untuk RPi)")
    p.add_argument("--imgsz", type=int, default=640,
                    help="Ukuran input gambar (kecilkan mis. 320/416 buat lebih ngebut di RPi)")
    p.add_argument("--int8", action="store_true",
                    help="Aktifkan quantization INT8 (butuh --data untuk kalibrasi, lebih akurat)")
    p.add_argument("--half", action="store_true",
                    help="Export FP16 (setengah presisi, aman untuk akurasi, size lebih kecil dari FP32)")
    p.add_argument("--data", type=str, default=None,
                    help="Path ke data.yaml (dipakai sebagai calibration set kalau --int8 aktif)")
    p.add_argument("--simplify", action="store_true", default=True,
                    help="Simplify graph ONNX (default aktif, mengurangi ukuran & mempercepat)")
    p.add_argument("--validate", action="store_true",
                    help="Validasi & bandingkan mAP model asli vs hasil export (butuh --data)")
    p.add_argument("--nms", action="store_true", default=True,
                    help="Sertakan NMS di dalam graph model export (default aktif)")
    return p.parse_args()


def export_model(args):
    from ultralytics import YOLO

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"[ERROR] File weights tidak ditemukan: {weights_path}")

    print(f"[INFO] Load model: {weights_path}")
    model = YOLO(str(weights_path))

    # Base kwargs yang valid untuk semua format
    export_kwargs = dict(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
    )

    # Beberapa parameter cuma didukung format tertentu di Ultralytics.
    # ncnn TIDAK support: nms, simplify -> jadi cuma ditambahkan untuk format yang support.
    if args.format in ("onnx", "engine", "coreml", "tflite", "edgetpu", "tfjs"):
        export_kwargs["nms"] = args.nms
    if args.format == "onnx":
        export_kwargs["simplify"] = args.simplify

    # INT8 quantization butuh dataset kalibrasi biar akurasi tetap terjaga
    if args.int8:
        if args.data is None:
            sys.exit("[ERROR] --int8 butuh --data <path/to/data.yaml> untuk kalibrasi")
        if args.format not in ("onnx", "tflite", "edgetpu", "engine", "openvino"):
            sys.exit(f"[ERROR] --int8 tidak didukung untuk format='{args.format}' di Ultralytics. "
                      f"NCNN belum ada opsi int8 langsung dari export ini.")
        export_kwargs["int8"] = True
        export_kwargs["data"] = args.data

    print(f"[INFO] Export ke format='{args.format}' | imgsz={args.imgsz} | "
          f"int8={args.int8} | half={args.half}")
    result_path = model.export(**export_kwargs)
    print(f"[DONE] Model hasil export: {result_path}")
    return result_path, model


def validate_models(original_model, exported_path, data_yaml, imgsz):
    """Bandingkan mAP model asli vs model hasil export, biar keliatan trade-off-nya."""
    from ultralytics import YOLO

    print("\n[VALIDATE] Mengevaluasi model ASLI (.pt)...")
    metrics_orig = original_model.val(data=data_yaml, imgsz=imgsz, verbose=False)

    print("[VALIDATE] Mengevaluasi model HASIL EXPORT...")
    exported_model = YOLO(str(exported_path))
    metrics_exp = exported_model.val(data=data_yaml, imgsz=imgsz, verbose=False)

    print("\n=== PERBANDINGAN AKURASI ===")
    print(f"{'Metric':<12}{'Original':<12}{'Exported':<12}{'Selisih':<10}")
    for key, orig_val, exp_val in [
        ("mAP50", metrics_orig.box.map50, metrics_exp.box.map50),
        ("mAP50-95", metrics_orig.box.map, metrics_exp.box.map),
    ]:
        diff = exp_val - orig_val
        print(f"{key:<12}{orig_val:<12.4f}{exp_val:<12.4f}{diff:+.4f}")


def print_raspi_tips(export_format):
    print("\n=== TIPS DEPLOY DI RASPBERRY PI ===")
    if export_format == "ncnn":
        print("- Install ncnn python binding: pip install ncnn")
        print("- Load pakai: model = YOLO('namamodel_ncnn_model')")
        print("- Pastikan pakai OpenMP/threads sesuai jumlah core RPi (export.set_num_threads)")
    elif export_format == "onnx":
        print("- Install: pip install onnxruntime (bukan onnxruntime-gpu, RPi gak ada CUDA)")
        print("- Gunakan session dengan providers=['CPUExecutionProvider']")
    elif export_format == "tflite":
        print("- Install: pip install tflite-runtime (lebih ringan dari full tensorflow)")
        print("- Cocok kalau nanti mau lanjut ke Coral Edge TPU")
    print("- Turunkan --imgsz (misal 416 atau 320) kalau masih kurang real-time")
    print("- Pakai RPi 4/5 dengan cooling yang baik biar gak throttle saat inference terus-menerus")
    print("- Kalau butuh lebih ngebut lagi: pertimbangkan model variant lebih kecil (n/s)")
    print("  daripada maksa quantize model besar (m/l), karena arsitektur kecil > kompresi berat")


def main():
    args = parse_args()
    exported_path, model = export_model(args)

    if args.validate:
        if args.data is None:
            print("[WARNING] --validate butuh --data, dilewati.")
        else:
            validate_models(model, exported_path, args.data, args.imgsz)

    print_raspi_tips(args.format)


if __name__ == "__main__":
    main()