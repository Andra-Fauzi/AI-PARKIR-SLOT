"""
label_slots.py
Tool GUI buat gambar area/polygon tiap slot parkir SECARA MANUAL pakai mouse
(klik titik-titik sudutnya satu-satu), kasih label per polygon (mis. "A1", "A2"),
lalu disimpan ke file JSON. File JSON ini dipakai di predict.py lewat --slots,
biar predict.py tau posisi & identitas tiap slot fisik LANGSUNG dari koordinatnya
(gak perlu OCR baca nomor sama sekali -- kameranya kan statis, jadi posisi tiap
slot di layar udah pasti/gak berubah-ubah).

Cara pakai (gambar polygon dari foto/snapshot statis):
    python label_slots.py --source foto_parkiran.jpg --output slots.json

Cara pakai (ambil frame dari kamera dulu, baru gambar polygon di atasnya):
    python label_slots.py --camera 0 --output slots.json
    python label_slots.py --camera "rtsp://user:pass@192.168.1.10:554/stream1" --output slots.json

Cara pakai (lanjutin/tambah slot ke file JSON yang udah ada sebelumnya):
    python label_slots.py --source foto_parkiran.jpg --input slots.json --output slots.json

Kontrol mouse & keyboard:
    - Klik kiri       : tambah titik sudut ke polygon yang lagi digambar
    - Klik kanan       : hapus titik terakhir (undo 1 titik)
    - Enter / 'n'      : TUTUP polygon yang lagi digambar (minimal 3 titik), lalu
                          diminta ngetik label-nya di terminal (mis. "A1")
    - 'u'              : undo titik terakhir (sama kayak klik kanan)
    - 'd'              : hapus SLOT TERAKHIR yang udah selesai/tersimpan (kalau salah)
    - 's'              : simpan progres sekarang ke file JSON (tanpa keluar)
    - 'q' / ESC        : simpan & keluar
    - 'c'              : (mode --camera doang, saat masih live preview) capture/bekukan
                          frame ini buat mulai digambar

Format output slots.json:
    {
      "image_width": 1920, "image_height": 1080,
      "slots": [
        {"label": "A1", "points": [[120, 80], [340, 80], [340, 260], [120, 260]]},
        {"label": "A2", "points": [[360, 80], [580, 80], [580, 260], [360, 260]]}
      ]
    }
"""

import argparse
import json
import os

import cv2
import numpy as np

WINDOW_NAME = "Label Slots - klik=titik, Enter/n=tutup polygon, u=undo, d=hapus slot, s=simpan, q=keluar"

POLYGON_COLOR = (0, 255, 0)       # hijau: polygon yang lagi digambar (belum ditutup)
SAVED_COLOR = (0, 200, 255)       # oranye: polygon yang udah tersimpan jadi slot
SAVED_FILL_ALPHA = 0.25
POINT_RADIUS = 4


def load_existing_slots(path):
    """Load daftar slot dari file JSON yang udah ada (buat dilanjutin/ditambah),
    return list kosong kalau path None atau file belum ada."""
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    slots = data.get("slots", [])
    for s in slots:
        s["points"] = [tuple(p) for p in s["points"]]
    return slots


def save_slots(path, slots, image_width, image_height):
    data = {
        "image_width": image_width,
        "image_height": image_height,
        "slots": [{"label": s["label"], "points": [list(p) for p in s["points"]]} for s in slots],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Simpan] {len(slots)} slot tersimpan di: {path}")


def _polygon_centroid(points):
    pts = np.array(points, dtype=np.float32)
    return int(pts[:, 0].mean()), int(pts[:, 1].mean())


def render(base_image, saved_slots, current_points):
    """Gambar ulang base_image + semua slot yang udah tersimpan + polygon yang lagi
    digambar (kalau ada titiknya), return frame siap ditampilkan."""
    canvas = base_image.copy()

    # slot yang udah tersimpan: isi semi-transparan + outline + label di tengah
    overlay = canvas.copy()
    for slot in saved_slots:
        pts = np.array(slot["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], SAVED_COLOR)
    canvas = cv2.addWeighted(overlay, SAVED_FILL_ALPHA, canvas, 1 - SAVED_FILL_ALPHA, 0)

    for slot in saved_slots:
        pts = np.array(slot["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=SAVED_COLOR, thickness=2)
        cx, cy = _polygon_centroid(slot["points"])
        cv2.putText(canvas, slot["label"], (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, slot["label"], (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # polygon yang LAGI digambar (belum ditutup): titik + garis penghubung
    if current_points:
        for i, p in enumerate(current_points):
            cv2.circle(canvas, p, POINT_RADIUS, POLYGON_COLOR, -1)
            if i > 0:
                cv2.line(canvas, current_points[i - 1], p, POLYGON_COLOR, 2)
        if len(current_points) > 1:
            # garis putus-putus dari titik terakhir balik ke titik pertama, preview
            # penutupan polygon (belum beneran ketutup sampai Enter/'n' ditekan)
            cv2.line(canvas, current_points[-1], current_points[0], POLYGON_COLOR, 1)

    # bar info di atas
    info = (f"Slot tersimpan: {len(saved_slots)}  |  Titik polygon aktif: {len(current_points)}  |  "
            f"Enter/n=tutup polygon  u=undo  d=hapus slot terakhir  s=simpan  q=keluar")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(canvas, info, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def edit_slots(base_image, initial_slots, output_path):
    """Loop utama editor: tampilkan base_image, biar user gambar polygon + label
    lewat mouse & keyboard, sampai user tekan 'q'/ESC. Return list slot final."""
    saved_slots = list(initial_slots)
    current_points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_points:
                current_points.pop()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    h, w = base_image.shape[:2]
    print(f"\nEditor slot aktif ({w}x{h}). Total slot awal: {len(saved_slots)}")
    print("Klik kiri = tambah titik, klik kanan/'u' = undo titik, Enter/'n' = tutup polygon & kasih label.")
    print("'d' = hapus slot terakhir, 's' = simpan, 'q'/ESC = simpan & keluar.\n")

    while True:
        frame = render(base_image, saved_slots, current_points)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, ord('n')):  # Enter atau 'n' -> tutup polygon
            if len(current_points) < 3:
                print("Polygon butuh minimal 3 titik dulu sebelum ditutup.")
                continue
            # tampilkan preview tertutup sebentar biar user lihat bentuknya sebelum diminta label
            preview = render(base_image, saved_slots, current_points)
            pts = np.array(current_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], isClosed=True, color=POLYGON_COLOR, thickness=2)
            cv2.imshow(WINDOW_NAME, preview)
            cv2.waitKey(1)

            label = input(f"Label buat polygon ini (mis. 'A1'), atau kosongkan buat batal: ").strip()
            if not label:
                print("Dibatalkan, polygon gak disimpan.")
            else:
                saved_slots.append({"label": label, "points": list(current_points)})
                print(f"Slot '{label}' tersimpan ({len(current_points)} titik). Total slot: {len(saved_slots)}")
            current_points = []

        elif key in (ord('u'),):
            if current_points:
                current_points.pop()

        elif key == ord('d'):
            if saved_slots:
                removed = saved_slots.pop()
                print(f"Slot '{removed['label']}' dihapus. Total slot: {len(saved_slots)}")
            else:
                print("Belum ada slot tersimpan buat dihapus.")

        elif key == ord('s'):
            save_slots(output_path, saved_slots, w, h)

        elif key in (ord('q'), 27):  # 'q' atau ESC
            break

    cv2.destroyAllWindows()
    return saved_slots, w, h


def grab_frame_from_camera(camera_arg):
    """Buka kamera, tampilkan live preview, tunggu user tekan 'c' buat capture/bekukan
    satu frame yang bakal dipakai sebagai dasar gambar polygon."""
    try:
        cam_source = int(camera_arg)
    except ValueError:
        cam_source = camera_arg  # URL RTSP/HTTP

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera/sumber: {camera_arg}")

    print("Live preview kamera. Tekan 'c' buat capture frame ini (dasar gambar polygon), 'q' buat batal.")
    captured = None
    window = "Live preview - tekan 'c' untuk capture"
    cv2.namedWindow(window)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Gagal membaca frame dari kamera. Cek koneksi/URL.")
            preview = frame.copy()
            cv2.putText(preview, "Tekan 'c' untuk capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                captured = frame.copy()
                break
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyWindow(window)

    if captured is None:
        raise RuntimeError("Gak ada frame yang di-capture, batal.")
    return captured


def main():
    parser = argparse.ArgumentParser(
        description="Gambar polygon slot parkir manual + kasih label, simpan ke JSON buat dipakai di predict.py --slots."
    )
    parser.add_argument("--source", default=None, help="Path gambar/foto statis dasar buat gambar polygon.")
    parser.add_argument("--camera", default=None,
                         help="Index kamera (mis. 0) atau URL RTSP/HTTP -- live preview dulu, tekan 'c' buat capture 1 frame.")
    parser.add_argument("--input", default=None,
                         help="File slots.json yang udah ada, buat dilanjutin/ditambah slot barunya (opsional).")
    parser.add_argument("--output", default="slots.json", help="Path file JSON hasil (default: slots.json)")
    args = parser.parse_args()

    if not args.source and not args.camera:
        raise ValueError("Isi salah satu: --source (path gambar) atau --camera (index/URL kamera).")
    if args.source and args.camera:
        raise ValueError("Pilih salah satu aja: --source ATAU --camera, jangan dua-duanya.")

    if args.source:
        base_image = cv2.imread(args.source)
        if base_image is None:
            raise RuntimeError(f"Tidak bisa membaca gambar: {args.source}")
    else:
        base_image = grab_frame_from_camera(args.camera)

    initial_slots = load_existing_slots(args.input)
    if initial_slots:
        print(f"Memuat {len(initial_slots)} slot dari: {args.input}")

    final_slots, w, h = edit_slots(base_image, initial_slots, args.output)
    save_slots(args.output, final_slots, w, h)


if __name__ == "__main__":
    main()