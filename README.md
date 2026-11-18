# 🅿️ AI-PARKIR-SLOT — Deteksi Slot Parkir Kosong dengan YOLO

Sistem deteksi **status slot parkir (kosong / terisi)** secara real-time dari kamera (webcam, CCTV/RTSP) atau gambar statis, menggunakan **YOLO (Ultralytics)**. Hasil deteksi bisa langsung dikirim ke **API parkir** dan/atau disimpan ke **database MySQL**. Model juga tersedia dalam format ringan (**NCNN / ONNX int8**) agar bisa berjalan di **Raspberry Pi**.

![Hasil deteksi](hasil.jpg)

---

## ✨ Fitur Utama

- **Deteksi 2 kelas**: `free` (slot kosong, nomor slot terlihat) dan `car` (slot terisi mobil).
- **Dua cara mengenali nomor slot**:
  - **OCR (EasyOCR)** membaca nomor slot (misalnya `P1`, `A12`) dari slot yang kosong, ditambah *slot position memory* untuk slot yang tertutup mobil.
  - **Slot manual (`--slots`)** dengan polygon yang kamu gambar sendiri lewat `label_slots.py`. Tanpa OCR, lebih akurat dan ringan (cocok untuk Raspberry Pi).
- **Smoothing deteksi**: box tidak berkedip dan tidak melompat saat confidence rendah atau kamera goyang.
- **Integrasi API**: otomatis `POST` status slot, dengan validasi ke daftar slot resmi, konfirmasi *min-hit* berturut-turut, dedup, dan rekonsiliasi jika data di server diubah pihak lain.
- **Simpan ke MySQL**: mode `manual`, `continuous`, atau `both`, dengan opsi overwrite atau append (log history).
- **Mode capture**: deteksi hanya saat menekan tombol, tanpa proses terus-menerus.
- **Peningkat kontras (CLAHE)** untuk kondisi cahaya kurang.
- **Model ringan** untuk edge device lewat `export_to_lightweight_model.py`.
- **Bonus ALPR** (`alpr_camera.py`): pembacaan plat nomor real-time dengan `fast-alpr`.

---

## 📁 Struktur Repository

```
AI-PARKIR-SLOT/
├── predict.py                      # Script utama: deteksi slot (gambar / kamera / CCTV)
├── label_slots.py                  # Tool menggambar polygon slot manual → slots.json
├── slots.json                      # Contoh hasil pelabelan slot
├── export_to_lightweight_model.py  # Export model ke NCNN / ONNX / TFLite
├── alpr_camera.py                  # Pembaca plat nomor (ALPR) real-time
├── best.pt / last.pt               # Bobot YOLO hasil training (PyTorch)
├── best_ncnn_model/                # Model best dalam format NCNN
├── last_ncnn_model/                # Model last dalam format NCNN
├── last_int8.onnx                  # Model ONNX terkuantisasi int8
├── yolo26n.pt / yolov8n.pt         # Bobot dasar (pretrained)
├── requirements-windows.txt
├── requirements-linux.txt
├── requirements-raspberrypi.txt
├── parkiran.jpg / images.jpg       # Contoh gambar input
├── hasil.jpg                       # Contoh hasil deteksi
└── LICENSE
```

---

## ⚙️ Instalasi

**1. Clone repository**

```bash
git clone https://github.com/Andra-Fauzi/AI-PARKIR-SLOT.git
cd AI-PARKIR-SLOT
```

**2. (Opsional) Buat virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

**3. Install dependency sesuai platform**

```bash
pip install -r requirements-windows.txt        # Windows
pip install -r requirements-linux.txt          # Linux
pip install -r requirements-raspberrypi.txt    # Raspberry Pi
```

> Jika memakai mode OCR, **EasyOCR** akan mengunduh model bahasa Inggris sekali saat run pertama. Dengan `--slots`, EasyOCR tidak dimuat sama sekali.

---

## 🚀 Cara Penggunaan

### 1. Deteksi dari gambar

```bash
python predict.py --source parkiran.jpg --model best.pt
python predict.py --source parkiran.jpg --model best.pt --conf 0.5 --iou 0.4
python predict.py --source parkiran.jpg --mode free      # tampilkan hanya slot kosong
python predict.py --source parkiran.jpg --mode car       # tampilkan hanya slot terisi
```

### 2. Deteksi dari kamera / CCTV

```bash
python predict.py --camera 0                                         # webcam
python predict.py --camera 0 --width 1920 --height 1080
python predict.py --camera "rtsp://user:pass@192.168.1.10:554/stream1"   # CCTV / IP cam
```

Kontrol saat mode kamera: tekan **`s`** untuk menyimpan frame hasil deteksi, **`q`** untuk keluar.

### 3. Mode capture (deteksi hanya saat tombol ditekan)

```bash
python predict.py --camera 0 --capture-mode
```

Tekan **`c`** untuk capture dan deteksi, **`s`** untuk menyimpan hasil, **`q`** untuk keluar.

### 4. Slot manual dengan polygon (tanpa OCR)

Gambar polygon tiap slot dan beri label (misalnya `A1`, `A2`) sekali saja:

```bash
python label_slots.py --source parkiran.jpg --output slots.json
python label_slots.py --camera 0 --output slots.json
```

Lalu pakai file `slots.json` saat deteksi:

```bash
python predict.py --camera 0 --slots slots.json
python predict.py --source parkiran.jpg --slots slots.json --no-draw-slots   # sembunyikan overlay polygon
```

Nomor slot diambil langsung dari polygon yang memuat titik tengah tiap deteksi. Mode ini lebih ringan dan cocok untuk Raspberry Pi.

### 5. Kirim status ke API parkir

```bash
python predict.py --camera 0 --api-enable
python predict.py --camera 0 --api-enable --api-url "https://api-parkir.xorbit.my.id/api/ml/update-slot"
python predict.py --camera 0 --api-enable --api-min-hits 5
python predict.py --camera 0 --slots slots.json --api-enable
```

Body request yang dikirim:

```json
{ "slot_number": "P1", "is_available": 1 }
```

Cara kerja pengiriman:

| Mekanisme | Penjelasan |
|---|---|
| **Min-hit** | Status harus konsisten terbaca `--api-min-hits` kali berturut-turut (default 3) sebelum dikirim, supaya salah baca sesaat tidak langsung dipercaya. |
| **Dedup** | Request hanya dikirim jika status slot berubah dari yang terakhir dikirim. |
| **Validasi slot resmi** | Sebelum kirim, nomor slot dicek ke daftar slot di API (`--api-slots-url`). Matikan dengan `--api-skip-slots-check`. |
| **Rekonsiliasi** | Jika status di server berbeda dari hasil deteksi kamera, status yang benar dikirim ulang. |
| **Background thread** | Request jaringan tidak membuat video atau deteksi ikut lag. |

### 6. Simpan ke database MySQL

```bash
python predict.py --camera 0 --db-enable --db-name parkir_db
python predict.py --camera 0 --db-enable --db-name parkir_db --db-mode both --db-interval 15
python predict.py --camera 0 --db-enable --db-name parkir_db --db-append   # simpan sebagai history
python predict.py --camera 0 --db-enable --db-name parkir_db --db-host 192.168.1.5 --db-user admin --db-password <password>
```

Tabel dibuat otomatis jika belum ada, tetapi **database-nya harus dibuat dulu**.

| `--db-mode` | Perilaku |
|---|---|
| `manual` (default) | Simpan hanya saat menekan `s` |
| `continuous` | Simpan otomatis tiap `--db-interval` frame (default 30) |
| `both` | Gabungan keduanya |

Secara default isi tabel **di-overwrite** dengan snapshot terbaru. Tambahkan `--db-append` untuk menyimpan sebagai riwayat.

### 7. Tips tambahan

```bash
# Naikkan kontras untuk kondisi cahaya kurang (diterapkan sebelum masuk model)
python predict.py --camera 0 --contrast-boost --contrast-clip-limit 3.0

# Atur smoothing deteksi
python predict.py --camera 0 --smooth-frames 15 --smooth-alpha 0.3 --smooth-min-hits 5
python predict.py --camera 0 --no-smooth

# Kustomisasi warna
python predict.py --source parkiran.jpg --class-colors "car:red,free:green" --class-text-colors "car:white,free:black"

# Pilih device
python predict.py --source parkiran.jpg --device 0      # GPU
python predict.py --source parkiran.jpg --device cpu
```

Jalankan `python predict.py --help` untuk melihat seluruh argumen.

---

## 🍓 Deployment di Raspberry Pi

Export model ke format ringan, lalu arahkan `--model` ke folder hasil export:

```bash
python export_to_lightweight_model.py

python predict.py --camera 0 --model best_ncnn_model --slots slots.json
python predict.py --source parkiran.jpg --model best_ncnn_model --mode car --conf 0.5
```

Catatan:
- Model hasil export berupa **folder**, bukan file `.pt` tunggal.
- Untuk model export (NCNN / ONNX / TFLite), `--device` diabaikan karena inference berjalan di CPU.
- Semua argumen lain (`--mode`, `--conf`, `--db-*`, smoothing, dan seterusnya) berfungsi sama seperti model `.pt`.
- Gunakan `requirements-raspberrypi.txt` dan mode `--slots` agar lebih ringan.

---

## 🔖 Bonus: ALPR (Pembaca Plat Nomor)

```bash
pip install fast-alpr[onnx]          # CPU
pip install fast-alpr[onnx-gpu]      # NVIDIA GPU (CUDA)
pip install fast-alpr[onnx-directml] # GPU Windows (DirectML)

python alpr_camera.py --source 0                               # webcam
python alpr_camera.py --source video.mp4                       # file video
python alpr_camera.py --source rtsp://user:pass@ip/stream      # CCTV
python alpr_camera.py --source 0 --save-output output.mp4 --no-display
```

---

## 🧠 Alur Kerja Singkat

```
Kamera / Gambar
      │
      ▼
 YOLO (best.pt / NCNN)  ──►  Deteksi kelas: free / car
      │
      ├── Mode OCR:    EasyOCR baca nomor slot + position memory
      └── Mode --slots: cocokkan titik tengah box ke polygon slot manual
      │
      ▼
 Smoothing + konfirmasi min-hit
      │
      ├──► POST ke API parkir  (is_available: 1 = kosong, 0 = terisi)
      └──► Simpan ke MySQL
```

---

## 📝 Catatan

- Kamera sebaiknya **statis** (tidak digeser), karena *slot position memory* dan polygon manual bergantung pada posisi tetap.
- Jangan menyimpan password database di dalam kode atau di-commit ke repository. Berikan lewat argumen `--db-password`.
- Pastikan nomor slot di `slots.json` atau hasil OCR terdaftar di API, jika tidak update akan dilewati.

---

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi [MIT](LICENSE).

## 👤 Author

**Andra Fauzi** — [@Andra-Fauzi](https://github.com/Andra-Fauzi)
