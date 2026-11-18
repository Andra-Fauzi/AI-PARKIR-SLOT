"""
predict.py
Prediksi objek pakai YOLO — simpel, tanpa slot/polygon apapun.
Mendukung input gambar statis maupun kamera (webcam/CCTV) secara live.

Catatan instalasi:
    OCR sekarang pakai EasyOCR (ganti dari RapidOCR, lebih akurat buat teks kecil/
    buram/miring kayak nomor slot parkir di kamera CCTV). Kalau belum keinstall:
        pip install easyocr
    (Model bahasa Inggris otomatis ke-download sekali di run pertama & di-cache lokal.)

Cara pakai (gambar):
    python predict.py --source gambar.jpg
    python predict.py --source gambar.jpg --model best.pt
    python predict.py --source gambar.jpg --model best.pt --conf 0.5
    python predict.py --source gambar.jpg --model best.pt --conf 0.5 --iou 0.4
    python predict.py --source gambar.jpg --model best.pt --mode free
    python predict.py --source gambar.jpg --model best.pt --mode car
    python predict.py --source gambar.jpg --color red --text-color white
    python predict.py --source gambar.jpg --class-colors "car:red,free:green" --class-text-colors "car:white,free:black"
    python predict.py --source gambar.jpg --ocr-color cyan --class-ocr-colors "car:white,free:orange"
    python predict.py --source gambar.jpg --device 0
    python predict.py --source gambar.jpg --device cpu

Cara pakai (naikin kontras gambar/kamera, mis. buat kondisi cahaya kurang):
    python predict.py --source gambar.jpg --contrast-boost
    python predict.py --camera 0 --contrast-boost --contrast-clip-limit 3.0
    # Kontras dinaikin SEBELUM masuk ke model, jadi selain tampilan lebih jelas,
    # ini juga bisa membantu akurasi deteksi di kondisi cahaya kurang/gak rata.

Cara pakai (kamera, misal webcam index 0):
    python predict.py --camera 0
    python predict.py --camera 0 --model best.pt --conf 0.5 --iou 0.4
    python predict.py --camera 0 --width 1920 --height 1080
    python predict.py --camera 0 --model best.pt --mode free
    python predict.py --camera 0 --ocr-interval 3
    python predict.py --camera 0 --no-camera-ocr

Cara pakai (CCTV/IP cam via RTSP/HTTP):
    python predict.py --camera "rtsp://user:pass@192.168.1.10:554/stream1"

Cara pakai (deteksi mulus, gak kedip-kedip & gak lompat-lompat saat confidence rendah/goyang):
    python predict.py --camera 0                                  # smoothing aktif by default
    python predict.py --camera 0 --smooth-frames 15                # box bertahan lebih lama sebelum dihapus
    python predict.py --camera 0 --smooth-iou 0.2                  # lebih toleran mencocokkan box antar frame
    python predict.py --camera 0 --smooth-alpha 0.3                # posisi box lebih halus 'meluncur' (default: 0.5)
    python predict.py --camera 0 --smooth-min-hits 5                # box baru harus stabil 5 frame dulu baru nongol (default: 3)
    python predict.py --camera 0 --no-smooth                       # matikan smoothing, kembali ke deteksi mentah per-frame

    Deteksi dengan confidence di ANTARA --conf dan --min-conf otomatis disembunyikan
    (diproses model tapi gak ditampilkan/dipakai), mis. --conf 0.3 --min-conf 0.4 artinya
    deteksi dengan confidence 0.30-0.39 gak akan muncul di layar.

Cara pakai (simpan hasil deteksi ke database MySQL):
    # tabel otomatis dibuat kalau belum ada (database-nya sendiri harus sudah ada duluan)
    python predict.py --camera 0 --db-enable --db-name parkir_db
    python predict.py --camera 0 --db-enable --db-name parkir_db --db-mode continuous
    python predict.py --camera 0 --db-enable --db-name parkir_db --db-mode both --db-interval 15
    python predict.py --camera 0 --db-enable --db-name parkir_db --db-host 192.168.1.5 --db-user admin --db-password rahasia123
    python predict.py --source gambar.jpg --db-enable --db-name parkir_db

    --db-mode manual     -> simpan ke DB cuma pas tekan 's' (default)
    --db-mode continuous -> simpan otomatis tiap --db-interval frame (default: 30)
    --db-mode both       -> gabungan keduanya
    (mode gambar --source selalu simpan sekali kalau --db-enable aktif, --db-mode diabaikan)

    Default: tabel MySQL di-OVERWRITE tiap simpan (cuma berisi snapshot deteksi terbaru,
    sama seperti file --save yang ikut tergantikan). Tambahkan --db-append kalau mau
    tabel nambah baris terus jadi history/log, bukan overwrite:
    python predict.py --camera 0 --db-enable --db-name parkir_db --db-append

Cara pakai (kirim update status slot ke API parkir, mis. api-parkir.xorbit.my.id):
    # Aktif di semua mode (gambar/kamera/capture). Setiap kali nomor slot (mis. "P1")
    # berhasil terbaca lewat OCR, script otomatis POST ke API:
    #   body: {"slot_number": "P1", "is_available": 1}
    # OCR cuma jalan buat slot class 'free' (slot kosong, nomornya kebaca jelas),
    # sedangkan class 'car' (nomor ketutupan mobil) di-skip OCR-nya --
    # jadi begitu nomor slot kedeteksi, otomatis artinya slot itu KOSONG (is_available=1).
    python predict.py --camera 0 --api-enable
    python predict.py --camera 0 --api-enable --api-url "https://api-parkir.xorbit.my.id/api/ml/update-slot"
    python predict.py --camera 0 --api-enable --api-min-hits 5    # lebih ketat, butuh 5x hit berturut-turut
    python predict.py --source gambar.jpg --api-enable
    python predict.py --camera 0 --capture-mode --api-enable

    Minimum hit (mode kamera live): status slot (kosong/terisi) harus konsisten
    kebaca --api-min-hits kali BERTURUT-TURUT dulu (default: 3) baru beneran
    dikirim ke API, biar salah baca OCR/tebakan posisi yang cuma kedip sekali
    gak langsung dipercaya jadi update. Mode --source gambar & --capture-mode
    selalu kirim langsung (cuma ada satu observasi tegas per aksi).

    Request cuma dikirim ulang kalau status slot itu BERUBAH dari yang terakhir dikirim
    (dedup otomatis), jadi gak nge-spam API tiap frame buat slot yang sama terus-terusan.
    Request dijalankan di background thread, jadi gak bikin video/deteksi nge-lag
    nunggu response network.

    Validasi ke daftar slot resmi: sebelum kirim update, script GET dulu daftar
    slot yang beneran terdaftar (default: https://api-parkir.xorbit.my.id/api/slots).
    Kalau nomor slot hasil OCR/deteksi gak ada di daftar itu (mis. OCR salah baca
    jadi nomor yang gak eksis), update-nya GAK DIKIRIM. Daftar ini di-cache &
    di-refresh berkala, bukan di-GET ulang tiap mau kirim satu update:
        python predict.py --camera 0 --api-enable
        python predict.py --camera 0 --api-enable --api-slots-url "https://api-parkir.xorbit.my.id/api/slots"
        python predict.py --camera 0 --api-enable --api-slots-refresh 60   # refresh tiap 1 menit
        python predict.py --camera 0 --api-enable --api-skip-slots-check  # matiin validasi ini

Cara pakai (slot manual, gambar polygon tiap slot sendiri pakai mouse, TANPA OCR sama sekali):
    # Langkah 1: gambar polygon tiap slot + kasih label (mis. "A1", "A2") pakai tool
    # label_slots.py -- klik titik-titik sudutnya satu-satu di foto/frame kamera,
    # simpan ke file JSON. Lihat "python label_slots.py --help" buat detail kontrolnya.
    python label_slots.py --source foto_parkiran.jpg --output slots.json
    python label_slots.py --camera 0 --output slots.json

    # Langkah 2: pakai file slots.json itu di predict.py lewat --slots. Identitas
    # slot (slot_number) diambil LANGSUNG dari polygon yang mengandung titik tengah
    # tiap box deteksi -- gak ada OCR/tebakan posisi otomatis sama sekali, karena
    # posisi semua slot udah pasti didefinisikan manual dari awal.
    python predict.py --camera 0 --slots slots.json
    python predict.py --camera 0 --slots slots.json --db-enable --db-name parkir_db
    python predict.py --camera 0 --slots slots.json --api-enable
    python predict.py --source gambar.jpg --slots slots.json --db-enable --db-name parkir_db --api-enable

    - EasyOCR TIDAK dimuat/dijalankan sama sekali kalau --slots dipakai (start-up
      lebih cepat & ringan, cocok buat Raspberry Pi).
    - Berlaku buat SEMUA class (baik 'free' maupun 'car') -- status kosong/terisi
      tetap ditentukan dari ada/tidaknya kata "free" di nama class hasil deteksi.
    - Overlay tipis polygon slot ikut digambar di hasil (buat verifikasi posisinya
      udah pas), matikan dengan --no-draw-slots kalau gak mau.
    - Hasilnya tetap bisa disimpan ke database (--db-enable) & dikirim ke API
      (--api-enable), persis seperti mode OCR biasa -- cuma sumber nomor slotnya
      beda (dari polygon manual, bukan hasil baca OCR).

Kontrol saat mode kamera:
    - Tekan 's' : simpan frame hasil deteksi saat ini ke file (menggantikan simpanan sebelumnya)
    - Tekan 'q' : keluar

Cara pakai (mode capture, deteksi HANYA saat pencet tombol, bukan tiap frame):
    python predict.py --camera 0 --capture-mode
    python predict.py --camera 0 --capture-mode --model best.pt --conf 0.5
    python predict.py --camera 0 --capture-mode --db-enable --db-name parkir_db

    Beda dengan mode kamera biasa (yang deteksi TERUS-MENERUS tiap frame),
    di mode ini kamera cuma nampilin live preview mentah (ringan, gak ada beban
    YOLO/OCR jalan tiap frame). Deteksi baru dijalankan SEKALI setiap kamu pencet 'c'.

    Kontrol mode capture:
    - Tekan 'c' : capture frame saat ini & jalankan deteksi (YOLO + OCR)
    - Tekan 's' : simpan hasil capture TERAKHIR ke file (& DB kalau --db-enable aktif)
    - Tekan 'q' : keluar

    Cocok buat skenario "kendaraan/orang berhenti dulu di depan kamera baru discan",
    bukan pemantauan real-time terus-menerus. Kalau --db-enable aktif, tiap capture
    otomatis tersimpan ke DB (terlepas dari --db-mode), karena momen capture itu sendiri
    yang jadi pemicunya, bukan interval frame seperti di mode kamera biasa.

Cara pakai model hasil export ringan (NCNN/ONNX/TFLite, mis. untuk Raspberry Pi):
    Model hasil export (dari export_to_lightweight_model.py) itu berupa FOLDER,
    bukan file .pt tunggal. Cukup arahkan --model ke folder tersebut:

    python predict.py --camera 0 --model best_ncnn_model
    python predict.py --source gambar.jpg --model best_ncnn_model
    python predict.py --camera 0 --model best_ncnn_model --mode car --conf 0.5

    Catatan:
    - Nama folder default = <nama_weights>_ncnn_model (atau _onnx_model, dst),
      cek isi direktori tempat kamu jalanin proses export tadi.
    - Untuk model hasil export, --device diabaikan untuk pemindahan device
      (script otomatis mendeteksi ini lewat ekstensi --model yang bukan .pt),
      karena format export (NCNN/ONNX/TFLite) adalah inference graph CPU-only.
      Ini normal dan memang sesuai untuk deployment di Raspberry Pi (tidak ada GPU/CUDA).
    - Semua argumen lain (--mode, --conf, --iou, --color, --db-*, smoothing, dst)
      tetap berfungsi sama persis seperti pakai model .pt biasa.
"""

import argparse
import json
import threading
import time
import os
import re
import cv2
import numpy as np
import torch
import requests
from ultralytics import YOLO
import easyocr

ocr = None  # diinisialisasi belakangan di main(), setelah kita tau mau pakai GPU atau CPU

# ---- Integrasi API update slot parkir (api-parkir.xorbit.my.id) ----
DEFAULT_SLOT_API_URL = "https://api-parkir.xorbit.my.id/api/ml/update-slot"
DEFAULT_SLOTS_LIST_URL = "https://api-parkir.xorbit.my.id/api/slots"

# regex nomor slot: 1-3 huruf diikuti 1-4 angka, mis. "P1", "P12", "A3".
# Spasi/strip opsional di antara huruf & angka ditoleransi (mis. "P 1", "P-1"),
# karena hasil OCR real-world kadang nyisipin whitespace/dash gitu.
SLOT_NUMBER_RE = re.compile(r"\b([A-Za-z])[\s\-]?(\d{1,4})\b")

# cache state terakhir yang udah dikirim ke API per slot_number, biar gak nge-spam
# request yang sama berulang-ulang tiap frame di mode kamera live.
_slot_api_last_state = {}
_slot_api_lock = threading.Lock()

# ---- Validasi slot_number ke daftar slot resmi (GET /api/slots) ----
# Sebelum kirim update, dicek dulu apakah slot_number itu BENERAN terdaftar di
# API (mis. hasil OCR salah baca jadi nomor yang gak ada, atau modelnya ngasal
# nebak di luar area parkiran valid). Kalau gak ketemu di daftar, gak dikirim.
# Daftar di-cache & di-refresh berkala (--api-slots-refresh) biar gak nge-GET
# ulang tiap kali mau kirim satu update.
_valid_slots_cache = None  # set slot_number (huruf besar), None = belum/gagal dimuat
_valid_slots_cache_time = 0.0
_valid_slots_lock = threading.Lock()

# ---- Minimum hit sebelum kirim ke API ----
# Di mode kamera live, satu frame kebaca salah (mis. OCR sesaat salah baca "P1"
# jadi "P7", atau tebakan posisi meleset) gak langsung dipercaya & dikirim ke API.
# Status (slot_number, is_available) harus konsisten kebaca N kali BERTURUT-TURUT
# dulu (--api-min-hits) baru beneran dikirim -> nyaring kedipan/salah baca sesaat,
# sama semangatnya kayak --smooth-min-hits buat box deteksi.
_slot_hit_state = {}  # {slot_number: {"pending_available": int, "hits": int}}
_slot_hit_lock = threading.Lock()


def extract_slot_number(text):
    """Cari pola nomor slot (1 huruf + angka, mis. 'P1', 'P12', 'A3') di dalam teks
    hasil OCR. Return nomor slot dalam huruf besar kalau ketemu, None kalau tidak ada.

    Sengaja dibatasi CUMA 1 huruf di depan (bukan 1-3 huruf) -- kalau dibiarkan
    sampai 3 huruf, token OCR yang ke-gabung gak sengaja (mis. ikon simbol 'P'
    parkir yang nempel/berdekatan sama label 'P3' kebaca EasyOCR jadi satu token
    literal 'PP3') bisa salah ke-parse jadi slot 'PP3' (padahal maksudnya 'P3'),
    yang ujung-ujungnya bikin slot itu ditolak terus sama validasi --api-slots-url
    (karena 'PP3' emang gak pernah kedaftar) dan nge-spam log 'gak terdaftar'.
    Dengan cuma 1 huruf, kasus begini balikin None (dilewatin) dan bukan
    slot_number yang salah -- masih ada kesempatan kebaca benar di frame lain.
    """
    if not text:
        return None
    match = SLOT_NUMBER_RE.search(text.upper())
    return f"{match.group(1)}{match.group(2)}" if match else None


def _slot_hit_confirmed(slot_number, is_available, min_hits):
    """Update hit-counter buat slot_number ini berdasarkan state (is_available)
    yang baru kebaca. Return True begitu state itu udah konsisten kebaca
    >= min_hits kali BERTURUT-TURUT (siap dikirim ke API); begitu state-nya
    beda dari yang lagi 'dihitung', counter direset ulang dari 1."""
    with _slot_hit_lock:
        state = _slot_hit_state.get(slot_number)
        if state is None or state["pending_available"] != is_available:
            state = {"pending_available": is_available, "hits": 1}
        else:
            state["hits"] += 1
        _slot_hit_state[slot_number] = state
        return state["hits"] >= min_hits


def fetch_slots_full(api_slots_url, timeout):
    """GET daftar slot resmi dari API (mis. https://api-parkir.xorbit.my.id/api/slots)
    & return RAW list of dict apa adanya (satu dict per slot, biasanya berisi
    minimal 'slot_number', kadang juga field status kayak 'is_available' kalau
    API-nya nyertain). Return None kalau gagal (network error, status bukan 2xx,
    atau bukan list)."""
    try:
        resp = requests.get(api_slots_url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            print(f"[API] Format response dari {api_slots_url} gak sesuai dugaan (bukan list).")
            return None
        return data
    except Exception as e:
        print(f"[API] Gagal ambil daftar slot resmi dari {api_slots_url}: {e}")
        return None


def fetch_valid_slots(api_slots_url, timeout):
    """GET daftar slot resmi dari API, return set slot_number (huruf besar) kalau
    sukses, None kalau gagal."""
    data = fetch_slots_full(api_slots_url, timeout)
    if data is None:
        return None
    return {
        str(item["slot_number"]).strip().upper()
        for item in data
        if isinstance(item, dict) and item.get("slot_number")
    }


def _parse_slots_status(data, status_field):
    """Dari raw list hasil GET --api-slots-url, bikin dict {slot_number: is_available}
    buat tiap slot yang punya field status_field (default 'is_available') keisi
    angka valid. Slot yang gak punya field itu (mis. API-nya cuma nyertain
    slot_number tanpa status) gak dimasukin -- gak bisa dibandingin kalau gak
    ada datanya."""
    status = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        slot_number = item.get("slot_number")
        if not slot_number:
            continue
        raw_val = item.get(status_field)
        if raw_val is None:
            continue
        try:
            status[str(slot_number).strip().upper()] = int(raw_val)
        except (TypeError, ValueError):
            continue
    return status


def reconcile_slot_state(args, slot_number, expected_available, actual_available):
    """Ketahuan slot_number ini statusnya BEDA di DB (actual_available, hasil GET
    --api-slots-url) dibanding status yang kita yakini bener dari deteksi kamera
    kita sendiri (expected_available -- state terakhir yang berhasil KITA kirim).
    Ini nandain kemungkinan ada pihak ketiga yang ubah data itu langsung di DB,
    di luar alur update otomatis kita -- makanya ditimpa balik ke status yang
    bener dengan ngirim ulang ke API."""
    print(f"[API] Slot '{slot_number}' -> di DB tercatat is_available={actual_available}, "
          f"tapi menurut deteksi kamera kita seharusnya is_available={expected_available} "
          f"(kemungkinan diubah pihak ketiga). Ngirim ulang status yang bener ke API...")
    threading.Thread(
        target=_post_slot_update,
        args=(args.api_url, slot_number, expected_available, args.api_timeout),
        daemon=True,
    ).start()


def slots_api_watcher(args, stop_event=None):
    """Thread background yang ngecek daftar slot resmi (GET --api-slots-url) tiap
    --api-slots-poll-interval detik (default 30), TERUS-MENERUS selama script
    jalan (gak nunggu ada slot_number yang mau dikirim kayak get_valid_slots).

    Begitu daftar berhasil diambil (API "ada"/reachable), langsung update cache
    (_valid_slots_cache) & kasih notifikasi -- tapi CUMA sekali pas transisi dari
    'gak tersedia' ke 'tersedia', bukan tiap 30 detik biar gak nge-spam log kalau
    statusnya emang gak berubah-ubah. Kalau tiba-tiba jadi gak bisa diakses lagi
    (misal API-nya down), juga dikasih tau sekali pas transisi itu terjadi.

    Rekonsiliasi status (--api-enable): tiap kali cek, status is_available per
    slot dari DB (kalau API-nya nyertain field --api-status-field, default
    'is_available') dibandingin ke state terakhir yang KITA kirim sendiri
    (_slot_api_last_state -- ini yang kita anggap sesuai kondisi real di
    parkiran, soalnya udah lolos min-hit confirmation dari deteksi kamera).
    Kalau beda -> kemungkinan ada pihak ketiga yang ubah manual di DB, jadi
    langsung ditimpa ulang (kirim lagi) balik ke status yang bener.
    """
    global _valid_slots_cache, _valid_slots_cache_time
    was_available = False
    status_field = getattr(args, "api_status_field", "is_available")
    reconcile = getattr(args, "api_enable", False) and not getattr(args, "api_no_reconcile", False)

    while stop_event is None or not stop_event.is_set():
        data = fetch_slots_full(args.api_slots_url, args.api_timeout)
        now = time.time()
        if data is not None:
            fetched_valid = {
                str(item["slot_number"]).strip().upper()
                for item in data
                if isinstance(item, dict) and item.get("slot_number")
            }
            with _valid_slots_lock:
                _valid_slots_cache = fetched_valid
                _valid_slots_cache_time = now
            if not was_available:
                print(f"[API] Daftar slot resmi dari {args.api_slots_url} sudah ADA "
                      f"({len(fetched_valid)} slot terdaftar): {sorted(fetched_valid)}")
            was_available = True

            if reconcile:
                fetched_status = _parse_slots_status(data, status_field)
                if fetched_status:
                    with _slot_api_lock:
                        expected_snapshot = dict(_slot_api_last_state)
                    for slot_number, expected in expected_snapshot.items():
                        actual = fetched_status.get(slot_number)
                        if actual is not None and actual != expected:
                            reconcile_slot_state(args, slot_number, expected, actual)
        else:
            if was_available:
                print(f"[API] Daftar slot resmi dari {args.api_slots_url} gak bisa "
                      f"diakses lagi, bakal terus dicoba tiap {args.api_slots_poll_interval:.0f} detik.")
            was_available = False

        if stop_event is not None and stop_event.wait(args.api_slots_poll_interval):
            break
        elif stop_event is None:
            time.sleep(args.api_slots_poll_interval)


def get_valid_slots(args, force_refresh=False):
    """Return set slot_number yang resmi terdaftar di API (buat validasi sebelum
    kirim update), dengan cache biar gak nge-GET ulang tiap mau kirim satu update.
    Auto-refresh tiap --api-slots-refresh detik (default 300).

    Return None kalau daftar belum pernah berhasil dimuat sama sekali (mis. API
    lagi down pas startup) -> caller (notify_slot_api) menganggap ini 'validasi
    gak tersedia' dan tetap ngirim tanpa filter, biar update gak macet total
    cuma gara-gara satu request GET gagal.
    """
    global _valid_slots_cache, _valid_slots_cache_time
    now = time.time()

    with _valid_slots_lock:
        stale = (now - _valid_slots_cache_time) >= max(1.0, args.api_slots_refresh)
        need_refresh = force_refresh or _valid_slots_cache is None or stale

    if not need_refresh:
        return _valid_slots_cache

    fetched = fetch_valid_slots(args.api_slots_url, args.api_timeout)
    with _valid_slots_lock:
        # timestamp selalu diupdate (sukses ATAU gagal), biar percobaan ulang yang
        # gagal beruntun juga di-throttle sesuai --api-slots-refresh -- gak nge-GET
        # ulang tiap kali mau kirim satu update kalau API-nya lagi down.
        _valid_slots_cache_time = now
        if fetched is not None:
            _valid_slots_cache = fetched
        return _valid_slots_cache


def _post_slot_update(api_url, slot_number, is_available, timeout):
    """Kirim POST request ke API update slot. Dijalankan di background thread
    (lewat notify_slot_api) supaya gak bikin video/deteksi nge-lag nunggu network."""
    try:
        resp = requests.post(
            api_url,
            json={"slot_number": slot_number, "is_available": is_available},
            timeout=timeout,
        )
        if resp.ok:
            print(f"[API] Slot '{slot_number}' -> is_available={is_available} (OK, status {resp.status_code})")
        else:
            print(f"[API] Slot '{slot_number}' -> is_available={is_available} GAGAL "
                  f"(status {resp.status_code}: {resp.text[:200]})")
    except requests.RequestException as e:
        print(f"[API] Gagal mengirim update slot '{slot_number}': {e}")


def notify_slot_api(args, slot_number, is_available, min_hits=None):
    """Kirim update ke API kalau --api-enable aktif.

    min_hits: berapa kali state (is_available) ini harus konsisten kebaca
    BERTURUT-TURUT sebelum beneran dikirim (default: ambil dari --api-min-hits).
    Mode gambar (--source) selalu pakai min_hits=1 secara eksplisit di caller-nya,
    karena cuma ada satu observasi sama sekali (gak ada 'frame berikutnya' buat
    dikonfirmasi ulang).

    Dedup: setelah lolos min-hit, tetap cuma kirim kalau state-nya beda dari
    yang terakhir BENERAN dikirim, biar gak spam request yang sama berulang-ulang.
    """
    if not getattr(args, "api_enable", False) or not slot_number:
        return

    # validasi slot_number ke daftar resmi (GET /api/slots) -> kalau gak kedaftar
    # sama sekali di sana, gak usah dikirim (kemungkinan hasil OCR salah baca).
    # Bisa dimatiin lewat --api-skip-slots-check.
    if not getattr(args, "api_skip_slots_check", False):
        valid_slots = get_valid_slots(args)
        if valid_slots is not None and slot_number not in valid_slots:
            print(f"[API] Slot '{slot_number}' gak terdaftar di {args.api_slots_url}, dilewatin (gak dikirim).")
            return

    if min_hits is None:
        min_hits = getattr(args, "api_min_hits", 1)
    min_hits = max(1, min_hits)

    debug = getattr(args, "api_debug", False)

    if not _slot_hit_confirmed(slot_number, is_available, min_hits):
        if debug:
            state = _slot_hit_state.get(slot_number, {})
            print(f"[API-DEBUG] Slot '{slot_number}' -> is_available={is_available} "
                  f"belum konsisten ({state.get('hits', '?')}/{min_hits} hit), belum dikirim.")
        return  # belum konsisten kebaca cukup banyak kali, tunggu dulu

    with _slot_api_lock:
        last = _slot_api_last_state.get(slot_number)
        if last == is_available:
            if debug:
                print(f"[API-DEBUG] Slot '{slot_number}' -> is_available={is_available} "
                      f"sama kayak state terakhir yang dikirim, di-skip (dedup).")
            return  # udah pernah dikirim state yang sama, skip
        _slot_api_last_state[slot_number] = is_available

    threading.Thread(
        target=_post_slot_update,
        args=(args.api_url, slot_number, is_available, args.api_timeout),
        daemon=True,
    ).start()


def slot_is_available_for_class(cls_name):
    """Tentuin nilai is_available berdasarkan class hasil deteksi:
    - 'free'  (slot kosong, nomornya kebaca jelas lewat OCR) -> 1 (tersedia)
    - selain itu (mis. 'car', nomor ketutupan mobil) -> 0 (terisi)
    """
    return 1 if cls_name == "free" else 0


# ---- "Slot position memory" ----
# Nomor slot cuma kebaca OCR pas slotnya KOSONG (class 'free'); begitu ada mobil
# nutupin (class 'car'), OCR di-skip -> gak ada nomor sama sekali.
# Supaya slot yang lagi keisi mobil tetap bisa dikirim update is_available=0 ke API
# (bukan cuma didiemin gitu aja), kita inget posisi (x,y) tiap nomor slot yang
# PERNAH kebaca, lalu kalau nanti muncul deteksi mobil di posisi yang sama,
# kita anggap itu slot yang sama juga. Ini valid selama kameranya statis (CCTV
# parkir gak digeser-geser), yang memang kondisi normal buat kasus ini.
SLOT_POSITION_TOLERANCE = 30  # px, toleransi jarak fallback (dipakai kalau containment gagal)
# Nilai ini relatif terhadap resolusi/imgsz yang dipakai -- makin kecil imgsz
# (mis. 320), makin kecil juga toleransi px yang masuk akal, karena satu slot
# cuma ditempatin sedikit pixel. Bisa di-override lewat --slot-position-tolerance.

_slot_position_memory = {}  # {slot_number: (center_x, center_y)}
_slot_position_lock = threading.Lock()


def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def remember_slot_position(box, slot_number):
    """Simpan/update posisi terakhir slot_number ini di frame."""
    with _slot_position_lock:
        _slot_position_memory[slot_number] = _box_center(box)


def find_slot_by_position(box, tolerance=SLOT_POSITION_TOLERANCE):
    """Cari slot_number yang posisinya paling cocok sama box ini. Dicek 2 tahap:
    1. Containment: kalau titik tengah slot yang diingat ADA DI DALAM box ini,
       langsung dianggap cocok (gak peduli tolerance) -- ini nolong kasus box
       'car' yang jauh lebih besar/bergeser dari box 'free' aslinya (mobil gak
       selalu pas presisi di tengah slot), selama posisi nomor slot yang pernah
       kebaca masih ketutupan badan mobil itu.
    2. Fallback: kalau gak ada yang containment, baru dicoba jarak titik tengah
       (dalam radius tolerance px) -- buat kasus box yang gak nutupin posisi
       nomor asli tapi masih deket (mis. box 'free' kecil di dekat box 'car').
    Return None kalau gak ada yang cocok (mis. mobil baru parkir di slot yang
    nomornya belum pernah kebaca sama sekali sebelumnya).
    """
    x1, y1, x2, y2 = box
    cx, cy = _box_center(box)
    with _slot_position_lock:
        for slot_number, (sx, sy) in _slot_position_memory.items():
            if x1 <= sx <= x2 and y1 <= sy <= y2:
                return slot_number

        best_slot, best_dist = None, tolerance
        for slot_number, (sx, sy) in _slot_position_memory.items():
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if dist <= best_dist:
                best_slot, best_dist = slot_number, dist
    return best_slot


def resolve_slot_and_availability(det, position_tolerance=SLOT_POSITION_TOLERANCE):
    """Gabungin OCR + slot position memory buat nentuin (slot_number, is_available)
    dari satu hasil deteksi. Return (None, None) kalau slot_number-nya sendiri gak
    bisa ditentuin sama sekali (mis. mobil parkir di posisi yang nomornya belum
    pernah kebaca sebelumnya -- gak ada cara tau itu slot nomor berapa).

    position_tolerance: toleransi px buat fallback pencocokan posisi (lihat
    find_slot_by_position), diambil dari --slot-position-tolerance.

    Status (is_available) SELALU ditentukan dari class hasil deteksi saat ini
    (slot_is_available_for_class), bukan dari berhasil/gagalnya OCR. OCR cuma
    dipakai buat nentuin NOMOR slotnya (slot_number) -- kalau OCR gagal baca di
    frame ini, nomornya ditebak dari posisi yang pernah dikenal sebelumnya
    (find_slot_by_position), tapi class saat ini (free/car) tetap yang dipakai
    buat status, jadi class 'free' yang OCR-nya sesaat gagal baca tetap dikirim
    is_available=1, bukan otomatis dianggap terisi.
    """
    cls_name = det["cls_name"]
    slot_number = extract_slot_number(det.get("ocr_text", ""))

    if slot_number:
        # nomor kebaca langsung -> update posisi memori + pakai class buat nentuin status
        remember_slot_position(det["box"], slot_number)
        return slot_number, slot_is_available_for_class(cls_name)

    # nomor gak kebaca sama sekali di frame ini (ketutupan mobil, ATAU class 'free'
    # tapi OCR gagal baca) -> coba tebak identitas slotnya dari posisi yang pernah
    # dikenal sebelumnya, status tetap ikut class saat ini (bukan dipaksa 0)
    remembered = find_slot_by_position(det["box"], tolerance=position_tolerance)
    if remembered:
        return remembered, slot_is_available_for_class(cls_name)

    return None, None


# ---- Slot polygon manual (--slots, hasil label_slots.py) ----
# Alternatif buat resolve_slot_and_availability di atas: identitas tiap slot fisik
# didefinisikan MANUAL sekali di depan (gambar polygon + label pakai label_slots.py),
# bukan ditebak dari OCR/posisi yang baru "dipelajari" sambil jalan. Lebih akurat &
# gak ada cold-start problem (mis. mobil parkir duluan di slot yang nomornya belum
# pernah kebaca OCR sama sekali) -- karena posisi semua slot udah pasti dari awal.
def load_slots(path):
    """Load daftar slot dari file JSON hasil label_slots.py.
    Return list of dict {"label": str, "points": np.ndarray shape (N,1,2) int32}
    (format points ini siap dipakai langsung ke cv2.pointPolygonTest)."""
    with open(path, "r") as f:
        data = json.load(f)
    slots = []
    for s in data.get("slots", []):
        pts = np.array(s["points"], dtype=np.int32).reshape((-1, 1, 2))
        slots.append({"label": s["label"], "points": pts})
    if not slots:
        raise ValueError(f"File --slots '{path}' tidak berisi slot apapun.")
    return slots


def resolve_slot_from_position(det, slots):
    """Cocokkan titik tengah box hasil deteksi ke salah satu polygon slot dari
    --slots (dibikin manual pakai label_slots.py, BUKAN OCR/tebakan otomatis).

    Slot pertama yang polygon-nya mengandung titik tengah box ini dianggap
    identitas slotnya. Karena polygon-nya didefinisikan manual sesuai posisi
    kamera statis, ini valid buat SEMUA class (baik 'free' maupun 'car', atau
    class custom apapun) -- gak perlu OCR ataupun position-memory otomatis lagi.

    Return (None, None) kalau titik tengah box-nya gak ada di dalam polygon
    slot manapun (mis. deteksi di luar semua area slot yang didefinisikan).
    """
    cx, cy = _box_center(det["box"])
    for slot in slots:
        if cv2.pointPolygonTest(slot["points"], (cx, cy), False) >= 0:
            return slot["label"], slot_is_available_for_class(det["cls_name"])
    return None, None


def draw_slots(frame, slots):
    """Gambar overlay tipis polygon slot dari --slots (buat verifikasi visual posisi
    slot udah bener/pas), sebelum box hasil deteksi digambar di atasnya."""
    annotated = frame.copy()
    for slot in slots:
        cv2.polylines(annotated, [slot["points"]], isClosed=True, color=(180, 180, 180), thickness=1)
        cx = int(slot["points"][:, 0, 0].mean())
        cy = int(slot["points"][:, 0, 1].mean())
        cv2.putText(annotated, slot["label"], (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return annotated


def resolve_device(device_arg):
    """Tentuin device final buat YOLO: 'auto' -> pilih GPU kalau ada, kalau gak ada jatuh ke CPU."""
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU terdeteksi: {gpu_name}")
        return "0"  # index GPU pertama
    print("Peringatan: GPU (CUDA) tidak terdeteksi, YOLO akan jalan di CPU.")
    return "cpu"


def init_ocr(use_gpu):
    """Inisialisasi EasyOCR. Jauh lebih akurat dibanding RapidOCR buat teks kecil/
    miring/pencahayaan kurang rata (recognizer-nya lebih robust), dengan harga
    model agak lebih berat & init pertama kali lebih lama (download model sekali,
    lalu ke-cache lokal). GPU otomatis dipakai kalau use_gpu=True dan CUDA kedetek
    lewat torch; kalau gagal, EasyOCR sendiri yang fallback ke CPU (gak perlu
    try/except manual kayak versi RapidOCR sebelumnya).
    """
    print(f"Memuat model EasyOCR ({'GPU' if use_gpu else 'CPU'})... "
          f"(unduhan model pertama kali bisa agak lama)")
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    print(f"EasyOCR siap ({'GPU' if use_gpu else 'CPU'}).")
    return reader


# Karakter yang diizinkan buat OCR: huruf + angka aja (nomor slot parkir kayak
# "P1", "A12" gak pernah pakai simbol/tanda baca). Ngasih allowlist ke EasyOCR
# ini SANGAT membantu akurasi -> recognizer gak lagi kebingungan/ke-distract
# nebak karakter aneh-aneh (titik, koma, garis bayangan, dll) yang sering muncul
# di crop kamera CCTV, jadi hasil bacanya jauh lebih bersih & konsisten.
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# confidence minimum dari EasyOCR biar teks "ngasal"/noise gak ikut kepakai
OCR_MIN_CONFIDENCE = 0.35

# target ukuran sisi terpendek crop sebelum di-OCR (px). Crop hasil bounding box
# YOLO sering kecil banget (teks cuma beberapa piksel tinggi) -> di-upscale dulu
# biar recognizer punya cukup detail buat baca, ini ningkatin akurasi signifikan
# dibanding OCR langsung ke crop mentah yang kecil.
OCR_UPSCALE_TARGET_PX = 120


def _preprocess_for_ocr(crop):
    """Upscale (kalau crop-nya kecil) + naikin kontras (CLAHE) sebelum OCR, biar
    teks kecil/buram/kurang kontras di kamera CCTV lebih gampang kebaca."""
    h, w = crop.shape[:2]
    short_side = min(h, w)
    if short_side > 0 and short_side < OCR_UPSCALE_TARGET_PX:
        scale = OCR_UPSCALE_TARGET_PX / short_side
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_CUBIC)
    return enhance_contrast(crop, clip_limit=2.0)


def run_ocr(crop):
    """Jalanin OCR pada crop gambar, return teks gabungan (kosong kalau gak ada teks kedeteksi).
    Crop di-preprocess dulu (upscale + kontras) dan hasil OCR difilter pakai
    OCR_MIN_CONFIDENCE biar noise/bacaan ngasal gak ikut kepakai.
    """
    if crop is None or crop.size == 0:
        return ""

    prep = _preprocess_for_ocr(crop)
    results = ocr.readtext(prep, allowlist=OCR_ALLOWLIST, detail=1, paragraph=False)
    if not results:
        return ""

    found_texts = []
    for (_bbox, text, conf) in results:
        if conf < OCR_MIN_CONFIDENCE:
            continue
        print(f"  Teks terdeteksi: {text} (conf OCR: {conf:.2f})")
        found_texts.append(text)
    return " ".join(found_texts)


class AsyncOcrWorker:
    """Jalanin OCR di background thread biar video kamera gak nge-freeze nunggu OCR selesai.

    Cara pakai: submit(crops) buat ngirim job baru (job lama yang belum sempat diproses
    otomatis dibuang, cuma job terbaru yang diproses). get_latest_result() ambil hasil OCR
    terakhir yang udah selesai diproses (non-blocking, gak nunggu apapun).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None  # job terbaru yang nunggu diproses: list (x1,y1,x2,y2,crop,cls_name)
        self._latest_result = []  # hasil OCR terakhir yang sudah jadi
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def submit(self, jobs):
        """Kirim job baru buat diproses. Kalau worker masih sibuk, job lama diganti job baru ini."""
        with self._lock:
            self._pending = jobs

    def get_latest_result(self):
        with self._lock:
            return self._latest_result

    def is_busy(self):
        with self._lock:
            return self._busy

    def stop(self):
        self._stop = True

    def _worker_loop(self):
        while not self._stop:
            with self._lock:
                jobs = self._pending
                self._pending = None
            if jobs is None:
                threading.Event().wait(0.01)  # gak ada job, tunggu sebentar
                continue

            with self._lock:
                self._busy = True

            result = []
            for (x1, y1, x2, y2, crop, cls_name) in jobs:
                detected_text = ""
                if cls_name == "car":
                    pass  # skip OCR buat class mobil
                elif crop.size == 0:
                    pass  # crop kosong, skip OCR
                else:
                    detected_text = run_ocr(crop)
                result.append((x1, y1, x2, y2, detected_text))

            with self._lock:
                self._latest_result = result
                self._busy = False


NAMED_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
}


def parse_color(color_str):
    """Parse warna dari nama (mis. 'red') atau hex (#RRGGBB / RRGGBB) jadi tuple BGR buat OpenCV."""
    if not color_str:
        return None
    color_str = color_str.strip().lower()

    if color_str in NAMED_COLORS:
        return NAMED_COLORS[color_str]

    hex_str = color_str.lstrip("#")
    if len(hex_str) == 6:
        try:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return (b, g, r)  # OpenCV pakai urutan BGR
        except ValueError:
            pass

    raise ValueError(
        f"Format warna salah: '{color_str}'. Gunakan nama warna ({', '.join(NAMED_COLORS.keys())}) "
        f"atau kode hex (mis. #00FF00)."
    )


def parse_class_colors(class_color_str):
    """Parse string 'car:red,free:#00FF00' jadi dict {nama_class: (b,g,r)}."""
    if not class_color_str:
        return {}
    mapping = {}
    for pair in class_color_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Format --class-colors salah di bagian '{pair}'. "
                f"Gunakan format nama_class:warna, mis. 'car:red,free:#00FF00'"
            )
        cls_name, color_str = pair.split(":", 1)
        mapping[cls_name.strip()] = parse_color(color_str.strip())
    return mapping


def _polygon_to_bbox(points):
    """Bikin bounding box lurus (x1,y1,x2,y2) yang ngelingkupin sekumpulan titik polygon.
    Tetap dibutuhkan walau kotaknya udah polygon/miring, karena beberapa bagian lain
    (crop OCR, IoU matching smoothing, dll) masih kerja pakai box lurus biasa."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def extract_detections(result, class_names):
    """Ubah hasil deteksi YOLO jadi list of dict sederhana:
    {box:(x1,y1,x2,y2), poly, cls_id, cls_name, conf}. Dipakai supaya fungsi lain
    (draw_detections, collect_detections, DetectionSmoother) gak perlu tau
    detail internal objek Ultralytics.

    'box' selalu berupa bounding box lurus (dipakai buat crop OCR & smoothing).
    'poly' berisi titik-titik polygon kalau modelnya mendukung bentuk yang gak
    lurus/axis-aligned (None kalau model deteksi biasa):
      - Model OBB (Oriented Bounding Box)  -> poly = 4 titik sudut kotak miring,
        ngikutin orientasi objek asli (cocok buat parkiran yang serong di kamera).
      - Model segmentasi (mask)            -> poly = kontur bentuk objek asli
        (disederhanakan), ngikutin bentuk nyata petak parkir/objeknya.
      - Model deteksi biasa (xyxy)         -> poly = None, tetap kotak lurus.
    """
    dets = []

    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        for i in range(len(obb)):
            pts = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
            poly = [(int(x), int(y)) for x, y in pts]
            cls_id = int(obb.cls[i])
            conf = float(obb.conf[i])
            dets.append({
                "box": _polygon_to_bbox(poly), "poly": poly, "cls_id": cls_id,
                "cls_name": class_names[cls_id], "conf": conf, "ocr_text": "",
            })
        return dets

    masks = getattr(result, "masks", None)
    boxes = result.boxes
    if masks is not None and boxes is not None and len(masks) > 0:
        for i in range(len(masks)):
            mask_pts = masks.xy[i]  # polygon mentah hasil mask, shape (N, 2)
            poly = None
            if mask_pts is not None and len(mask_pts) >= 3:
                contour = mask_pts.reshape(-1, 1, 2).astype("float32")
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
                poly = [(int(p[0][0]), int(p[0][1])) for p in approx]
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            dets.append({
                "box": (x1, y1, x2, y2), "poly": poly, "cls_id": cls_id,
                "cls_name": class_names[cls_id], "conf": conf, "ocr_text": "",
            })
        return dets

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        dets.append({
            "box": (x1, y1, x2, y2), "poly": None, "cls_id": cls_id,
            "cls_name": class_names[cls_id], "conf": conf, "ocr_text": "",
        })
    return dets


def _draw_text_high_contrast(img, text, pos, font_scale, color, thickness):
    """Gambar teks dengan outline hitam tebal di belakangnya, biar kontras tinggi
    dan tetap kebaca jelas di background apapun (terang, gelap, atau rame kayak
    plat nomor/aspal). Teknik ini umum dipakai buat overlay video real-time.
    """
    outline_thickness = thickness + 3
    # outline hitam dulu (digambar 4x sedikit offset biar solid, bukan cuma tipis)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), outline_thickness, cv2.LINE_AA)
    # baru teks warna aslinya di atas outline
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA)


def draw_detections(frame, detections, box_color, text_color, boxes_only=False,
                     class_colors=None, class_text_colors=None,
                     ocr_color=None, class_ocr_colors=None):
    """Gambar bounding box (+ label class/conf & teks OCR kalau ada) pakai warna custom.

    detections: list of dict {box:(x1,y1,x2,y2), cls_name, conf, ocr_text}
    (lihat extract_detections / DetectionSmoother.update untuk bentuk datanya)

    class_colors / class_text_colors: dict {nama_class: (b,g,r)} buat override warna
    kotak & label class/conf per class. Kalau class gak ada di dict, jatuh balik ke
    box_color / text_color default.

    ocr_color: warna default buat teks hasil OCR (jatuh balik ke text_color kalau None).
    class_ocr_colors: dict {nama_class: (b,g,r)} buat override warna teks OCR per class.
    """
    class_colors = class_colors or {}
    class_text_colors = class_text_colors or {}
    class_ocr_colors = class_ocr_colors or {}
    base_ocr_color = ocr_color if ocr_color is not None else text_color

    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        poly = det.get("poly")
        cls_name = det["cls_name"]
        conf = det["conf"]
        ocr_text = det.get("ocr_text", "")

        this_box_color = class_colors.get(cls_name, box_color)
        this_text_color = class_text_colors.get(cls_name, text_color)
        this_ocr_color = class_ocr_colors.get(cls_name, base_ocr_color)

        if poly:
            # kotak ngikutin bentuk asli (miring/serong sesuai OBB, atau kontur mask)
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=this_box_color, thickness=3)
        else:
            # model deteksi biasa -> tetap kotak lurus (axis-aligned)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), this_box_color, 3)

        if not boxes_only:
            label = f"{cls_name} {conf:.2f}"

            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2

            label_font_scale = 0.45
            ocr_font_scale = 0.5
            thickness = 1

            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, thickness
            )
            if ocr_text:
                (ocr_w, ocr_h), _ = cv2.getTextSize(
                    ocr_text, cv2.FONT_HERSHEY_SIMPLEX, ocr_font_scale, thickness
                )
            else:
                ocr_w, ocr_h = 0, 0

            gap = 6 if ocr_text else 0
            total_h = label_h + gap + ocr_h

            label_baseline_y = box_cy - total_h // 2 + label_h
            label_pos = (box_cx - label_w // 2, label_baseline_y)
            _draw_text_high_contrast(
                annotated, label, label_pos, label_font_scale, this_text_color, thickness,
            )

            if ocr_text:
                ocr_baseline_y = label_baseline_y + gap + ocr_h
                ocr_pos = (box_cx - ocr_w // 2, ocr_baseline_y)
                _draw_text_high_contrast(
                    annotated, ocr_text, ocr_pos, ocr_font_scale, this_ocr_color, thickness,
                )

    return annotated


def _iou(box_a, box_b):
    """Hitung IoU (Intersection over Union) antara 2 box (x1,y1,x2,y2)."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class DetectionSmoother:
    """Bikin deteksi tampil mulus, gak kedip-kedip (muncul-hilang-muncul) DAN gak
    lompat-lompat posisinya tiap frame, walau confidence-nya naik-turun tipis di
    sekitar threshold.

    Tiga mekanisme:
    1. Confirmation (min_hits): box BARU (belum pernah dilacak sebelumnya) gak langsung
       ditampilkan. Harus konsisten kedeteksi minimal min_hits frame berturut-turut dulu
       baru dianggap 'valid' dan mulai ditampilkan. Ini yang bikin deteksi 'kedip sekali
       terus ilang lagi' (noise/false positive sesaat) gak pernah nongol sama sekali di layar.
    2. Persistence: kalau suatu objek yang SUDAH confirmed TIDAK terdeteksi lagi di
       frame ini, box-nya TETAP ditampilkan (pakai posisi & confidence terakhir) selama
       beberapa frame (grace period), sebelum akhirnya beneran dihapus.
    3. Position smoothing (EMA): posisi box gak langsung "lompat" ke koordinat baru,
       tapi bergerak halus dari posisi lama menuju posisi baru (exponential moving
       average). alpha makin kecil = makin halus tapi makin "lambat" ngikutin gerakan
       objek asli; alpha=1.0 = gak ada smoothing posisi sama sekali.
    """

    def __init__(self, iou_thresh=0.3, max_missed_frames=10, alpha=0.5, min_hits=3):
        self.iou_thresh = iou_thresh
        self.max_missed_frames = max_missed_frames
        self.alpha = alpha
        self.min_hits = min_hits
        self._tracks = []  # list of dict: {box, cls_id, cls_name, conf, ocr_text, missed, hits}

    def update(self, current_boxes):
        """current_boxes: list of dict {box:(x1,y1,x2,y2), cls_id, cls_name, conf, ocr_text}
        Return: list dengan format sama, HANYA berisi track yang sudah 'confirmed'
        (hits >= min_hits) -- box yang baru muncul dan belum stabil gak ikut ditampilkan."""
        matched_track_idx = set()
        a = self.alpha

        for det in current_boxes:
            best_iou, best_idx = 0.0, -1
            for i, track in enumerate(self._tracks):
                if i in matched_track_idx:
                    continue
                if track["cls_id"] != det["cls_id"]:
                    continue  # cuma cocokin ke track dengan class yang sama
                iou = _iou(det["box"], track["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_idx != -1 and best_iou >= self.iou_thresh:
                # ketemu track lama yang cocok -> posisi baru di-blend halus (EMA) dengan
                # posisi lama, bukan langsung ditimpa, biar box gak lompat-lompat
                track = self._tracks[best_idx]
                ox1, oy1, ox2, oy2 = track["box"]
                nx1, ny1, nx2, ny2 = det["box"]
                smoothed_box = (
                    int(a * nx1 + (1 - a) * ox1), int(a * ny1 + (1 - a) * oy1),
                    int(a * nx2 + (1 - a) * ox2), int(a * ny2 + (1 - a) * oy2),
                )
                track["box"] = smoothed_box
                # poly (kotak miring/kontur) dipakai apa adanya dari deteksi terbaru
                # (gak di-EMA seperti box lurus, karena jumlah titiknya bisa beda tiap frame)
                track["poly"] = det.get("poly")
                # confidence juga di-smooth dikit biar angkanya gak kedip-kedip di label
                track["conf"] = a * det["conf"] + (1 - a) * track["conf"]
                if det.get("ocr_text"):
                    track["ocr_text"] = det["ocr_text"]  # OCR text baru gak ditimpa kosong
                track["missed"] = 0
                track["hits"] = min(track["hits"] + 1, self.min_hits)  # cap di min_hits, gak perlu naik terus
                matched_track_idx.add(best_idx)
            else:
                # objek baru, belum pernah dilacak -> bikin track baru, mulai dari hits=1
                # (belum confirmed, belum ditampilkan sampai kena update() min_hits kali berturut-turut)
                self._tracks.append({
                    "box": det["box"], "poly": det.get("poly"), "cls_id": det["cls_id"], "cls_name": det["cls_name"],
                    "conf": det["conf"], "ocr_text": det.get("ocr_text", ""), "missed": 0, "hits": 1,
                })
                matched_track_idx.add(len(self._tracks) - 1)

        # track yang gak ke-match frame ini -> tambah hitungan 'missed', buang kalau kelamaan hilang.
        # Track yang BELUM confirmed (hits < min_hits) dan gak ke-match langsung dibuang saat itu juga
        # (gak dikasih grace period, karena belum terbukti stabil/valid dari awal).
        alive_tracks = []
        for i, track in enumerate(self._tracks):
            if i not in matched_track_idx:
                if track["hits"] < self.min_hits:
                    continue  # belum sempat confirmed, langsung hilang tanpa grace period
                track["missed"] += 1
                if track["missed"] > self.max_missed_frames:
                    continue  # sudah kelamaan hilang, buang beneran
            alive_tracks.append(track)
        self._tracks = alive_tracks

        return [
            {"box": t["box"], "poly": t.get("poly"), "cls_id": t["cls_id"], "cls_name": t["cls_name"],
             "conf": t["conf"], "ocr_text": t["ocr_text"]}
            for t in self._tracks
            if t["hits"] >= self.min_hits  # cuma tampilkan track yang sudah 'confirmed'/stabil
        ]


def filter_by_conf(result, min_conf):
    """Filter tambahan setelah inference: cuma pertahankan box dengan confidence >= min_conf,
    walaupun model dijalankan dengan --conf yang lebih rendah dari itu.

    Kenapa dipisah dari --conf (threshold model)? Supaya model tetap 'melihat' deteksi
    lemah (mis. conf 0.3-0.4) saat proses internal (NMS, dll), tapi yang ditampilkan/dipakai
    ke user cuma yang confidence-nya di atas min_conf ini.
    """
    if min_conf is None:
        return result

    obb = getattr(result, "obb", None)
    if obb is not None and len(obb) > 0:
        keep = obb.conf >= min_conf
        result.obb = obb[keep]
        return result

    keep = result.boxes.conf >= min_conf
    masks = getattr(result, "masks", None)
    if masks is not None:
        result.masks = masks[keep]
    result.boxes = result.boxes[keep]
    return result


def get_db_connection(args):
    """Buka koneksi ke MySQL. mysql-connector-python di-import di sini (lazy import),
    biar orang yang gak pakai fitur DB gak wajib install library ini.
    """
    import mysql.connector
    return mysql.connector.connect(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_password,
        database=args.db_name,
    )


def ensure_table(conn, table_name):
    """Bikin tabel kalau belum ada. Nama tabel divalidasi dulu (cuma huruf/angka/underscore)
    biar aman dari SQL injection, karena nama tabel gak bisa di-parameterize kayak value biasa.
    """
    if not re.match(r"^[A-Za-z0-9_]+$", table_name):
        raise ValueError(f"Nama tabel tidak valid: '{table_name}' (cuma boleh huruf/angka/underscore)")

    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            class_name VARCHAR(50) NOT NULL,
            confidence FLOAT NOT NULL,
            ocr_text VARCHAR(255) DEFAULT NULL,
            source VARCHAR(255) DEFAULT NULL,
            detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()


def collect_detections(detections):
    """Ubah list deteksi (dari extract_detections / DetectionSmoother) jadi format
    siap simpan ke database: list of dict {class_name, confidence, ocr_text}."""
    return [
        {"class_name": d["cls_name"], "confidence": d["conf"], "ocr_text": d.get("ocr_text", "")}
        for d in detections
    ]


def insert_detections(conn, table_name, detections, source, overwrite=True):
    """Simpan list hasil deteksi ke tabel MySQL.

    overwrite=True (default): tabel dikosongkan dulu (TRUNCATE, id ikut reset ke 1)
    sebelum insert data baru, jadi tabel selalu cuma berisi snapshot deteksi TERBARU
    (sama seperti file --save yang ikut tergantikan tiap kali disimpan ulang) --
    bukan numpuk history dari waktu ke waktu.

    overwrite=False: insert nambah baris baru tanpa hapus data lama (mode history/log),
    id tetap lanjut naik seperti biasa.

    Catatan: TRUNCATE butuh privilege DROP di MySQL (bukan cuma DELETE/INSERT biasa).
    Kalau user MySQL kamu gak punya privilege itu, bakal muncul error izin akses --
    tinggal kasih privilege TRUNCATE/DROP ke user tersebut, atau pakai --db-append
    buat menghindari TRUNCATE sama sekali.
    """
    cursor = conn.cursor()
    if overwrite:
        # TRUNCATE dipakai (bukan DELETE FROM) supaya kolom id (AUTO_INCREMENT) ikut
        # ke-reset balik ke 1, bukan cuma ngosongin isi tabel doang.
        cursor.execute(f"TRUNCATE TABLE {table_name}")

    if detections:
        query = f"INSERT INTO {table_name} (class_name, confidence, ocr_text, source) VALUES (%s, %s, %s, %s)"
        values = [(d["class_name"], d["confidence"], d["ocr_text"] or None, source) for d in detections]
        cursor.executemany(query, values)

    conn.commit()
    cursor.close()

    if overwrite:
        print(f"[DB] Tabel '{table_name}' di-overwrite: {len(detections)} deteksi terbaru tersimpan.")
    elif detections:
        print(f"[DB] {len(detections)} deteksi ditambahkan ke tabel '{table_name}'.")


def parse_resize(resize_str):
    """Parse string 'WIDTHxHEIGHT' (mis. '640x480') jadi tuple (width, height)."""
    if not resize_str:
        return None
    try:
        w_str, h_str = resize_str.lower().split("x")
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return (w, h)
    except ValueError:
        raise ValueError(
            f"Format --resize salah: '{resize_str}'. Gunakan format WIDTHxHEIGHT, mis. 640x480"
        )


def resize_frame(frame, target_size):
    """Resize frame ke target_size (width, height) kalau target_size di-set."""
    if target_size is None:
        return frame
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def enhance_contrast(frame, clip_limit=2.0):
    """Naikin kontras frame pakai CLAHE (Contrast Limited Adaptive Histogram
    Equalization), diterapkan cuma di channel L (lightness) di color space LAB.

    Kenapa CLAHE, bukan contrast biasa (cv2.convertScaleAbs)?
    - CLAHE bersifat ADAPTIF per area kecil di gambar, jadi area gelap dan
      terang sama-sama kebantu kontrasnya, gak cuma nge-boost yang udah terang.
    - Diterapkan cuma di channel L (bukan langsung ke B/G/R), jadi warna asli
      (mis. warna mobil, kotak deteksi) gak berubah/pudar/oversaturasi.
    - clip_limit ngontrol seberapa agresif; makin tinggi makin kontras tapi
      makin gampang muncul noise di area gelap. 2.0-3.0 biasanya cukup aman.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def predict_image(args, model, db_conn=None):
    print(f"Menjalankan prediksi pada: {args.source}")

    # baca gambar asli (untuk crop OCR sekaligus jadi input model, biar konsisten sama resize)
    img = cv2.imread(args.source)
    if img is None:
        raise RuntimeError(f"Tidak bisa membaca gambar: {args.source}")

    img = resize_frame(img, args.resize)
    if args.resize:
        print(f"Gambar di-resize ke: {args.resize[0]}x{args.resize[1]}")

    if args.contrast_boost:
        img = enhance_contrast(img, clip_limit=args.contrast_clip_limit)
        print(f"Kontras gambar ditingkatkan (CLAHE, clip_limit={args.contrast_clip_limit})")

    results = model(img, conf=args.conf, iou=args.iou, classes=args.classes, device=args.device,
                     agnostic_nms=args.agnostic_nms)

    # ambil hasil pertama (karena cuma 1 gambar)
    result = results[0]
    result = filter_by_conf(result, args.min_conf)
    detections = extract_detections(result, model.names)

    # print daftar objek yang terdeteksi
    print("\nHasil deteksi:")

    if not detections:
        print("  Tidak ada objek terdeteksi.")
    else:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            crop = img[y1:y2, x1:x2]  # crop area deteksi untuk OCR dari gambar asli
            cls_name = det["cls_name"]

            if args.slots is not None:
                # --slots aktif: slot_number dari polygon manual, OCR gak dijalankan
                slot_number, is_available = resolve_slot_from_position(det, args.slots)
                det["ocr_text"] = slot_number or ""
            else:
                detected_text = ""
                if args.boxes_only:
                    pass  # mode boxes-only: skip OCR sepenuhnya
                elif cls_name == "car":
                    print(f"  {cls_name}: skip OCR")
                elif crop.size == 0:
                    print("  (crop kosong, skip OCR)")
                else:
                    detected_text = run_ocr(crop)

                det["ocr_text"] = detected_text
                slot_number, is_available = resolve_slot_and_availability(det, position_tolerance=args.slot_position_tolerance)

            print(f"  {cls_name} (conf: {det['conf']:.2f})")

            if slot_number is not None:
                # mode gambar: cuma 1 observasi, gak ada "frame berikutnya" buat
                # dikonfirmasi ulang -> min_hits dipaksa 1 (abaikan --api-min-hits)
                notify_slot_api(args, slot_number, is_available, min_hits=1)

    # overlay polygon --slots (kalau ada) sebelum box hasil deteksi digambar di atasnya
    if args.slots is not None and args.draw_slots:
        img = draw_slots(img, args.slots)

    # gambar hasil deteksi pakai warna custom
    annotated_frame = draw_detections(
        img, detections, args.color, args.text_color,
        boxes_only=args.boxes_only,
        class_colors=args.class_colors, class_text_colors=args.class_text_colors,
        ocr_color=args.ocr_color, class_ocr_colors=args.class_ocr_colors,
    )

    if args.db_enable and db_conn is not None:
        db_detections = collect_detections(detections)
        insert_detections(db_conn, args.db_table, db_detections, source=f"image:{args.source}", overwrite=not args.db_append)

    cv2.imwrite(args.save, annotated_frame)
    print(f"\nGambar hasil disimpan di: {args.save}")

    cv2.imshow("Hasil Prediksi YOLO", annotated_frame)
    print("Tekan tombol apa saja untuk keluar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_camera(args, model, db_conn=None):
    try:
        cam_source = int(args.camera)
    except ValueError:
        cam_source = args.camera  # URL string, misal RTSP CCTV

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera/sumber: {args.camera}")

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.width or args.height:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolusi diminta: {args.width}x{args.height}  |  Resolusi aktual: {actual_w}x{actual_h}")

    window_name = "Prediksi YOLO Live - [s]=simpan frame [q]=keluar"
    cv2.namedWindow(window_name)

    print("\nMode kamera aktif.")
    print(" - Tekan 's' untuk menyimpan frame hasil deteksi saat ini")
    print(" - Tekan 'q' untuk keluar\n")

    frame_count = 0

    slots_mode = args.slots is not None
    ocr_enabled = not slots_mode and not args.boxes_only and not args.no_camera_ocr
    ocr_worker = AsyncOcrWorker() if ocr_enabled else None

    smoother = (
        DetectionSmoother(
            iou_thresh=args.smooth_iou, max_missed_frames=args.smooth_frames,
            alpha=args.smooth_alpha, min_hits=args.smooth_min_hits,
        )
        if args.smooth else None
    )

    do_db_continuous = args.db_enable and args.db_mode in ("continuous", "both")
    do_db_manual = args.db_enable and args.db_mode in ("manual", "both")
    camera_source_label = f"camera:{args.camera}"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari kamera. Cek koneksi/URL.")
                break

            frame = resize_frame(frame, args.resize)
            if args.contrast_boost:
                frame = enhance_contrast(frame, clip_limit=args.contrast_clip_limit)
            frame_count += 1

            results = model(frame, conf=args.conf, iou=args.iou, classes=args.classes, device=args.device,
                             agnostic_nms=args.agnostic_nms, verbose=False)
            result = results[0]
            result = filter_by_conf(result, args.min_conf)
            detections = extract_detections(result, model.names)

            if slots_mode:
                # --slots aktif: slot_number dari polygon manual, langsung tiap frame,
                # gak ada OCR/interval/worker sama sekali.
                for det in detections:
                    slot_number, is_available = resolve_slot_from_position(det, args.slots)
                    det["ocr_text"] = slot_number or ""
                    if args.api_enable:
                        if slot_number is not None:
                            notify_slot_api(args, slot_number, is_available)
                        elif args.api_debug:
                            print(f"[API-DEBUG] cls={det['cls_name']} box={det['box']} "
                                  f"-> titik tengah box gak ada di polygon slot manapun, gak ada yang dikirim")
            elif ocr_enabled:
                # kirim job OCR baru tiap --ocr-interval frame, tapi TIDAK nunggu hasilnya
                # (worker jalan di background, video tetap lanjut render frame ini juga)
                if frame_count % args.ocr_interval == 0 and not ocr_worker.is_busy():
                    jobs = []
                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        crop = frame[y1:y2, x1:x2].copy()
                        jobs.append((x1, y1, x2, y2, crop, det["cls_name"]))
                    ocr_worker.submit(jobs)

                # pakai hasil OCR terakhir yang udah selesai diproses (bisa dari beberapa frame lalu),
                # dicocokkan ke box saat ini berdasarkan koordinat
                ocr_map = {(x1, y1, x2, y2): text for (x1, y1, x2, y2, text) in ocr_worker.get_latest_result()}
                for det in detections:
                    text = ocr_map.get(det["box"])
                    if text:
                        det["ocr_text"] = text

                    # resolve_slot_and_availability juga nutupin kasus 'car' tanpa OCR text
                    # (nomor ketutupan mobil) lewat slot position memory, biar tetap bisa
                    # kirim is_available=0 walau nomornya lagi gak kebaca.
                    if args.api_enable:
                        slot_number, is_available = resolve_slot_and_availability(det, position_tolerance=args.slot_position_tolerance)
                        if slot_number is not None:
                            notify_slot_api(args, slot_number, is_available)
                        elif args.api_debug:
                            print(f"[API-DEBUG] cls={det['cls_name']} box={det['box']} "
                                  f"ocr_text='{det.get('ocr_text', '')}' -> slot_number=None "
                                  f"(gak ketemu posisi yang cocok di memori, gak ada yang dikirim)")

            # smoothing: gabungkan deteksi frame ini dengan track yang masih dalam grace period,
            # biar box gak kedip-kedip pas confidence naik-turun tipis di sekitar threshold
            if smoother is not None:
                detections = smoother.update(detections)

            # overlay polygon --slots (kalau ada) sebelum box hasil deteksi digambar di atasnya
            if slots_mode and args.draw_slots:
                frame = draw_slots(frame, args.slots)

            annotated_frame = draw_detections(
                frame, detections, args.color, args.text_color, boxes_only=args.boxes_only,
                class_colors=args.class_colors, class_text_colors=args.class_text_colors,
                ocr_color=args.ocr_color, class_ocr_colors=args.class_ocr_colors,
            )

            # kumpulkan detections sekali di sini, dipakai buat mode continuous maupun manual (tekan 's')
            current_detections = collect_detections(detections) if args.db_enable else []

            if do_db_continuous and frame_count % args.db_interval == 0:
                insert_detections(db_conn, args.db_table, current_detections, source=camera_source_label, overwrite=not args.db_append)

            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # selalu simpan ke path yang sama (args.save) -> file lama otomatis
                # tergantikan/kehapus, gak numpuk jadi hasil_1.jpg, hasil_2.jpg, dst.
                save_path = args.save
                if os.path.exists(save_path):
                    os.remove(save_path)
                cv2.imwrite(save_path, annotated_frame)
                print(f"Frame disimpan di: {save_path} (menggantikan simpanan sebelumnya)")

                if do_db_manual:
                    insert_detections(
                        db_conn, args.db_table, current_detections,
                        source=f"{camera_source_label} (manual save: {save_path})",
                        overwrite=not args.db_append,
                    )

            elif key == ord('q'):
                print("Keluar.")
                break
    finally:
        if ocr_worker is not None:
            ocr_worker.stop()
        cap.release()
        cv2.destroyAllWindows()


def predict_camera_capture(args, model, db_conn=None):
    """Mode BARU: kamera cuma nampilin live preview mentah (tanpa deteksi tiap frame).
    Deteksi (YOLO + OCR) baru dijalankan SEKALI setiap kali user pencet 'c' (capture).
    Hasil capture terakhir ditampilkan sampai capture berikutnya, dan bisa disimpan
    ke file/DB dengan 's', sama seperti mode kamera live.
    """
    try:
        cam_source = int(args.camera)
    except ValueError:
        cam_source = args.camera  # URL string, misal RTSP CCTV

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera/sumber: {args.camera}")

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if args.width or args.height:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolusi diminta: {args.width}x{args.height}  |  Resolusi aktual: {actual_w}x{actual_h}")

    window_name = "Mode Capture YOLO - [c]=capture&detect [s]=simpan [q]=keluar"
    cv2.namedWindow(window_name)

    print("\nMode capture aktif (deteksi HANYA jalan saat capture, bukan tiap frame).")
    print(" - Tekan 'c' untuk capture frame saat ini & jalankan deteksi")
    print(" - Tekan 's' untuk menyimpan hasil capture terakhir ke file (& DB kalau aktif)")
    print(" - Tekan 'q' untuk keluar\n")

    slots_mode = args.slots is not None
    ocr_enabled = not slots_mode and not args.boxes_only and not args.no_camera_ocr
    camera_source_label = f"camera:{args.camera}"

    # state hasil capture terakhir; ditampilkan terus sampai capture berikutnya
    last_annotated_frame = None
    last_detections = []
    has_capture = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari kamera. Cek koneksi/URL.")
                break

            frame = resize_frame(frame, args.resize)
            if args.contrast_boost:
                frame = enhance_contrast(frame, clip_limit=args.contrast_clip_limit)

            # tampilkan hasil capture terakhir kalau ada, kalau belum ada capture
            # sama sekali, tampilkan live preview mentah + overlay instruksi
            if has_capture:
                display_frame = last_annotated_frame
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, "LIVE - tekan 'c' untuk capture & deteksi",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                print("Capture! Menjalankan deteksi...")
                capture_frame = frame.copy()

                results = model(capture_frame, conf=args.conf, iou=args.iou,
                                 classes=args.classes, device=args.device,
                                 agnostic_nms=args.agnostic_nms, verbose=False)
                result = results[0]
                result = filter_by_conf(result, args.min_conf)
                detections = extract_detections(result, model.names)

                if slots_mode:
                    # --slots aktif: slot_number dari polygon manual, OCR gak dijalankan
                    for det in detections:
                        slot_number, is_available = resolve_slot_from_position(det, args.slots)
                        det["ocr_text"] = slot_number or ""
                        if slot_number is not None:
                            # mode capture: tiap capture itu aksi sengaja/manual (tekan 'c'),
                            # bukan continuous frame -> min_hits dipaksa 1 juga
                            notify_slot_api(args, slot_number, is_available, min_hits=1)
                # OCR sinkron per crop (bukan async/background) karena ini cuma
                # dijalankan sekali per capture, bukan tiap frame kayak mode live
                elif ocr_enabled:
                    for det in detections:
                        x1, y1, x2, y2 = det["box"]
                        crop = capture_frame[y1:y2, x1:x2]
                        cls_name = det["cls_name"]
                        if cls_name == "car":
                            det["ocr_text"] = ""
                        elif crop.size == 0:
                            det["ocr_text"] = ""
                        else:
                            det["ocr_text"] = run_ocr(crop)

                        slot_number, is_available = resolve_slot_and_availability(det, position_tolerance=args.slot_position_tolerance)
                        if slot_number is not None:
                            # mode capture: tiap capture itu aksi sengaja/manual (tekan 'c'),
                            # bukan continuous frame -> min_hits dipaksa 1 juga
                            notify_slot_api(args, slot_number, is_available, min_hits=1)

                if slots_mode and args.draw_slots:
                    capture_frame = draw_slots(capture_frame, args.slots)

                annotated_frame = draw_detections(
                    capture_frame, detections, args.color, args.text_color, boxes_only=args.boxes_only,
                    class_colors=args.class_colors, class_text_colors=args.class_text_colors,
                    ocr_color=args.ocr_color, class_ocr_colors=args.class_ocr_colors,
                )

                last_annotated_frame = annotated_frame
                last_detections = detections
                has_capture = True

                if not detections:
                    print("  Tidak ada objek terdeteksi.")
                else:
                    for det in detections:
                        print(f"  {det['cls_name']} (conf: {det['conf']:.2f})"
                              + (f" | OCR: {det['ocr_text']}" if det.get("ocr_text") else ""))

                # mode capture: simpan ke DB otomatis tiap kali capture kalau db aktif
                # (di mode ini, capture = momen deteksinya, beda dari mode live yang punya
                # opsi continuous/manual terpisah)
                if args.db_enable and args.db_mode in ("manual", "both", "continuous"):
                    db_detections = collect_detections(detections)
                    insert_detections(db_conn, args.db_table, db_detections,
                                       source=f"{camera_source_label} (capture)",
                                       overwrite=not args.db_append)

            elif key == ord('s'):
                if not has_capture:
                    print("Belum ada hasil capture. Tekan 'c' dulu sebelum 's'.")
                else:
                    save_path = args.save
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    cv2.imwrite(save_path, last_annotated_frame)
                    print(f"Hasil capture disimpan di: {save_path} (menggantikan simpanan sebelumnya)")

            elif key == ord('q'):
                print("Keluar.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--source", help="Path gambar yang mau diprediksi")
    src_group.add_argument("--camera", help="Index kamera (mis. 0) atau URL kamera (RTSP/HTTP)")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Path model YOLO: file .pt biasa, ATAU folder hasil export "
                              "(mis. best_ncnn_model, best_onnx_model) untuk deployment ringan "
                              "seperti di Raspberry Pi.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold saat inference model (bisa rendah, filter tambahan di --min-conf)")
    parser.add_argument("--min-conf", type=float, default=0.4,
                         help="Confidence minimum agar deteksi benar-benar ditampilkan/dipakai (default: 0.4). "
                              "Deteksi dengan conf antara --conf dan --min-conf tetap diproses model tapi disembunyikan.")
    parser.add_argument("--iou", type=float, default=0.5,
                         help="IoU threshold buat NMS (makin kecil, makin agresif buang box tumpang tindih, mis. 0.4-0.5 buat objek rapat kayak mobil parkir)")
    parser.add_argument("--no-agnostic-nms", dest="agnostic_nms", action="store_false",
                         help="Matikan class-agnostic NMS (default: AKTIF). Secara default, NMS menekan box "
                              "tumpang tindih LINTAS CLASS juga (mis. 'car' dan 'free' yang overlap di slot "
                              "fisik yang sama akan disaring jadi cuma 1 box, yang confidence-nya tertinggi "
                              "yang dipertahankan). Ini penting buat kasus 1 slot yang kedeteksi 2x dengan "
                              "class berbeda -> tanpa ini, kedua box bisa lolos bareng dan kirim status "
                              "is_available yang SALING BERTENTANGAN ke API buat slot yang sama. Matiin ini "
                              "cuma kalau memang butuh box overlap dari class berbeda tetap muncul bareng.")
    parser.set_defaults(agnostic_nms=True)
    parser.add_argument("--no-smooth", dest="smooth", action="store_false",
                         help="Matikan smoothing deteksi di mode kamera (default: smoothing aktif). "
                              "Smoothing bikin box gak kedip-kedip pas confidence naik-turun tipis di sekitar threshold, "
                              "dengan cara tetap nampilin box beberapa frame walau deteksi sempat 'putus' sesaat.")
    parser.add_argument("--smooth-frames", type=int, default=10,
                         help="Di mode kamera dengan smoothing aktif: berapa frame 'grace period' box tetap "
                              "ditampilkan setelah deteksi hilang, sebelum beneran dihapus (default: 10).")
    parser.add_argument("--smooth-iou", type=float, default=0.3,
                         help="Di mode kamera dengan smoothing aktif: ambang IoU buat mencocokkan box baru "
                              "ke box yang sudah dilacak dari frame sebelumnya (default: 0.3).")
    parser.add_argument("--smooth-alpha", type=float, default=0.5,
                         help="Di mode kamera dengan smoothing aktif: seberapa cepat box mengikuti posisi "
                              "deteksi baru, rentang 0-1 (default: 0.5). Makin kecil = posisi box makin halus "
                              "'meluncur' pelan, tapi makin lambat ngikutin gerakan objek asli. "
                              "1.0 = gak ada smoothing posisi sama sekali.")
    parser.add_argument("--smooth-min-hits", type=int, default=3,
                         help="Di mode kamera dengan smoothing aktif: berapa frame berturut-turut suatu "
                              "objek BARU harus konsisten kedeteksi dulu sebelum mulai ditampilkan (default: 3). "
                              "Ini yang bikin deteksi 'kedip sekali terus ilang lagi' (noise sesaat) gak pernah "
                              "nongol sama sekali. Makin besar = makin ketat/gak gampang nongol box baru, tapi "
                              "makin 'lambat' merespon objek yang beneran baru muncul.")
    parser.add_argument("--mode", choices=["free", "car", "all"], default="all",
                         help="Mode deteksi: 'free' (cuma slot kosong), 'car' (cuma mobil), atau 'all' (keduanya)")
    parser.add_argument("--capture-mode", action="store_true",
                         help="Mode BARU untuk kamera: TIDAK mendeteksi tiap frame (live preview biasa aja). "
                              "Deteksi (YOLO + OCR) baru dijalankan sekali tiap kamu pencet 'c' (capture). "
                              "Hasil capture terakhir tetap ditampilkan sampai capture berikutnya. "
                              "Tombol 's' menyimpan hasil capture terakhir ke file/DB, 'q' keluar. "
                              "Cocok buat skenario 'berhenti di depan kamera baru discan', bukan real-time terus-menerus.")
    parser.add_argument("--save", default="hasil.jpg", help="Path output gambar hasil")
    parser.add_argument("--boxes-only", action="store_true",
                         help="Cuma tampilkan kotak deteksi, tanpa label class/confidence maupun teks OCR")
    parser.add_argument("--width", type=int, default=None,
                         help="Resolusi lebar frame kamera yang diminta (mis. 1920)")
    parser.add_argument("--height", type=int, default=None,
                         help="Resolusi tinggi frame kamera yang diminta (mis. 1080)")
    parser.add_argument("--resize", default=None,
                         help="Ubah ukuran gambar/frame sebelum diproses, format WIDTHxHEIGHT (mis. --resize 640x480)")
    parser.add_argument("--contrast-boost", action="store_true",
                         help="Naikin kontras gambar/frame kamera pakai CLAHE sebelum dideteksi & ditampilkan. "
                              "Berguna kalau kondisi cahaya kurang/pencahayaan gak rata, sekaligus bisa "
                              "membantu akurasi deteksi (bukan cuma tampilan).")
    parser.add_argument("--contrast-clip-limit", type=float, default=2.0,
                         help="Kekuatan CLAHE saat --contrast-boost aktif. Makin tinggi makin kontras, "
                              "tapi makin gampang muncul noise di area gelap (default: 2.0, rentang aman 1.5-4.0)")
    parser.add_argument("--color", default="green",
                         help=f"Warna bounding box, nama ({', '.join(NAMED_COLORS.keys())}) atau hex mis. #00FF00 (default: green)")
    parser.add_argument("--text-color", default="yellow",
                         help=f"Warna teks label/OCR, nama ({', '.join(NAMED_COLORS.keys())}) atau hex mis. #FFFF00 (default: yellow)")
    parser.add_argument("--class-colors", default=None,
                         help="Warna kotak per class, format 'nama_class:warna,nama_class2:warna2', "
                              "mis. 'car:red,free:#00FF00'. Class yang gak disebut pakai --color.")
    parser.add_argument("--class-text-colors", default=None,
                         help="Warna teks label per class, format sama seperti --class-colors. "
                              "Class yang gak disebut pakai --text-color.")
    parser.add_argument("--ocr-color", default=None,
                         help="Warna teks hasil OCR (default: sama dengan --text-color kalau tidak diisi)")
    parser.add_argument("--class-ocr-colors", default=None,
                         help="Warna teks OCR per class, format 'nama_class:warna,nama_class2:warna2'. "
                              "Class yang gak disebut pakai --ocr-color (atau --text-color kalau --ocr-color juga kosong).")
    parser.add_argument("--no-camera-ocr", action="store_true",
                         help="Matikan OCR sepenuhnya di mode kamera (OCR tetap aktif di mode --source gambar).")
    parser.add_argument("--ocr-interval", type=int, default=5,
                         help="Di mode kamera, kirim job OCR baru tiap N frame (default: 5). OCR jalan di "
                              "background thread jadi video gak nge-freeze; makin kecil N, makin sering OCR "
                              "diupdate tapi makin banyak beban CPU/GPU.")
    parser.add_argument("--device", default="auto",
                         help="Device buat YOLO: 'auto' (pilih GPU kalau ada), '0' (GPU index 0), 'cpu', dst. (default: auto)")
    parser.add_argument("--no-ocr-gpu", action="store_true",
                         help="Matikan GPU khusus buat EasyOCR, paksa jalan di CPU (YOLO tetap ikut --device).")

    # ---- Argumen database MySQL ----
    parser.add_argument("--db-enable", action="store_true",
                         help="Aktifkan penyimpanan hasil deteksi ke database MySQL.")
    parser.add_argument("--db-mode", choices=["manual", "continuous", "both"], default="manual",
                         help="Kapan simpan ke DB di mode kamera: 'manual' (cuma saat tekan 's'), "
                              "'continuous' (otomatis tiap --db-interval frame), atau 'both'. "
                              "Di mode --source gambar, selalu tersimpan sekali kalau --db-enable aktif. (default: manual)")
    parser.add_argument("--db-interval", type=int, default=30,
                         help="Di mode 'continuous'/'both', simpan ke DB tiap N frame (default: 30, biar gak membanjiri DB tiap frame).")
    parser.add_argument("--db-host", default="localhost", help="Host MySQL (default: localhost)")
    parser.add_argument("--db-port", type=int, default=3306, help="Port MySQL (default: 3306)")
    parser.add_argument("--db-user", default="root", help="Username MySQL (default: root)")
    parser.add_argument("--db-password", default="", help="Password MySQL (default: kosong)")
    parser.add_argument("--db-name", default=None, help="Nama database MySQL (wajib diisi kalau --db-enable aktif)")
    parser.add_argument("--db-table", default="detections", help="Nama tabel MySQL (default: detections)")
    parser.add_argument("--db-append", action="store_true",
                         help="Simpan sebagai history (nambah baris baru terus), BUKAN overwrite. "
                              "Default (tanpa flag ini): tabel di-overwrite, cuma berisi snapshot deteksi terbaru.")

    # ---- Argumen integrasi API update slot parkir ----
    parser.add_argument("--api-enable", action="store_true",
                         help="Aktifkan pengiriman update status slot (is_available) ke API parkir "
                              "setiap kali nomor slot (mis. 'P1') berhasil terbaca lewat OCR.")
    parser.add_argument("--api-url", default=DEFAULT_SLOT_API_URL,
                         help=f"Endpoint API update slot, body JSON {{slot_number, is_available}} "
                              f"(default: {DEFAULT_SLOT_API_URL})")
    parser.add_argument("--api-timeout", type=float, default=5.0,
                         help="Timeout (detik) buat tiap request ke API update slot (default: 5.0)")
    parser.add_argument("--api-min-hits", type=int, default=3,
                         help="Di mode kamera live, status slot (kosong/terisi) harus konsisten "
                              "kebaca N kali BERTURUT-TURUT dulu baru dikirim ke API (default: 3). "
                              "Ini nyaring salah baca OCR/tebakan posisi sesaat biar gak nge-spam "
                              "API sama data yang keliru cuma karena kedip sekali. Mode --source "
                              "gambar dan --capture-mode SELALU kirim langsung (min_hits=1) karena "
                              "cuma ada satu observasi tegas, gak ada 'frame berikutnya' buat "
                              "dikonfirmasi ulang.")
    parser.add_argument("--api-slots-url", default=DEFAULT_SLOTS_LIST_URL,
                         help="Endpoint GET buat ambil daftar slot resmi (array JSON berisi "
                              f"'slot_number', dst). Dipakai buat validasi sebelum kirim update -- "
                              f"kalau nomor slot hasil OCR/deteksi gak ada di daftar ini, gak "
                              f"dikirim. (default: {DEFAULT_SLOTS_LIST_URL})")
    parser.add_argument("--api-slots-refresh", type=float, default=300.0,
                         help="Seberapa sering (detik) daftar slot resmi di-refresh ulang dari "
                              "--api-slots-url (default: 300 = 5 menit). Di-cache di antara refresh "
                              "biar gak nge-GET ulang tiap mau kirim satu update.")
    parser.add_argument("--api-slots-poll-interval", type=float, default=30.0,
                         help="Seberapa sering (detik) thread background ngecek daftar slot resmi "
                              "dari --api-slots-url TERUS-MENERUS selama script jalan (default: 30). "
                              "Beda dari --api-slots-refresh (yang cuma nge-refresh cache pas-pasan "
                              "mau kirim update) -- ini jalan independen di background & cuma nge-print "
                              "notifikasi pas statusnya BERUBAH jadi 'ada'/'gak ada' (gak diulang-ulang "
                              "tiap 30 detik kalau statusnya gak berubah).")
    parser.add_argument("--api-status-field", default="is_available",
                         help="Nama field status di response GET --api-slots-url yang dipakai buat "
                              "rekonsiliasi (default: 'is_available'). Kalau field ini ADA & nilainya "
                              "beda dari status terakhir yang kita kirim sendiri (kemungkinan diubah "
                              "manual/pihak ketiga langsung di DB), status yang bener otomatis dikirim "
                              "ulang ke API tiap --api-slots-poll-interval detik. Sesuaikan kalau nama "
                              "field di API kamu beda.")
    parser.add_argument("--api-no-reconcile", action="store_true",
                         help="Matikan rekonsiliasi otomatis (default: aktif kalau --api-enable). "
                              "Tanpa flag ini, tiap --api-slots-poll-interval detik status slot di DB "
                              "dibandingin ke status hasil deteksi kamera kita sendiri -- kalau beda "
                              "(mis. diubah pihak ketiga langsung ke DB), otomatis dikirim ulang "
                              "buat nimpa balik ke status yang bener.")
    parser.add_argument("--api-skip-slots-check", action="store_true",
                         help="Matikan validasi ke daftar slot resmi -- semua nomor slot hasil "
                              "OCR/deteksi langsung dikirim ke API tanpa dicek dulu ada di daftar "
                              "atau enggak (perilaku versi lama, sebelum --api-slots-url ada).")
    parser.add_argument("--api-debug", action="store_true",
                         help="Print detail proses resolve slot per deteksi (class, teks OCR, "
                              "slot_number & is_available hasil resolve, status hit-counter) "
                              "buat diagnosa kenapa suatu update gak/telat terkirim ke API.")
    parser.add_argument("--slot-position-tolerance", type=float, default=SLOT_POSITION_TOLERANCE,
                         help=f"Toleransi jarak (px) buat nyocokin posisi slot yang lagi ketutupan "
                              f"mobil ke posisi nomor slot yang pernah kebaca (dipakai kalau box-nya "
                              f"gak persis nutupin/containment titik tengah slot). Sesuaikan sama "
                              f"resolusi/imgsz -- makin kecil imgsz (mis. 320), makin kecil juga nilai "
                              f"yang wajar (default: {SLOT_POSITION_TOLERANCE}). Diabaikan kalau --slots dipakai.")
    parser.add_argument("--slots", default=None,
                         help="Path file JSON hasil label_slots.py (polygon slot yang digambar & dilabel "
                              "manual pakai mouse). Kalau diisi: identitas slot_number diambil LANGSUNG "
                              "dari polygon yang mengandung titik tengah tiap box deteksi -- OCR TIDAK "
                              "dijalankan/dimuat sama sekali, dan --slot-position-tolerance diabaikan "
                              "(gak perlu tebak-tebakan posisi lagi, polygon-nya udah pasti). Status "
                              "kosong/terisi tetap ditentukan dari nama class hasil deteksi.")
    parser.add_argument("--no-draw-slots", dest="draw_slots", action="store_false",
                         help="Matikan overlay garis tipis polygon slot dari --slots di video/gambar hasil "
                              "(default: overlay ditampilkan, berguna buat verifikasi posisi slot udah pas).")
    parser.set_defaults(draw_slots=True)

    args = parser.parse_args()

    args.resize = parse_resize(args.resize)
    args.color = parse_color(args.color)
    args.text_color = parse_color(args.text_color)
    args.class_colors = parse_class_colors(args.class_colors)
    args.class_text_colors = parse_class_colors(args.class_text_colors)
    args.ocr_color = parse_color(args.ocr_color) if args.ocr_color else None
    args.class_ocr_colors = parse_class_colors(args.class_ocr_colors)
    if args.ocr_interval < 1:
        raise ValueError("--ocr-interval harus >= 1")
    if args.db_enable and not args.db_name:
        raise ValueError("--db-name wajib diisi kalau --db-enable aktif.")
    if args.db_interval < 1:
        raise ValueError("--db-interval harus >= 1")

    args.slots = load_slots(args.slots) if args.slots else None
    if args.slots is not None:
        print(f"[Slots] {len(args.slots)} slot dimuat dari file --slots: "
              f"{[s['label'] for s in args.slots]}")

    print("Memuat model...")
    args.device = resolve_device(args.device)
    print(f"YOLO akan jalan di device: {args.device}")
    model = YOLO(args.model)

    # Model hasil export (NCNN/ONNX/TFLite/dll) itu inference graph yang CPU-only
    # (di Raspberry Pi memang gak ada CUDA juga), jadi gak support .to(device)
    # kayak model PyTorch (.pt) biasa. Deteksi otomatis dari task/nama file model.
    is_exported_model = not str(args.model).endswith(".pt")
    if is_exported_model:
        print(f"Model terdeteksi format export ({args.model}) -> lewati model.to(), jalan native di CPU.")
    else:
        torch_device = f"cuda:{args.device}" if args.device.isdigit() else args.device
        model.to(torch_device)

    model_task = getattr(model, "task", None)
    if model_task == "obb":
        print("Model terdeteksi: OBB (Oriented Bounding Box) -> kotak akan digambar MIRING "
              "mengikuti orientasi objek asli (cocok buat parkiran yang serong/menyudut di kamera).")
    elif model_task == "segment":
        print("Model terdeteksi: Segmentasi -> kotak akan digambar mengikuti KONTUR bentuk objek asli (dari mask).")
    else:
        print(f"Model terdeteksi: '{model_task or 'detect'}' (deteksi biasa) -> kotak tetap lurus (axis-aligned), "
              "TIDAK mengikuti kemiringan parkiran. Kalau mau kotak mengikuti sudut/bentuk parkiran, "
              "latih ulang modelmu sebagai model OBB (task=obb) atau segmentasi (task=segment) di Ultralytics.")

    if args.slots is not None:
        # --slots aktif: identitas slot dari polygon manual, EasyOCR gak dibutuhkan
        # sama sekali -> gak usah dimuat (start-up lebih cepat & ringan, apalagi
        # kalau jalan di Raspberry Pi).
        print("[Slots] EasyOCR TIDAK dimuat/dijalankan sama sekali (identitas slot "
              "diambil langsung dari polygon --slots, bukan OCR).")
    else:
        global ocr
        ocr = init_ocr(use_gpu=(args.device != "cpu" and not args.no_ocr_gpu))

    # tentukan filter class berdasarkan --mode
    if args.mode == "all":
        args.classes = None  # None = deteksi semua class
    else:
        # cari class id yang namanya cocok sama args.mode (mis. "car" atau "free")
        matching_ids = [cid for cid, name in model.names.items() if name == args.mode]
        if not matching_ids:
            raise ValueError(
                f"Class '{args.mode}' tidak ditemukan di model. "
                f"Class yang tersedia: {list(model.names.values())}"
            )
        args.classes = matching_ids
        print(f"Mode: '{args.mode}' -> hanya mendeteksi class id {matching_ids}")

    if args.api_enable:
        print(f"[API] Integrasi update slot aktif -> POST ke {args.api_url} "
              f"tiap kali nomor slot (mis. 'P1') terbaca lewat OCR pada slot kosong ('free').")
        if not args.source:
            print(f"[API] Minimum hit (mode kamera live): status slot harus konsisten kebaca "
                  f"{args.api_min_hits}x berturut-turut dulu baru dikirim.")

        if args.api_skip_slots_check:
            print("[API] Validasi ke daftar slot resmi DIMATIKAN (--api-skip-slots-check) "
                  "-> semua nomor slot langsung dikirim tanpa dicek.")
        else:
            print(f"[API] Mulai ngecek daftar slot resmi dari {args.api_slots_url} tiap "
                  f"{args.api_slots_poll_interval:.0f} detik di background...")
            slots_watcher_thread = threading.Thread(
                target=slots_api_watcher, args=(args,), daemon=True
            )
            slots_watcher_thread.start()

    db_conn = None
    if args.db_enable:
        print(f"Menghubungkan ke database MySQL '{args.db_name}' di {args.db_host}:{args.db_port}...")
        db_conn = get_db_connection(args)
        ensure_table(db_conn, args.db_table)
        print(f"Terhubung. Tabel '{args.db_table}' siap dipakai.")

    try:
        if args.source:
            predict_image(args, model, db_conn=db_conn)
        elif args.capture_mode:
            predict_camera_capture(args, model, db_conn=db_conn)
        else:
            predict_camera(args, model, db_conn=db_conn)
    finally:
        if db_conn is not None:
            db_conn.close()
            print("Koneksi database ditutup.")


if __name__ == "__main__":
    main()