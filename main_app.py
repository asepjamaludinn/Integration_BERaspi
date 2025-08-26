# main_app.py (Lengkap dengan Pelaporan Status Online/Offline)

import cv2
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO



# --- Konfigurasi MQTT ---
MQTT_BROKER = "192.168.0.174"
MQTT_PORT = 1883
MQTT_USERNAME = "cpsmagang"
MQTT_PASSWORD = "cpsjaya123"

# --- Konfigurasi Perangkat ---
DEVICE_IP_ADDRESS = "192.168.0.174"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status" # <-- Topik baru untuk status

# --- Konfigurasi Model & Kamera ---
MODEL_PATH = 'yolov8n-pose.pt'
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Konfigurasi Pin GPIO ---
LAMP_PIN = 26 



# Variabel global
lamp = None
gpio_state = 0 # Menggunakan gpio_state Anda untuk melacak status lampu

# --- Logika Kontrol Hardware ---
def control_device(device, action):
    global gpio_state
    if device == "lamp":
        if action == "turn_on":
            led.on()
            gpio_state = 1
        elif action == "turn_off":
            led.off()
            gpio_state = 0
        print(f"🚀 AKSI DARI BACKEND: Menjalankan '{action}' pada '{device}'")

# --- Fungsi-fungsi MQTT ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"✅ Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        # Setelah terhubung, subscribe ke topik aksi
        client.subscribe(ACTION_TOPIC)
        print(f"👂 SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
        
        # === KIRIM STATUS ONLINE SETELAH BERHASIL KONEK ===
        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload, retain=True)
        print(f"📡 PUBLISH ke {STATUS_TOPIC}: Status Online")
    else:
        print(f"❌ Gagal terhubung ke MQTT, kode error: {rc}")

def on_message(client, userdata, msg):
    print(f"📩 PESAN DITERIMA di topik {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        device = payload.get("device")
        action = payload.get("action")
        if device and action:
            control_device(device, action)
    except Exception as e:
        print(f"Error memproses pesan: {e}")

# ====================================================================
# PROGRAM UTAMA
# ====================================================================

# --- Inisialisasi MQTT Client ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# === SET "LAST WILL" UNTUK OTOMATIS OFFLINE ===
# Ini adalah pesan yang akan dikirim oleh Broker jika Pi mati mendadak
offline_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=offline_payload, qos=1, retain=True)

client.on_connect = on_connect
client.on_message = on_message

# --- Inisialisasi Hardware ---
print("⏳ Mempersiapkan GPIO...")
led = LED(gpio_pin)
print("✅ GPIO siap.")

# --- Inisialisasi Kamera & Model ---
print("⏳ Mempersiapkan kamera dan memuat model YOLO...")
if "usb" in cam_source:
    cam_idx = int(cam_source[3:])
    cam = cv2.VideoCapture(cam_idx)
    cam.set(3, resW)
    cam.set(4, resH)
    if not cam.isOpened():
        print("❌ Gagal membuka kamera.")
        exit()
else:
    print("❌ Sumber kamera tidak valid!")
    exit()
model = YOLO(model_pose, task="pose")
print("✅ Kamera dan model siap.")

# --- Try...Finally block untuk menjalankan semuanya ---
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # Variabel dari kode Anda
    consecutive_detections = 0
    is_person_reported = False
    fps_buffer = []
    fps_avg_len = 50

    print("\n🚀 Sistem deteksi mulai berjalan. Tekan 'Q' untuk berhenti.")
    while True:
        # Loop utama Anda untuk deteksi, publish sensor, dan visualisasi
        # ... (seluruh logika 'while' Anda tetap sama persis)
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0
        if pose_found:
            consecutive_detections = min(consecutive_detections + 1, 10)
        else:
            consecutive_detections = max(consecutive_detections - 1, 0)
        
        should_be_active = consecutive_detections >= 8
        should_be_inactive = consecutive_detections <= 0

        # Logika aksi lokal & publish digabung
        if should_be_active and gpio_state == 0:
            gpio_state = 1
            led.on()
            if not is_person_reported:
                is_person_reported = True
                client.publish(SENSOR_TOPIC, json.dumps({"motion_detected": True}))
                print(f"📡 PUBLISH: Pose Terdeteksi!")

        elif should_be_inactive and gpio_state == 1:
            gpio_state = 0
            led.off()
            if is_person_reported:
                is_person_reported = False
                client.publish(SENSOR_TOPIC, json.dumps({"motion_cleared": True}))
                print(f"📡 PUBLISH: Pose Tidak Terdeteksi!")

        # Visualisasi
        if gpio_state == 0:
            cv2.putText(annotated_frame, "Light OFF", (20,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
        else:
            cv2.putText(annotated_frame, "Light ON", (20,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
        
        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            # Kalkulasi dan tampilkan FPS
            pass # Kode FPS Anda di sini

        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- Cleanup ---
    print("\n🧹 Membersihkan sumber daya...")
    # Mengirim status offline secara manual saat program berhenti normal
    client.publish(STATUS_TOPIC, offline_payload, retain=True)
    print(f"📡 PUBLISH ke {STATUS_TOPIC}: Status Offline")
    time.sleep(0.5) # Beri sedikit waktu agar pesan terkirim

    cam.release()
    cv2.destroyAllWindows()
    led.close()
    client.loop_stop()
    client.disconnect()
    print("Selesai.")