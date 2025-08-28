import cv2
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO
from datetime import datetime

# --- KONFIGURASI ---
MQTT_BROKER = "10.193.35.108"
MQTT_PORT = 1883
MQTT_USERNAME = "cpsmagang"
MQTT_PASSWORD = "cpsjaya123"
DEVICE_IP_ADDRESS = "10.193.35.108"

# --- TOPIK MQTT ---
STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"

# --- Inisialisasi Perangkat dan Model ---
try:
    model_pose = YOLO("yolo11n-pose_ncnn_model", task="pose")
    cam_source = "usb0"
    resW, resH = 640, 480

    # Struktur data baru untuk mengelola perangkat secara independen
    devices = {
        "light": {
            "instance": LED(26),
            "state": 0,  # 0: OFF, 1: ON
            "mode": "auto",  # 'auto', 'manual', 'scheduled'
            "schedule_on": None, # Contoh: "18:00"
            "schedule_off": None, # Contoh: "06:00"
            "is_person_reported": False
        },
        "fan": {
            "instance": LED(19),
            "state": 0,
            "mode": "auto",
            "schedule_on": None,
            "schedule_off": None,
            "is_person_reported": False
        }
    }

    # Inisialisasi Kamera
    if "usb" in cam_source:
        cam_idx = int(cam_source[3:])
        cam = cv2.VideoCapture(cam_idx)
        cam.set(3, resW)
        cam.set(4, resH)
        if not cam.isOpened():
            print("Gagal membuka kamera.")
            exit()
    else:
        print("Tidak ada kamera terdeteksi.")
        exit()
    print("Kamera siap.")

except Exception as e:
    print(f"Error saat inisialisasi: {e}")
    exit()

# Variabel untuk logika deteksi
consecutive_detections = 0
fps_buffer = []
fps_avg_len = 50

# --- FUNGSI CALLBACK MQTT ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        print(f"SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        print(f"SUBSCRIBE ke topik settings: {SETTINGS_UPDATE_TOPIC}")

        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload)
        print(f"PUBLISH: Mengirim status ONLINE ke {STATUS_TOPIC}")
    else:
        print(f"Gagal terhubung ke MQTT, kode error: {rc}")

def on_message(client, userdata, msg):
    global devices
    print(f"PESAN DITERIMA di topik {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        
        # Penanganan untuk perintah aksi manual
        if msg.topic == ACTION_TOPIC:
            device_name = payload.get("device")
            action = payload.get("action")
            
            if device_name in devices and action in ["turn_on", "turn_off"]:
                # Aksi manual akan override mode saat ini menjadi 'manual'
                devices[device_name]["mode"] = "manual"
                if action == "turn_on":
                    devices[device_name]["instance"].on()
                    devices[device_name]["state"] = 1
                elif action == "turn_off":
                    devices[device_name]["instance"].off()
                    devices[device_name]["state"] = 0
                print(f"AKSI MANUAL: '{action}' pada '{device_name}'. Mode diubah ke MANUAL.")
        
        # Penanganan untuk pembaruan pengaturan
        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device_name = payload.get("device")
            if device_name in devices:
                # Update mode
                if "mode" in payload:
                    new_mode = payload["mode"]
                    if new_mode in ["auto", "manual", "scheduled"]:
                        devices[device_name]["mode"] = new_mode
                        print(f"SETTINGS UPDATE: Mode '{device_name}' diubah menjadi {new_mode.upper()}")
                
                # Update jadwal (hanya berlaku jika mode 'scheduled')
                if "schedule_on" in payload:
                    devices[device_name]["schedule_on"] = payload["schedule_on"]
                    print(f"SETTINGS UPDATE: Jadwal ON '{device_name}' diatur ke {payload['schedule_on']}")
                if "schedule_off" in payload:
                    devices[device_name]["schedule_off"] = payload["schedule_off"]
                    print(f"SETTINGS UPDATE: Jadwal OFF '{device_name}' diatur ke {payload['schedule_off']}")

    except Exception as e:
        print(f"Error memproses pesan: {e}")

# Inisialisasi Klien MQTT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print("\nSistem deteksi mulai berjalan. Tekan 'Q' untuk berhenti.")
try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            print("Peringatan: Gagal mengambil frame.")
            break

        # Proses deteksi hanya dilakukan sekali per frame
        results = model_pose.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0

        if pose_found:
            consecutive_detections = min(consecutive_detections + 1, 10)
        else:
            consecutive_detections = max(consecutive_detections - 1, 0)
        
        should_be_active = consecutive_detections >= 8
        should_be_inactive = consecutive_detections <= 0
        
        # Dapatkan waktu saat ini untuk mode terjadwal
        now = datetime.now().time()

        # Iterasi melalui setiap perangkat untuk menerapkan logika modenya
        for name, device in devices.items():
            # --- LOGIKA MODE OTOMATIS ---
            if device["mode"] == "auto":
                if should_be_active and device["state"] == 0:
                    device["instance"].on()
                    device["state"] = 1
                elif should_be_inactive and device["state"] == 1:
                    device["instance"].off()
                    device["state"] = 0
                
                # Kirim laporan MQTT hanya saat status berubah
                if should_be_active and not device["is_person_reported"]:
                    device["is_person_reported"] = True
                    payload = json.dumps({"device": name, "motion_detected": True})
                    client.publish(SENSOR_TOPIC, payload)
                    print(f"PUBLISH (AUTO '{name}'): Pose Terdeteksi!")
                elif should_be_inactive and device["is_person_reported"]:
                    device["is_person_reported"] = False
                    payload = json.dumps({"device": name, "motion_cleared": True})
                    client.publish(SENSOR_TOPIC, payload)
                    print(f"PUBLISH (AUTO '{name}'): Pose Tidak Terdeteksi!")

            # --- LOGIKA MODE TERJADWAL ---
            elif device["mode"] == "scheduled":
                try:
                    on_time = datetime.strptime(device["schedule_on"], "%H:%M").time()
                    off_time = datetime.strptime(device["schedule_off"], "%H:%M").time()
                    
                    is_active_time = False
                    if on_time < off_time: # Jadwal di hari yang sama (e.g., 08:00 - 17:00)
                        if on_time <= now < off_time:
                            is_active_time = True
                    else: # Jadwal melewati tengah malam (e.g., 22:00 - 06:00)
                        if now >= on_time or now < off_time:
                            is_active_time = True
                    
                    if is_active_time and device["state"] == 0:
                        device["instance"].on()
                        device["state"] = 1
                    elif not is_active_time and device["state"] == 1:
                        device["instance"].off()
                        device["state"] = 0
                except (ValueError, TypeError):
                    # Jadwal tidak valid atau belum diatur, biarkan perangkat mati
                    if device["state"] == 1:
                        device["instance"].off()
                        device["state"] = 0

            # --- MODE MANUAL tidak melakukan apa-apa di loop ini ---

        # --- Tampilkan Informasi di Layar ---
        y_pos = 30
        for name, device in devices.items():
            mode_text = f"{name.upper()} Mode: {device['mode'].upper()}"
            status_text = f"{name.upper()} Status: {'ON' if device['state'] == 1 else 'OFF'}"
            
            # Warna berdasarkan status
            color_mode = (0, 255, 255) # Kuning untuk mode
            color_status = (0, 255, 0) if device['state'] == 1 else (0, 0, 255) # Hijau/Merah untuk status
            
            cv2.putText(annotated_frame, mode_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, .6, color_mode, 2)
            y_pos += 30
            cv2.putText(annotated_frame, status_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, .6, color_status, 2)
            y_pos += 40

        # Kalkulasi dan tampilkan FPS
        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            frame_rate_calc = 1 / (t_stop - t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (resW - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)

        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("\nMembersihkan sumber daya...")
    status_payload = json.dumps({"status": "offline"})
    client.publish(STATUS_TOPIC, status_payload)
    print(f"PUBLISH: Mengirim status OFFLINE ke {STATUS_TOPIC}")
    time.sleep(0.5)

    cam.release()
    cv2.destroyAllWindows()
    # Tutup semua instance perangkat
    for device in devices.values():
        device["instance"].close()
    client.loop_stop()
    client.disconnect()
    print("Selesai.")