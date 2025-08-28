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