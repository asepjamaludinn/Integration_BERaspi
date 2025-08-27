import cv2
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED  
from ultralytics import YOLO

# --- KONFIGURASI ---
MQTT_BROKER = "192.168.0.174"
MQTT_PORT = 1883
MQTT_USERNAME = "cpsmagang"
MQTT_PASSWORD = "cpsjaya123"
DEVICE_IP_ADDRESS = "192.168.0.174"

# --- TOPIK MQTT ---
STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"


try:
    model_pose = YOLO("yolo12n-pose_ncnn_model", task="pose")
    lamp = LED(19) 
    fan = LED(26)  

    cam_source = "usb0"
    resW, resH = 640, 480
    
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

auto_mode_enabled = True
lamp_state = 0  # 0 = OFF, 1 = ON
fan_state = 0   # 0 = OFF, 1 = ON
consecutive_detections = 0
is_person_reported = False
fps_buffer = []
fps_avg_len = 50


def control_device(device, action):
    """Fungsi ini sekarang menangani 'lamp' dan 'fan' secara terpisah."""
    global lamp_state, fan_state, auto_mode_enabled
    
    # Logika untuk Lampu
    if device == "lamp":
        if action == "turn_on":
            lamp.on()
            lamp_state = 1
        elif action == "turn_off":
            lamp.off()
            lamp_state = 0
            
    # Logika untuk Kipas
    elif device == "fan":
        if action == "turn_on":
            fan.on()
            fan_state = 1
        elif action == "turn_off":
            fan.off()
            fan_state = 0
    
    # Aksi manual apapun akan mematikan mode otomatis
    auto_mode_enabled = False
    print(f"AKSI MANUAL: Menjalankan '{action}' pada '{device}'. Mode Otomatis kini DINONAKTIFKAN.")

# --- FUNGSI CALLBACK MQTT (Tidak ada perubahan di sini) ---
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
    global auto_mode_enabled
    print(f"PESAN DITERIMA di topik {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        if msg.topic == ACTION_TOPIC:
            device = payload.get("device")
            action = payload.get("action")
            if device and action:
                control_device(device, action)
        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            if "auto_mode_enabled" in payload:
                auto_mode_enabled = payload["auto_mode_enabled"]
                mode_status = "DIAKTIFKAN" if auto_mode_enabled else "DINONAKTIFKAN"
                print(f"SETTINGS UPDATE: Mode Otomatis kini {mode_status}")
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

        # --- [DIUBAH] Mode otomatis HANYA mengontrol lampu ---
        if auto_mode_enabled:
            results = model_pose.predict(frame, verbose=False)
            annotated_frame = results[0].plot()
            pose_found = len(results) > 0 and len(results[0].keypoints) > 0

            if pose_found:
                consecutive_detections = min(consecutive_detections + 1, 10)
            else:
                consecutive_detections = max(consecutive_detections - 1, 0)
            
            # Logika hanya memeriksa dan mengubah 'lamp_state'
            should_be_active = consecutive_detections >= 8 and lamp_state == 0
            should_be_inactive = consecutive_detections <= 0 and lamp_state == 1

            if should_be_active:
                lamp_state = 1
                lamp.on()
            elif should_be_inactive:
                lamp_state = 0
                lamp.off()
            
            if should_be_active and not is_person_reported:
                is_person_reported = True
                payload = json.dumps({"motion_detected": True})
                client.publish(SENSOR_TOPIC, payload)
                print(f"PUBLISH (AUTO): Pose Terdeteksi, lampu menyala!")
            elif should_be_inactive and is_person_reported:
                is_person_reported = False
                payload = json.dumps({"motion_cleared": True})
                client.publish(SENSOR_TOPIC, payload)
                print(f"PUBLISH (AUTO): Pose Tidak Terdeteksi, lampu mati!")
        else:
            annotated_frame = frame

        # --- [DIUBAH] Tampilkan status terpisah untuk setiap perangkat ---
        # Tampilkan status Lampu
        lamp_status_text = f"Lamp: {'ON' if lamp_state == 1 else 'OFF'}"
        lamp_color = (0, 255, 0) if lamp_state == 1 else (0, 0, 255)
        cv2.putText(annotated_frame, lamp_status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, lamp_color, 2)

        # Tampilkan status Kipas
        fan_status_text = f"Fan: {'ON' if fan_state == 1 else 'OFF'}"
        fan_color = (0, 255, 0) if fan_state == 1 else (0, 0, 255)
        cv2.putText(annotated_frame, fan_status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, fan_color, 2)

        # Tampilkan status Mode
        mode_text = f"MODE: {'AUTO' if auto_mode_enabled else 'MANUAL OVERRIDE'}"
        mode_color = (0, 255, 255) if auto_mode_enabled else (255, 0, 255)
        cv2.putText(annotated_frame, mode_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, .7, mode_color, 2)
        
        # Kalkulasi dan tampilkan FPS
        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            frame_rate_calc = 1 / (t_stop - t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)

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
    # --- [DIUBAH] Tutup koneksi GPIO terpisah ---
    lamp.close()
    fan.close()
    client.loop_stop()
    client.disconnect()
    print("Selesai.")
