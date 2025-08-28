import cv2
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO


MQTT_BROKER = "192.168.0.174"
MQTT_PORT = 1883
MQTT_USERNAME = "cpsmagang"
MQTT_PASSWORD = "cpsjaya123"
DEVICE_IP_ADDRESS = "192.168.0.174"

# --- Konfigurasi Perangkat Keras ---
LAMP_PIN = 26
FAN_PIN = 19
CAM_SOURCE = "usb0"
CAM_WIDTH = 640
CAM_HEIGHT = 480
YOLO_MODEL_PATH = "yolo12n-pose_ncnn_model"

# --- Konfigurasi Logika Deteksi ---
# Hanya proses 1 dari setiap N frame untuk efisiensi
PROCESS_EVERY_N_FRAMES = 5 
# Berapa frame deteksi berturut-turut dibutuhkan untuk menyalakan perangkat
DETECTION_THRESHOLD_ON = 8
# Berapa frame non-deteksi berturut-turut dibutuhkan untuk mematikan perangkat
DETECTION_THRESHOLD_OFF = 0


# --- Topik MQTT ---
STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"

try:
    print("Menginisialisasi perangkat keras dan model AI...")
    model_pose = YOLO(YOLO_MODEL_PATH, task="pose")
    lamp = LED(LAMP_PIN)
    fan = LED(FAN_PIN)
    
    cam = cv2.VideoCapture(int(CAM_SOURCE[3:]))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cam.isOpened(): raise Exception("Gagal membuka kamera.")
    print("Kamera, GPIO, dan Model AI siap.")
except Exception as e:
    print(f"Error fatal saat inisialisasi: {e}")
    exit()

# --- Inisialisasi State Awal ---
# Dictionary untuk menyimpan mode otomatis setiap perangkat
device_auto_modes = { "lamp": True, "fan": True }
# Dictionary untuk menyimpan status ON/OFF setiap perangkat
device_states = { "lamp": 0, "fan": 0 } # 0 = OFF, 1 = ON

consecutive_detections = 0
is_person_reported = False
fps_buffer = []
fps_avg_len = 50

# Variabel untuk optimasi frame processing
frame_counter = 0
last_pose_found = False


def control_device(device, action):
    """Menangani perintah manual dari backend."""
    global device_states, device_auto_modes
    
    target_device_obj = None
    if device == "lamp": target_device_obj = lamp
    elif device == "fan": target_device_obj = fan
    
    if target_device_obj:
        if action == "turn_on":
            target_device_obj.on()
            device_states[device] = 1
        else:
            target_device_obj.off()
            device_states[device] = 0
    
    # Aksi manual akan mengubah mode perangkat yang bersangkutan menjadi MANUAL
    if device in device_auto_modes:
        device_auto_modes[device] = False
        print(f"AKSI MANUAL: Mode Otomatis untuk '{device}' kini DINONAKTIFKAN.")

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback saat berhasil terhubung ke MQTT Broker."""
    if rc == 0:
        print(f"Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        # Status online sebaiknya retain, agar backend tahu status terakhir saat reconnect
        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload, qos=1, retain=True)
    else:
        print(f"Gagal terhubung ke MQTT, kode error: {rc}")

def on_message(client, userdata, msg):
    """Callback saat menerima pesan dari MQTT Broker."""
    global device_auto_modes
    print(f"PESAN DITERIMA di topik {msg.topic}")
    try:
        if not msg.payload: return # Abaikan pesan kosong

        payload = json.loads(msg.payload.decode())
        
        if msg.topic == ACTION_TOPIC:
            device = payload.get("device")
            action = payload.get("action")
            if device and action: control_device(device, action)

        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device = payload.get("device")
            if device in device_auto_modes and "auto_mode_enabled" in payload:
                new_mode = payload["auto_mode_enabled"]
                device_auto_modes[device] = new_mode
                mode_status = "DIAKTIFKAN" if new_mode else "DINONAKTIFKAN"
                print(f"SETTINGS UPDATE: Mode Otomatis untuk '{device}' kini {mode_status}")
    except json.JSONDecodeError:
        print(f"Gagal mem-parsing JSON dari payload: {msg.payload}")
    except Exception as e:
        print(f"Error memproses pesan: {e}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
# Last Will: jika koneksi terputus, kirim status "offline"
last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()


print("\nSistem deteksi mulai berjalan. Tekan 'Q' di jendela video untuk berhenti.")
try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret: 
            print("Peringatan: Gagal mengambil frame dari kamera.")
            time.sleep(1)
            continue

        annotated_frame = frame.copy()

        # Jalankan deteksi hanya jika ada setidaknya satu perangkat dalam mode auto
        if any(device_auto_modes.values()):
            frame_counter += 1
            
            # Optimasi: Hanya jalankan model AI pada frame ke-N
            if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                results = model_pose.predict(frame, verbose=False)
                if len(results) > 0 and results[0].keypoints is not None:
                    last_pose_found = True
                    annotated_frame = results[0].plot() # Gambar anotasi hanya saat diproses
                else:
                    last_pose_found = False

            # Logika debouncing tetap berjalan di setiap frame untuk responsivitas
            if last_pose_found:
                consecutive_detections = min(consecutive_detections + 1, 10)
            else:
                consecutive_detections = max(consecutive_detections - 1, 0)
            
            should_be_active = consecutive_detections >= DETECTION_THRESHOLD_ON
            should_be_inactive = consecutive_detections <= DETECTION_THRESHOLD_OFF
            
            # Kontrol setiap perangkat secara individual berdasarkan modenya
            if should_be_active:
                if device_auto_modes["lamp"] and device_states["lamp"] == 0: device_states["lamp"] = 1; lamp.on()
                if device_auto_modes["fan"] and device_states["fan"] == 0: device_states["fan"] = 1; fan.on()
            elif should_be_inactive:
                if device_auto_modes["lamp"] and device_states["lamp"] == 1: device_states["lamp"] = 0; lamp.off()
                if device_auto_modes["fan"] and device_states["fan"] == 1: device_states["fan"] = 0; fan.off()
            
            # Kirim laporan status deteksi ke backend
            if should_be_active and not is_person_reported:
                is_person_reported = True
                client.publish(SENSOR_TOPIC, json.dumps({"motion_detected": True}))
                print(f"PUBLISH (AUTO): Pose Terdeteksi!")
            elif should_be_inactive and is_person_reported:
                is_person_reported = False
                client.publish(SENSOR_TOPIC, json.dumps({"motion_cleared": True}))
                print(f"PUBLISH (AUTO): Pose Tidak Terdeteksi!")
        
        # --- Tampilan Status pada Layar ---
        lamp_mode = "AUTO" if device_auto_modes["lamp"] else "MANUAL"
        lamp_status_text = f"Lamp: {'ON' if device_states['lamp'] else 'OFF'} ({lamp_mode})"
        cv2.putText(annotated_frame, lamp_status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0) if device_states['lamp'] else (0, 0, 255), 2)

        fan_mode = "AUTO" if device_auto_modes["fan"] else "MANUAL"
        fan_status_text = f"Fan: {'ON' if device_states['fan'] else 'OFF'} ({fan_mode})"
        cv2.putText(annotated_frame, fan_status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0) if device_states['fan'] else (0, 0, 255), 2)
        
        # Tampilan FPS
        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            frame_rate_calc = 1/(t_stop-t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len: fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)

        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    print("\nMembersihkan sumber daya...")
    status_payload = json.dumps({"status": "offline"})
    client.publish(STATUS_TOPIC, status_payload)
    time.sleep(0.5)
    cam.release()
    cv2.destroyAllWindows()
    lamp.close()
    fan.close()
    client.loop_stop()
    client.disconnect()
    print("Selesai.")