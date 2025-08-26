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

# --- Konfigurasi Model & Kamera ---
MODEL_PATH = 'yolov8n-pose.pt'
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Konfigurasi Pin GPIO ---
LAMP_PIN = 26 


# Variabel global
lamp = None
lamp_is_on = False # Variabel untuk melacak status lampu sebenarnya

# --- Logika Kontrol Hardware (dipanggil oleh MQTT) ---
def setup_gpio():
    global lamp
    lamp = LED(LAMP_PIN)
    print("✅ GPIO untuk Lampu siap...")

def control_device(device, action):
    global lamp_is_on
    if device == "lamp":
        if action == "turn_on":
            lamp.on()
            lamp_is_on = True
        elif action == "turn_off":
            lamp.off()
            lamp_is_on = False
        print(f"🚀 AKSI DARI BACKEND: Menjalankan '{action}' pada '{device}'")
    else:
        print(f"Peringatan: Perangkat '{device}' tidak dikenali.")

# --- Fungsi-fungsi MQTT ---
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"✅ Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        print(f"👂 SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
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

# --- Program Utama ---
if __name__ == "__main__":
    # Inisialisasi MQTT
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Inisialisasi Hardware
    setup_gpio()

    # Inisialisasi Kamera & Model
    print("⏳ Mempersiapkan kamera dan memuat model YOLOv8-Pose...")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera.isOpened():
        raise IOError("❌ Tidak bisa membuka kamera.")
    model = YOLO(MODEL_PATH)
    print("✅ Kamera dan model siap.")

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()

        # Variabel untuk logika deteksi canggih Anda
        consecutive_detections = 0
        is_person_reported = False # Melacak status terakhir yang dilaporkan ke backend
        fps_buffer = []
        fps_avg_len = 50

        print("\n🚀 Sistem deteksi mulai berjalan. Tekan 'Q' untuk berhenti.")
        while True:
            t_start = time.perf_counter()
            
            ret, frame = camera.read()
            if not ret:
                print("Peringatan: Gagal mengambil frame.")
                break

            # Menjalankan deteksi
            results = model.predict(frame, verbose=False)
            
            # Mendapatkan frame dengan anotasi (bounding box, keypoints)
            annotated_frame = results[0].plot()

            # Logika deteksi pose
            pose_found = len(results) > 0 and len(results[0].keypoints) > 0

            # Logika smoothing
            if pose_found:
                consecutive_detections = min(consecutive_detections + 1, 10)
            else:
                consecutive_detections = max(consecutive_detections - 1, 0)
            
            should_be_active = consecutive_detections >= 8
            should_be_inactive = consecutive_detections == 0

            # Daripada menyalakan lampu langsung, kita PUBLISH status ke backend
            if should_be_active and not is_person_reported:
                is_person_reported = True
                payload = json.dumps({"motion_detected": True})
                client.publish(SENSOR_TOPIC, payload)
                print(f"📡 PUBLISH ke {SENSOR_TOPIC}: Pose Terdeteksi!")
            
            elif should_be_inactive and is_person_reported:
                is_person_reported = False
                payload = json.dumps({"motion_cleared": True})
                client.publish(SENSOR_TOPIC, payload)
                print(f"📡 PUBLISH ke {SENSOR_TOPIC}: Pose Tidak Terdeteksi!")

            # Menampilkan FPS dan Status Lampu di layar
            t_stop = time.perf_counter()
            frame_rate_calc = 1 / (t_stop - t_start)
            fps_buffer.append(frame_rate_calc)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)
            avg_frame_rate = np.mean(fps_buffer)
            
            cv2.putText(annotated_frame, f'FPS: {avg_frame_rate:.2f}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # Status lampu dibaca dari variabel global yang diubah oleh MQTT
            if lamp_is_on:
                cv2.putText(annotated_frame, "Light ON", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(annotated_frame, "Light OFF", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Smart Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Program dihentikan oleh pengguna.")
    except Exception as e:
        print(f"\n Terjadi error: {e}")
    finally:
        print("🧹 Membersihkan sumber daya...")
        camera.release()
        cv2.destroyAllWindows()
        if lamp:
            lamp.close()
        client.loop_stop()
        client.disconnect()
        print("Selesai.")