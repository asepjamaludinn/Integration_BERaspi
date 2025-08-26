# main_app.py
# -----------------------------
# Import semua library yang dibutuhkan
# -----------------------------
import paho.mqtt.client as mqtt
import json
import time
import cv2
import RPi.GPIO as GPIO
from ultralytics import YOLO

# ====================================================================
# PENTING: UBAH KONFIGURASI DI BAWAH INI SESUAI DENGAN SETUP ANDA
# ====================================================================

# --- Konfigurasi MQTT ---
MQTT_BROKER = "192.168.1.5"  # Ganti dengan IP Address komputer/server tempat MQTT Broker berjalan
MQTT_PORT = 1883
MQTT_USERNAME = "your_mqtt_username" # Sesuaikan dengan .env backend
MQTT_PASSWORD = "your_mqtt_password" # Sesuaikan dengan .env backend

# --- Konfigurasi Perangkat ---
# Ganti dengan IP Address unik dari Raspberry Pi ini
DEVICE_IP_ADDRESS = "192.168.1.101" 
# Topik MQTT akan dibuat secara otomatis berdasarkan IP ini
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action" 

# --- Konfigurasi Model & Kamera ---
MODEL_PATH = 'yolov8n.pt' # Model nano, yang paling ringan
CONF_THRESHOLD = 0.45    # Tingkat kepercayaan minimum (45%) untuk mendeteksi orang
CAMERA_INDEX = 0         # 0 untuk webcam USB atau Pi Camera

# --- Konfigurasi Pin GPIO ---
# Sesuaikan nomor pin ini dengan koneksi relay Anda
LAMP_PIN = 23
FAN_PIN = 24

# ====================================================================
# KODE UTAMA (UMUMNYA TIDAK PERLU DIUBAH)
# ====================================================================

# --- Logika Kontrol Hardware ---
def setup_gpio():
    """Mengatur mode pin GPIO dan menginisialisasi relay."""
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LAMP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(FAN_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("✅ GPIO dan Relay siap...")

def control_device(device, action):
    """Mengontrol relay berdasarkan perintah dari MQTT."""
    pin = None
    if device == "lamp":
        pin = LAMP_PIN
    elif device == "fan":
        pin = FAN_PIN
    else:
        print(f"Peringatan: Tipe perangkat '{device}' tidak dikenali.")
        return

    state = GPIO.HIGH if action == "turn_on" else GPIO.LOW
    GPIO.output(pin, state)
    print(f"🚀 AKSI: Menjalankan '{action}' pada '{device}' (Pin {pin})")

# --- Logika Deteksi Orang dengan YOLO ---
def setup_camera_and_model():
    """Mempersiapkan kamera dan memuat model YOLO."""
    print("⏳ Mempersiapkan kamera dan memuat model YOLOv8...")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not camera.isOpened():
        raise IOError("❌ Tidak bisa membuka kamera.")
        
    model = YOLO(MODEL_PATH)
    print("✅ Kamera dan model siap.")
    return camera, model

def detect_person(camera, model):
    """Mengambil frame dan mendeteksi objek 'person'."""
    ret, frame = camera.read()
    if not ret:
        print("Gagal mengambil frame.")
        return False

    results = model.predict(frame, verbose=False)
    
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0 and box.conf > CONF_THRESHOLD:
                return True # Ditemukan orang
    return False # Tidak ditemukan orang

# --- Fungsi-fungsi MQTT ---
def on_connect(client, userdata, flags, rc):
    """Callback saat berhasil terhubung ke broker."""
    if rc == 0:
        print(f"✅ Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        print(f"👂 SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
    else:
        print(f"❌ Gagal terhubung ke MQTT, kode error: {rc}")

def on_message(client, userdata, msg):
    """Callback saat menerima pesan dari topik yang di-subscribe."""
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
    # Inisialisasi MQTT Client
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Inisialisasi Hardware & Software
    setup_gpio()
    camera, model = setup_camera_and_model()
    
    try:
        # Coba terhubung ke broker
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start() # Menjalankan loop MQTT di thread terpisah
        
        is_person_active = False
        print("\n🚀 Sistem deteksi mulai berjalan. Tekan CTRL+C untuk berhenti.")
        while True:
            person_found = detect_person(camera, model)
            
            # Kirim pesan ke backend HANYA saat status deteksi berubah
            if person_found and not is_person_active:
                is_person_active = True
                payload = json.dumps({"motion_detected": True})
                client.publish(SENSOR_TOPIC, payload)
                print(f"📡 PUBLISH ke {SENSOR_TOPIC}: Orang Terdeteksi!")
            
            elif not person_found and is_person_active:
                is_person_active = False
                payload = json.dumps({"motion_cleared": True})
                client.publish(SENSOR_TOPIC, payload)
                print(f"📡 PUBLISH ke {SENSOR_TOPIC}: Orang Tidak Terdeteksi!")

    except KeyboardInterrupt:
        print("\n🛑 Program dihentikan oleh pengguna.")
    except Exception as e:
        print(f"\n Terjadi error: {e}")
    finally:
        # Membersihkan semua sumber daya saat program berhenti
        print("🧹 Membersihkan sumber daya...")
        camera.release()
        GPIO.cleanup()
        client.loop_stop()
        client.disconnect()
        print("Selesai.")