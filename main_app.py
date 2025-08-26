# main_app.py (Diperbarui untuk Raspberry Pi 5 dengan gpiod)
# -----------------------------
# Import semua library yang dibutuhkan
# -----------------------------
import paho.mqtt.client as mqtt
import json
import time
import cv2
import gpiod  # <-- Ganti RPi.GPIO dengan gpiod
from ultralytics import YOLO

# ====================================================================
# PENTING: UBAH KONFIGURASI DI BAWAH INI SESUAI DENGAN SETUP ANDA
# ====================================================================

# --- Konfigurasi MQTT ---
MQTT_BROKER = "192.168.1.5"
MQTT_PORT = 1883
MQTT_USERNAME = "your_mqtt_username"
MQTT_PASSWORD = "your_mqtt_password"

# --- Konfigurasi Perangkat ---
DEVICE_IP_ADDRESS = "192.168.1.101" 
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action" 

# --- Konfigurasi Model & Kamera ---
MODEL_PATH = 'yolov8n.pt'
CONF_THRESHOLD = 0.45
CAMERA_INDEX = 0

# --- Konfigurasi Pin GPIO ---
# Nomor pin tetap sama (mode BCM)
LAMP_PIN = 23
FAN_PIN = 24
# Nama GPIO chip untuk Raspberry Pi 5 biasanya 'gpiochip4'
GPIO_CHIP = 'gpiochip4'

# ====================================================================
# KODE UTAMA (UMUMNYA TIDAK PERLU DIUBAH)
# ====================================================================

# Variabel global untuk menampung objek line GPIO
lamp_line = None
fan_line = None
chip = None

# --- Logika Kontrol Hardware dengan gpiod ---
def setup_gpio():
    """Mengatur pin GPIO menggunakan gpiod."""
    global lamp_line, fan_line, chip
    try:
        chip = gpiod.Chip(GPIO_CHIP)
        # Minta akses ke line/pin yang akan digunakan sebagai OUTPUT
        lamp_line = chip.get_line(LAMP_PIN)
        fan_line = chip.get_line(FAN_PIN)
        
        lamp_line.request(consumer="lamp", type=gpiod.Line.DIRECTION_OUTPUT)
        fan_line.request(consumer="fan", type=gpiod.Line.DIRECTION_OUTPUT)

        # Matikan relay saat program dimulai
        lamp_line.set_value(0)
        fan_line.set_value(0)
        
        print("✅ GPIO dan Relay siap (menggunakan gpiod)...")
    except Exception as e:
        print(f"❌ Gagal setup GPIO: {e}")
        print("Pastikan nama GPIO_CHIP sudah benar dan Anda memiliki izin.")
        exit()

def control_device(device, action):
    """Mengontrol relay berdasarkan perintah dari MQTT."""
    line = None
    if device == "lamp":
        line = lamp_line
    elif device == "fan":
        line = fan_line
    else:
        print(f"Peringatan: Tipe perangkat '{device}' tidak dikenali.")
        return

    # 1 untuk ON (HIGH), 0 untuk OFF (LOW)
    value = 1 if action == "turn_on" else 0
    line.set_value(value)
    print(f"🚀 AKSI: Menjalankan '{action}' pada '{device}' (Pin {line.offset()})")

# --- Logika Deteksi Orang dengan YOLO (Tetap Sama) ---
def setup_camera_and_model():
    # ... (fungsi ini tidak berubah) ...
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
    # ... (fungsi ini tidak berubah) ...
    ret, frame = camera.read()
    if not ret:
        return False
    results = model.predict(frame, verbose=False)
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0 and box.conf > CONF_THRESHOLD:
                return True
    return False

# --- Fungsi-fungsi MQTT (dengan perbaikan warning) ---
def on_connect(client, userdata, flags, rc, properties=None):
    # ... (fungsi ini tidak berubah) ...
    if rc == 0:
        print(f"✅ Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        print(f"👂 SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
    else:
        print(f"❌ Gagal terhubung ke MQTT, kode error: {rc}")

def on_message(client, userdata, msg):
    # ... (fungsi ini tidak berubah) ...
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
    # Perbaikan untuk DeprecationWarning dari paho-mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    
    setup_gpio()
    camera, model = setup_camera_and_model()
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start() 
        
        is_person_active = False
        print("\n🚀 Sistem deteksi mulai berjalan. Tekan CTRL+C untuk berhenti.")
        while True:
            person_found = detect_person(camera, model)
            
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
        print("🧹 Membersihkan sumber daya...")
        camera.release()
        # Melepaskan line GPIO saat program selesai
        if lamp_line:
            lamp_line.release()
        if fan_line:
            fan_line.release()
        if chip:
            chip.close()
        client.loop_stop()
        client.disconnect()
        print("Selesai.")