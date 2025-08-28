import cv2
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO


MQTT_BROKER = "10.249.70.108"
MQTT_PORT = 1883
MQTT_USERNAME = "cpsmagang"
MQTT_PASSWORD = "cpsjaya123"
DEVICE_IP_ADDRESS = "10.249.70.108" 


STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"

try:
    print("Menginisialisasi model YOLO...")
    model_pose = YOLO("yolo11n-pose_ncnn_model", task="pose")
    
    cam_source = "usb0"
    resW, resH = 640, 480

    
    devices = {
        "lamp": {
            "instance": LED(26),
            "state": 0,      # 0: OFF, 1: ON
            "mode": "auto",  # 'auto' atau 'manual'. Mode 'scheduled' dikelola server.
            "is_person_reported": False
        },
        "fan": {
            "instance": LED(19),
            "state": 0,
            "mode": "auto",
            "is_person_reported": False
        }
    }
    print("Perangkat (lamp, fan) telah diinisialisasi.")


    print("Mencari kamera...")
    if "usb" in cam_source:
        cam_idx = int(cam_source[3:])
        cam = cv2.VideoCapture(cam_idx)
        cam.set(3, resW)
        cam.set(4, resH)
        if not cam.isOpened():
            print("ERROR: Gagal membuka kamera.")
            exit()
    else:
        print("ERROR: Tidak ada kamera terdeteksi.")
        exit()
    print("Kamera siap.")

except Exception as e:
    print(f"FATAL ERROR saat inisialisasi: {e}")
    exit()


consecutive_detections = 0
DETECTION_THRESHOLD_ON = 8   # Jumlah frame berurutan untuk menyalakan
DETECTION_THRESHOLD_OFF = 0  # Jumlah frame berurutan untuk mematikan
DETECTION_MAX_FRAMES = 10    # Buffer frame deteksi

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback yang dipanggil saat berhasil terhubung ke broker MQTT."""
    if rc == 0:
        print(f"Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        print(f"SUBSCRIBE ke topik aksi: {ACTION_TOPIC}")
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        print(f"SUBSCRIBE ke topik settings: {SETTINGS_UPDATE_TOPIC}")


        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload, qos=1, retain=True)
        print(f"PUBLISH: Mengirim status ONLINE ke {STATUS_TOPIC}")
    else:
        print(f"Gagal terhubung ke MQTT, kode error: {rc}. Mencoba lagi...")

def on_message(client, userdata, msg):
    """Callback yang dipanggil saat menerima pesan dari topik yang di-subscribe."""
    global devices
    print(f"PESAN DITERIMA di topik {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        
     
        if msg.topic == ACTION_TOPIC:
            device_name = payload.get("device")
            action = payload.get("action")
            
            if device_name in devices and action in ["turn_on", "turn_off"]:
            
                devices[device_name]["mode"] = "manual"
                
                if action == "turn_on":
                    devices[device_name]["instance"].on()
                    devices[device_name]["state"] = 1
                elif action == "turn_off":
                    devices[device_name]["instance"].off()
                    devices[device_name]["state"] = 0
                
                print(f"AKSI EKSTERNAL: '{action}' pada '{device_name}'. Mode diubah ke MANUAL.")
        
        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device_name = payload.get("device")
            if device_name in devices and "mode" in payload:
                new_mode = payload["mode"]
                if new_mode in ["auto", "manual", "scheduled"]:
                    # Jika server menyetel mode 'scheduled', perangkat akan menganggapnya 'manual'
                    # karena perangkat hanya menunggu perintah, bukan menjalankan logika jadwal.
                    effective_mode = "manual" if new_mode == "scheduled" else new_mode
                    devices[device_name]["mode"] = effective_mode
                    print(f"SETTINGS UPDATE: Mode '{device_name}' diubah menjadi {effective_mode.upper()}")

    except Exception as e:
        print(f"Error memproses pesan MQTT: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print("\nSistem deteksi mulai berjalan. Tekan 'Q' di jendela CV2 untuk berhenti.")
try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret:
            print("Peringatan: Gagal mengambil frame dari kamera.")
            time.sleep(1)
            continue

     
        results = model_pose.predict(frame, verbose=False)
        annotated_frame = results[0].plot()
        pose_found = len(results) > 0 and len(results[0].keypoints) > 0

       
        if pose_found:
            consecutive_detections = min(consecutive_detections + 1, DETECTION_MAX_FRAMES)
        else:
            consecutive_detections = max(consecutive_detections - 1, 0)
        
        should_be_active = consecutive_detections >= DETECTION_THRESHOLD_ON
        should_be_inactive = consecutive_detections <= DETECTION_THRESHOLD_OFF
        
      
        for name, device in devices.items():
          
            if device["mode"] == "auto":
                action_taken = False
                if should_be_active and device["state"] == 0:
                    device["instance"].on()
                    device["state"] = 1
                    action_taken = True
                elif should_be_inactive and device["state"] == 1:
                    device["instance"].off()
                    device["state"] = 0
                    action_taken = True
                
                if action_taken:
                    print(f"AKSI OTOMATIS: Perangkat '{name}' diubah ke status {'ON' if device['state'] == 1 else 'OFF'}")

              
                if should_be_active and not device["is_person_reported"]:
                    device["is_person_reported"] = True
                    payload = json.dumps({"motion_detected": True}) 
                    client.publish(SENSOR_TOPIC, payload)
                    print(f"PUBLISH: Laporan 'motion_detected' dikirim ke {SENSOR_TOPIC}")
                elif should_be_inactive and device["is_person_reported"]:
                    device["is_person_reported"] = False
                    payload = json.dumps({"motion_cleared": True}) 
                    client.publish(SENSOR_TOPIC, payload)
                    print(f"PUBLISH: Laporan 'motion_cleared' dikirim ke {SENSOR_TOPIC}")
      
        y_pos = 30
        for name, device in devices.items():
            mode_text = f"{name.upper()} Mode: {device['mode'].upper()}"
            status_text = f"{name.upper()} Status: {'ON' if device['state'] == 1 else 'OFF'}"
            
            color_mode = (0, 255, 255) 
            color_status = (0, 255, 0) if device['state'] == 1 else (0, 0, 255)
            
            cv2.putText(annotated_frame, mode_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, .6, color_mode, 2)
            y_pos += 30
            cv2.putText(annotated_frame, status_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, .6, color_status, 2)
            y_pos += 40

        cv2.imshow("Smart Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("\nMembersihkan sumber daya dan mematikan sistem...")
    
    
    status_payload = json.dumps({"status": "offline"})
    client.publish(STATUS_TOPIC, status_payload, qos=1, retain=True)
    print(f"PUBLISH: Mengirim status OFFLINE terakhir ke {STATUS_TOPIC}")
    time.sleep(0.5) 

    cam.release()
    cv2.destroyAllWindows()
    
   
    for name, device in devices.items():
        print(f"Mematikan {name}...")
        device["instance"].off()
        device["instance"].close()
        
    client.loop_stop()
    client.disconnect()
    print("Selesai.")
