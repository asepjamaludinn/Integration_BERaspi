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
    lamp = LED(26)
    fan = LED(19)
    cam_source = "usb0"
    resW, resH = 640, 480
    cam = cv2.VideoCapture(int(cam_source[3:]))
    cam.set(3, resW)
    cam.set(4, resH)
    if not cam.isOpened(): raise Exception("Gagal membuka kamera.")
    print("Kamera siap.")
except Exception as e:
    print(f"Error saat inisialisasi: {e}")
    exit()


device_auto_modes = {
    "lamp": True,
    "fan": True
}
lamp_state = 0
fan_state = 0
consecutive_detections = 0
is_person_reported = False
fps_buffer = []
fps_avg_len = 50


def control_device(device, action):
    global lamp_state, fan_state, device_auto_modes
    
    if device == "lamp":
        if action == "turn_on": lamp.on(); lamp_state = 1
        else: lamp.off(); lamp_state = 0
            
    elif device == "fan":
        if action == "turn_on": fan.on(); fan_state = 1
        else: fan.off(); fan_state = 0
    
    if device in device_auto_modes:
        device_auto_modes[device] = False
        print(f"AKSI MANUAL: Mode Otomatis untuk '{device}' kini DINONAKTIFKAN.")

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Terhubung ke MQTT Broker di {MQTT_BROKER}!")
        client.subscribe(ACTION_TOPIC)
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        status_payload = json.dumps({"status": "online"})
        client.publish(STATUS_TOPIC, status_payload)
    else:
        print(f"Gagal terhubung ke MQTT, kode error: {rc}")

def on_message(client, userdata, msg):
    global device_auto_modes
    print(f"PESAN DITERIMA di topik {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        if msg.topic == ACTION_TOPIC:
            device = payload.get("device")
            action = payload.get("action")
            if device and action:
                control_device(device, action)
        elif msg.topic == SETTINGS_UPDATE_TOPIC:
            device = payload.get("device")
            if device in device_auto_modes and "auto_mode_enabled" in payload:
                new_mode = payload["auto_mode_enabled"]
                device_auto_modes[device] = new_mode
                mode_status = "DIAKTIFKAN" if new_mode else "DINONAKTIFKAN"
                print(f"SETTINGS UPDATE: Mode Otomatis untuk '{device}' kini {mode_status}")
    except Exception as e:
        print(f"Error memproses pesan: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

print("\nSistem deteksi mulai berjalan.")
try:
    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret: break

        if any(device_auto_modes.values()):
            results = model_pose.predict(frame, verbose=False)
            annotated_frame = results[0].plot()
            pose_found = len(results) > 0 and results[0].keypoints is not None

            if pose_found:
                consecutive_detections = min(consecutive_detections + 1, 10)
            else:
                consecutive_detections = max(consecutive_detections - 1, 0)
            
            should_be_active = consecutive_detections >= 8
            should_be_inactive = consecutive_detections <= 0
            
            # Periksa dan kontrol setiap perangkat secara individual
            if should_be_active:
                if device_auto_modes["lamp"] and lamp_state == 0:
                    lamp_state = 1; lamp.on()
                if device_auto_modes["fan"] and fan_state == 0:
                    fan_state = 1; fan.on()
            elif should_be_inactive:
                if device_auto_modes["lamp"] and lamp_state == 1:
                    lamp_state = 0; lamp.off()
                if device_auto_modes["fan"] and fan_state == 1:
                    fan_state = 0; fan.off()
            
            if should_be_active and not is_person_reported:
                is_person_reported = True
                client.publish(SENSOR_TOPIC, json.dumps({"motion_detected": True}))
                print(f"PUBLISH (AUTO): Pose Terdeteksi!")
            elif should_be_inactive and is_person_reported:
                is_person_reported = False
                client.publish(SENSOR_TOPIC, json.dumps({"motion_cleared": True}))
                print(f"PUBLISH (AUTO): Pose Tidak Terdeteksi!")
        else:
            annotated_frame = frame

       
        lamp_mode = "AUTO" if device_auto_modes["lamp"] else "MANUAL"
        lamp_status_text = f"Lamp: {'ON' if lamp_state == 1 else 'OFF'} ({lamp_mode})"
        cv2.putText(annotated_frame, lamp_status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0) if lamp_state else (0, 0, 255), 2)

        fan_mode = "AUTO" if device_auto_modes["fan"] else "MANUAL"
        fan_status_text = f"Fan: {'ON' if fan_state == 1 else 'OFF'} ({fan_mode})"
        cv2.putText(annotated_frame, fan_status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0) if fan_state else (0, 0, 255), 2)
        
      
        t_stop = time.perf_counter()
        if (t_stop - t_start) > 0:
            avg_frame_rate = np.mean(fps_buffer) if fps_buffer else 0
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
