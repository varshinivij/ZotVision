npm# ZotVision

Real-time firefighter monitoring system. ESP32-CAM modules stream live video, an ESP32-C3 GPS tracker reports location, a Python Flask backend runs ML inference, and a React dashboard displays everything.

---

## System Overview

```
ESP32-CAM  ──┐
              ├──► Flask Backend (Python) ──► React Dashboard (browser)
ESP32-C3   ──┘     (ML inference)              (live feeds + GPS)
(GPS)
```

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/) | Flash ESP32-CAM firmware |
| [Arduino IDE 2.x](https://www.arduino.cc/en/software) | Flash GPS tracker firmware |
| Python 3.10+ | Run backend |
| Node.js 18+ | Run frontend |

---

## 1. ESP32-CAM Firmware (ESP-IDF)

The camera module uses the ESP-IDF build system — **do not use Arduino IDE for this one**.

### Setup

1. Install ESP-IDF and open an **ESP-IDF terminal** (the one that sets up the environment variables).

2. Navigate to the firmware directory:
   ```bash
   cd zot-vision/firmware
   ```

3. Configure your WiFi credentials and server IP in `main/main.cpp`:
   ```cpp
   #define WIFI_SSID     "your_wifi_name"
   #define WIFI_PASSWORD "your_wifi_password"
   #define SERVER_URL    "http://<your-pc-ip>:5000/api/image"
   ```
   > The server IP should be the machine running the Python backend. On Windows you can find it with `ipconfig` — look for the hotspot or LAN adapter address.

4. Build and flash (replace `PORT` with your COM port, e.g. `COM3`):
   ```bash
   idf.py build
   idf.py -p PORT flash
   idf.py -p PORT monitor
   ```
   The monitor output will show WiFi connection status and frame-send logs.

---

## 2. GPS Tracker Firmware (Arduino IDE)

The GPS tracker runs on an ESP32-C3 and is flashed via **Arduino IDE**.

### Required Libraries

Install these via Arduino IDE → **Library Manager** (Sketch → Include Library → Manage Libraries):

- `TinyGPSPlus` by Mikal Hart
- `Adafruit SSD1306`
- `Adafruit GFX Library`

### Board Setup

1. In Arduino IDE, go to **File → Preferences** and add this URL to "Additional boards manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
2. Go to **Tools → Board → Boards Manager**, search for `esp32`, and install the Espressif package.
3. Select **Tools → Board → ESP32C3 Dev Module**.

### Flash

1. Open `zot-vision/firmware/gps/gps_tracker.ino` in Arduino IDE.

2. Update the WiFi credentials and server IP near the top of the file:
   ```cpp
   const char* ssid     = "your_wifi_name";
   const char* password = "your_wifi_password";
   const char* serverUrl = "http://<your-pc-ip>:5000/api/gps";
   ```

3. Select the correct COM port under **Tools → Port**.

4. Click **Upload**.

**Wiring:**
| ESP32-C3 Pin | Connects To |
|---|---|
| GPIO4 (RX) | GPS module TX |
| GPIO5 (TX) | GPS module RX |
| 3.3V / GND | GPS module power |
| SDA / SCL | SSD1306 OLED display |

---

## 3. Python Backend

### Install Dependencies

```bash
cd zot-vision/backend
pip install flask flask-cors torch torchvision transformers efficientnet_pytorch opencv-python Pillow
```

> If you have a CUDA GPU, install the CUDA version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for faster inference.

### ML Model Weights

The backend requires a trained model at `datasets/results/model_weights.pth`. If you don't have weights yet, run training first:

```bash
cd zot-vision/backend
python transformer.py
```

This trains on the images in `datasets/images/` using labels from `datasets/results/labels.txt`. Training takes a while depending on your hardware.

### Run the Backend

```bash
cd zot-vision/backend
python api.py
```

The server starts on `http://0.0.0.0:5000`. You should see:
```
 * Running on http://0.0.0.0:5000
```

Leave this terminal open — it receives frames from the ESP32-CAMs and GPS data from the trackers.

---

## 4. React Frontend (Vite)

### Install Dependencies

```bash
cd zot-vision
npm install
```

### Run the Dev Server

```bash
npm run dev
```

Open the URL shown in the terminal (usually `http://localhost:5173`) in your browser. The dashboard polls the backend every 100ms and displays live feeds for up to 5 firefighters with ML predictions and GPS coordinates.

> The frontend proxies API requests to `http://localhost:5000` — make sure the backend is running first.

---

## Startup Order

Run these in order:

1. **Backend** — `python api.py` (must be running before devices connect)
2. **Frontend** — `npm run dev` (open in browser)
3. **Power on ESP32-CAM** — it will connect to WiFi and start streaming
4. **Power on GPS tracker** — it will connect and start posting location

---

## Troubleshooting

**ESP32-CAM won't connect to WiFi**
- Double-check SSID/password in `main.cpp`
- Make sure the PC and ESP32 are on the same network
- Check the IDF monitor output for error messages

**Backend receives no frames**
- Verify `SERVER_URL` in firmware points to the correct IP and port
- Check that port 5000 is not blocked by a firewall
- On Windows: allow Python through Windows Defender Firewall

**Frontend shows all firefighters as offline**
- Confirm the backend is running on port 5000
- Check browser console for CORS or network errors

**GPS not showing on dashboard**
- Check OLED display — if it shows "No Fix", the GPS module needs a clear view of the sky
- Verify the serial wiring (RX/TX are often swapped)
