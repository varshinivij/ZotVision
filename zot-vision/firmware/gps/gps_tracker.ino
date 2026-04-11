#include <TinyGPS++.h>
#include <HardwareSerial.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ── Credentials / server ──────────────────────────────────────
// CHANGE THESE:
#define WIFI_SSID        "KAUSHIK25"
#define WIFI_PASSWORD    "12345678"
#define SERVER_URL       "http://192.168.137.1:5000/api/gps"
#define FIREFIGHTER_ID   0          // which firefighter slot this GPS belongs to
#define GPS_SEND_INTERVAL_MS 1000   // how often to POST (ms), when fix is valid
// ─────────────────────────────────────────────────────────────

// ── OLED config ───────────────────────────────────────────────
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64
#define OLED_RESET     -1
#define OLED_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ── GPS config (ESP32-C3: UART1) ──────────────────────────────
#define GPS_RX_PIN  4
#define GPS_TX_PIN  5
#define GPS_BAUD 9600

HardwareSerial GPSSerial(1);
TinyGPSPlus    gps;

// ── State ─────────────────────────────────────────────────────
enum Page { PAGE_LOCATION, PAGE_ALTITUDE, PAGE_SPEED, PAGE_SATS };
Page     currentPage  = PAGE_LOCATION;
uint32_t lastPageSwap = 0;
uint32_t lastGpsSend  = 0;
const uint16_t PAGE_DURATION = 4000;

// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  GPSSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  Wire.begin(8, 9);  // SDA = GPIO8, SCL = GPIO9

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("SSD1306 not found – check wiring/address");
    while (true);
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  splashScreen();

  // Connect to WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  uint32_t wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 10000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWiFi failed – GPS display will still work");
  }
}

// ─────────────────────────────────────────────────────────────
void loop() {
  while (GPSSerial.available())
    gps.encode(GPSSerial.read());

  if (millis() - lastPageSwap >= PAGE_DURATION) {
    lastPageSwap = millis();
    currentPage  = static_cast<Page>((currentPage + 1) % 4);
  }

  drawPage(currentPage);

  // Send GPS to backend when we have a valid fix
  if (gps.location.isValid() &&
      gps.location.age() < 2000 &&
      millis() - lastGpsSend >= GPS_SEND_INTERVAL_MS &&
      WiFi.status() == WL_CONNECTED) {
    sendGPS();
    lastGpsSend = millis();
  }

  delay(250);
}

// ── HTTP POST GPS to backend ──────────────────────────────────
void sendGPS() {
  double lat = gps.location.lat();
  double lon = gps.location.lng();
  double alt = gps.altitude.isValid() ? gps.altitude.meters() : 0.0;

  char body[128];
  snprintf(body, sizeof(body),
           "{\"firefighter_id\":%d,\"lat\":%.6f,\"lon\":%.6f,\"alt\":%.2f}",
           FIREFIGHTER_ID, lat, lon, alt);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  if (code > 0) {
    Serial.printf("GPS POST %d  lat=%.6f lon=%.6f alt=%.1f\n", code, lat, lon, alt);
  } else {
    Serial.printf("GPS POST failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();
}

// ── Splash ────────────────────────────────────────────────────
void splashScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(20, 10);  display.print("ESP32-C3 GPS");
  display.setCursor(35, 28);  display.print("GT-U7 + OLED");
  display.setCursor(22, 46);  display.print("Waiting for fix...");
  display.display();
  delay(2000);
}

// ── Draw page ─────────────────────────────────────────────────
void drawPage(Page page) {
  display.clearDisplay();
  drawStatusBar();

  switch (page) {
    case PAGE_LOCATION:   drawLocation();   break;
    case PAGE_ALTITUDE:   drawAltitude();   break;
    case PAGE_SPEED:      drawSpeed();      break;
    case PAGE_SATS:       drawSatellites(); break;
  }

  for (int i = 0; i < 4; i++) {
    int x = 56 + i * 6;
    if (i == static_cast<int>(page))
      display.fillCircle(x, 62, 2, SSD1306_WHITE);
    else
      display.drawCircle(x, 62, 2, SSD1306_WHITE);
  }

  display.display();
}

// ── Status bar ────────────────────────────────────────────────
void drawStatusBar() {
  bool hasFix = gps.location.isValid() && gps.location.age() < 2000;
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("GPS");
  if (hasFix)
    display.fillRect(22, 1, 5, 5, SSD1306_WHITE);
  else
    display.drawRect(22, 1, 5, 5, SSD1306_WHITE);

  if (gps.time.isValid()) {
    char buf[9];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d",
             gps.time.hour(), gps.time.minute(), gps.time.second());
    display.setCursor(74, 0);
    display.print(buf);
  } else {
    display.setCursor(62, 0);
    display.print("--:--:--");
  }
  display.drawLine(0, 9, 127, 9, SSD1306_WHITE);
}

// ── Page 1: Location ─────────────────────────────────────────
void drawLocation() {
  display.setTextSize(1);
  display.setCursor(0, 13);  display.print("Latitude");
  display.setCursor(0, 37);  display.print("Longitude");
  if (gps.location.isValid()) {
    display.setCursor(0, 22);  display.print(gps.location.lat(), 6);
    display.setCursor(0, 46);  display.print(gps.location.lng(), 6);
  } else {
    display.setCursor(0, 22);  display.print("Acquiring...");
    display.setCursor(0, 46);  display.print("Acquiring...");
  }
}

// ── Page 2: Altitude ─────────────────────────────────────────
void drawAltitude() {
  display.setTextSize(1);
  display.setCursor(30, 13);  display.print("Altitude");
  if (gps.altitude.isValid()) {
    display.setTextSize(2);
    display.setCursor(10, 28);
    display.print(gps.altitude.meters(), 1);
    display.setTextSize(1);
    display.print(" m");
  } else {
    display.setCursor(20, 32);  display.print("No data");
  }
}

// ── Page 3: Speed ────────────────────────────────────────────
void drawSpeed() {
  display.setTextSize(1);
  display.setCursor(38, 13);  display.print("Speed");
  if (gps.speed.isValid()) {
    display.setTextSize(2);
    display.setCursor(10, 28);
    display.print(gps.speed.kmph(), 1);
    display.setTextSize(1);
    display.print(" km/h");
    int barWidth = min((int)(gps.speed.kmph() / 150.0 * 110), 110);
    display.drawRect(8, 52, 112, 6, SSD1306_WHITE);
    display.fillRect(8, 52, barWidth, 6, SSD1306_WHITE);
  } else {
    display.setCursor(20, 32);  display.print("No data");
  }
}

// ── Page 4: Satellites ───────────────────────────────────────
void drawSatellites() {
  display.setTextSize(1);
  display.setCursor(20, 13);  display.print("Satellites / HDOP");
  if (gps.satellites.isValid()) {
    display.setTextSize(2);
    display.setCursor(18, 28);
    display.print(gps.satellites.value());
    display.setTextSize(1);
    display.print(" sats");
    if (gps.hdop.isValid()) {
      double h = gps.hdop.hdop();
      display.setCursor(0, 50);
      display.print("HDOP: ");
      display.print(h, 1);
      display.print(h < 1.0 ? " Ideal" :
                    h < 2.0 ? " Excellent" :
                    h < 5.0 ? " Good" : " Fair");
    }
  } else {
    display.setTextSize(1);
    display.setCursor(20, 32);  display.print("Searching...");
  }
}
