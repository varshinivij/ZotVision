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
#define FIREFIGHTER_ID   0
#define GPS_SEND_INTERVAL_MS 1000
// ─────────────────────────────────────────────────────────────

// ── OLED config — 0.91" SSD1306 = 128×32 ─────────────────────
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  32
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
enum Page { PAGE_IDENTITY, PAGE_LOCATION, PAGE_SPEED, PAGE_STATUS };
Page     currentPage  = PAGE_IDENTITY;
uint32_t lastPageSwap = 0;
uint32_t lastGpsSend  = 0;
const uint16_t PAGE_DURATION = 4000;

// Layout for 128×32
//   y 0–7   : status bar (1 row, textSize 1)
//   y 8     : divider line
//   y 9–27  : content area (19 px tall)
//   y 30    : page-indicator dots (radius 1)
//
// Content-area centre:
const int16_t CCX = 64;
const int16_t CCY = 18;   // midpoint of y 9–27

// ─────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  GPSSerial.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  Wire.begin(8, 9);  // SDA = GPIO8, SCL = GPIO9

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("SSD1306 not found – check wiring/address");
    while (true);
  }

  // Mirror horizontally — flip segment remap from 0xA1 (default) to 0xA0
  display.ssd1306_command(SSD1306_SEGREMAP);

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  splashScreen();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  uint32_t wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 10000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED)
    Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
  else
    Serial.println("\nWiFi failed – GPS display will still work");
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

  if (gps.location.isValid() &&
      gps.location.age() < 2000 &&
      millis() - lastGpsSend >= GPS_SEND_INTERVAL_MS &&
      WiFi.status() == WL_CONNECTED) {
    sendGPS();
    lastGpsSend = millis();
  }

  delay(250);
}

// ── HTTP POST ─────────────────────────────────────────────────
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
  if (code > 0)
    Serial.printf("GPS POST %d  lat=%.6f lon=%.6f alt=%.1f\n", code, lat, lon, alt);
  else
    Serial.printf("GPS POST failed: %s\n", http.errorToString(code).c_str());
  http.end();
}

// ═══════════════════════════════════════════════════════════════
//  ICONS  (all sized for the 19 px content area, y 9–27)
// ═══════════════════════════════════════════════════════════════

// Stick figure centred at (cx, cy) — fits in ~19 px tall
//   head top  = cy - 9 = 9   (≥ y 9 ✓)
//   leg bottom = cy + 9 = 27  (≤ y 27 ✓)
void iconStickFigure(int16_t cx, int16_t cy) {
  display.drawCircle(cx, cy - 6, 3, SSD1306_WHITE);          // head
  display.drawLine(cx, cy - 3, cx, cy + 3, SSD1306_WHITE);   // body
  display.drawLine(cx - 6, cy,  cx + 6, cy, SSD1306_WHITE);  // arms
  display.drawLine(cx, cy + 3, cx - 5, cy + 9, SSD1306_WHITE); // left leg
  display.drawLine(cx, cy + 3, cx + 5, cy + 9, SSD1306_WHITE); // right leg
}

// Horizontal right arrow centred at (cx, cy), half-length len
void iconRightArrow(int16_t cx, int16_t cy, int16_t len) {
  int16_t head = max((int16_t)4, (int16_t)(len / 4));
  display.drawLine(cx - len, cy, cx + len, cy, SSD1306_WHITE);
  display.drawLine(cx + len - head, cy - head, cx + len, cy, SSD1306_WHITE);
  display.drawLine(cx + len - head, cy + head, cx + len, cy, SSD1306_WHITE);
}

// Horizontal left arrow centred at (cx, cy), half-length len
void iconLeftArrow(int16_t cx, int16_t cy, int16_t len) {
  int16_t head = max((int16_t)4, (int16_t)(len / 4));
  display.drawLine(cx - len, cy, cx + len, cy, SSD1306_WHITE);
  display.drawLine(cx - len + head, cy - head, cx - len, cy, SSD1306_WHITE);
  display.drawLine(cx - len + head, cy + head, cx - len, cy, SSD1306_WHITE);
}

// Exclamation mark centred at (cx, cy) — fits 19 px tall
void iconExclamation(int16_t cx, int16_t cy) {
  display.drawLine(cx, cy - 8, cx, cy + 1, SSD1306_WHITE); // bar
  display.fillCircle(cx, cy + 6, 2, SSD1306_WHITE);         // dot
}

// 4-bar signal strength, bottom-left corner at (x, y), bar width bw
// Bar heights: 3, 6, 9, 13 — tallest bar is 13 px, fits in content area
void iconSignalBars(int16_t x, int16_t y, uint8_t filled, int16_t bw) {
  const int16_t heights[] = { 3, 6, 9, 13 };
  for (int i = 0; i < 4; i++) {
    int16_t bx = x + i * (bw + 2);
    int16_t by = y - heights[i];
    if (i < filled)
      display.fillRect(bx, by, bw, heights[i], SSD1306_WHITE);
    else
      display.drawRect(bx, by, bw, heights[i], SSD1306_WHITE);
  }
}

// ═══════════════════════════════════════════════════════════════
//  PAGES
// ═══════════════════════════════════════════════════════════════

// Page 1 – Firefighter identity
//   Left  : stick figure (cx=18)
//   Right : "FF {N}" in large text
void drawIdentity() {
  iconStickFigure(18, CCY);

  display.drawLine(33, 9, 33, 27, SSD1306_WHITE); // divider

  display.setTextSize(1);
  display.setCursor(38, 9);
  display.print("FF");

  // ID number: size 2 (16 px tall) fits in content area (19 px)
  display.setTextSize(2);
  display.setCursor(38, 17);
  display.print(FIREFIGHTER_ID);

  // WiFi dot on far right
  bool wifi = (WiFi.status() == WL_CONNECTED);
  display.setTextSize(1);
  display.setCursor(88, 9);
  display.print(wifi ? "WiFi" : "    ");
  display.setCursor(88, 19);
  display.print(wifi ? "OK" : "----");
}

// Page 2 – Location
//   Two compact rows: "LAT value" / "LON value"
void drawLocation() {
  display.setTextSize(1);
  if (gps.location.isValid()) {
    display.setCursor(0, 10);
    display.print("LA ");
    display.print(gps.location.lat(), 5);

    display.setCursor(0, 20);
    display.print("LO ");
    display.print(gps.location.lng(), 5);
  } else {
    display.setCursor(0, 10);  display.print("LAT  Acquiring...");
    display.setCursor(0, 20);  display.print("LON  Acquiring...");
  }
}

// Page 3 – Speed + direction
//   Arrow across content area; speed value above shaft
void drawSpeedDir() {
  bool hasSpeed  = gps.speed.isValid();
  double spd     = hasSpeed ? gps.speed.kmph() : 0.0;
  double course  = gps.course.isValid() ? gps.course.deg() : 0.0;

  if (hasSpeed && spd > 0.5) {
    // Speed number above the arrow shaft
    char buf[10];
    snprintf(buf, sizeof(buf), "%.1f", spd);
    display.setTextSize(1);
    int16_t tw = strlen(buf) * 6;
    display.setCursor(CCX - tw / 2, 10);
    display.print(buf);
    display.print(" km/h");

    // Arrow at y=22 (below the text row)
    bool goingRight = (course < 180.0);
    if (goingRight) iconRightArrow(CCX, 22, 46);
    else            iconLeftArrow (CCX, 22, 46);

  } else {
    iconExclamation(CCX, CCY);
    display.setTextSize(1);
    display.setCursor(CCX - 22, 10);
    display.print("STATIONARY");
  }
}

// Page 4 – Signal status
//   No fix  → exclamation + "NO FIX"
//   Has fix → signal bars (left) + sats + HDOP label (right)
void drawStatus() {
  bool hasFix = gps.location.isValid() && gps.location.age() < 2000;

  if (!hasFix) {
    iconExclamation(CCX, CCY);
    display.setTextSize(1);
    display.setCursor(CCX + 10, 13);
    display.print("NO FIX");
    return;
  }

  uint8_t bars = 1;
  if (gps.hdop.isValid()) {
    double h = gps.hdop.hdop();
    bars = (h < 1.0) ? 4 : (h < 2.0) ? 3 : (h < 5.0) ? 2 : 1;
  }
  iconSignalBars(4, 27, bars, 6);  // bottom-left, bar-width 6

  display.setTextSize(1);
  // Satellite count
  display.setCursor(40, 10);
  if (gps.satellites.isValid()) {
    display.print(gps.satellites.value());
    display.print(" sats");
  } else {
    display.print("? sats");
  }
  // HDOP quality
  if (gps.hdop.isValid()) {
    double h = gps.hdop.hdop();
    const char* q = (h < 1.0) ? "Ideal" :
                    (h < 2.0) ? "Excl" :
                    (h < 5.0) ? "Good" : "Fair";
    display.setCursor(40, 20);
    display.print("HDOP ");
    display.print(q);
  }
}

// ═══════════════════════════════════════════════════════════════
//  FRAME
// ═══════════════════════════════════════════════════════════════

void drawStatusBar() {
  bool hasFix = gps.location.isValid() && gps.location.age() < 2000;

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("GPS");

  // Solid square = fix  |  hollow = searching
  if (hasFix) display.fillRect(22, 1, 5, 5, SSD1306_WHITE);
  else        display.drawRect(22, 1, 5, 5, SSD1306_WHITE);

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
  display.drawLine(0, 8, 127, 8, SSD1306_WHITE);
}

void drawPage(Page page) {
  display.clearDisplay();
  drawStatusBar();

  switch (page) {
    case PAGE_IDENTITY: drawIdentity(); break;
    case PAGE_LOCATION: drawLocation(); break;
    case PAGE_SPEED:    drawSpeedDir(); break;
    case PAGE_STATUS:   drawStatus();   break;
  }

  // Page indicator dots — radius 1, centred at y=30
  for (int i = 0; i < 4; i++) {
    int x = 58 + i * 5;
    if (i == static_cast<int>(page))
      display.fillCircle(x, 30, 1, SSD1306_WHITE);
    else
      display.drawCircle(x, 30, 1, SSD1306_WHITE);
  }

  display.display();
}

// ── Splash ────────────────────────────────────────────────────
void splashScreen() {
  display.clearDisplay();

  // Tiny stick figure on left half
  iconStickFigure(18, 16);

  display.drawLine(33, 0, 33, 31, SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(38, 2);   display.print("ZotVision");
  display.setCursor(38, 12);  display.print("GPS Unit");
  char buf[8];
  snprintf(buf, sizeof(buf), "FF %d", FIREFIGHTER_ID);
  display.setCursor(38, 22);  display.print(buf);

  display.display();
  delay(2000);
}
