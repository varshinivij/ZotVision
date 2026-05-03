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
#define CONTROL_URL      "http://192.168.137.1:5000/api/control/" STRINGIFY(FIREFIGHTER_ID)
#define FIREFIGHTER_ID   0
#define GPS_SEND_INTERVAL_MS  1000
#define CONTROL_POLL_MS       500   // how often to check for OLED commands
// ─────────────────────────────────────────────────────────────

// Stringify helper so FIREFIGHTER_ID (an int) becomes part of a string literal
#define STRINGIFY2(x) #x
#define STRINGIFY(x)  STRINGIFY2(x)

// ── OLED config — 128×64 SSD1306 ─────────────────────────────
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
uint32_t lastGpsSend    = 0;
uint32_t lastControlPoll = 0;

bool   serverConnected = false;  // true after first successful control poll

// Command received from the web dashboard ("none" | "left" | "right" | "warning")
String currentCommand = "none";

// Chat message received from the web dashboard (empty = no message)
String currentMessage = "";

// Screen centre
const int16_t CCX = 64;
const int16_t CCY = 32;

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

  // Poll server for OLED command
  if (WiFi.status() == WL_CONNECTED &&
      millis() - lastControlPoll >= CONTROL_POLL_MS) {
    pollControl();
    lastControlPoll = millis();
  }

  // Display
  if (!serverConnected) {
    drawConnecting();
  } else if (currentCommand != "none") {
    drawCommandPage();
  } else if (currentMessage.length() > 0) {
    drawMessage();
  } else {
    drawIdle();
  }

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

// ── Poll server for OLED command ──────────────────────────────
void pollControl() {
  HTTPClient http;
  http.begin(CONTROL_URL);
  int code = http.GET();
  if (code == 200) {
    serverConnected = true;  // first successful reply = connected
    String body = http.getString();

    // Parse "command" — body looks like: {"command":"left","message":"hello"}
    int idx = body.indexOf("\"command\":");
    if (idx >= 0) {
      int start = body.indexOf('"', idx + 10) + 1;
      int end   = body.indexOf('"', start);
      if (start > 0 && end > start)
        currentCommand = body.substring(start, end);
    }

    // Parse "message"
    idx = body.indexOf("\"message\":");
    if (idx >= 0) {
      int start = body.indexOf('"', idx + 10) + 1;
      int end   = body.indexOf('"', start);
      if (start > 0 && end > start)
        currentMessage = body.substring(start, end);
      else
        currentMessage = "";
    }
  }
  http.end();
}

// ── Waiting for server ────────────────────────────────────────
void drawConnecting() {
  display.clearDisplay();
  display.setTextSize(1);

  if (WiFi.status() != WL_CONNECTED) {
    display.setCursor(16, 20);
    display.print("Connecting to WiFi");
    display.setCursor(40, 34);
    display.print(WIFI_SSID);
  } else {
    display.setCursor(16, 20);
    display.print("Waiting for server");
    display.setCursor(28, 34);
    display.print("192.168.137.1");
  }

  display.display();
}

// ── Neutral / idle screen ─────────────────────────────────────
void drawIdle() {
  display.clearDisplay();

  // Time — size 2 (each char 12×16 px), centred horizontally
  char timeBuf[9];
  if (gps.time.isValid()) {
    snprintf(timeBuf, sizeof(timeBuf), "%02d:%02d:%02d",
             gps.time.hour(), gps.time.minute(), gps.time.second());
  } else {
    snprintf(timeBuf, sizeof(timeBuf), "--:--:--");
  }
  display.setTextSize(2);
  int16_t timeW = strlen(timeBuf) * 12;
  display.setCursor((128 - timeW) / 2, 4);
  display.print(timeBuf);

  // Divider
  display.drawLine(0, 24, 127, 24, SSD1306_WHITE);

  // GPS coords — size 1 below divider
  display.setTextSize(1);
  if (gps.location.isValid()) {
    display.setCursor(0, 28);
    display.print("LA ");
    display.print(gps.location.lat(), 5);
    display.setCursor(0, 40);
    display.print("LO ");
    display.print(gps.location.lng(), 5);
  } else {
    display.setCursor(18, 32);
    display.print("GPS Acquiring...");
  }

  display.display();
}

// ── Full-screen command icons ─────────────────────────────────
void drawCommandPage() {
  display.clearDisplay();

  if (currentCommand == "left") {
    iconLeftArrow(CCX, CCY, 55);

  } else if (currentCommand == "right") {
    iconRightArrow(CCX, CCY, 55);

  } else if (currentCommand == "warning") {
    iconExclamation(CCX, CCY);
  }

  display.display();
}

// ── Chat message from web dashboard ──────────────────────────
void drawMessage() {
  display.clearDisplay();

  // "MSG:" label + divider
  display.setTextSize(1);
  display.setCursor(0, 3);
  display.print("MSG:");
  display.drawLine(0, 13, 127, 13, SSD1306_WHITE);

  // Message text — size 2 (12 px wide per char), max 10 chars per line
  display.setTextSize(2);
  int len = currentMessage.length();

  if (len <= 10) {
    // Single line, horizontally centred
    int16_t x = (128 - len * 12) / 2;
    display.setCursor(x < 0 ? 0 : x, 24);
    display.print(currentMessage);
  } else {
    // Two lines, each centred
    String line1 = currentMessage.substring(0, 10);
    String line2 = currentMessage.substring(10);
    int16_t x1 = (128 - (int16_t)line1.length() * 12) / 2;
    int16_t x2 = (128 - (int16_t)line2.length() * 12) / 2;
    display.setCursor(x1 < 0 ? 0 : x1, 18);
    display.print(line1);
    display.setCursor(x2 < 0 ? 0 : x2, 38);
    display.print(line2);
  }

  display.display();
}

// ═══════════════════════════════════════════════════════════════
//  ICONS  (all sized for the 45 px content area, y 10–54)
// ═══════════════════════════════════════════════════════════════

// Stick figure centred at (cx, cy) — fits in ~44 px tall
//   head top  = cy - 18 = 14  (≥ y 10 ✓)
//   leg bottom = cy + 18 = 50  (≤ y 54 ✓)
void iconStickFigure(int16_t cx, int16_t cy) {
  display.drawCircle(cx, cy - 12, 6, SSD1306_WHITE);            // head
  display.drawLine(cx, cy - 6,  cx, cy + 6,  SSD1306_WHITE);   // body
  display.drawLine(cx - 10, cy, cx + 10, cy, SSD1306_WHITE);   // arms
  display.drawLine(cx, cy + 6, cx - 9, cy + 18, SSD1306_WHITE); // left leg
  display.drawLine(cx, cy + 6, cx + 9, cy + 18, SSD1306_WHITE); // right leg
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

// Exclamation mark centred at (cx, cy) — scaled for 45 px content area
void iconExclamation(int16_t cx, int16_t cy) {
  display.drawLine(cx, cy - 16, cx, cy + 2, SSD1306_WHITE); // bar
  display.fillCircle(cx, cy + 10, 3, SSD1306_WHITE);         // dot
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

// ── Splash ────────────────────────────────────────────────────
void splashScreen() {
  display.clearDisplay();

  // Stick figure on left half
  iconStickFigure(18, 32);

  display.drawLine(38, 0, 38, 63, SSD1306_WHITE);

  display.setTextSize(2);
  display.setCursor(44, 6);   display.print("ZotVision");
  display.setTextSize(1);
  display.setCursor(44, 26);  display.print("GPS Unit");
  char buf[8];
  snprintf(buf, sizeof(buf), "FF %d", FIREFIGHTER_ID);
  display.setCursor(44, 38);  display.print(buf);

  display.display();
  delay(2000);
}
