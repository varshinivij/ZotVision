#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "camera.h"
#include "wifi.h"
#include "http_sender.h"
#include "esp_log.h"

static const char* TAG = "main";

// --- CHANGE THESE ---
#define WIFI_SSID     "KAUSHIK25"
#define WIFI_PASSWORD "12345678"
// --------------------

void capture_task(void* pvParameters) {
    TickType_t last_wake = xTaskGetTickCount();

    while (true) {
        camera_fb_t* fb = camera_capture();

        if (fb) {
            ESP_LOGI(TAG, "Frame: %zu bytes, %dx%d",
                     fb->len, fb->width, fb->height);

            // send frame to Flask server
            http_send_frame(fb->buf, fb->len);

            camera_release(fb);  // never skip this
        }

        // No artificial delay — loop as fast as WiFi allows.
        // A minimal yield keeps the watchdog happy without capping FPS.
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

extern "C" void app_main() {
    wifi_init(WIFI_SSID, WIFI_PASSWORD);  // connect to WiFi first
    ESP_ERROR_CHECK(camera_init());        // then init camera

    xTaskCreate(
        capture_task,   // function
        "capture_task", // name
        16384,          // increased stack size for HTTP
        NULL,           // params
        5,              // priority
        NULL            // handle
    );
}