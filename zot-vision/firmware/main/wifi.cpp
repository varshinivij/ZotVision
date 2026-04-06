#include "wifi.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include <cstring>

static const char* TAG = "wifi";
static EventGroupHandle_t wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

static int retry_count = 0;
#define MAX_RETRY_BACKOFF_MS 30000

static void event_handler(void* arg, esp_event_base_t event_base,
                           int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        int delay_ms = (1 << retry_count) * 1000;
        if (delay_ms > MAX_RETRY_BACKOFF_MS) delay_ms = MAX_RETRY_BACKOFF_MS;
        ESP_LOGW(TAG, "Disconnected, retrying in %d ms...", delay_ms);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
        retry_count++;
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        retry_count = 0;
        ESP_LOGI(TAG, "WiFi connected");
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

esp_err_t wifi_init(const char* ssid, const char* password) {
    // create event group before anything that could trigger events
    wifi_event_group = xEventGroupCreate();

    // initialize non-volatile storage (required by wifi driver)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    // initialize network interface and event loop
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    // initialize wifi driver with default config
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // register event handlers
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL);

    // set ssid and password
    wifi_config_t wifi_config = {};
    strlcpy((char*)wifi_config.sta.ssid,     ssid,     sizeof(wifi_config.sta.ssid));
    strlcpy((char*)wifi_config.sta.password, password, sizeof(wifi_config.sta.password));

    // start wifi
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // block here until connected
    xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT,
                        false, true, portMAX_DELAY);
    return ESP_OK;
}
