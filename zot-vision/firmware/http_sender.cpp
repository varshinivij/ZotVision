#include "http_sender.h"
#include "esp_http_client.h"
#include "esp_log.h"

static const char* TAG = "http_sender";

// --- CHANGE THIS ---
#define SERVER_URL  "http://192.168.1.100:8000/classify"
// -------------------

esp_err_t http_send_frame(const uint8_t* buf, size_t len) {
    esp_http_client_config_t config = {
        .url    = SERVER_URL,
        .method = HTTP_METHOD_POST,
    };

    // initialize HTTP client
    esp_http_client_handle_t client = esp_http_client_init(&config);

    // set headers and attach JPEG bytes as the body
    esp_http_client_set_header(client, "Content-Type", "image/jpeg");
    esp_http_client_set_post_field(client, (const char*)buf, len);

    // perform the request
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        int status = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "HTTP POST status: %d", status);
    } else {
        ESP_LOGE(TAG, "HTTP POST failed: %s", esp_err_to_name(err));
    }

    // always clean up
    esp_http_client_cleanup(client);
    return err;
}