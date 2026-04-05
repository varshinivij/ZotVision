idf_component_register(
    SRCS "main.cpp" "camera.cpp" "wifi.cpp" "http_sender.cpp"
    INCLUDE_DIRS "."
    REQUIRES
        esp32-camera
        esp_http_client
        esp_wifi
        nvs_flash
        esp_netif
        esp_event
)
