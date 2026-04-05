idf_component_register(
    SRCS "main.cpp" "camera.cpp" "wifi.cpp" "http_sender.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp32-camera esp_wifi esp_http_client nvs_flash esp_netif esp_event
)