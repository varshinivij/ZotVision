idf_component_register(
    SRCS "main.cpp" "camera.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp32-camera
)