#pragma once
#include "esp_camera.h"

esp_err_t camera_init();
camera_fb_t* camera_capture();
void camera_release(camera_fb_t* fb);