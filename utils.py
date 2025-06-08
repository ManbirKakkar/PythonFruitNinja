import os
import cv2
import pygame
import numpy as np

def preload_assets(asset_dir: str = "assets") -> dict:
    pygame.mixer.init()
    assets = {}
    fruit_types = ["apple", "banana", "orange", "bomb"]
    target_size = (40, 40)

    for ft in fruit_types:
        path = os.path.join(asset_dir, f"{ft}.png")
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError
                
            img = cv2.resize(img, target_size)
            
            if img.shape[2] == 3:
                b_channel, g_channel, r_channel = cv2.split(img)
                alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
                img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            elif img.shape[2] == 4:
                b, g, r, a = cv2.split(img)
                img = cv2.merge((b, g, r, a))
                
            assets[ft] = img
        except:
            if ft == "bomb":
                color = (0, 0, 0, 255)
            else:
                color = (100, 100, 255, 255)
            assets[ft] = np.full((*target_size, 4), color, dtype=np.uint8)

    assets["slice_sound"] = None
    return assets