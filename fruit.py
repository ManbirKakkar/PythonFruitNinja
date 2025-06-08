import random
import time
import cv2
import numpy as np
from typing import Tuple, List, Dict
import math

class Fruit:
    def __init__(
        self,
        fruit_type: str,
        screen_size: Tuple[int, int],
        assets: dict
    ):
        self.type = fruit_type
        self.image = assets[fruit_type]
        self.height, self.width = self.image.shape[:2]
        w, h = screen_size
        self.x = random.randint(self.width // 2, w - self.width // 2)
        self.y = h
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-18, -14)
        self.gravity = 0.35
        self.sliced = False
        self.sliced_pieces: List[Dict] = []
        self.sliced_time: float = 0.0

    def update(self) -> None:
        if not self.sliced:
            self.vy += self.gravity
            self.x += self.vx
            self.y += self.vy
        else:
            for piece in self.sliced_pieces:
                piece['vy'] += self.gravity
                piece['x'] += piece['vx']
                piece['y'] += piece['vy']
                # Add rotation effect
                piece['rotation'] += piece['rotation_speed']

    def check_collision(self, point: Tuple[int, int]) -> bool:
        px, py = point
        return (abs(px - self.x) < self.width//2 and 
                abs(py - self.y) < self.height//2)

    def slice(self) -> None:
        if self.sliced:
            return
        self.sliced = True
        self.sliced_time = time.time()
        
        # Create sliced pieces with initial rotation
        self.sliced_pieces = [
            {
                'image': self.image[:, :self.width//2],
                'x': self.x, 
                'y': self.y, 
                'vx': -5, 
                'vy': self.vy,
                'rotation': 0,
                'rotation_speed': random.uniform(-10, -5)
            },
            {
                'image': self.image[:, self.width//2:],
                'x': self.x, 
                'y': self.y, 
                'vx': 5, 
                'vy': self.vy,
                'rotation': 0,
                'rotation_speed': random.uniform(5, 10)
            }
        ]

    def draw(self, frame: np.ndarray) -> None:
        if not self.sliced:
            self.draw_image(frame, self.image, self.x, self.y)
        else:
            for piece in self.sliced_pieces:
                self.draw_rotated_image(
                    frame, 
                    piece['image'], 
                    piece['x'], 
                    piece['y'], 
                    piece['rotation']
                )
    
    def draw_rotated_image(self, frame: np.ndarray, img: np.ndarray, 
                          x: float, y: float, angle: float) -> None:
        """Draw an image rotated around its center"""
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Create rotation matrix
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate the image with border values set to transparent
        rotated_img = cv2.warpAffine(
            img, 
            rot_mat, 
            (w, h), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0, 0))
        
        # Now draw the rotated image
        self.draw_image(frame, rotated_img, x, y)
    
    def draw_image(self, frame: np.ndarray, img: np.ndarray, 
                  x: float, y: float) -> None:
        h, w = img.shape[:2]
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = x1 + w
        y2 = y1 + h
        
        # Skip if image is completely off-screen
        if x1 >= frame.shape[1] or y1 >= frame.shape[0] or x2 < 0 or y2 < 0:
            return
            
        # Calculate cropping coordinates
        crop_x1 = max(0, -x1)
        crop_y1 = max(0, -y1)
        crop_x2 = w - max(0, x2 - frame.shape[1])
        crop_y2 = h - max(0, y2 - frame.shape[0])
        
        # Adjust coordinates to stay within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Skip if nothing to draw
        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            return
            
        # Get region of interest and cropped image
        roi = frame[y1:y2, x1:x2]
        img_cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Handle alpha channel if present
        if img_cropped.shape[2] == 4:
            b, g, r, a = cv2.split(img_cropped)
            a = a.astype(float) / 255.0
            color = cv2.merge((b, g, r))
            
            # Blend with background using alpha
            for c in range(0, 3):
                roi[:, :, c] = (a * color[:, :, c] + 
                               (1 - a) * roi[:, :, c])
        else:
            # No alpha channel - just overwrite
            roi[:] = img_cropped[:, :, :3]