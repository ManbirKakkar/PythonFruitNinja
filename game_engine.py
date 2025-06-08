import random
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List
from fruit import Fruit
from hand_tracker import HandTracker

class GameEngine:
    def __init__(self, screen_size: Tuple[int, int], assets: dict):
        self.width, self.height = screen_size
        self.assets = assets
        self.hand_tracker = HandTracker(processing_size=256)
        self.reset()
        
    def reset(self) -> None:
        self.fruits: List[Fruit] = []
        self.score = 0
        self.missed = 0
        self.max_missed = 10
        self.game_over = False
        self.last_spawn = time.time()
        self.started = False
        self.last_update_time = time.time()
        self.last_hand_process = 0
        self.last_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def spawn_fruit(self) -> None:
        current_time = time.time()
        # Count only unsliced fruits for spawning limit
        unsliced_count = sum(1 for fruit in self.fruits if not fruit.sliced)
        
        if (current_time - self.last_spawn > random.uniform(0.2, 0.8) and 
            not self.game_over and 
            unsliced_count < 5):  # Changed to only count unsliced fruits
            
            t = random.random()
            fruit_type = 'bomb' if t < 0.1 else random.choice(['apple', 'banana', 'orange'])
            self.fruits.append(Fruit(fruit_type, (self.width, self.height), self.assets))
            self.last_spawn = current_time

    def update(self, frame: np.ndarray) -> None:
        self.last_frame = frame.copy()
        
        current_time = time.time()
        if current_time - self.last_hand_process > 0.05:
            self.hand_tracker.process(frame)
            self.last_hand_process = current_time
            
        fingertip = self.hand_tracker.get_fingertip(frame.shape)
        slicing = self.hand_tracker.is_slicing()

        if not self.started and slicing:
            self.started = True
            self.started_time = current_time

        if not self.started or self.game_over:
            self.draw_static()
        else:
            self.draw_game()

        self.spawn_fruit()
        for fruit in self.fruits[:]:
            fruit.update()
            # Remove fruits that are off-screen and their pieces are also off-screen
            if fruit.sliced:
                pieces_off_screen = all(
                    piece['y'] > self.height + 100 
                    for piece in fruit.sliced_pieces
                )
                if pieces_off_screen:
                    self.fruits.remove(fruit)
            elif fruit.y > self.height + 100:
                self.fruits.remove(fruit)
                self.missed += 1
                if self.missed >= self.max_missed:
                    self.game_over = True
            
            if fingertip and slicing and not fruit.sliced:
                if fruit.check_collision(fingertip):
                    fruit.slice()
                    if fruit.type == 'bomb':
                        self.game_over = True
                    else:
                        self.score += 1

        self.draw_game()

    def draw_static(self) -> None:
        if not self.started:
            cv2.putText(self.last_frame, "SLICE TO START", 
                       (self.width//2-150, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        elif self.game_over:
            cv2.putText(self.last_frame, "GAME OVER", 
                       (self.width//2-150, self.height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

    def draw_game(self) -> None:
        for fruit in self.fruits:
            fruit.draw(self.last_frame)
        
        cv2.putText(self.last_frame, f"Score: {self.score}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(self.last_frame, f"Missed: {self.missed}/{self.max_missed}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if self.started:
            elapsed = int(time.time() - self.started_time)
            cv2.putText(self.last_frame, f"Time: {elapsed}s", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)