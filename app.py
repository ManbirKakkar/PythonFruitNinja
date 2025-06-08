import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import threading
import numpy as np
import av
import time
import queue
import logging

from utils import preload_assets
from game_engine import GameEngine

ASSETS = preload_assets()
logger = logging.getLogger(__name__)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.engine = None
        self.frame_count = 0
        self.frame_queue = queue.Queue(maxsize=2)  # Per-instance queue
        self.thread_lock = threading.Lock()
        
        # Start game thread for this transformer
        self.game_thread = threading.Thread(target=self._game_thread, daemon=True)
        self.game_thread.start()

    def _game_thread(self):
        """Separate thread for game logic"""
        engine = None
        while True:
            try:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame = self.frame_queue.get()
                with self.thread_lock:
                    if not engine:
                        h, w, _ = frame.shape
                        engine = GameEngine((w, h), ASSETS)
                        self.engine = engine  # Set reference for recv
                    
                    # Process at 30 FPS max
                    current_time = time.time()
                    if current_time - engine.last_update_time > 0.033:
                        engine.update(frame)
                        engine.last_update_time = current_time
            except Exception as e:
                logger.error(f"Game thread error: {str(e)}")
                time.sleep(0.1)

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            
            # Put frame in queue for game thread
            if not self.frame_queue.full():
                self.frame_queue.put(img.copy())
            
            # Draw game state
            if self.engine and not self.engine.game_over:
                with self.thread_lock:
                    if self.engine and hasattr(self.engine, 'last_frame'):
                        img = self.engine.last_frame.copy()
            
            self.frame_count += 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            return frame

# Streamlit UI
st.set_page_config(page_title="🎯 Fruit Ninja", layout="wide")
st.title("🎯 Real-Time Fruit Ninja")

ctx = webrtc_streamer(
    key="fruit-ninja",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }),
    media_stream_constraints={
        "video": {"width": 640, "height": 480},
        "audio": False
    },
    async_processing=True
)

def reset_game():
    with game_lock:
        frame_queue.queue.clear()
        # Recreate engine in game thread

with st.sidebar:
    st.header("How to Play")
    st.markdown("1. Make a slicing motion with your finger")
    st.markdown("2. Slice fruits, avoid bombs")
    st.markdown("3. Don't miss more than 10 fruits")
    if st.button("🔄 Restart Game"):
        reset_game()