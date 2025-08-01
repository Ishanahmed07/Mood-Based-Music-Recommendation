import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from threading import Thread, Lock
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# --- Global Variables ---
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Load weights after model architecture is defined
try:
    emotion_model.load_weights('model.weights.h5')
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Please ensure 'model.weights.h5' exists in the same directory as camera.py")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised",
    "No Face": "No Face",
    "Processing...": "Processing..."
}

show_text = [4]
user_day_description = [""]
emotion_lock = Lock()
description_lock = Lock()


# --- Spotify API Credentials and Initialization ---
SPOTIPY_CLIENT_ID = os.environ.get('SPOTIPY_CLIENT_ID', 'bbbc93468ee94649a12b5f58dfcb31b3')
SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET', 'f21bdba4f80144e5b7f0049b87110314')

try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET
    ))
except Exception as e:
    print(f"Error initializing Spotify API: {e}")
    sp = None


# --- Webcam Stream Class (for threading video capture) ---
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            self.grabbed = False
            self.frame = None
        else:
            (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
            
    def update(self):
        while True:
            if self.stopped:
                if self.stream.isOpened():
                    self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- Video Camera Class (handles frame processing and emotion detection) ---
class VideoCamera(object):
    def __init__(self):
        self.cap = WebcamVideoStream(src=0).start()
        self.frame_count = 0
        self.detection_interval = 5

    def __del__(self):
        self.cap.stop()

    def get_frame(self):
        frame = self.cap.read()

        if frame is None:
            with emotion_lock:
                show_text[0] = "No Face"
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank_frame)
            return jpeg.tobytes(), None

        display_frame = cv2.resize(frame, (640, 480))
        gray_frame_for_detection = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

        self.frame_count += 1
        if self.frame_count % self.detection_interval == 0:
            face_rects = face_cascade.detectMultiScale(gray_frame_for_detection, 1.3, 5)
            
            if len(face_rects) > 0:
                (x, y, w, h) = face_rects[0]
                
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(gray_frame_for_detection.shape[1], x + w), min(gray_frame_for_detection.shape[0], y + h)
                
                roi_gray_frame = gray_frame_for_detection[y1:y2, x1:x2]
                
                if roi_gray_frame.size == 0:
                    with emotion_lock:
                        show_text[0] = "No Face"
                else:
                    cropped_img = cv2.resize(roi_gray_frame, (48, 48))
                    cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
                    
                    try:
                        prediction = emotion_model.predict(cropped_img)
                        maxindex = int(np.argmax(prediction))
                        with emotion_lock:
                            show_text[0] = maxindex
                        
                        cv2.rectangle(display_frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
                        cv2.putText(display_frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                    except Exception as e:
                        with emotion_lock:
                            show_text[0] = "Processing..."
            else:
                with emotion_lock:
                    show_text[0] = "No Face"
            
        ret, jpeg = cv2.imencode('.jpg', display_frame)
        return jpeg.tobytes(), None

# --- Music Recommendation Function (Spotify API Integration) ---
def music_rec(detected_emotion_str: str, day_description: str):
    if sp is None:
        return pd.DataFrame(columns=['Name', 'Artist', 'spotify_url'])

    primary_mood = detected_emotion_str.lower()
    description_keywords = day_description.lower().split()
    
    sentiment_map = {
        'happy': 'happy', 'great': 'happy', 'good': 'happy', 'joyful': 'happy', 'positive': 'happy',
        'sad': 'sad', 'stressed': 'sad', 'tired': 'sad', 'bad': 'sad', 'down': 'sad', 'negative': 'sad',
        'angry': 'angry', 'frustrated': 'angry', 'irritated': 'angry',
        'calm': 'neutral', 'relaxing': 'neutral', 'chill': 'neutral', 'peaceful': 'neutral',
        'energetic': 'happy', 'excited': 'happy', 'pumped': 'happy',
        'anxious': 'fearful', 'nervous': 'fearful',
        'surprised': 'surprised',
        'disgusted': 'disgusted'
    }

    user_sentiment_mood = None
    for word in description_keywords:
        if word in sentiment_map:
            user_sentiment_mood = sentiment_map[word]
            break

    final_mood_for_search = primary_mood

    if user_sentiment_mood:
        final_mood_for_search = user_sentiment_mood
    elif primary_mood in ["no face", "processing...", "detecting..."]:
        final_mood_for_search = "neutral"
    
    query_map = {
        "happy": "upbeat dance pop",
        "sad": "melancholy acoustic indie",
        "angry": "aggressive rock metal",
        "neutral": "chill instrumental ambient",
        "fearful": "calm soothing instrumental",
        "surprised": "upbeat catchy pop",
        "disgusted": "punk alternative aggressive"
    }
    
    search_query = query_map.get(final_mood_for_search, f"{final_mood_for_search} music")
    
    results = sp.search(q=search_query, type='track', limit=15)
    
    data = []
    for item in results['tracks']['items']:
        name = item['name']
        artist = ', '.join([artist_obj['name'] for artist_obj in item['artists']])
        spotify_url = item['external_urls']['spotify']
        data.append({"Name": name, "Artist": artist, "spotify_url": spotify_url})

    df = pd.DataFrame(data)
    
    if df.empty:
        results_fallback = sp.search(q="popular songs", type='track', limit=5)
        for item in results_fallback['tracks']['items']:
            name = item['name']
            artist = ', '.join([artist_obj['name'] for artist_obj in item['artists']])
            spotify_url = item['external_urls']['spotify']
            data.append({"Name": name, "Artist": artist, "spotify_url": spotify_url})
        df = pd.DataFrame(data)

    return df.head(5) 