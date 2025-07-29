import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import datetime
from threading import Thread, Lock # Import Lock for thread-safe global variable access
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import re # Import regex for text processing

# --- Global Variables ---
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
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
    print("Emotion model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Please ensure 'model.weights.h5' exists in the same directory as camera.py")
    # Exit or handle gracefully if model weights are critical and missing
    # sys.exit(1) # Uncomment if you want the program to exit if weights are not found

cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised",
    "No Face": "No Face", # String key for UI feedback
    "Processing...": "Processing..." # String key for UI feedback
}

# Global list to hold the current emotion index (numeric) for inter-thread communication
# Initialized to Neutral (key 4)
show_text = [4]
# Global variable to store the user's day description
user_day_description = [""]
# Lock for safely updating global variables from different threads/contexts
emotion_lock = Lock()
description_lock = Lock()


# --- Spotify API Credentials and Initialization ---
# IMPORTANT: Replace with your actual Spotify API Client ID and Client Secret
# You should ideally load these from environment variables for security.
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIPY_CLIENT_ID', 'bbbc93468ee94649a12b5f58dfcb31b3')
SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET', 'f21bdba4f80144e5b7f0049b87110314')

try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET
    ))
    print("Spotify API initialized successfully.")
except Exception as e:
    print(f"Error initializing Spotify API: {e}")
    print("Please check your Spotify CLIENT_ID and CLIENT_SECRET.")
    sp = None # Set to None to handle gracefully later


# --- Webcam Stream Class (for threading video capture) ---
class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            print(f"Error: Could not open video stream with source {src}")
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
            # A small sleep can help reduce CPU usage if frames are read too fast,
            # but might introduce slight latency. Adjust as needed.
            # time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- Video Camera Class (handles frame processing and emotion detection) ---
class VideoCamera(object):
    def __init__(self):
        self.cap = WebcamVideoStream(src=0).start()
        # Ensure model is loaded before first use
        self.frame_count = 0
        self.detection_interval = 5 # Process emotion every 5 frames

    def __del__(self):
        self.cap.stop()

    def get_frame(self):
        frame = self.cap.read()

        if frame is None:
            # If no frame, return a black image and signal no face detected
            with emotion_lock:
                show_text[0] = "No Face"
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank_frame)
            return jpeg.tobytes(), None # Return None for df_songs as no detection occurred

        # Resize the frame for consistent display and faster processing
        display_frame = cv2.resize(frame, (640, 480)) # Standard 640x480 for display
        gray_frame_for_detection = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

        # Only run emotion detection logic every `detection_interval` frames
        self.frame_count += 1
        if self.frame_count % self.detection_interval == 0:
            face_rects = face_cascade.detectMultiScale(gray_frame_for_detection, 1.3, 5)
            
            if len(face_rects) > 0:
                (x, y, w, h) = face_rects[0] # Process the first detected face
                
                # Ensure ROI is within bounds
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(gray_frame_for_detection.shape[1], x + w), min(gray_frame_for_detection.shape[0], y + h)
                
                roi_gray_frame = gray_frame_for_detection[y1:y2, x1:x2]
                
                if roi_gray_frame.size == 0: # Check if ROI is empty
                    with emotion_lock:
                        show_text[0] = "No Face" # Handle empty ROI case
                else:
                    cropped_img = cv2.resize(roi_gray_frame, (48, 48))
                    cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
                    
                    try:
                        prediction = emotion_model.predict(cropped_img)
                        maxindex = int(np.argmax(prediction))
                        with emotion_lock:
                            show_text[0] = maxindex # Update global emotion index
                        
                        # Draw rectangle and text on the display frame
                        cv2.rectangle(display_frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
                        cv2.putText(display_frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        with emotion_lock:
                            show_text[0] = "Processing..." # Indicate a temporary issue
            else:
                with emotion_lock:
                    show_text[0] = "No Face" # Set emotion to "No Face" if none detected
        else:
            # Continue to draw the last known emotion on the frame
            with emotion_lock:
                current_emotion_val = show_text[0]
            if isinstance(current_emotion_val, int):
                # We need to know where the face was to draw the box, but we don't re-detect here.
                # So just draw the emotion text without a new box.
                # A more advanced approach would involve tracking the face between detections.
                pass # Don't redraw the box/text if not re-detecting
            
        # Encode the frame (with or without detected emotion drawn)
        ret, jpeg = cv2.imencode('.jpg', display_frame)
        
        return jpeg.tobytes(), None # df_songs is handled by the /get_emotion_and_songs route

# --- Music Recommendation Function (Spotify API Integration) ---
def music_rec(detected_emotion_str: str, day_description: str):
    """
    Fetches music recommendations from Spotify based on the detected emotion
    and the user's description of their day.
    """
    if sp is None:
        print("Spotify API not initialized. Cannot fetch recommendations.")
        return pd.DataFrame(columns=['Name', 'Artist', 'spotify_url'])

    # --- Step 1: Determine the primary mood based on detected emotion ---
    primary_mood = detected_emotion_str.lower()
    
    # --- Step 2: Analyze user's day description for sentiment/keywords ---
    # This is a simple keyword-based analysis. For better results, consider NLP.
    description_keywords = day_description.lower().split()
    
    # Map common sentiment words to generalized moods.
    # Expand this dictionary with more words and refined mappings based on your music data.
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
            break # Take the first strong sentiment word found

    # --- Step 3: Combine Detected Emotion and User Sentiment ---
    # Logic:
    # 1. If user provides a strong sentiment, use that as the primary driver.
    # 2. Otherwise, rely on the detected facial emotion.
    # 3. Handle "No Face" or "Processing" by defaulting to neutral or user sentiment.

    final_mood_for_search = primary_mood

    if user_sentiment_mood:
        final_mood_for_search = user_sentiment_mood
        print(f"User description suggests '{user_sentiment_mood}'. Prioritizing over detected '{primary_mood}'.")
    elif primary_mood in ["no face", "processing...", "detecting..."]:
        final_mood_for_search = "neutral" # Default if no face or still processing
        print(f"No clear detected emotion or user sentiment. Defaulting to 'neutral'.")
    
    print(f"Combined Mood for Search: {final_mood_for_search}")

    # --- Step 4: Construct Spotify Search Query ---
    query_map = {
        "happy": "upbeat dance pop",
        "sad": "melancholy acoustic indie",
        "angry": "aggressive rock metal",
        "neutral": "chill instrumental ambient",
        "fearful": "calm soothing instrumental",
        "surprised": "upbeat catchy pop",
        "disgusted": "punk alternative aggressive"
    }
    
    # Get a search query or genre list based on the final determined mood
    search_query = query_map.get(final_mood_for_search, f"{final_mood_for_search} music")
    
    print(f"Spotify Search Query: '{search_query}'")

    results = sp.search(q=search_query, type='track', limit=15) # Fetch 15 tracks
    
    data = []
    # Process results, including Spotify URL
    for item in results['tracks']['items']:
        name = item['name']
        artist = ', '.join([artist_obj['name'] for artist_obj in item['artists']])
        spotify_url = item['external_urls']['spotify'] # Correct key for track URL
        data.append({"Name": name, "Artist": artist, "spotify_url": spotify_url})

    df = pd.DataFrame(data)
    
    if df.empty:
        print(f"Warning: No songs found for query '{search_query}'.")
        # Fallback to general popular music if specific search fails
        results_fallback = sp.search(q="popular songs", type='track', limit=5)
        for item in results_fallback['tracks']['items']:
            name = item['name']
            artist = ', '.join([artist_obj['name'] for artist_obj in item['artists']])
            spotify_url = item['external_urls']['spotify']
            data.append({"Name": name, "Artist": artist, "spotify_url": spotify_url})
        df = pd.DataFrame(data)

    return df.head(5) # Return top 5 recommendations