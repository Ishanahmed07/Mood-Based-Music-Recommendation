import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from pandastable import Table, TableModel # Not used, can be removed
from tensorflow.keras.preprocessing import image
import datetime
from threading import Thread
import time
import pandas as pd
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6 # This variable is not used anywhere, can be removed

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
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Please ensure 'model.weights.h5' exists in the same directory as camera.py")


cv2.ocl.setUseOpenCL(False) # Keep this if you're not using OpenCL

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
# music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"} # This is commented out/not used, can be removed if Spotify API is always used
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 # This variable is not used, can be removed
show_text=[0] # Global list to hold the current emotion index for inter-thread communication

# Class for calculating FPS - not directly used in the web app logic but good for debugging performance
class FPS:
	def __init__(self):
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		self._end = datetime.datetime.now()
	def update(self):
		self._numFrames += 1
	def elapsed(self):
		return (self._end - self._start).total_seconds()
	def fps(self):
		return self._numFrames / self.elapsed()


# Class for using another thread for video streaming to boost performance
class WebcamVideoStream:
		def __init__(self, src=0):
			# Ensure the camera is opened properly
			self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			if not self.stream.isOpened():
				print(f"Error: Could not open video stream with source {src}")
				self.grabbed = False
				self.frame = None
			else:
				(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
			# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					self.stream.release() # Release camera resources
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()
				# Small delay to prevent 100% CPU usage if frames are read too fast
				# time.sleep(0.01) # This can be adjusted based on performance needs

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

# Class for reading video stream, generating prediction and recommendations
class VideoCamera(object):
    def __init__(self):
        # Initialize the webcam stream once
        self.cap = WebcamVideoStream(src=0).start()

    def __del__(self):
        # Ensure the stream is stopped when the object is deleted
        self.cap.stop()

    def get_frame(self):
        # image = self.cap.read() # The image variable is unused here
        frame = self.cap.read() # Read frame from the threaded stream

        if frame is None:
            # Return an empty byte string or a placeholder image if no frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank_frame)
            return jpeg.tobytes(), None # Also return None for df1 if no frame

        # Resize the frame for display and processing
        image_display = cv2.resize(frame, (600, 500))
        gray_processing = cv2.cvtColor(image_display, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(gray_processing, 1.3, 5)
        
        # Initialize df1 here so it's always defined
        df1 = None 

        if len(face_rects) > 0:
            # Process only the first detected face for simplicity
            (x, y, w, h) = face_rects[0] 
            cv2.rectangle(image_display, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            roi_gray_frame = gray_processing[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            
            # Predict emotion
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            show_text[0] = maxindex # Update the global emotion index
            
            # Display emotion text on the frame
            cv2.putText(image_display, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get music recommendations based on the detected emotion
            df1 = music_rec() # This function internally uses show_text[0]
        else:
            # If no face is detected, set emotion to Neutral or last detected
            # For simplicity, let's revert to Neutral if no face
            show_text[0] = 4 # Neutral emotion index
            df1 = music_rec() # Get recommendations for neutral emotion

        # Convert the image for streaming
        ret, jpeg = cv2.imencode('.jpg', image_display)
        
        return jpeg.tobytes(), df1


# Spotify API credentials
SPOTIFY_CLIENT_ID = 'bbbc93468ee94649a12b5f58dfcb31b3'
SPOTIFY_CLIENT_SECRET = 'f21bdba4f80144e5b7f0049b87110314'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# Placeholder for features (not used in current music_rec but good to have if you implement get_song_recommendations)
EMOTION_TO_FEATURES = {
    "Happy": {'valence': 0.9, 'energy': 0.8, 'danceability': 0.7},
    "Sad": {'valence': 0.2, 'energy': 0.3, 'danceability': 0.4},
    "Angry": {'valence': 0.1, 'energy': 0.9, 'danceability': 0.6},
    "Neutral": {'valence': 0.5, 'energy': 0.5, 'danceability': 0.5},
    "Fearful": {'valence': 0.3, 'energy': 0.7, 'danceability': 0.5},
    "Surprised": {'valence': 0.7, 'energy': 0.7, 'danceability': 0.6},
    "Disgusted": {'valence': 0.2, 'energy': 0.6, 'danceability': 0.3},
}


def music_rec():
    emotion_name = emotion_dict.get(show_text[0], "Neutral")  # Use detected emotion, default to Neutral
    
    # Use emotion as a search keyword
    # It's better to search for a genre or mood rather than just "emotion_name songs" for better results
    # For now, let's stick to your current approach.
    
    # Try searching with the emotion name directly
    results = sp.search(q=f"{emotion_name} music", type='track', limit=15)
    
    # If direct emotion search yields few results, fallback to broader terms or genres
    if not results['tracks']['items']:
        print(f"No tracks found for '{emotion_name} music', trying general mood genres.")
        # Fallback to general mood genres based on emotion
        if emotion_name == "Happy":
            results = sp.search(q="upbeat happy pop", type='track', limit=15)
        elif emotion_name == "Sad":
            results = sp.search(q="melancholy slow music", type='track', limit=15)
        elif emotion_name == "Angry":
            results = sp.search(q="rock metal energetic", type='track', limit=15)
        elif emotion_name == "Fearful":
            results = sp.search(q="ambient calm music", type='track', limit=15)
        elif emotion_name == "Surprised":
            results = sp.search(q="upbeat catchy", type='track', limit=15)
        elif emotion_name == "Disgusted":
             results = sp.search(q="aggressive punk", type='track', limit=15)
        else: # Neutral
            results = sp.search(q="chill instrumental background", type='track', limit=15)


    data = []
    for item in results['tracks']['items']:
        name = item['name']
        album = item['album']['name']
        artist = ', '.join([artist['name'] for artist in item['artists']])
        url = item['external_urls']['spotify']
        data.append({"Name": name, "Artist": artist, "URL": url}) # Include URL in the dictionary
    
    df = pd.DataFrame(data)
    return df

# This function is not currently used but is a good example of using Spotify's recommendation engine
def get_song_recommendations(sp, emotion):
    features = EMOTION_TO_FEATURES.get(emotion, EMOTION_TO_FEATURES['Neutral'])
    results = sp.recommendations(
        seed_genres=['pop', 'rock'], # You can make this dynamic based on emotion
        limit=5,
        target_valence=features['valence'],
        target_energy=features['energy'],
        target_danceability=features['danceability']
    )
    return [(track['name'], track['artists'][0]['name'], track['external_urls']['spotify']) for track in results['tracks']]




import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd # Needed for creating DataFrame
# You will also need emotion_dict and show_text which are defined elsewhere in camera.py

# --- Spotify API Credentials and Initialization ---
# IMPORTANT: Replace with your actual Spotify API Client ID and Client Secret
SPOTIFY_CLIENT_ID = 'bbbc93468ee94649a12b5f58dfcb31b3'
SPOTIFY_CLIENT_SECRET = 'f21bdba4f80144e5b7f0049b87110314'

# Authenticate with Spotify using Client Credentials Flow (for server-side applications)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# --- Music Recommendation Function using Spotify API ---
def music_rec():
    """
    Fetches music recommendations from Spotify based on the current detected emotion.
    Assumes 'emotion_dict' and 'show_text' are globally accessible (from other parts of camera.py).
    Returns a Pandas DataFrame of recommended songs with Name, Artist, and Spotify URL.
    """
    # Get the human-readable emotion name from the global emotion_dict
    # show_text[0] is assumed to hold the numerical index of the detected emotion
    emotion_name = emotion_dict.get(show_text[0], "Neutral")  # Default to "Neutral" if index is not found

    # Attempt to search for music directly related to the emotion
    # Searches for tracks matching "[emotion_name] music"
    results = sp.search(q=f"{emotion_name} music", type='track', limit=15)
    
    # Fallback logic if the direct emotion search yields no results or poor results
    # This provides more robust recommendations even if the initial search is too specific
    if not results['tracks']['items']:
        print(f"No tracks found for '{emotion_name} music' on direct search, trying general mood genres.")
        # Try broader, mood-based searches as a fallback
        if emotion_name == "Happy":
            results = sp.search(q="upbeat happy pop", type='track', limit=15)
        elif emotion_name == "Sad":
            results = sp.search(q="melancholy slow music", type='track', limit=15)
        elif emotion_name == "Angry":
            results = sp.search(q="rock metal energetic", type='track', limit=15)
        elif emotion_name == "Fearful":
            results = sp.search(q="ambient calm music", type='track', limit=15)
        elif emotion_name == "Surprised":
            results = sp.search(q="upbeat catchy", type='track', limit=15)
        elif emotion_name == "Disgusted":
            results = sp.search(q="aggressive punk", type='track', limit=15)
        else: # For "Neutral" or any other unrecognized emotion
            results = sp.search(q="chill instrumental background", type='track', limit=15)

    data = []
    # Process the search results to extract song details
    for item in results['tracks']['items']:
        name = item['name']
        album = item['album']['name']
        # Join multiple artists if a song has them
        artist = ', '.join([artist_obj['name'] for artist_obj in item['artists']])
        url = item['external_urls']['spotify'] # Get the direct Spotify URL for the track
        data.append({"Name": name, "Artist": artist, "URL": url}) # Store Name, Artist, and URL

    # Convert the list of song dictionaries into a Pandas DataFrame
    df = pd.DataFrame(data)
    return df

# Note: The 'get_song_recommendations' function commented out in previous responses
# is an alternative way to get recommendations using Spotify's recommendation engine
# based on audio features. The 'music_rec' function above uses search, which is simpler
# for direct emotion-to-genre mapping. You can use either, but 'music_rec' is the one
# currently integrated into your VideoCamera's get_frame and app.py's /get_emotion_and_songs route.