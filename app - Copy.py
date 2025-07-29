from flask import Flask, render_template, Response, jsonify
# import gunicorn # This import is generally not needed for running with `python app.py`
from camera import VideoCamera, emotion_dict, show_text, music_rec # Import VideoCamera and other necessary globals
import time # Import time for potential delay if no frame is returned

app = Flask(__name__)

# No global camera object here to prevent early initialization

headings = ("Name","Album","Artist")

@app.route('/')
def index():
    # Initial state for emotion and songs before detection starts
    detected_emotion = "Click 'Start' to begin"
    initial_songs_data = [] # Empty list for initial display
    return render_template('index.html', headings=headings, data=initial_songs_data, emotion=detected_emotion)

def gen(camera):
    # This loop continuously yields frames from the camera
    while True:
        frame_bytes, df_songs = camera.get_frame() # get_frame returns bytes and dataframe
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        else:
            # If no frame is returned, wait a bit to avoid busy-waiting and try again
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    # Initialize the camera only when this route is accessed
    # This ensures the camera opens when the client requests the video feed
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion_and_songs')
def get_emotion_and_songs():
    # This route is called by JavaScript to get the latest emotion and song recommendations.
    # It doesn't initiate video capture itself.
    
    # We need a way to get the *latest* emotion from the active VideoCamera instance
    # that is feeding the /video_feed route.
    # The 'show_text' list is used for this global communication.
    
    detected_emotion = emotion_dict.get(show_text[0], "Detecting...")
    
    # Call music_rec to get new recommendations for the current emotion
    # Ensure music_rec uses the updated show_text[0]
    df = music_rec()
    
    songs = []
    # Ensure df is a DataFrame before iterating
    if df is not None:
        for _, row in df.iterrows():
            # music_rec in camera.py now returns a DataFrame directly with Name, Artist, URL
            # assuming 'URL' is a column with the Spotify link
            songs.append([row['Name'], row['Artist'], row['URL']]) # Directly use URL column
    
    return jsonify({'emotion': detected_emotion, 'songs': songs})

if __name__ == '__main__':
    app.debug = True
    app.run()