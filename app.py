from flask import Flask, render_template, Response, jsonify, request # Import request
# Make sure to import the updated music_rec
from camera import VideoCamera, emotion_dict, show_text, music_rec 
import time

app = Flask(__name__)

headings = ("Name","Artist","URL") # Added URL to headings

@app.route('/')
def index():
    detected_emotion = "Click 'Start Listening' to begin"
    initial_songs_data = []
    return render_template('index.html', headings=headings, data=initial_songs_data, emotion=detected_emotion)

def gen(camera):
    while True:
        frame_bytes, _ = camera.get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion_and_songs')
def get_emotion_and_songs():
    # Get the day description from the query parameters
    day_description = request.args.get('day_description', '') # Default to empty string

    # Get the detected emotion string using the stored key
    current_emotion_key = show_text[0]
    detected_emotion_str = emotion_dict.get(current_emotion_key, "Detecting...")

    # Call music_rec with both emotion and day description
    df = music_rec(detected_emotion_str, day_description) # Pass emotion string and description
    
    songs = []
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            songs.append([row['Name'], row['Artist'], row['spotify_url']]) 
    
    return jsonify({'emotion': detected_emotion_str, 'songs': songs})

if __name__ == '__main__':
    app.debug = True
    app.run()