# Music Recommendation Based on Facial Expression

MoodTunes is a Python web application that recommends music from Spotify based on your facial emotions and a description of your day. It uses a convolutional neural network (CNN) to detect emotions from webcam video in real-time and suggests personalized songs to match your mood.

## Features
- **Real-Time Emotion Detection**: Detects emotions (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised) using a pre-trained CNN model.
- **Spotify Integration**: Fetches song recommendations from Spotify based on detected emotions and user-provided text input.
- **Web Interface**: A Flask-based interface displays the webcam feed, detected emotion, and recommended songs with Spotify links.
- **Custom Mood Input**: Users can describe their day to refine music recommendations.
- **Model Training**: Includes a script to train the emotion detection model using FER2013 and RAF-DB datasets.

## Technologies Used
- **Python 3.8+**: Core language.
- **OpenCV**: For webcam capture and face detection.
- **TensorFlow/Keras**: For the emotion detection CNN model.
- **Spotipy**: For Spotify API integration.
- **Flask**: For the web application.
- **Pandas**: For managing song recommendation data.
- **HTML/CSS**: For the front-end interface.

## Prerequisites
- **Python 3.8 or higher**: Install from [python.org](https://www.python.org/downloads/).
- **Webcam**: A functional webcam for video capture.
- **Spotify API Credentials**: Obtain Client ID and Client Secret from the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
- **Haar Cascade File**: `haarcascade_frontalface_default.xml` for face detection (included with OpenCV).
- **Model Weights**: `model.weights.h5` (generate with `train.py` or obtain separately).
- **Dataset (Optional)**: FER2013 and RAF-DB datasets in `combined_fer2013_rafdb` for training the model.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Music-Recommendation-Based-on-Facial-Expression.git
   cd Music-Recommendation-Based-on-Facial-Expression-Spotify-API-Python-main
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Spotify API Credentials**:
   - Register an app on the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/).
   - Set environment variables:
     ```bash
     export SPOTIPY_CLIENT_ID='your-client-id'
     export SPOTIPY_CLIENT_SECRET='your-client-secret'
     ```
     On Windows (Command Prompt):
     ```bash
     set SPOTIPY_CLIENT_ID=your-client-id
     set SPOTIPY_CLIENT_SECRET=your-client-secret
     ```
     On Windows (MINGW64/Git Bash):
     ```bash
     export SPOTIPY_CLIENT_ID='your-client-id'
     export SPOTIPY_CLIENT_SECRET='your-client-secret'
     ```

5. **Obtain Haar Cascade File**:
   - Download `haarcascade_frontalface_default.xml` from [OpenCV’s GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades).
   - Place it in the project root directory.

6. **Obtain Model Weights**:
   - Place `model.weights.h5` in the project root if you have it.
   - Otherwise, train the model using `train.py` (see "Training the Model").

## Usage

### Running the Web Application
1. Activate the virtual environment:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in your browser.
4. Click "Start Listening" to enable webcam capture and emotion detection.
5. Enter a day description and click "Get Smarter Songs" for refined recommendations.
6. View recommended songs with Spotify links in the table.

### Training the Model
1. Ensure the `combined_fer2013_rafdb` directory contains `train` and `test` subdirectories with FER2013 and RAF-DB datasets.
2. Run:
   ```bash
   python train.py
   ```
3. The script trains the CNN for 75 epochs and saves weights to `model.weights.h5`.

## Project Structure
```
Music-Recommendation-Based-on-Facial-Expression-Spotify-API-Python-main/
├── camera.py                # Emotion detection and Spotify API logic
├── app.py                   # Flask web application
├── train.py                 # CNN model training script
├── templates/
│   └── index.html           # Web interface template
├── requirements.txt         # Python dependencies
├── haarcascade_frontalface_default.xml  # Face detection cascade
├── model.weights.h5         # Pre-trained model weights (optional)
└── README.md                # Project documentation
```

## Troubleshooting

### Git Issues
If `git add .` does nothing but `git status` shows untracked/modified files:
- **Check `.gitignore`**:
  ```bash
  cat .gitignore
  ```
  Verify that files like `model.weights.h5`, `venv/`, or `combined_fer2013_rafdb/` are intentionally ignored. Update `.gitignore` if needed:
  ```bash
  nano .gitignore
  ```
- **Check for Lock Files**:
  ```bash
  cd .git
  ls *.lock
  rm <lock_file_name>  # Remove any lock files (e.g., HEAD.lock)
  cd ..
  ```
- **Test Adding Specific Files**:
  ```bash
  git add README.md
  git status
  ```
  If this works, try adding other files individually:
  ```bash
  git add camera.py app.py train.py templates/index.html
  ```
- **Verbose Output**:
  ```bash
  GIT_TRACE=1 git add .
  ```
  Inspect the output for errors or skipped files.
- **Check Permissions (MINGW64/Windows)**:
  Run the terminal as administrator:
  ```bash
  icacls .git /grant Everyone:F
  ```
  Or check the file system:
  ```bash
  chkdsk D: /f
  ```
- **Check for Running Git Processes**:
  ```bash
  tasklist | findstr git
  taskkill /F /PID <process_id>
  ```
- **Reclone if Necessary**:
  If the repository is corrupted:
  ```bash
  cd ..
  mv Music-Recommendation-Based-on-Facial-Expression-Spotify-API-Python-main Backup
  git clone <repository_url>
  ```

### Other Issues
- **Webcam Errors**: Ensure no other applications are using the webcam.
- **Spotify API Errors**: Verify Client ID and Client Secret are set correctly.
- **Model Errors**: Ensure `model.weights.h5` exists or run `train.py`.
- **Flask Errors**: Check if port 5000 is free (`netstat -a -n -o | find "5000"`) or change it in `app.py`.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub.

## License
MIT License. See [LICENSE](LICENSE) for details.