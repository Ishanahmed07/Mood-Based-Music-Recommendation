import os
import shutil
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Configuration (unchanged, ensure paths are correct) ---
RAF_DB_BASE_DIR = 'path/to/your/RAF-DB/basic'
RAF_DB_IMAGE_DIR = os.path.join(RAF_DB_BASE_DIR, 'Image/aligned')
RAF_DB_LABEL_FILE = os.path.join(RAF_DB_BASE_DIR, 'list_patition_label.txt')

FER2013_TRAIN_DIR = 'data/train'
FER2013_VAL_DIR = 'data/test'

COMBINED_BASE_DIR = 'combined_fer2013_rafdb'
COMBINED_TRAIN_DIR = os.path.join(COMBINED_BASE_DIR, 'train')
COMBINED_VAL_DIR = os.path.join(COMBINED_BASE_DIR, 'test')

EMOTION_MAP = {
    1: 'surprise',
    2: 'fear',
    3: 'disgust',
    4: 'happy',
    5: 'sad',
    6: 'angry',
    7: 'neutral'
}

# --- 1. Create the new combined directory structure ---
print(f"Creating combined data directory structure under: {COMBINED_BASE_DIR}")
for emotion_folder_name_lower in EMOTION_MAP.values(): # Use the consistent lowercase names
    os.makedirs(os.path.join(COMBINED_TRAIN_DIR, emotion_folder_name_lower), exist_ok=True)
    os.makedirs(os.path.join(COMBINED_VAL_DIR, emotion_folder_name_lower), exist_ok=True)

# --- 2. Copy FER2013 data to the combined directories ---
print("\n--- Copying FER2013 data ---")
# Changed variable name to avoid confusion: src_phase_dir and dest_phase_dir
for src_phase_dir, dest_phase_dir in [(FER2013_TRAIN_DIR, COMBINED_TRAIN_DIR), (FER2013_VAL_DIR, COMBINED_VAL_DIR)]:
    print(f"Copying from {src_phase_dir} to {dest_phase_dir}")
    for fer_emotion_folder in os.listdir(src_phase_dir): # This `fer_emotion_folder` might be 'Angry', 'Disgust', etc.
        src_folder_path = os.path.join(src_phase_dir, fer_emotion_folder)

        # IMPORTANT FIX: Canonicalize the destination folder name to lowercase
        # This assumes your EMOTION_MAP values are all lowercase.
        dest_emotion_folder = fer_emotion_folder.lower() # Convert to lowercase for destination
        dest_folder_path = os.path.join(dest_phase_dir, dest_emotion_folder)

        if os.path.isdir(src_folder_path): # Ensure it's a directory
            print(f"  Processing {fer_emotion_folder} -> to {dest_emotion_folder}...") # Added verbose print
            for filename in tqdm(os.listdir(src_folder_path)):
                # Added check to ensure the destination folder actually exists before copying
                # This should prevent the FileNotFoundError if the previous `makedirs` was consistent
                # but it's good for debugging. The real fix is the .lower() above.
                if not os.path.exists(dest_folder_path):
                     print(f"ERROR: Destination folder does not exist: {dest_folder_path}")
                     continue # Skip copying if dest folder is missing

                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    shutil.copy2(os.path.join(src_folder_path, filename), os.path.join(dest_folder_path, filename))
print("Finished copying FER2013 data.")

# --- 3. Process and Copy RAF-DB data (remaining part is mostly unchanged) ---
print("\n--- Processing and copying RAF-DB data ---")
raf_df = pd.read_csv(RAF_DB_LABEL_FILE, sep=' ', header=None, names=['image_name_raw', 'emotion_code'])

raf_df['image_name_full'] = raf_df['image_name_raw'].str.replace('.jpg', '_aligned.jpg', regex=False)

train_raf_df, val_raf_df = train_test_split(
    raf_df, test_size=0.2, random_state=42, stratify=raf_df['emotion_code']
)

# Copy RAF-DB training images
print("  Copying RAF-DB training images...")
for index, row in tqdm(train_raf_df.iterrows(), total=len(train_raf_df)):
    image_name = row['image_name_full']
    emotion_code = row['emotion_code']
    emotion_folder_name = EMOTION_MAP[emotion_code] # This name is already lowercase from EMOTION_MAP

    src_path = os.path.join(RAF_DB_IMAGE_DIR, image_name)
    dest_path = os.path.join(COMBINED_TRAIN_DIR, emotion_folder_name, image_name)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
    else:
        print(f"WARNING: RAF-DB image not found: {src_path}. Skipping.")

# Copy RAF-DB validation images
print("  Copying RAF-DB validation images...")
for index, row in tqdm(val_raf_df.iterrows(), total=len(val_raf_df)):
    image_name = row['image_name_full']
    emotion_code = row['emotion_code']
    emotion_folder_name = EMOTION_MAP[emotion_code]

    src_path = os.path.join(RAF_DB_IMAGE_DIR, image_name)
    dest_path = os.path.join(COMBINED_VAL_DIR, emotion_folder_name, image_name)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
    else:
        print(f"WARNING: RAF-DB image not found: {src_path}. Skipping.")

print("Finished processing and copying RAF-DB data.")
print(f"\nYour combined dataset is now prepared in: {COMBINED_BASE_DIR}")
print("You can now point your Keras ImageDataGenerator to these new directories.")