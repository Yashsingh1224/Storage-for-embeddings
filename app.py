import os
import git
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from scipy.spatial.distance import cosine

app = FastAPI()

# Set REPO_DIR to your current project directory
REPO_DIR = os.getcwd()  # This will point to the current directory where app.py is located
GIT_REPO_URL = "https://github.com/Yashsingh1224/Storage-for-embeddings"  # Your GitHub repo URL

# Initialize the local repo if not already done
repo = git.Repo(REPO_DIR)
if not repo.bare:
    repo.remotes.origin.fetch()

def preprocess_audio(file, sr=16000):
    audio, _ = librosa.load(file, sr=sr)
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio)
    return audio

def extract_features(audio, sr=16000, max_pad_len=100):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    pad_width = max_pad_len - features.shape[1]
    if pad_width > 0:
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_pad_len]
    return np.mean(features, axis=1)

@app.post("/register-user")
async def register_user(username: str = Form(...), file1: UploadFile = File(...), file2: UploadFile = File(...), file3: UploadFile = File(...)):
    files = [file1, file2, file3]
    embeddings = []

    for file in files:
        with open(file.filename, 'wb') as f:
            f.write(await file.read())
        audio = preprocess_audio(file.filename)
        features = extract_features(audio)
        embeddings.append(features)
        os.remove(file.filename)

    avg_embedding = np.mean(np.array(embeddings), axis=0)
    np.save(f"{REPO_DIR}/{username}.npy", avg_embedding)  # Save the .npy file in your current project directory

    # Commit and push to GitHub
    repo.index.add([f"{username}.npy"])
    repo.index.commit(f"Added {username}'s embeddings")
    repo.remotes.origin.push()

    return JSONResponse({"message": f"User {username} registered successfully."})

@app.post("/verify-voice")
async def verify_voice(username: str = Form(...), file: UploadFile = File(...)):
    # Fetch the embedding from GitHub repository
    local_file_path = f"{REPO_DIR}/{username}.npy"
    if not os.path.exists(local_file_path):
        return JSONResponse({"match": False, "message": "User not found."})

    with open(file.filename, 'wb') as f:
        f.write(await file.read())
    audio = preprocess_audio(file.filename)
    test_embedding = extract_features(audio)
    os.remove(file.filename)

    saved_embedding = np.load(local_file_path)
    similarity = 1 - cosine(saved_embedding, test_embedding)

    match = similarity > 0.75
    return JSONResponse({"match": match, "similarity": similarity})
