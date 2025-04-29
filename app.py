from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import git
from scipy.spatial.distance import cosine
import os

app = FastAPI()

# Define the path for the embeddings folder
REPO_DIR = os.path.join(os.getcwd(), "Embeddings")

# Ensure the Embeddings directory exists
if not os.path.exists(REPO_DIR):
    os.makedirs(REPO_DIR)

GIT_REPO_URL = "https://github.com/Yashsingh1224/Storage-for-embeddings"
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    repo = git.Repo.init(REPO_DIR)
    repo.create_remote('origin', GIT_REPO_URL)
else:
    repo = git.Repo(REPO_DIR)

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
async def register_user(username: str = Form(...), file1: UploadFile = File(...), file2: UploadFile = File(...), file3: UploadFile = File(...), file4: UploadFile = File(...), file5: UploadFile = File(...)):
    files = [file1, file2, file3, file4, file5]  # Handle 5 files
    embeddings = []

    for file in files:
        with open(file.filename, 'wb') as f:
            f.write(await file.read())
        audio = preprocess_audio(file.filename)
        features = extract_features(audio)
        embeddings.append(features)
        os.remove(file.filename)

    avg_embedding = np.mean(np.array(embeddings), axis=0)
    np.save(f"{REPO_DIR}/{username}.npy", avg_embedding)

    # Commit and push to GitHub
    try:
        repo.index.add([f"{username}.npy"])
        repo.index.commit(f"Added {username}'s embeddings")
        repo.remotes.origin.push()
    except git.exc.GitCommandError as e:
        return JSONResponse({"error": f"Failed to push to GitHub: {str(e)}"})


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
