from fastapi import FastAPI

app = FastAPI(
    title="Music Profanity Remover API",
    description="APIs for removing profanity from music lyrics.",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Music-Profanity-Remover"}