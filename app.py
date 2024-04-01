import os , sys
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import  Mood.predict as mood_pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from starlette.concurrency import run_in_threadpool
import uvicorn

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class process_Mood_pd(BaseModel):
    TestData: str

@app.get("/")
async def root():
    return {"message": "Welcome to the AudioCraft API. Use /mood_analyze/ to perform mood analysis and /music_generate/ to generate music based on mood."}

@app.post("/mood_analyze/")
async def perform_mood_pd(request: process_Mood_pd):
    try:
        result = await run_in_threadpool(mood_pd.prediction, request.TestData)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8061))
    uvicorn.run(app , host = '0.0.0.0' , port = port)
    