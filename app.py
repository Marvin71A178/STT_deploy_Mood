import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import uvicorn
import Mood.predict as mood_pd

app = FastAPI()

# 設定允許的來源
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]



# 定義請求體模型
class ProcessMoodRequest(BaseModel):
    TestData: str

# 根路由，歡迎訊息
@app.get("/")
async def root():
    return {"message": "Welcome to the AudioCraft API. Use /mood_analyze/ to perform mood analysis and /music_generate/ to generate music based on mood."}

# 處理 OPTIONS 請求
@app.options("/mood_analyze/")
async def options_mood_analyze():
    return {}

# 情緒分析路由
@app.post("/mood_analyze/")
async def perform_mood_pd(request: ProcessMoodRequest):
    try:
        result = await run_in_threadpool(mood_pd.prediction, request.TestData)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 添加CORS中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 80))
    uvicorn.run(app, host='0.0.0.0', port=port)
