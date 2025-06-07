from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from inference_model import predict_disease
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ frontend 디렉토리 경로
frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")

# ✅ 정적 파일은 /static으로 마운트
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# ✅ index.html 직접 반환
@app.get("/")
def root():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# ✅ 예측 API (POST /predict)
@app.post("/predict")
async def predict(
    gender: str = Form(...),
    age: int = Form(...),
    body_part: str = Form(...),
    image: UploadFile = File(...)
):
    image_data = await image.read()

    result = predict_disease(image_data, {
        "gender": gender,
        "age": age,
        "body_part": body_part
    })
    print(f"Prediction result: {result}")  # 디버깅용 로그

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)