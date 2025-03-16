from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
import io
import cv2
import numpy as np
from PIL import Image
import httpx
import traceback

# ✅ ตั้งค่า FastAPI
app = FastAPI()

# ✅ ตั้งค่า CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ กำหนดโฟลเดอร์อัปโหลด
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_image(image_bytes: bytes) -> bytes:
    """✅ แปลงภาพเป็นขาวดำด้วย OpenCV และคืนค่าเป็นไฟล์"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("ไม่สามารถอ่านไฟล์รูปภาพได้")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, buffer = cv2.imencode(".png", gray_image)
    return buffer.tobytes()

@app.post("/api-ai/upload-eye-predict", summary="Upload and analyze eye images")
async def upload_eye_predict(
    user_id: int = Form(...),
    line_right: str = Form(...),
    line_left: str = Form(...),
    va_right: str = Form(...),
    va_left: str = Form(...),
    near_description: str = Form(...),
    right_eye: UploadFile = File(...),
    left_eye: UploadFile = File(...)
):
    try:
        # ✅ อ่านไฟล์ภาพ
        right_eye_bytes = await right_eye.read()
        left_eye_bytes = await left_eye.read()

        # ✅ ประมวลผลภาพด้วย OpenCV
        ai_right_image = process_image(right_eye_bytes)
        ai_left_image = process_image(left_eye_bytes)

        # ✅ ส่งไฟล์ไปยัง Node.js API
        form_data = {
            "user_id": (None, str(user_id)),
            "line_right": (None, line_right),
            "line_left": (None, line_left),
            "va_right": (None, va_right),
            "va_left": (None, va_left),
            "near_description": (None, near_description),
            "right_eye": (right_eye.filename, io.BytesIO(right_eye_bytes), "image/png"),
            "left_eye": (left_eye.filename, io.BytesIO(left_eye_bytes), "image/png"),
            "ai_right": ("ai_right.png", io.BytesIO(ai_right_image), "image/png"),
            "ai_left": ("ai_left.png", io.BytesIO(ai_left_image), "image/png"),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:3000/api/savescanlog", files=form_data)
            response.raise_for_status()

        return JSONResponse(content={"success": True, "message": "Scan log sent successfully."})

    except Exception as e:
        error_trace = traceback.format_exc() 
        print(f"🔥 ERROR: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# ✅ เริ่มต้นเซิร์ฟเวอร์ FastAPI
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)