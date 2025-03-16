from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from roboflow import Roboflow
import cv2
import numpy as np
import io
import os
import base64
import shutil
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import uvicorn
import asyncio

matplotlib.use('Agg')

app = FastAPI()

# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MIME_TYPES = {"jpg": "image/jpg", "jpeg": "image/jpeg", "png": "image/png"}

# โฟลเดอร์สำหรับบันทึกไฟล์
UPLOAD_DIR = "uploadss"
OUTPUT_DIR = "outputss"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ตั้งค่า Roboflow
rf = Roboflow(api_key="mJRJtBYRhInoDZPAzrv3")
project = rf.workspace("mongta-swaxu").project("mongta")
model = project.version("3").model

class ScanResult(BaseModel):
    user_id: int
    description: str
    pic_description: str
    ai_right_image_url: str
    ai_left_image_url: str
    ai_right_image_base64: str
    ai_left_image_base64: str

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image: Image.Image) -> Image.Image:
    """Resize the image to 640x640 if it is larger."""
    width, height = image.size
    if width < 640 or height < 640:
        return image
    return image.resize((640, 640))

def draw_boxes(image: Image.Image, predictions: dict) -> Image.Image:
    """ วาดกรอบรอบวัตถุที่ตรวจพบ """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    for pred in predictions['predictions']:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        x1, y1, x2, y2 = x - width/2, y - height/2, x + width/2, y + height/2
        rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"{pred['class']}", color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def detect_eyes(image: Image.Image) -> Image.Image:
    """Detect eyes in the image and return an enhanced zoomed-in region if detected."""
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    img_resized = image.resize((640, 640))
    img_resized = np.array(img_resized)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        zoomed_eye = img_resized[y:y+h, x:x+w]
        zoomed_eye_resized = cv2.resize(zoomed_eye, (640, 640))
        return Image.fromarray(zoomed_eye_resized)
    return image

async def generate_ai_analysis(image_path: str, output_path: str):
    """
    Uses Roboflow to analyze the image and generate annotated results.
    This function runs asynchronously to avoid blocking the server.
    """
    try:
        # Run the AI model asynchronously
        result = await asyncio.to_thread(model.predict, image_path, confidence=40, overlap=30)
        predictions = result.json()

        # Open the original image
        image = Image.open(image_path)

        # Draw bounding boxes on the image
        annotated_image = draw_boxes(image, predictions)

        # Save the processed image
        annotated_image.save(output_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in AI analysis: {str(e)}")

@app.post("/api-ai/upload-eye-predict", summary="Upload eye images for AI analysis")
async def upload_eye_predict(
    right_eye: UploadFile = File(..., description="Image of the right eye"),
    left_eye: UploadFile = File(..., description="Image of the left eye"),
    user_id: str = Form(..., description="User's unique ID"),
    va_right: str = Form(..., description="Visual acuity of the right eye"),
    va_left: str = Form(..., description="Visual acuity of the left eye"),
    near_description: str = Form("", description="Description of near vision test"),
    line_right: str = Form("", description="Right eye line information"),
    line_left: str = Form("", description="Left eye line information")
):
    try:
        # 📌 1. บันทึกภาพที่อัปโหลด
        right_eye_path = os.path.join(UPLOAD_DIR, f"{user_id}_right.png")
        left_eye_path = os.path.join(UPLOAD_DIR, f"{user_id}_left.png")
        with open(right_eye_path, "wb") as buffer:
            shutil.copyfileobj(right_eye.file, buffer)
        with open(left_eye_path, "wb") as buffer:
            shutil.copyfileobj(left_eye.file, buffer)
        
        # 📌 2. ใช้ Roboflow วิเคราะห์ภาพ
        predictions_right = model.predict(right_eye_path, confidence=40, overlap=30).json()
        predictions_left = model.predict(left_eye_path, confidence=40, overlap=30).json()

        # 📌 3. วาดกรอบรอบวัตถุที่พบ
        right_eye_img = Image.open(right_eye_path).convert("RGB")
        left_eye_img = Image.open(left_eye_path).convert("RGB")
        
        right_eye_annotated = draw_boxes(right_eye_img, predictions_right)
        left_eye_annotated = draw_boxes(left_eye_img, predictions_left)

        # 📌 4. บันทึกภาพผลลัพธ์
        ai_right_path = os.path.join(OUTPUT_DIR, f"{user_id}_ai_right.png")
        ai_left_path = os.path.join(OUTPUT_DIR, f"{user_id}_ai_left.png")
        right_eye_annotated.save(ai_right_path, format="PNG")
        left_eye_annotated.save(ai_left_path, format="PNG")

        # 📌 5. แปลงภาพเป็น Base64
        ai_right_image_base64 = base64.b64encode(open(ai_right_path, "rb").read()).decode()
        ai_left_image_base64 = base64.b64encode(open(ai_left_path, "rb").read()).decode()

        # 📌 6. ตรวจสอบว่ามีความผิดปกติหรือไม่
        right_eye_classes = [pred['class'] for pred in predictions_right['predictions']]
        left_eye_classes = [pred['class'] for pred in predictions_left['predictions']]
        
        if all(cls == "Normal" for cls in right_eye_classes) and all(cls == "Normal" for cls in left_eye_classes):
            pic_description = "ยังไม่พบสิ่งผิดปกติบนดวงตาข้างซ้ายและข้างขวา"
        elif any(cls != "Normal" for cls in right_eye_classes) and any(cls != "Normal" for cls in left_eye_classes):
            pic_description = "พบความผิดปกติที่ดวงตาข้างซ้ายและข้างขวา"
        elif any(cls != "Normal" for cls in right_eye_classes):
            pic_description = "พบความผิดปกติที่ดวงตาข้างขวา"
        elif any(cls != "Normal" for cls in left_eye_classes):
            pic_description = "พบความผิดปกติที่ดวงตาข้างซ้าย"
        else:
            pic_description = "ยังไม่สามารถระบุผลลัพธ์ได้"

        # 📌 7. ส่งผลลัพธ์กลับไปยังผู้ใช้
        return ScanResult(
            user_id=user_id,
            description="AI analysis completed successfully",
            pic_description=pic_description,
            ai_right_image_url=f"http://127.0.0.1:8000/output/{user_id}_ai_right.png",
            ai_left_image_url=f"http://127.0.0.1:8000/output/{user_id}_ai_left.png",
            ai_right_image_base64=ai_right_image_base64,
            ai_left_image_base64=ai_left_image_base64,
        )

    except Exception as e:
        return {"success": False, "message": "Error analyzing eye scan", "error": str(e)}

# 📌 8. เสิร์ฟไฟล์ที่เซิร์ฟเวอร์ให้ผู้ใช้เรียกดูภาพผ่าน URL
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)