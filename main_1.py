from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from roboflow import Roboflow
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import os
from PIL import Image

app = FastAPI()

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MIME_TYPES = {"jpg": "image/jpg", "jpeg": "image/jpeg", "png": "image/png"}

rf = Roboflow(api_key="mJRJtBYRhInoDZPAzrv3")
project = rf.workspace("mongta-swaxu").project("mongta")
model = project.version("6").model

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width < 640 or height < 640:
        return image
    return image.resize((640, 640))

def enhance_image(image: np.ndarray) -> np.ndarray:
    enhanced = cv2.detailEnhance(image, sigma_s=1, sigma_r=0.15)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened

def detect_eyes(image: Image.Image) -> Image.Image:
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        zoomed_eye = img_array[y:y+h, x:x+w]
        zoomed_eye_resized = cv2.resize(zoomed_eye, (640, 640))
        return Image.fromarray(enhance_image(zoomed_eye_resized))
    return image

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # ตรวจสอบประเภทไฟล์
    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg, .jpeg, and .png are allowed.")
    
    # ดึงนามสกุลไฟล์ (jpg, jpeg, png)
    image_extension = image.filename.rsplit('.', 1)[-1].lower()
    
    # อ่านไฟล์ภาพ
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_image = resize_image(image_pil)
    
    # แปลงนามสกุลเป็น "JPEG" ถ้าเป็น "jpg"
    pil_format = "JPEG" if image_extension in ["jpg", "jpeg"] else "PNG"

    # บันทึกไฟล์ชั่วคราว
    temp_path = f"temp.{image_extension}"
    resized_image.save(temp_path, format=pil_format)

    # ใช้โมเดล Roboflow ตรวจจับ
    predictions = model.predict(temp_path, confidence=40, overlap=30).json()
    
    # ถ้าไม่เจอ ให้ลองตรวจจับจากดวงตา
    if not predictions['predictions']:
        eye_image = detect_eyes(resized_image)
        eye_image.save(temp_path, format=pil_format)
        predictions = model.predict(temp_path, confidence=40, overlap=30).json()
        if not predictions['predictions']:
            os.remove(temp_path)
            raise HTTPException(status_code=404, detail="No predictions found.")
    
    # วาดกรอบบนภาพ
    image_np = np.array(resized_image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    for pred in predictions['predictions']:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        x1, y1, x2, y2 = x - width / 2, y - height / 2, x + width / 2, y + height / 2
        plt.gca().add_patch(plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none'))
        plt.text(x1, y1 - 10, f"{pred['class']} ({pred['confidence']:.2f})", color='red', fontsize=12,
                 bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    
    # บันทึกผลลัพธ์ลงในไฟล์
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี
    output_filename = os.path.join(output_dir, f"predicted_{image.filename}")
    plt_format = "jpeg" if image_extension in ["jpg", "jpeg"] else "png"
    plt.savefig(output_filename, format=plt_format, bbox_inches='tight', pad_inches=0)
    plt.close()

    # ลบไฟล์ชั่วคราว
    os.remove(temp_path)

    # ส่งไฟล์ที่บันทึกไปยังผู้ใช้
    return StreamingResponse(open(output_filename, "rb"), media_type=MIME_TYPES[image_extension], headers={"Content-Disposition": f"attachment; filename={output_filename}"})

