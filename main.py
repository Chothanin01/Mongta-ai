from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from requests_toolbelt.multipart.encoder import MultipartEncoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from roboflow import Roboflow
import os
import uvicorn
from pydantic import BaseModel
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import asyncio
import io
import cv2
import numpy as np
import httpx
import base64
import tempfile

# Prevents Matplotlib from requiring a display
matplotlib.use('Agg')

# Initialize FastAPI application
app = FastAPI()

# Enable CORS to allow cross-origin requests from Node.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories for uploads and outputs
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allowed file types for uploaded images
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MIME_TYPES = {"jpg": "image/jpg", "jpeg": "image/jpeg", "png": "image/png"}

class ScanResult(BaseModel):
    """Data model for storing eye scan results"""
    user_id: int
    description: str
    pic_description: str
    ai_right_image_url: str
    ai_left_image_url: str

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a valid extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to Base64 string for easier transport"""
    try:
        # Open image and ensure it's in RGB mode before encoding
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Save to a byte buffer in JPEG format
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        
        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        # Return a placeholder if encoding fails
        return ""

def resize_image(image: Image.Image) -> Image.Image:
    """Resize the image to 256x256 if it is larger"""
    width, height = image.size
    if width > 640 or height > 640:
        return image.resize((640, 640))
    return image

def detect_eyes(image: Image.Image) -> Image.Image:
    """Detect eyes and return a zoomed-in region if detected"""
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    img_resized = image.resize((640, 640))
    img_array = np.array(img_resized)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        zoomed_eye = img_array[y:y+h, x:x+w]
        zoomed_eye_resized = cv2.resize(zoomed_eye, (640, 640))
        return Image.fromarray(zoomed_eye_resized)
    return image

async def generate_ai_analysis(image_path: str, output_path: str):
    """Runs Roboflow AI asynchronously and saves the analyzed image"""
    try:
        result = await asyncio.to_thread(model.predict, image_path, confidence=40, overlap=30)
        predictions = result.json()
        image = Image.open(image_path)
        annotated_image = draw_boxes(image, predictions)
        annotated_image.convert("RGB").save(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in AI analysis: {str(e)}")

def draw_boxes(image: Image.Image, predictions: dict) -> Image.Image:
    """Draw bounding boxes around detected objects in the image using Matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for pred in predictions['predictions']:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        x1, y1 = x - width / 2, y - height / 2

        # ใช้สี #12358F สำหรับกรอบและข้อความ
        custom_color = '#12358F'
        rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=custom_color, facecolor='none')
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 10, f"{pred['class']}", color=custom_color, fontsize=12,
            bbox=dict(facecolor='white', edgecolor=custom_color, boxstyle='round,pad=0.3', linewidth=1.5)
        )
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    # Convert RGBA to RGB before returning
    return Image.open(buf).convert("RGB")

async def predict_image_memory(image: Image.Image, confidence=40, overlap=30):
    """Predict by temporarily saving image to a secure temp file path"""
    with tempfile.NamedTemporaryFile(suffix=".png") as temp:
        image.save(temp.name, format="PNG")
        result = await asyncio.to_thread(model.predict, temp.name, confidence=confidence, overlap=overlap)
    return result.json()

@app.on_event("startup")
async def startup_event():
    global model
    rf = Roboflow(api_key="mJRJtBYRhInoDZPAzrv3")
    project = rf.workspace("mongta-swaxu").project("mongta")
    model = project.version("3").model

@app.post("/api-ai/upload-eye-predict", summary="Upload and analyze eye images")
async def analyze_eye_scan(
    user_id: int = Form(...),
    line_right: str = Form(...),
    line_left: str = Form(...),
    va_right: str = Form(...),
    va_left: str = Form(...),
    near_description: str = Form(...),
    right_eye: UploadFile = File(...),
    left_eye: UploadFile = File(...),
):
    try:
        if not right_eye or not left_eye:
            raise HTTPException(status_code=400, detail="Right eye and left eye images are required.")

        # Validate file extensions
        if not (allowed_file(right_eye.filename) and allowed_file(left_eye.filename)):
            raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, JPEG, and PNG are allowed.")

        # Read uploaded images into memory
        right_eye_bytes = await right_eye.read()
        left_eye_bytes = await left_eye.read()

        right_eye_path = os.path.join(UPLOAD_DIR, right_eye.filename)
        left_eye_path = os.path.join(UPLOAD_DIR, left_eye.filename)

        right_eye_pil = Image.open(io.BytesIO(right_eye_bytes)).convert("RGB")
        left_eye_pil = Image.open(io.BytesIO(left_eye_bytes)).convert("RGB")

        # Determine file format
        image_extension = right_eye.filename.rsplit('.', 1)[-1].lower()
        pil_format = "JPEG" if image_extension in ["jpg", "jpeg"] else "PNG"

        # Resize images
        resized_right_image = resize_image(right_eye_pil)
        resized_left_image = resize_image(left_eye_pil)

        predictions_right, predictions_left = await asyncio.gather(
        predict_image_memory(resized_right_image),
        predict_image_memory(resized_left_image)
        )
        # AI Processing for Right Eye
        if not predictions_right['predictions']:
            detected_right = detect_eyes(right_eye_pil).convert("RGB")
            predictions_right_retry = await predict_image_memory(detected_right)

            if predictions_right_retry['predictions']:
                final_right = detected_right
                predictions_right = predictions_right_retry
                image_for_drawing_right = detected_right
            else:
                final_right = resized_right_image
                image_for_drawing_right = resized_right_image
        else:
            final_right = resized_right_image
            image_for_drawing_right = resized_right_image

        # AI Processing for Left Eye
        if not predictions_left['predictions']:
            detected_left = detect_eyes(left_eye_pil).convert("RGB")
            predictions_left_retry = await predict_image_memory(detected_left)

            if predictions_left_retry['predictions']:
                final_left = detected_left
                predictions_left = predictions_left_retry
                image_for_drawing_left = detected_left
            else:
                final_left = resized_left_image
                image_for_drawing_left = resized_left_image
        else:
            final_left = resized_left_image
            image_for_drawing_left = resized_left_image

        # Draw AI Predictions
        predicted_right_path = os.path.join(OUTPUT_DIR, f"predicted_{user_id}_right.{image_extension}")
        predicted_left_path = os.path.join(OUTPUT_DIR, f"predicted_{user_id}_left.{image_extension}")

        # output_right_predicted = draw_boxes(final_right, predictions_right)
        output_left_predicted = draw_boxes(image_for_drawing_left, predictions_left)
        output_right_predicted = draw_boxes(image_for_drawing_right, predictions_right)
        # output_left_predicted = draw_boxes(image_for_drawing_left, predictions_left)

        output_right_predicted.convert("RGB").save(predicted_right_path, format=pil_format)
        output_left_predicted.convert("RGB").save(predicted_left_path, format=pil_format)

        # Save original and AI-processed images
        original_right_path = os.path.join(OUTPUT_DIR, f"{user_id}_right.{image_extension}")
        original_left_path = os.path.join(OUTPUT_DIR, f"{user_id}_left.{image_extension}")
        final_right.convert("RGB").save(original_right_path, format="PNG")
        final_left.convert("RGB").save(original_left_path, format="PNG")

        await asyncio.gather(
        generate_ai_analysis(original_right_path, predicted_right_path),
        generate_ai_analysis(original_left_path, predicted_left_path)
        )

        # # Generate eye health descriptions
        right_eye_classes = [pred['class'] for pred in predictions_right['predictions']]
        left_eye_classes = [pred['class'] for pred in predictions_left['predictions']]

        # รายชื่อโรคเป็นภาษาไทย
        disease_map = {
            "Cataract": "ต้อกระจก",
            "Conjunctivitis": "ตาแดง",
            "Normal": "ปกติ",
            "Pterygium": "ต้อเนื้อ",
            "Stye": "ตากุ้งยิง"
        }

        # แปลชื่อโรค
        right_eye_translated = [disease_map.get(cls, cls) for cls in right_eye_classes]
        left_eye_translated = [disease_map.get(cls, cls) for cls in left_eye_classes]

        # รายงานภาพรวม
        if all(cls == "Normal" for cls in right_eye_classes) and all(cls == "Normal" for cls in left_eye_classes):
            pic_description = "ยังไม่พบสิ่งผิดปกติบนดวงตาข้างซ้ายเเละข้างขวา"
        elif any(cls != "Normal" for cls in right_eye_classes) and any(cls != "Normal" for cls in left_eye_classes):
            pic_description = "พบความผิดปกติที่ดวงตาข้างซ้ายและดวงตาข้างขวา"
        elif any(cls != "Normal" for cls in right_eye_classes):
            pic_description = "พบความผิดปกติที่ดวงตาข้างขวา"
        elif any(cls != "Normal" for cls in left_eye_classes):
            pic_description = "พบความผิดปกติที่ดวงตาข้างซ้าย"
        else:
            pic_description = "ยังไม่สามารถระบุผลลัพธ์ได้"

        # รายงานรายตา
        right_eye_description = "การสเเกนดวงตาขวายังไม่พบเจอสิ่งผิดปกติ" if all(cls == "ปกติ" for cls in right_eye_translated) else \
            f"การสเเกนดวงตาขวาพบความผิดปกติ อาจมีอาการ {' '.join(set(right_eye_translated))}"

        left_eye_description = "การสเเกนดวงตาซ้ายยังไม่พบเจอสิ่งผิดปกติ" if all(cls == "ปกติ" for cls in left_eye_translated) else \
            f"การสเเกนดวงตาซ้ายพบความผิดปกติ อาจมีอาการ {' '.join(set(left_eye_translated))}"

        # ระดับความเสี่ยง
        risk_levels = ["ปกติ", "เริ่มมีความผิดปกติ", "มีความผิดปกติ"]
        if "มีความผิดปกติ" in near_description:
            risk_level = risk_levels[2]
        elif "เริ่มมีความผิดปกติ" in near_description:
            risk_level = risk_levels[1]
        else:
            risk_level = risk_levels[0]

        # รายงานสรุป
        if all(cls == "Normal" for cls in right_eye_classes) and all(cls == "Normal" for cls in left_eye_classes):
            description = f"เบื้องต้นพบว่าค่าสายตาทั้งสองข้างอยู่ในระดับ{risk_level} และไม่พบความผิดปกติจากการสแกนดวงตา"
        elif any(cls != "Normal" for cls in right_eye_classes) and any(cls != "Normal" for cls in left_eye_classes):
            description = f"เบื้องต้นพบว่าค่าสายตาทั้งสองข้างอยู่ในระดับ{risk_level} และพบความผิดปกติที่ดวงตาทั้งสองข้าง อาจมีอาการ {' '.join(set(right_eye_translated + left_eye_translated))}"
        elif any(cls != "Normal" for cls in right_eye_classes):
            description = f"เบื้องต้นพบว่าค่าสายตาทั้งสองข้างอยู่ในระดับ{risk_level} และพบความผิดปกติที่ดวงตาข้างขวา อาจมีอาการ {' '.join(set(right_eye_translated))}"
        elif any(cls != "Normal" for cls in left_eye_classes):
            description = f"เบื้องต้นพบว่าค่าสายตาทั้งสองข้างอยู่ในระดับ{risk_level} และพบความผิดปกติที่ดวงตาข้างซ้าย อาจมีอาการ {' '.join(set(left_eye_translated))}"
        else:
            description = f"เบื้องต้นพบว่าค่าสายตาทั้งสองข้างอยู่ในระดับ{risk_level}แต่ยังไม่สามารถระบุผลลัพธ์จากการสแกนได้"
        
        result = {
            "right_eye": encode_image_to_base64(original_right_path),
            "left_eye": encode_image_to_base64(original_left_path),
            "ai_left_image_base64": encode_image_to_base64(predicted_left_path),
            "ai_right_image_base64": encode_image_to_base64(predicted_right_path),
            "user_id": user_id,
            "description": description,
            "va_right": va_right,
            "va_left": va_left,
            "near_description": near_description,
            "pic_description": pic_description,
            "pic_right_description": right_eye_description,
            "pic_left_description": left_eye_description,
            "line_right": line_right,
            "line_left": line_left,
        }
        
        # POST back to http://localhost:3000/api/savescanlog
        # async with httpx.AsyncClient() as server:
        #     response = await server.post(
        #         "http://localhost:3000/api/savescanlog",
        #         json=result
        #         # content=multipart_data.to_string(),
        #         # headers=headers
        #         )

        # # Verify that the Node.js server is responding correctly
        # if response.status_code != 200:
        #     raise HTTPException(status_code=500, detail="Failed to send data to Node.js API")

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing eye scan: {str(e)}")

# Serve processed images
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
