from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from requests_toolbelt.multipart.encoder import MultipartEncoder
import base64


# ✅ Prevents Matplotlib from requiring a display
matplotlib.use('Agg')

app = FastAPI()

# Enable CORS to allow cross-origin requests from Node.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Allowed file types
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MIME_TYPES = {"jpg": "image/jpg", "jpeg": "image/jpeg", "png": "image/png"}

# Roboflow AI Model Setup
rf = Roboflow(api_key="mJRJtBYRhInoDZPAzrv3")
project = rf.workspace("mongta-swaxu").project("mongta")
model = project.version("3").model

class ScanResult(BaseModel):
    user_id: int
    description: str
    pic_description: str
    ai_right_image_url: str
    ai_left_image_url: str

def allowed_file(filename: str) -> bool:
    """✅ Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image: Image.Image) -> Image.Image:
    """✅ Resize the image to 640x640 if it is larger."""
    width, height = image.size
    if width > 640 or height > 640:
        return image.resize((640, 640))
    return image

def detect_eyes(image: Image.Image) -> Image.Image:
    """✅ Detect eyes and return a zoomed-in region if detected."""
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

def process_image(image_bytes: bytes) -> str:
    """ประมวลผลภาพโดยใช้ OpenCV และแปลงกลับเป็น Base64"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("ไม่สามารถอ่านไฟล์รูปภาพได้")

    # แปลงเป็นภาพขาวดำ (ตัวอย่าง)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # แปลงกลับไปเป็น Base64
    _, buffer = cv2.imencode(".png", gray_image)
    return base64.b64encode(buffer).decode("utf-8")

def draw_boxes(image: Image.Image, predictions: dict) -> Image.Image:
    """✅ Draw bounding boxes around detected objects in the image using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    for pred in predictions['predictions']:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        x1, y1, x2, y2 = x - width/2, y - height/2, x + width/2, y + height/2
        rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"{pred['class']}", color='red', fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

async def generate_ai_analysis(image_path: str, output_path: str):
    """✅ Runs Roboflow AI asynchronously and saves the analyzed image."""
    try:
        result = await asyncio.to_thread(model.predict, image_path, confidence=40, overlap=30)
        predictions = result.json()
        image = Image.open(image_path)
        annotated_image = draw_boxes(image, predictions)
        annotated_image.save(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in AI analysis: {str(e)}")

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
            raise HTTPException(status_code=400, detail="Missing required photo.")

        # ✅ Validate file extensions
        if not (allowed_file(right_eye.filename) and allowed_file(left_eye.filename)):
            raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, JPEG, and PNG are allowed.")

        # ✅ Read uploaded images into memory
        right_eye_bytes = await right_eye.read()
        left_eye_bytes = await left_eye.read()

        # right_eye_pil = Image.open(io.BytesIO(right_eye_bytes)).convert("RGB")
        # left_eye_pil = Image.open(io.BytesIO(left_eye_bytes)).convert("RGB")
        try:
            right_eye_pil = Image.open(io.BytesIO(right_eye_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file for right_eye: {str(e)}")

        try:
            left_eye_pil = Image.open(io.BytesIO(left_eye_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file for left_eye: {str(e)}")

        right_eye_path = os.path.join(UPLOAD_DIR, f"{user_id}_right.png")
        left_eye_path = os.path.join(UPLOAD_DIR, f"{user_id}_left.png")

        with open(right_eye_path, "wb") as f:
            f.write(await right_eye.read())

        with open(left_eye_path, "wb") as f:
            f.write(await left_eye.read())

        if not os.path.exists(right_eye_path) or not os.path.exists(left_eye_path):
            raise HTTPException(status_code=500, detail="File saving error. The images were not saved correctly.")

        # ✅ Determine file format
        image_extension = right_eye.filename.rsplit('.', 1)[-1].lower()
        pil_format = "JPEG" if image_extension in ["jpg", "jpeg"] else "PNG"

        # ✅ Resize images
        resized_right_image = resize_image(right_eye_pil)
        resized_left_image = resize_image(left_eye_pil)

        # ✅ AI Processing for Right Eye
        temp_right_path = os.path.join(OUTPUT_DIR, f"temp_right.{image_extension}")
        resized_right_image.save(temp_right_path, format=pil_format)
        predictions_right = model.predict(temp_right_path, confidence=40, overlap=30).json()

        if not predictions_right['predictions']:
            # If no AI detection, use `detect_eyes`
            detected_right = detect_eyes(right_eye_pil)
            detected_right.save(temp_right_path, format=pil_format)
            predictions_right = model.predict(temp_right_path, confidence=40, overlap=30).json()
            final_right = detected_right if predictions_right['predictions'] else resized_right_image
        else:
            final_right = resized_right_image

        # ✅ AI Processing for Left Eye
        temp_left_path = os.path.join(OUTPUT_DIR, f"temp_left.{image_extension}")
        resized_left_image.save(temp_left_path, format=pil_format)
        predictions_left = model.predict(temp_left_path, confidence=40, overlap=30).json()

        if not predictions_left['predictions']:
            detected_left = detect_eyes(left_eye_pil)
            detected_left.save(temp_left_path, format=pil_format)
            predictions_left = model.predict(temp_left_path, confidence=40, overlap=30).json()
            final_left = detected_left if predictions_left['predictions'] else resized_left_image
        else:
            final_left = resized_left_image

        # ✅ Draw AI Predictions
        predicted_right_filename = os.path.join(OUTPUT_DIR, f"predicted_{user_id}_right.{image_extension}")
        predicted_left_filename = os.path.join(OUTPUT_DIR, f"predicted_{user_id}_left.{image_extension}")

        output_right_predicted = draw_boxes(final_right, predictions_right)
        output_left_predicted = draw_boxes(final_left, predictions_left)

        output_right_predicted.save(predicted_right_filename, format=pil_format)
        output_left_predicted.save(predicted_left_filename, format=pil_format)

        # ✅ Remove temporary files
        if os.path.exists(temp_right_path): os.remove(temp_right_path)
        if os.path.exists(temp_left_path): os.remove(temp_left_path)

        # ✅ Save original and AI-processed images
        original_right_path = os.path.join(OUTPUT_DIR, f"{user_id}_right.png")
        original_left_path = os.path.join(OUTPUT_DIR, f"{user_id}_left.png")
        right_eye_pil.save(original_right_path, format="PNG")
        left_eye_pil.save(original_left_path, format="PNG")

        ai_right_path = os.path.join(OUTPUT_DIR, f"{user_id}_ai_right.png")
        ai_left_path = os.path.join(OUTPUT_DIR, f"{user_id}_ai_left.png")
        await generate_ai_analysis(original_right_path, ai_right_path)
        await generate_ai_analysis(original_left_path, ai_left_path)

        # ✅ Generate eye health descriptions
        right_eye_classes = [pred['class'] for pred in predictions_right['predictions']]
        left_eye_classes = [pred['class'] for pred in predictions_left['predictions']]

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
        
        # ✅ Generate eye health descriptions
        right_eye_classes = [pred.get('class', 'Unknown') for pred in predictions_right.get('predictions', [])]
        left_eye_classes = [pred.get('class', 'Unknown') for pred in predictions_left.get('predictions', [])]

        # ✅ Format descriptions for better clarity
        right_eye_description = "การสเเกนดวงตาขวายังไม่พบเจอสิ่งผิดปกติ" if all(cls == "Normal" for cls in right_eye_classes) else \
                                f"การสเเกนดวงตาขวาพบความผิดปกติ ({', '.join(set(right_eye_classes))})"
                                
        left_eye_description = "การสเเกนดวงตาซ้ายยังไม่พบเจอสิ่งผิดปกติ" if all(cls == "Normal" for cls in left_eye_classes) else \
                               f"การสเเกนดวงตาซ้ายพบความผิดปกติ ({', '.join(set(left_eye_classes))})"
        
        # ✅ ระดับความเสี่ยงของสายตา
        risk_levels = ["ปกติ", "เริ่มมีความผิดปกติ", "มีความผิดปกติ"]

        # ✅ ระดับความเสี่ยงของสายตา
        risk_levels = ["ปกติ", "เริ่มมีความผิดปกติ", "มีความผิดปกติ"]

        # ✅ ตรวจจับระดับความผิดปกติจาก `near_description`
        if "มีความผิดปกติ" in near_description:
            risk_level = risk_levels[2]  # มีความผิดปกติ
        elif "เริ่มมีความผิดปกติ" in near_description:
            risk_level = risk_levels[1]  # เริ่มมีความผิดปกติ
        else:
            risk_level = risk_levels[0]  # ปกติ

        # ✅ ตรวจสอบผลการสแกนดวงตา
        if all(cls == "Normal" for cls in right_eye_classes) and all(cls == "Normal" for cls in left_eye_classes):
            description = f"เบื้องต้นมีค่าสายตาทั้งซ้ายและขวา อยู่ในระดับ{risk_level}แต่การสแกนดวงตายังไม่พบสิ่งผิดปกติ"
        elif any(cls != "Normal" for cls in right_eye_classes) and any(cls != "Normal" for cls in left_eye_classes):
            description = f"เบื้องต้นมีค่าสายตาทั้งซ้ายและขวา อยู่ในระดับ{risk_level}พบความผิดปกติที่ดวงตาทั้งสองข้าง ({', '.join(set(right_eye_classes + left_eye_classes))})"
        elif any(cls != "Normal" for cls in right_eye_classes):
            description = f"เบื้องต้นมีค่าสายตาทั้งซ้ายและขวา อยู่ในระดับ{risk_level}พบความผิดปกติที่ดวงตาข้างขวา ({', '.join(set(right_eye_classes))})"
        elif any(cls != "Normal" for cls in left_eye_classes):
            description = f"เบื้องต้นมีค่าสายตาทั้งซ้ายและขวา อยู่ในระดับ{risk_level}พบความผิดปกติที่ดวงตาข้างซ้าย ({', '.join(set(left_eye_classes))})"
        else:
            description = f"เบื้องต้นมีค่าสายตาทั้งซ้ายและขวา อยู่ในระดับ{risk_level}แต่ยังไม่สามารถระบุผลลัพธ์ได้"

        # result = {
        #     "status": "success",
        #     "right_eye": f"http://127.0.0.1:8000/output/{user_id}_right.png",
        #     "left_eye": f"http://127.0.0.1:8000/output/{user_id}_left.png",
        #     "ai_right": f"http://127.0.0.1:8000/output/{os.path.basename(predicted_right_filename)}",
        #     "ai_left": f"http://127.0.0.1:8000/output/{os.path.basename(predicted_left_filename)}",
        #     "user_id": user_id,
        #     "description": description,
        #     "va_right": va_right,
        #     "va_left": va_left,
        #     "near_description": near_description,
        #     "pic_description": pic_description,
        #     "pic_right_description": right_eye_description,
        #     "pic_left_description": left_eye_description,
        #     "line_right": line_right,
        #     "line_left": line_left,
        # }
        # print("📤 กำลังส่งข้อมูลไปยัง Node.js API:")
        # print(json.dumps(result, indent=4, ensure_ascii=False))
                # ✅ สร้าง JSON Payload
        payload = {
            "user_id": str(user_id),
            "line_right": line_right,
            "line_left": line_left,
            "va_right": va_right,
            "va_left": va_left,
            "near_description": near_description,
        }

        # ✅ ใช้ MultipartEncoder เพื่อส่งไฟล์ไปยัง Node.js API
        multipart_data = MultipartEncoder(
            fields={
                "user_id": str(user_id),
                "line_right": line_right,
                "line_left": line_left,
                "va_right": va_right,
                "va_left": va_left,
                "near_description": near_description,
                "right_eye": (right_eye.filename, open(right_eye_path, "rb"), "image/png"),
                "left_eye": (left_eye.filename, open(left_eye_path, "rb"), "image/png"),
            }
        )

        headers = {"Content-Type": multipart_data.content_type}


        # ✅ **POST กลับไปที่ http://localhost:3000/api/savescanlog**
        async with httpx.AsyncClient() as client:
            # response = await client.post("http://localhost:3000/api/savescanlog", json=result)
            response  = await client.post(
                "http://localhost:3000/api/savescanlog", 
                content=multipart_data.to_string(), 
                headers=headers
            )
            response.raise_for_status()

        os.remove(right_eye_path)
        os.remove(left_eye_path)

        # ✅ ตรวจสอบว่าเซิร์ฟเวอร์ Node.js ตอบกลับถูกต้อง
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to send data to Node.js API")

        # return JSONResponse(content=result)

        return JSONResponse(content={"success": True, "message": "Scan log saved successfully."})

    except httpx.HTTPStatusError as e:
        print(f"🔥 HTTP Error: {e.response.text}")
        return JSONResponse(status_code=500, content={"error": e.response.text})

    except Exception as e:
        print(f"🔥 ERROR: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve processed images
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)