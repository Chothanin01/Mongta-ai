from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from roboflow import Roboflow
import cv2
import numpy as np
import io
import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import uvicorn

# Use the Agg backend for Matplotlib (no display required)
matplotlib.use('Agg')

app = FastAPI()

# Define directories for uploading and storing processed files
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MIME_TYPES = {"jpg": "image/jpg", "jpeg": "image/jpeg", "png": "image/png"}

# Initialize the Roboflow model
rf = Roboflow(api_key="mJRJtBYRhInoDZPAzrv3")
project = rf.workspace("mongta-swaxu").project("mongta")
model = project.version("3").model


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_image(image: Image.Image) -> Image.Image:
    """Resize the image to 640x640 if it is larger."""
    width, height = image.size
    if width < 640 or height < 640:
        return image
    return image.resize((640, 640))

def generate_ai_analysis(image_path: str, output_path: str):
    """
    Use the Roboflow model to generate an annotated image (AI Analysis).
    Calls model.predict (without save=True since it is not supported).
    Then, annotates the image using draw_boxes and saves it to output_path.
    """
    result = model.predict(image_path, confidence=40, overlap=30)
    predictions = result.json()
    # Load original image
    image = Image.open(image_path)
    annotated_image = draw_boxes(image, predictions)
    annotated_image.save(output_path)

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

def draw_boxes(image: Image.Image, predictions: dict) -> Image.Image:
    """Draw bounding boxes around detected objects in the image using Matplotlib."""
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


@app.post("/upload-eye-predict")
async def upload_eye_predict(
    right_eye: UploadFile = File(...),
    left_eye: UploadFile = File(...),
    user_id: str = Form(...),
    va_right: str = Form(...),
    va_left: str = Form(...),
    near_description: str = Form(""),
    line_right: str = Form(""),
    line_left: str = Form("")
):
    """
    Endpoint for processing right and left eye images:
      - Read image files.
      - Convert them to PIL images.
      - Resize and save temporary files for object detection.
      - Use the Roboflow model to detect objects in each eye image.
      - If no objects are detected, try detect_eyes for further analysis.
      - Draw bounding boxes and save predicted images.
      - Annotate images using the Roboflow model.
      - Return URLs of processed images and predictions as JSONResponse.
    """
    # Read the uploaded images and convert them to PIL format
    right_eye_bytes = await right_eye.read()
    left_eye_bytes = await left_eye.read()

    right_eye_pil = Image.open(io.BytesIO(right_eye_bytes)).convert("RGB")
    left_eye_pil = Image.open(io.BytesIO(left_eye_bytes)).convert("RGB")

    # Determine the file extension and format
    image_extension = right_eye.filename.rsplit('.', 1)[-1].lower()
    pil_format = "JPEG" if image_extension in ["jpg", "jpeg"] else "PNG"

    # Process the right eye image
    temp_right_path = os.path.join(OUTPUT_DIR, f"temp_right.{image_extension}")
    resized_right_image = resize_image(right_eye_pil)
    resized_right_image.save(temp_right_path, format=pil_format)
    predictions_right = model.predict(temp_right_path, confidence=40, overlap=30).json()
    if not predictions_right['predictions']:
        # If no predictions are detected, try using detect_eyes on the right eye image
        detected_right = detect_eyes(right_eye_pil)
        detected_right.save(temp_right_path, format=pil_format)
        predictions_right = model.predict(temp_right_path, confidence=40, overlap=30).json()
        if predictions_right['predictions']:
            final_right = detected_right
        else:
            raise HTTPException(status_code=404, detail="No predictions detected in right eye image.")
    else:
        final_right = resized_right_image

    predicted_right_filename = os.path.join(OUTPUT_DIR, f"predicted_{user_id}_right.{image_extension}")
    output_right_predicted = draw_boxes(final_right, predictions_right)
    if output_right_predicted.mode == 'RGBA':
        output_right_predicted = output_right_predicted.convert('RGB')
    output_right_predicted.save(predicted_right_filename, format=pil_format)

    # Process the left eye image
    temp_left_path = os.path.join(OUTPUT_DIR, f"temp_left.{image_extension}")
    resized_left_image = resize_image(left_eye_pil)
    resized_left_image.save(temp_left_path, format=pil_format)
    predictions_left = model.predict(temp_left_path, confidence=40, overlap=30).json()
    if not predictions_left['predictions']:
        # If no predictions are detected, try using detect_eyes on the left eye image
        detected_left = detect_eyes(left_eye_pil)
        detected_left.save(temp_left_path, format=pil_format)
        predictions_left = model.predict(temp_left_path, confidence=40, overlap=30).json()
        if predictions_left['predictions']:
            final_left = detected_left
        else:
            raise HTTPException(status_code=404, detail="No predictions detected in left eye image.")
    else:
        final_left = resized_left_image

    predicted_left_filename = os.path.join(OUTPUT_DIR, f"predicted_{user_id}_left.{image_extension}")
    output_left_predicted = draw_boxes(final_left, predictions_left)
    if output_left_predicted.mode == 'RGBA':
        output_left_predicted = output_left_predicted.convert('RGB')
    output_left_predicted.save(predicted_left_filename, format=pil_format)

    # Remove temporary files
    if os.path.exists(temp_right_path): os.remove(temp_right_path)
    if os.path.exists(temp_left_path): os.remove(temp_left_path)


    # Save the original and AI analysis images
    original_right_path = os.path.join(OUTPUT_DIR, f"{user_id}_right.png")
    original_left_path = os.path.join(OUTPUT_DIR, f"{user_id}_left.png")
    right_eye_pil.save(original_right_path, format="PNG")
    left_eye_pil.save(original_left_path, format="PNG")

    ai_right_path = os.path.join(OUTPUT_DIR, f"{user_id}_ai_right.png")
    ai_left_path = os.path.join(OUTPUT_DIR, f"{user_id}_ai_left.png")
    generate_ai_analysis(original_right_path, ai_right_path)
    generate_ai_analysis(original_left_path, ai_left_path)

    
    right_eye_classes = [pred['class'] for pred in predictions_right['predictions']]
    left_eye_classes = [pred['class'] for pred in predictions_left['predictions']]

    # Check if the eyes are normal or have abnormalities
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


    # check if the eyes are normal or have abnormalities
    return JSONResponse({
        "status": "success",
        "right_eye": f"http://127.0.0.1:8000/output/{user_id}_right.png",
        "left_eye": f"http://127.0.0.1:8000/output/{user_id}_left.png",
        "ai_right": f"http://127.0.0.1:8000/output/{os.path.basename(predicted_right_filename)}",
        "ai_left": f"http://127.0.0.1:8000/output/{os.path.basename(predicted_left_filename)}",
        "user_id": user_id,
        "description": "Eye test and object detection results",
        "va_right": va_right,
        "va_left": va_left,
        "near_description": near_description,
        "pic_description": pic_description,
        "line_right": line_right,
        "line_left": line_left,
    })

# Mount the output directory to serve the processed images
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
