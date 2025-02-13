from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from roboflow import Roboflow
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import os
from PIL import Image
import matplotlib

matplotlib.use('Agg')  # Use the Agg backend which does not require a display.

app = FastAPI()

# Allowed file extensions and MIME types for image uploads
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MIME_TYPES = {"jpg": "image/jpg", "jpeg": "image/jpeg", "png": "image/png"}

# Initialize Roboflow model
rf = Roboflow(api_key="mJRJtBYRhInoDZPAzrv3")
project = rf.workspace("mongta-swaxu").project("mongta")
model = project.version("6").model

def allowed_file(filename: str) -> bool:
    """ Check if the uploaded file has a valid image extension. """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image(image: Image.Image) -> Image.Image:
    """ Resize the image to 640x640 if it is larger. """
    width, height = image.size
    if width < 640 or height < 640:
        return image
    return image.resize((640, 640))

def enhance_image(image: np.ndarray) -> np.ndarray:
    """ Apply enhancement techniques to improve image clarity. """
    enhanced = cv2.detailEnhance(image, sigma_s=1, sigma_r=0.15)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return sharpened

def detect_eyes(image: Image.Image) -> Image.Image:
    """ Detect eyes in the image and return an enhanced zoomed-in region if detected. """
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Resize and convert the image for processing
    img_resized = image.resize((640, 640))
    img_resized = np.array(img_resized)

    # Convert to grayscale for better eye detection
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If at least one eye is detected, process it
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        zoomed_eye = img_resized[y:y+h, x:x+w]
        zoomed_eye_resized = cv2.resize(zoomed_eye, (640, 640))
        zoomed_eye_enhanced = enhance_image(zoomed_eye_resized)
        return Image.fromarray(zoomed_eye_enhanced)

    # If no eyes detected, return the original image
    return image

def draw_boxes(image: Image.Image, predictions: dict) -> Image.Image:
    """ Draw bounding boxes around detected objects in the image using Matplotlib. """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for pred in predictions['predictions']:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        x1, y1, x2, y2 = x - width / 2, y - height / 2, x + width / 2, y + height / 2

        # Draw bounding box
        rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Label with class name and confidence score
        ax.text(x1, y1 - 10, f"{pred['class']} ({pred['confidence']:.2f})",
                color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

    ax.axis("off")

    # Save to buffer and convert back to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """ Process an uploaded image, detect objects, and return the modified image. """
    
    # Validate file type
    if not allowed_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .jpg, .jpeg, and .png are allowed.")
    
    image_extension = image.filename.rsplit('.', 1)[-1].lower()
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resized_image = resize_image(image_pil)

    # Determine the appropriate format for saving
    pil_format = "JPEG" if image_extension in ["jpg", "jpeg"] else "PNG"
    temp_path = f"temp.{image_extension}"
    resized_image.save(temp_path, format=pil_format)

    # Perform object detection
    predictions = model.predict(temp_path, confidence=40, overlap=30).json()
    # print(predictions)

    # If no objects detected, try eye detection
    if not predictions['predictions']:
        eye_image = detect_eyes(resized_image)

        # Save detected eye region before running model again
        eye_image_path = os.path.join("images", f"eye_{image.filename}")
        os.makedirs("images", exist_ok=True)
        eye_image.save(eye_image_path, format=pil_format)

        # Run model on eye detection result
        eye_image.save(temp_path, format=pil_format)
        predictions = model.predict(temp_path, confidence=40, overlap=30).json()
        # print(predictions)

        # If still no predictions, return an error
        if not predictions['predictions']:
            os.remove(temp_path)
            raise HTTPException(status_code=404, detail="No predictions found.")

    # Use eye_image if eye detection was performed; otherwise, use resized_image
    final_image = eye_image if 'eye_image' in locals() else resized_image
    output_image = draw_boxes(final_image, predictions)

    # Convert RGBA to RGB to avoid transparency issues
    if output_image.mode == 'RGBA':
        output_image = output_image.convert('RGB')

    # Save final output image
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"predicted_{image.filename}")
    output_image.save(output_filename, format=pil_format)

    # Clean up temporary file
    os.remove(temp_path)

    # Return processed image as response
    return StreamingResponse(open(output_filename, "rb"), 
                             media_type=MIME_TYPES[image_extension], 
                             headers={"Content-Disposition": f"attachment; filename={output_filename}"})
