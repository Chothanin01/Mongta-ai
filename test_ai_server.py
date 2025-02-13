import pytest
from fastapi.testclient import TestClient
from main_3 import app
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which does not require a display.

client = TestClient(app)

# Test สำหรับการทำนายภาพที่มีไฟล์ถูกต้อง
@pytest.mark.parametrize("image_file, expected_status_code", [
    ("tests/images/valid_image.jpg", 200),  # ใช้ไฟล์ jpg ตัวอย่างที่ถูกต้อง
])
def test_predict_valid_image(image_file, expected_status_code):
    with open(image_file, "rb") as img:
        response = client.post("/predict", files={"image": (image_file, img, "image/jpeg")})

    assert response.status_code == expected_status_code

    if response.status_code == 200:
        # ตรวจสอบว่าผลลัพธ์มี "predictions"
        assert "predictions" in response.json()

# Test สำหรับไฟล์ที่ไม่รองรับ (เช่น .txt)
@pytest.mark.parametrize("image_file, expected_status_code", [
    ("tests/images/invalid_image.txt", 400),  # ไฟล์ที่ไม่ถูกต้อง
])
def test_predict_invalid_file(image_file, expected_status_code):
    with open(image_file, "rb") as img:
        response = client.post("/predict", files={"image": (image_file, img, "image/jpeg")})

    assert response.status_code == expected_status_code

# Test สำหรับกรณีที่ไม่พบการทำนาย
def test_predict_no_prediction():
    # ใช้ภาพที่ไม่สามารถทำการทำนายได้
    with open("tests/images/no_prediction_image.jpg", "rb") as img:
        response = client.post("/predict", files={"image": ("no_prediction_image.jpg", img, "image/jpeg")})

    assert response.status_code == 404
    assert response.json() == {"detail": "No predictions found."}
