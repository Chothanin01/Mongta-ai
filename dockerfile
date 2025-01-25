# Use an official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the application with Uvicorn
CMD ["uvicorn", "ai_api:app", "--host", "0.0.0.0", "--port", "8000"]
