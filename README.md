## Overview

This project is a development grade FastAPI for the Vehicle Damage Detection Assignment

## Project Structure
- **app/app.py**: Contains the FastAPI application and endpoints for vehicle damage detection
- **app/models/**: Directory housing YOLOv5 model files (`best-s.pt`, `best-m.pt`, `best-x.pt`) and training results for the medium model

## Installation

### Steps

1. **Clone the Repository**

2. **Create a Virtual Environment**

    It's recommended to use a virtual environment to manage project dependencies.

    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

    - **Windows**:

        ```bash
        venv\Scripts\activate
        ```

    - **MacOS/Linux** TODO

4. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the FastAPI server, follow these steps:

1. **Run the Uvicorn Server**

    Execute the following command from the project root directory:

    ```bash
    uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
    ```

## API Documentation

FastAPI provides interactive API documentation accessible via:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

These interfaces allow you to explore and interact with the API endpoints directly from your browser.

## API Endpoints

### Root Endpoint

### Predict Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Upload an image to perform vehicle damage detection.
- **Request**:

    - **Content-Type**: `multipart/form-data`
    - **Form Data**:
        - `file`: The image file to be uploaded.

- **Response**:

    - **Content-Type**: `application/json`
    - **Body**:

        ```json
        {
          "detections": [
            {
              "class_id": 0,
              "class_name": "minor-dent",
              "confidence": 0.95,
              "bbox": [100.0, 150.0, 200.0, 250.0]
            }
            // ... more detections
          ]
        }
        ```

    - **Fields**:
        - `detections`: List of detected objects.
            - `class_id`: Integer representing the class ID.
            - `class_name`: String representing the class name.
            - `confidence`: Float representing the confidence score.
            - `bbox`: List of four floats representing the bounding box coordinates `[x_min, y_min, x_max, y_max]`.

## Testing the API

### Using cURL

Replace `path_to_image.jpg` with the actual path to your image file.

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_image.jpg" -H "accept: application/json" -H "Content-Type: multipart/form-data"
