# MAIN.PY: BACKEND API WITH FASTAPI AND S3 PERSISTENCE
# IMPORTS ================================================================================
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, BackgroundTasks, Body, APIRouter
from fastapi.responses import Response, JSONResponse, StreamingResponse, FileResponse
import subprocess
import base64
import cv2
import boto3
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from my_functions import head_region, torso_region, bbox_intersects, analyze_video_dataframe
import datetime
from typing import List
import tempfile
import json
import shutil
import uuid 
import io
import os
from botocore.exceptions import ClientError


# --- S3 CONFIGURATION FOR PERSISTENCE ---
# Initialize S3 client. AWS will automatically use the IAM Task Role in ECS.
s3_client = boto3.client('s3')
BUCKET_NAME = "tfm-s3" #First, create the bucket in AWS S3 and set the name here

app = FastAPI(title="PPE Detection API - Capstone Project", version="1.0.0")

# Directory for processed video results
OUTPUT_DIR = "processed"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model
model_path = "best_YOLOv8n_TFM_Yolo11s_Dec22_con_Data_Aug.pt"
model = YOLO(model_path)

# FUNCTIONS ================================================================================================
# It ensures that the S3 key is consistent for all video uploads, making it easier to manage files in the bucket
def get_s3_key(filename: str):
    """Standardizes the S3 path for processed videos"""
    return f"processed/{filename}"

# It converts the received bytes from the image to a format that OpenCV can process (BGR) and then converts it to RGB, 
# which is the standard format for YOLO training and inference.
def read_image(file_bytes):
    """Converts received bytes to OpenCV format (BGR) and then to RGB"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert BGR to RGB (YOLO training standard)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

video_writer = None
current_video_id = None

# It initializes a VideoWriter for live recording with a unique timestamp, 
# ensuring that each live session is saved as a separate file without overwriting previous recordings.
def get_video_writer(width, height, fps=20):
    """Initializes a VideoWriter for live recording with unique timestamp"""
    global video_writer, current_video_id
    if video_writer is None:
        current_video_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"live_record_{current_video_id}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return video_writer


## ROOT ====================================================================================================
@app.get("/")
async def root():
    message = {
        "message": "Model YOLOv8n Trained for Object Detection - TFM",
        "version": "1.0.0",
        "model": "YOLOv8n"
    }
    return message

# Endpoint that creates the output directory if it doesn't exist in a S3 bucket or locally,
# ensuring that the application can save processed videos without errors related to missing directories.
@app.put("/prepare-output-dir")
async def prepare_output_dir(storage_type: str = "local"):
    """Ensures the output directory exists for saving processed videos"""
    try:
        if storage_type == "local":
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            return {
                "status": "success", 
                "storage_type": "local",
                "message": f"Local output directory '{OUTPUT_DIR}' is ready."
                }
        elif storage_type == "s3":
            try:
            # For S3, it won't create directories but we can check if we have access to the bucket
                s3_client.head_bucket(Bucket=BUCKET_NAME)
                return {
                    "status": "success", 
                    "storage_type": "s3",
                    "message": f"S3 bucket '{BUCKET_NAME}' is accessible and ready for use."
                    }
            except ClientError as e:
                error_code = e.response['Error']['Code']
                # if the error is 404, it meanse the bucket doesn't exist. 
                # If it's 403, it means it doesn't have permissions to access it.
                if error_code == '404':
                    raise HTTPException(status_code=404, detail=f"S3 Bucket '{BUCKET_NAME}' not found: {str(e)}")
                elif error_code == '403':
                    raise HTTPException(status_code=403, detail=f"Access denied to S3 Bucket '{BUCKET_NAME}': {str(e)}")
                else:
                    raise HTTPException(status_code=500, detail=f"Error accessing S3 Bucket '{BUCKET_NAME}': {str(e)}")
        else: 
            raise HTTPException(status_code=400, detail=f"Invalid storage type specified: {storage_type}. Use 'local' or 's3'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing output directory: {str(e)}")
        


@app.get("/health")
async def health_check():
    return {"status": "ok"}


## IMAGENES ================================================================================================
# Endpoint with metadata (image)
@app.post("/image-specs")
async def image_specs(image: UploadFile = File(...)):
    """Extracts metadata from the uploaded image"""
    contents = await image.read()
    img = read_image(contents)
    height, width, channels = img.shape
    file_size_bytes = len(contents)
    file_size_kb = round(file_size_bytes / 1024, 2)
    file_size_mb = round((file_size_bytes / (1024 * 1024)), 2)

    return JSONResponse(content={
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_kb": file_size_kb,
        "file_size_mb": file_size_mb
    })

# Endpoint /predict-image-all
# it returns the image with base64 + dataframe
# Without Base64, we would need 2 endpoints (one for image and another for df)
@app.post("/predict_image")
async def predict_image(image: UploadFile = File(...), conf: float = 0.15, iou: float = 0.50, max_det: int = 1000, imgsz: int = 640, gpu_usage: str = 'cpu'):
    # 1. Read and process image
    contents = await image.read()
    img = read_image(contents)

    # CPU/GPU
    device_to_use = gpu_usage
    if gpu_usage == "None" or gpu_usage is None:
        device_to_use = None  # Auto-detection YOLO 
    elif gpu_usage == "0":
        device_to_use = 0 
    elif gpu_usage.lower() == "cpu":
        device_to_use = "cpu"           
    elif gpu_usage.lower() == "mps":
        device_to_use = "mps"

    # 3. Inference
    results = model.predict(
        img, 
        conf=conf, 
        iou=iou, 
        max_det=max_det, 
        imgsz=imgsz, 
        device=device_to_use)
    
    # 4. Detection processing (JSON)
    detections = []
    result = results[0]
    nombres = result.names
    
    # Use of properties
    for box in result.boxes:
        coords = box.xyxy[0].cpu().numpy()
        detections.append({
            "class": nombres[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "xmin": float(coords[0]),
            "ymin": float(coords[1]),
            "xmax": float(coords[2]),
            "ymax": float(coords[3]),
            "width": float(coords[2] - coords[0]),
            "height": float(coords[3] - coords[1])
        })

    # 5. Base64: Annotated image processing
    # imenconde needs BGR. results[0].plot() is in BGR
    annotated_img_bgr = results[0].plot() 
    _, buffer = cv2.imencode(".png", annotated_img_bgr)
    
    # Coinverting bytes from an image to a Base64 string
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # 6. Response
    return {
        "status": "success",
        "count": len(detections),
        "detections": detections,
        "image_base64": img_base64
    }


## VIDEOS ================================================================================================
# Endpoint para saber el metadata del video subido por el usuario
@app.post("/video-feed-df-specs") 
async def video_feed_specs(video: UploadFile = File(...)):
    # 1. Save the video in a temp file
    """Retrieves metadata from the uploaded video file"""
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(await video.read())
    temp_input.close()

    # 2. Debugging 1
    cap = cv2.VideoCapture(temp_input.name)
    if not cap.isOpened():
        os.unlink(temp_input.name)
        return JSONResponse(status_code=500, content={"error": "Impossible to open video codec"})

    # Metadata video
    # Getting the video properties to create the output video with same size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Size of the file
    file_size_mb = round(os.path.getsize(temp_input.name) / (1024 * 1024), 2)
    duration_seconds = total_frames / fps if fps > 0 else 0
        
    return JSONResponse(content={
        "video_width": width,
        "video_height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration_seconds, 2),
        "file_size_mb": file_size_mb
    })

# Endpoint to inference a video
# It returns a JSON with multiple elements: success message, detections according business logic,
# output_filename for future API references
@app.post("/process-video_df")
async def process_video_df(
    video: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.50,
    imgsz: int = 640,
    frame_stride: int = 3,
    gpu_usage: str = "cpu",
    head_ratio: float = 0.35,
    lower_part_ratio: float = 0.2,
    higher_part_ratio: float = 0.8
):
    """Processes video, saves it locally, and UPLOADS it to S3 for persistence"""
    suffix = os.path.splitext(video.filename)[1]
    t_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(video.file, t_input)
        t_input.close() 
        input_path = t_input.name

        output_filename = f"pred_{video.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Device selection logic
        device_to_use = gpu_usage
        if gpu_usage == "None" or gpu_usage is None:
            device_to_use = None
        elif gpu_usage == "0":
            device_to_use = 0
        elif gpu_usage.lower() == "cpu":
            device_to_use = "cpu"           

        # Start Video Inference
        results_df, classes = analyze_video_dataframe(
            video=input_path,
            my_model=model, 
            conf=conf, 
            iou=iou, 
            imgsz=imgsz,
            frame_stride=frame_stride,
            output_path=output_path,
            device=device_to_use,
            head_ratio=head_ratio, 
            lower_part_ratio=lower_part_ratio, 
            higher_part_ratio=higher_part_ratio
        )
        
        # --- PERSISTENCE LOGIC (UPLOAD TO S3) ---
        # After video is written locally, upload to S3 bucket
        try:
            s3_key = get_s3_key(output_filename) # Use the helper function
            s3_client.upload_file(output_path, BUCKET_NAME, s3_key)
            print(f"DEBUG: Successfully uploaded {s3_key} to S3")
        except Exception as s3_err:
            print(f"WARNING: S3 Upload failed: {s3_err}")
        
        # DataFrame post-processing
        expected_cols = ['Helmet', 'Vest']
        for col in expected_cols:
            if col not in results_df.columns:
                results_df[col] = 0

        results_df['wo_Helmet'] = 1 - results_df['Helmet']
        results_df['wo_Vest'] = 1 - results_df['Vest']

        df_as_json = results_df.to_json(orient="records")
        detections_cleaned = json.loads(df_as_json)


        # Save Dataframe in th directory with the same name as the video (but .json) and upload to S3
        df_output_path = output_path.replace('.mp4', '.json')
        results_df_json = results_df.to_json(df_output_path, orient="records", indent=4)
        
        try:
            s3_key_df = get_s3_key(os.path.basename(df_output_path))
            s3_client.upload_file(df_output_path, BUCKET_NAME, s3_key_df)
            print(f"DEBUG: Successfully uploaded {s3_key_df} to S3")
        except Exception as s3_err:
            print(f"WARNING: S3 Upload of DataFrame failed: {s3_err}")

        print (f"Video saved in {output_path} and uploaded to S3 with key {s3_key}")
        print(f"Dataframe saved in: {output_path.replace('.mp4', '.json')}")
        
        return {
            "status": "success",
            "detections": detections_cleaned,
            "video_url": output_filename
        }

    except Exception as e:
        print(f"Error detected: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

# Endpoint that downloads the video inferenced by the previous endpoint
@app.get("/download-video/{filename}")
async def download_video(filename: str):
    """Downloads the video file (checks local storage first, then S3)"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    # Download from S3 if missing locally
    if not os.path.exists(file_path):
        try:
            s3_key = get_s3_key(filename)
            s3_client.download_file(BUCKET_NAME, s3_key, file_path)
        except:
            raise HTTPException(status_code=404, detail="File missing in S3")

    return FileResponse(path=file_path, filename=filename, media_type='video/mp4')

# Endpoint that shows a video 
@app.get("/show-video/{video}")
async def show_video(video: str):
    """Streams video for playback (downloads from S3 if necessary)"""
    video_path = os.path.join(OUTPUT_DIR, video)
    
    # --- PERSISTENCE LOGIC (DOWNLOAD FROM S3) ---
    # If the video is not found locally, it tries to download it from S3. If it's not in S3 either, it returns a 404 error.
    if not os.path.exists(video_path):
        try:
            s3_key = get_s3_key(video)
            s3_client.download_file(BUCKET_NAME, s3_key, video_path)
        except:
            raise HTTPException(status_code=404, detail="File missing in S3")

    return FileResponse(video_path, media_type="video/mp4")

# Endpoint that shows a specific frame in a video
# Used to take a frame and see where the model failed
@app.post("/visualize-frame/{video_name}/{n_frame}")
async def get_frame_viz(video_name: str, n_frame: int, detections: List[dict] = Body(...)):
    """Extracts a specific frame from a video (downloads from S3 if necessary)"""
    video_path = os.path.join(OUTPUT_DIR, video_name)
    
    # --- PERSISTENCE LOGIC (DOWNLOAD FROM S3) ---
    if not os.path.exists(video_path):
        try:
            # Debugging: Print the S3 key being accessed to ensure it's correct
            s3_key = get_s3_key(video_name)
            print(f"DEBUG: Downloading {s3_key} from bucket {BUCKET_NAME} to {video_path}")
            
            # Attempt to download the video from S3. If it fails, 
            # it will raise an exception that we catch to return a 404 error.
            s3_client.download_file(BUCKET_NAME, s3_key, video_path)
        except Exception as e:
            print(f"ERROR: S3 Download failed: {e}")
            raise HTTPException(status_code=404, detail=f"Video missing in S3: {e}")

    # It uses OpenCV to read the video and extract the specified frame.
    # If the frame cannot be read, it returns a 404 error.
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=404, detail="Could not read frame from video")

    # Drawing Logic for Bounding Boxes
    for det in detections:
        # It extracts bbox
        bbox = det['Person_Box']
        if isinstance(bbox, str): 
            import ast
            bbox = ast.literal_eval(bbox)
        
        x1, y1, x2, y2 = map(int, bbox)
        # Red if Violation (1), Green otherwise (0)
        color = (0, 0, 255) if det.get('Violation') == 1 else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"ID: {det['Person_ID']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # It converts the image to a format compatible with the browser (JPEG)
    _, img_encoded = cv2.imencode('.jpg', frame)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")


## LIVE INFERENCE ==========================================================================================
# Endpoint that shows predictions in live videos
@app.post("/predict-video_live")
async def predict_single_frame(
            video: UploadFile = File(...), 
            session_id: str = Query(...), # To identify the live session and save the video with a unique name
            conf: float = 0.25, 
            iou: float = 0.50, 
            imgsz: int = 640, 
            gpu_usage: str = "cpu", 
            save_video: bool = Query(False),
            head_ratio: float = 0.35,
            lower_part_ratio: float = 0.2,
            higher_part_ratio: float = 0.8
            ):
    """Performs real-time inference on a single frame sent from Streamlit"""
    # 1. It reads the frame sent from Streamlit
    contents = await video.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    device_to_use = gpu_usage
    if gpu_usage == "None" or gpu_usage is None:
        device_to_use = None
    elif gpu_usage == "0":
        device_to_use = 0
    elif gpu_usage.lower() == "cpu":
        device_to_use = "cpu"           

    # 2. Inference model
    results = model.track(frame, persist=True, tracker="botsort.yaml", conf=conf, iou=iou, imgsz=imgsz, device=device_to_use, verbose=False)        
    
    # 3. It extracts data from the model itself
    raw_detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        raw_detections.append({
            "box": box.xyxy[0].tolist(),
            "label": model.names[cls_id],
            "confidence": float(box.conf[0]),
            "color": colors(cls_id, bgr=True) 
        })

    annotated_frame = results[0].plot()

    if save_video:
        h, w, _ = annotated_frame.shape
        writer = get_video_writer(w, h)
        writer.write(annotated_frame)

    # Business logic for the dataframe
    business_logic = []
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clases = results[0].boxes.cls.cpu().numpy().astype(int)
        scores = results[0].boxes.conf.cpu().numpy()
        names = model.names

        # Creating dict/list to save bboxes, score...
        # For people, it will save bbox, score indexed by ID
        # Filter entities by class
        frame_persons = {ids[i]: (boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'person'}
        
        # For PPE Equipment, it will use tuples --> (bbox, score)
        frame_helmets = [(boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'helmet']
        frame_vests = [(boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'vest']
        frame_no_helmets = [(boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'no-helmet']
        frame_no_vests = [(boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'no-vest']

        for p_id, (p_bbox, p_score) in frame_persons.items():
            h_zone = head_region(p_bbox, head_ratio=head_ratio)
            t_zone = torso_region(p_bbox, lower_part_ratio=lower_part_ratio, higher_part_ratio=higher_part_ratio)
            
            # Intersection logic for PPE detection
            # helments / no helmets
            found_h = next(((hb.tolist(), hs) for hb, hs in frame_helmets if bbox_intersects(h_zone, hb)), (None, 0.0))
            found_nh = next(((nhb.tolist(), nhs) for nhb, nhs in frame_no_helmets if bbox_intersects(h_zone, nhb)), (None, 0.0))
            
            # vests / no vests
            found_v = next(((vb.tolist(), vs) for vb, vs in frame_vests if bbox_intersects(t_zone, vb)), (None, 0.0))
            found_nv = next(((nvb.tolist(), nvs) for nvb, nvs in frame_no_vests if bbox_intersects(t_zone, nvb)), (None, 0.0))

            # 1 if PPE, 0 if not
            has_helmet = 1 if found_h[0] else 0
            has_vest = 1 if found_v[0] else 0

            det_no_helmet = 1 if found_nh[0] else 0
            det_no_vest = 1 if found_nv[0] else 0

            business_logic.append({
                "Person_ID": int(p_id),
                "Person_Conf": round(float(p_score), 3),
                "Person_Box": p_bbox.tolist(),
                
                "Helmet_Detected": has_helmet,
                "Helmet_Conf": round(float(found_h[1]), 3) if has_helmet else 0.0,
                "Helmet_Box": found_h[0], # Helmet coordinates
                "No_Helmet_Detected": det_no_helmet,
                "No_Helmet_Conf": round(float(found_nh[1]), 3) if det_no_helmet else 0.0,
                "No_Helmet_Box": found_nh[0], # No helmet coordinates
                
                "Vest_Detected": has_vest,
                "Vest_Conf": round(float(found_v[1]), 3) if has_vest else 0.0,
                "Vest_Box": found_v[0],
                "No_Vest_Detected": det_no_vest,
                "No_Vest_Conf": round(float(found_nv[1]), 3) if det_no_vest else 0.0,
                "No_Vest_Box": found_nv[0],
                
                "Violation": 1 if (has_helmet == 0 or has_vest == 0) else 0,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            })

    # Save live detections to log file
    if business_logic:
        # It creates a file with the name live_inference_log + session_id
        # Each live session will have a different file.
        session_log_file = os.path.join(OUTPUT_DIR, f"live_inference_{session_id}.csv")

        df_to_save = pd.DataFrame(business_logic)
        # it creates a timestamp column to know when the detection was made
        df_to_save['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if the file already exists
        file_exists = os.path.isfile(session_log_file)
        
        # Save the df to a CSV file. 
        # If the file already exists, it appends the new data without writing the header again.
        df_to_save.to_csv(session_log_file, mode='a', index=False, header=not file_exists)
        
        print(f"Live detections saved to {session_log_file}")

        # # If the file doesn't exist, it will create it
        # # If it exists, it will add the results at the end (mode = )
        # file_exists = os.path.isfile(LIVE_LOG_FILE)

        # # It saves the dataframe in a CSV file. If the file already exists, it appends the new data without writing the header again.
        # df_to_save.to_csv(LIVE_LOG_FILE, mode='a', index=False, header=not file_exists)

    return {
        "raw_detections": raw_detections,
        "business_logic": business_logic
    }
