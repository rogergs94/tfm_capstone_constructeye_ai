import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from PIL import Image
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import httpx 
import datetime

## FUNCTIONS AND CLASSES ==================================================

# 1. PERSISTENT CLIENT (Outside functions)
ip_aws_address = "98.81.50.97"  # AWS IP address
local_host = "http://127.0.0.1:8000" # Localhost for testing

#backend_url = os.getenv("BACKEND_URL", f"http://{ip_address}:8000")
http_client = httpx.Client(base_url=local_host, timeout=120.0)



# SESSION STATES INITIALIZATION ===================================================
# --- MANDATORY SESSION STATE INITIALIZATION ---
# This must be at the very top to avoid AttributeErrors
if "last_detections" not in st.session_state:
    st.session_state.last_detections = []
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a unique session ID
current_session_id = st.session_state.session_id

# --- VIDEO SESSION STATE INITIALIZATION ---
if 'prediction_results' not in st.session_state:
    st.session_state['prediction_results'] = None
if 'original_image_np' not in st.session_state:
    st.session_state['original_image_np'] = None

# --- PROCESSING SESSION STATE INITIALIZATION ---
if "video_df" not in st.session_state:
    st.session_state.video_df = None
if "video_endpoint" not in st.session_state:
    st.session_state.video_endpoint = None
if "specs_json" not in st.session_state:
    st.session_state.specs_json = None
if "inference_status" not in st.session_state:
    st.session_state.inference_status = None

# --- LIVE SESSION STATE INITIALIZATION ---
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0



# STREAMLIT APP ==================================================
## --- STREAMLIT INTERFACE ---
st.title("Object Detection and Tracking App")
st.subheader("Created and developed by Roger González")
st.text("This model has been trained on a dataset of thousands of images with different classes (person, helmet, no-helmet, vest, no-vest). The model belongs to the YOLOv8n family.")

st.sidebar.markdown(f"**Session ID:** {current_session_id}")
st.sidebar.button("Reset Session ID", on_click=lambda: st.session_state.update({"session_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}))
# I create a button that selects local or s3 storage and then calls the endpoint to prepare the directory accordingly.
directory_button = st.sidebar.radio("Select storage type for outputs:", ("local", "s3"),\
                                    help="Choose 'local' to save outputs on the server's local storage, or 's3' to save them in an AWS S3 bucket. Make sure to configure the backend accordingly for S3 usage.")

# Seeing whether the directory has been created in the backend
# Call to endpoint prepare-output-dir to create the directory if it doesn't exist.
try:
    response = http_client.put(f"/prepare-output-dir?storage_type={directory_button}") # Change to 's3' if using S3 storage
    status = response.json()
    st.sidebar.success(status["message"])
except:    
    st.sidebar.error("Directory has not been found")


input_method = st.sidebar.radio("Choose input method:", ("Live Inference", "Upload image", "Upload video"))

st.sidebar.title("Model Configuration")
# GET endpoint to show model version and name
try:
    app_specs = http_client.get(f"/")                
    if app_specs.status_code == 200:
        specs_data = app_specs.json()
        st.sidebar.markdown(f'*{specs_data["model"]} v{specs_data["version"]}*')
except:
    st.sidebar.error("Model API is offline.")


conf_level = st.sidebar.slider("Confidence", 0.0, 1.0, 0.15)
iou_level = st.sidebar.slider("IoU", 0.0, 1.0, 0.5)
imgsz = st.sidebar.selectbox("Image Size", [160, 320, 480, 640, 800, 960, 1280, 1440, 1600, 1760], index=2)
max_det = st.sidebar.number_input("Max Detections", min_value=1, max_value=5000, value=1000, step=1)
gpu_usage = st.sidebar.selectbox(label="GPU Usage", options=[None, 0, 'mps', 'cpu'], help="None for default, 0 for NVIDIA, mps for Mac GPU, cpu for Intel/AMD")

st.markdown("---")

## IMAGE INFERENCE ------------------------------------
if input_method == "Upload image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # FIX: Added check to prevent 'NoneType' errors when no file is uploaded
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            # Read and show image
            st.image(uploaded_file, caption="Original Image")
            
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.asarray(image)
            
            _, img_encoded = cv2.imencode('.jpg', image_np)
            files = {'image': (uploaded_file.name, img_encoded.tobytes(), 'image/jpeg')}

        with col3:
            # Endpoint /image-specs -----------------
            response_image_specs = http_client.post(f"/image-specs", files=files)
            if response_image_specs.status_code == 200:
                specs = response_image_specs.json()
                specs_df = pd.DataFrame.from_dict(specs, orient='index', columns=['Value'])    
                st.write("Image Specs:", specs_df)
            else:
                st.error("Error retrieving image specs.")

        if st.button("Run Inference"):
            with st.spinner("Initializing image processing..."):
                try:
                    # Endpoint /predict_image -----------------
                    response = http_client.post(f"/predict_image?conf={conf_level}&iou={iou_level}&imgsz={imgsz}&max_det={max_det}&gpu_usage={gpu_usage}", files=files)
                    if response.status_code == 200:
                        data_image_predicted = response.json()                
                        df_image = data_image_predicted['detections']
                        img_data = base64.b64decode(data_image_predicted['image_base64'])
                        
                        with col2:
                            st.image(img_data, caption="Predicted Image")
                        with col4:
                            st.write("Prediction Dataframe")
                            st.dataframe(df_image)
                        st.write(f"Instances found: {len(df_image)}")
                    else:
                        st.error("Error from API.")
                except Exception as e:
                    st.error(f"Processing error: {e}")

## VIDEO INFERENCE ------------------------------------
elif input_method == "Upload video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    
    col1, col2 = st.columns(2)

    if uploaded_file:
        
        
        # API_1: Get video specs -----------------
        files = {"video": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
        response_video_specs = http_client.post(f"/video-feed-df-specs", files=files)
        if response_video_specs.status_code == 200:
            st.session_state.specs_json = response_video_specs.json()
            specs_df = pd.DataFrame.from_dict(st.session_state.specs_json, orient='index', columns=['Value'])
            st.write("Video Specs:", specs_df)
        
        with col1:
            st.write("Original Video")
            st.video(uploaded_file, format="video/mp4", muted=True)

        frame_stride = st.number_input("Frame stride", min_value=1, max_value=100, value=3, step=1)
        head_region = st.slider("Head Region Ratio", 0.0, 1.0, 0.35, step=0.05, 
                                help="Defines the upper part of the person bounding box considered as 'head' for helmet detection. For example, 0.35 means the top 35% of the bounding box height is the head region.")
        lower_part_ratio = st.slider("Lower Part Ratio", 0.0, 1.0, 0.2, step=0.05, 
                                help="Defines the lower boundary of the torso region for vest detection. For example, 0.2 means the torso region starts at 20% of the bounding box height from the top.")
        higher_part_ratio = st.slider("Higher Part Ratio", 0.0, 1.0, 0.8, step=0.05, 
                                help="Defines the upper boundary of the torso region for vest detection. For example, 0.8 means the torso region ends at 80% of the bounding box height from the top.")
        
        if st.button("Run Inference"):        
            with st.spinner(f"Processing inference..."):            
                # API_2: Process video and get results -----------------
                files = {"video": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
                url = f"/process-video_df?conf={conf_level}&iou={iou_level}&imgsz={imgsz}&frame_stride={frame_stride}&gpu_usage={gpu_usage}&head_ratio={head_region}&lower_part_ratio={lower_part_ratio}&higher_part_ratio={higher_part_ratio}"
                response_video_df = http_client.post(url, files=files)
                
                if response_video_df.status_code == 200:                     
                    return_json = response_video_df.json()
                    st.session_state.inference_status = return_json["status"]
                    st.session_state.video_endpoint = return_json["video_url"].split("/")[-1] 
                    st.session_state.video_df = pd.DataFrame(return_json["detections"])
                else:
                    st.error(f"Inference error: {response_video_df.text}")

    # --- PERSISTENT RESULTS RENDERING ---
    if st.session_state.video_df is not None:
        st.success(st.session_state.inference_status)
        v_df = st.session_state.video_df
        v_endpoint = st.session_state.video_endpoint

        with col2:
            # GET /show-video/{video_df} to play the processed video
            video_watch = http_client.get(f"/show-video/{v_endpoint}")                
            if video_watch.status_code == 200:
                st.write("Predicted Video")
                st.video(video_watch.content, autoplay=True, loop=True)

        # DOWNLOAD TOOL
        response = http_client.get(f"/download-video/{v_endpoint}")
        if response.status_code == 200:
            st.download_button(label="Download Video", data=response.content, file_name=v_endpoint, mime="video/mp4")

        # ANALYTICS GRAPH
        inf_by_frame = v_df.groupby("Frame")["Violation"].sum().sort_index().reset_index()
        fig = px.line(inf_by_frame, x="Frame", y="Violation", markers=True, title="Violations per Frame")
        fig.update_layout(xaxis_title="Frame", yaxis_title="Number of Violations", height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.info("The graph above shows the number of PPE violations detected in each frame. Spikes indicate frames with more violations, which can help identify critical moments in the video for further review.")

        st.dataframe(v_df)

        # FRAME INSPECTION TOOL
        st.divider()
        st.subheader("Frame Inspection")
        max_frames = st.session_state.specs_json["total_frames"]
        n_frame = st.number_input("Select Frame:", min_value=1, max_value=max_frames, value=1)
        
        # API_3: Visualize specific frame -----------------
        frame_data = v_df[v_df["Frame"] == n_frame].to_dict(orient="records")
        url_viz = f"/visualize-frame/{v_endpoint}/{n_frame}"
        response_viz = http_client.post(url_viz, json=frame_data)

        if response_viz.status_code == 200:
            st.image(response_viz.content, caption=f"Frame {n_frame} Analysis", use_container_width=True)

## LIVE INFERENCE ------------------------------------
elif input_method == "Live Inference":

    head_region = st.slider("Head Region Ratio", 0.0, 1.0, 0.35, step=0.05, 
                                help="Defines the upper part of the person bounding box considered as 'head' for helmet detection. For example, 0.35 means the top 35% of the bounding box height is the head region.")
    lower_part_ratio = st.slider("Lower Part Ratio", 0.0, 1.0, 0.2, step=0.05, 
                                help="Defines the lower boundary of the torso region for vest detection. For example, 0.2 means the torso region starts at 20% of the bounding box height from the top.")
    higher_part_ratio = st.slider("Higher Part Ratio", 0.0, 1.0, 0.8, step=0.05, 
                                help="Defines the upper boundary of the torso region for vest detection. For example, 0.8 means the torso region ends at 80% of the bounding box height from the top.")


    # 2. CENTRALIZED DRAWING FUNCTIONS ==================================================
    def draw_detections(img, detections):
        """Draws all instances detected by the model (raw)"""
        for det in detections:
            # Extract coordinates
            x1, y1, x2, y2 = [int(coord) for coord in det['box']]
            label = det.get('label', 'obj')
            conf = det.get('confidence', 0.0)
            
            # Default to gray if color is missing
            color = det.get('color', (128, 128, 128))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Encode frame
        _, img_encoded = cv2.imencode('.jpg', img)
        files = {'video': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        try:
            # 2. Backend API call (AWS IP)
            response = http_client.post(
                f"/predict-video_live?conf={conf_level}&iou={iou_level}&imgsz={imgsz}&gpu_usage={gpu_usage}&head_ratio={head_region}&lower_part_ratio={lower_part_ratio}&higher_part_ratio={higher_part_ratio}&session_id={current_session_id}", 
                files=files)
            
            if response.status_code == 200:
                res_json = response.json()
                # 3. DRAW: Use raw detections coming from AWS
                raw_dets = res_json.get("raw_detections", [])
                if raw_dets:
                    img = draw_detections(img, raw_dets)
                
        except Exception as e:
            # Use standard print as st.error doesn't work inside threads
            print(f"Error in live inference (AWS): {e}")
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")



    st.subheader("Live Inference (Real-time PPE Detection)")
    webrtc_streamer(
        key="tfm-live",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}
    )

    
# FOOTER ------------------------------------
st.markdown("---")
st.markdown("**Follow me:**")
github_col, linkedin_col, _ = st.columns([1, 1, 10])

with github_col:
    st.markdown('<a href="https://github.com/rogergs94" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="32"></a>', unsafe_allow_html=True)
with linkedin_col:
    st.markdown('<a href="https://www.linkedin.com/in/rogergonzalezsanchez/" target="_blank"><img src="https://images.seeklogo.com/logo-png/38/2/linkedin-black-icon-logo-png_seeklogo-387472.png" width="32"></a>', unsafe_allow_html=True)