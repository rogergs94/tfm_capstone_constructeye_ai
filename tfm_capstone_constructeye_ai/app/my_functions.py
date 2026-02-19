
import urllib.request
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ast

# Business logic functions to define head and torso regions, check intersections, and analyze video frames for PPE detection.

def head_region(person_bbox, head_ratio=0.35):
    # First, define the head region as the top portion of the person's bounding box. 
    x1, y1, x2, y2 = person_bbox
    height = y2 - y1
    # The head is usually the top 35% of the bounding box
    return (x1, y1, x2, y1 + head_ratio * height)

def torso_region(person_bbox, lower_part_ratio=0.2, higher_part_ratio=0.8):
    x1, y1, x2, y2 = person_bbox
    height = y2 - y1
    # The vest is usually between 20% and 80% of the height
    return (x1, y1 + lower_part_ratio * height, x2, y1 + higher_part_ratio * height)

def bbox_intersects(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    return xA < xB and yA < yB

def analyze_video_dataframe(video, my_model, conf, iou, imgsz, frame_stride, st_placeholder=None, progress_placeholder=None, progress_bar=None, output_path=None, device=0, head_ratio=0.35, lower_part_ratio=0.2, higher_part_ratio=0.8):
    data_list = [] 
    frame_idx = 0
    cap = cv2.VideoCapture(video)

    # Video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if output_path:
        # Using mp4v for maximum compatibility across platforms.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    classes_detected = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break
        
        frame_idx += 1

        # --- 1. PROGRESS UPDATE (SAFE) ---
        if total_frames > 0:
            progress = min(frame_idx / total_frames, 1.0)
            if progress_bar is not None:
                progress_bar.progress(progress)
            if progress_placeholder is not None:
                progress_placeholder.markdown(f"**Frame:** {frame_idx} / {total_frames}")

        # It will only process frames that are multiples of the frame_stride to speed up inference.
        if frame_idx % frame_stride != 0 and frame_idx != total_frames:
            # If we're saving the video, we should write the frame even if it's not processed, 
            # or we could choose to skip it for speed, but the final video would look sped up.
            if out:
                out.write(frame)
            continue

        # --- INFERENCE Y TRACKING ---
        results = my_model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml",
            verbose=False, 
            conf=conf, 
            iou=iou,
            device=device, 
            imgsz=imgsz
        )
    
        # It extracts the class IDs from the detected boxes and maps them to class names using
        if results[0].boxes:
            detected_cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
            classes_detected = list(set(classes_detected + [my_model.names[i] for i in detected_cls_ids]))

        # Render the annotated frame with bounding boxes and labels.
        annotated_frame = results[0].plot()

        # --- 3. SAVE AND DISPLAY ---
        if out:
            out.write(annotated_frame)

        if st_placeholder is not None:
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

        # --- 4. DETECTION LOGIC (LOG DATAFRAME) ---

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clases = results[0].boxes.cls.cpu().numpy().astype(int)
            scores = results[0].boxes.conf.cpu().numpy() # It extracts confidence scores
            names = my_model.names

            # 1. It creates dictionaries/lists that store (bbox, score)
            # For people, we need to keep track of their IDs to associate them with PPE detections.
            frame_persons = {ids[i]: (boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'person'}
            
            # For PPE, we just need to know if they intersect with the person's head/torso regions, so we can keep them in lists.
            frame_helmets = [(boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'helmet']
            frame_vests = [(boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'vest']
            frame_no_helmets = [(boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'no-helmet']
            frame_no_vests = [(boxes[i], scores[i]) for i, c in enumerate(clases) if names[c] == 'no-vest']

            for p_id, (p_bbox, p_score) in frame_persons.items():
                                    h_zone = head_region(p_bbox, head_ratio=head_ratio)
                                    t_zone = torso_region(p_bbox, lower_part_ratio=lower_part_ratio, higher_part_ratio=higher_part_ratio)
                                    
                                    # 1. We look for coincidences (Box and Confidence) using bbox_intersects 
                                    # to check if PPE boxes intersect with head/torso zones.
                                    found_h = next(((hb.tolist(), hs) for hb, hs in frame_helmets if bbox_intersects(h_zone, hb)), (None, 0.0))
                                    found_nh = next(((nhb.tolist(), nhs) for nhb, nhs in frame_no_helmets if bbox_intersects(h_zone, nhb)), (None, 0.0))
                                    
                                    found_v = next(((vb.tolist(), vs) for vb, vs in frame_vests if bbox_intersects(t_zone, vb)), (None, 0.0))
                                    found_nv = next(((nvb.tolist(), nvs) for nvb, nvs in frame_no_vests if bbox_intersects(t_zone, nvb)), (None, 0.0))

                                    # 2. Setting flags based on detections
                                    has_helmet = 1 if found_h[0] else 0
                                    has_vest = 1 if found_v[0] else 0
                                    det_no_helmet = 1 if found_nh[0] else 0
                                    det_no_vest = 1 if found_nv[0] else 0

                                    data_list.append({
                                        "Frame": frame_idx,
                                        "Person_ID": int(p_id),
                                        "Person_Conf": round(float(p_score), 3),
                                        "Person_Box": p_bbox.tolist(),
                                        
                                        # Helmet / No Helmet
                                        "Helmet_Detected": has_helmet,
                                        "Helmet_Conf": round(float(found_h[1]), 3) if has_helmet else 0.0,
                                        "Helmet_Box": found_h[0], # Helmet Coordinates
                                        "No_Helmet_Detected": det_no_helmet,
                                        "No_Helmet_Conf": round(float(found_nh[1]), 3) if det_no_helmet else 0.0,
                                        "No_Helmet_Box": found_nh[0], # No Helmet Coordinates
                                        
                                        # Vest / No Vest
                                        "Vest_Detected": has_vest,
                                        "Vest_Conf": round(float(found_v[1]), 3) if has_vest else 0.0,
                                        "Vest_Box": found_v[0],
                                        "No_Vest_Detected": det_no_vest,
                                        "No_Vest_Conf": round(float(found_nv[1]), 3) if det_no_vest else 0.0,
                                        "No_Vest_Box": found_nv[0],
                                        
                                        # Final Violation Logic:
                                        "Violation": 1 if (has_helmet == 0 or has_vest == 0) else 0
                                    })



    # --- 5. FINAL CLEANUP ---
    cap.release()
    if out:
        out.release()

    if progress_bar is not None:
        progress_bar.progress(1.0)

    return pd.DataFrame(data_list), classes_detected
