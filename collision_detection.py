import cv2 as cv
from ultralytics import YOLO

def collision_warning(detections, frame_width, frame_height):
    warnings = []
    
    # split screen into zones
    left_zone = frame_width * 0.33    
    right_zone = frame_width * 0.67   
    front_zone_top = frame_height * 0.6 
    
    large_object_threshold = (frame_width * frame_height) * 0.05  # 5% of frame
    medium_object_threshold = (frame_width * frame_height) * 0.02  # 2% of frame
    
    # detection loop
    for detection in detections:
        box = detection.boxes.xyxy[0]
        class_id = int(detection.boxes.cls[0])
        class_name = detection.names[class_id]
        confidence = detection.boxes.conf[0]
        
        if confidence < 0.5:
            continue
            
        # calculate object bounding box
        obj_center_x = (box[0] + box[2]) / 2
        obj_center_y = (box[1] + box[3]) / 2
        area = (box[2] - box[0]) * (box[3] - box[1])
        
        warning_type = None
        danger_level = "LOW"
        
        # LEFT SIDE DETECTION
        if obj_center_x < left_zone:
            if area > large_object_threshold:
                warning_type = "INCOMING FROM LEFT"
                danger_level = "HIGH"
            elif area > medium_object_threshold:
                warning_type = "VEHICLE ON LEFT"
                danger_level = "MEDIUM"
        
        # RIGHT SIDE DETECTION  
        elif obj_center_x > right_zone:
            if area > large_object_threshold:
                warning_type = "INCOMING FROM RIGHT"
                danger_level = "HIGH"
            elif area > medium_object_threshold:
                warning_type = "VEHICLE ON RIGHT"
                danger_level = "MEDIUM"
        
        # FRONT DETECTION 
        elif left_zone <= obj_center_x <= right_zone and obj_center_y > front_zone_top:
            if area > large_object_threshold:
                warning_type = "COLLISION AHEAD"
                danger_level = "CRITICAL"
            elif area > medium_object_threshold:
                warning_type = "VEHICLE AHEAD"
                danger_level = "HIGH"
        
        # Add warning if detected
        if warning_type:
            warnings.append({
                'type': warning_type,
                'danger': danger_level,
                'object': class_name,
                'confidence': confidence,
                'area': area
            })
    
    return warnings

model = YOLO('yolo_project_backup/best.pt')
VIDEO_PATH = "test_videos/dashcam_footage.mp4"
OUTPUT_PATH = "output_with_collision_detection.mp4"
FRAME_SKIP = 3

cap = cv.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter.fourcc(*'mp4v')
out = cv.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1        
        if frame_count % FRAME_SKIP == 0:
            results = model(frame)
            warnings = collision_warning(results[0], width, height)
            
            annotated_frame = results[0].plot()
            
            y_offset = 30
            for warning in warnings:
                color = (0, 0, 255) if warning['danger'] == 'CRITICAL' else \
                        (0, 165, 255) if warning['danger'] == 'HIGH' else \
                        (0, 255, 255) 
                
                text = f"{warning['type']}: {warning['object']} ({warning['danger']})"
                cv.putText(annotated_frame, text, (10, y_offset), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
            
            cv.line(annotated_frame, (int(width*0.33), 0), (int(width*0.33), height), (255,255,255), 1)
            cv.line(annotated_frame, (int(width*0.67), 0), (int(width*0.67), height), (255,255,255), 1)
            cv.line(annotated_frame, (0, int(height*0.6)), (width, int(height*0.6)), (255,255,255), 1)
            
            cv.imshow('Collision Detection System', annotated_frame)
            out.write(annotated_frame)
            
            if warnings:
                print(f"Frame {frame_count}: {len(warnings)} warnings")
                for w in warnings:
                    print(f"  - {w['type']}: {w['object']} ({w['danger']})")
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    out.release()
    cv.destroyAllWindows()
    print(f"Collision detection video saved: {OUTPUT_PATH}")