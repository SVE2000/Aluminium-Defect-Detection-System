import asyncio
import websockets
import cv2
import numpy as np
from collections import defaultdict
import time
import torch
import os
import sys
import yaml
import requests
import signal

# Add the fasterrcnn pipeline to path
sys.path.append('fasterrcnn-pytorch-training-pipeline')

from models.create_fasterrcnn_model import create_model
from utils.transforms import infer_transforms, resize
from utils.annotations import convert_detections

# =============================================================================
# CONFIGURATION
# =============================================================================
# Classes to ignore/neglect from detection results
IGNORED_CLASSES = [
    'pitted_surface',  # Add class names here that should be filtered out
]

# Conveyor Belt API Configuration
CONVEYOR_BASE_URL = "http://192.168.1.198"
CONVEYOR_SPEED = 2000

# Initial bucket position: 'healthy' or 'anomaly'
INITIAL_BUCKET = 'healthy'  # Change this to set which bucket faces initially
# =============================================================================


class ConveyorBeltController:
    """Controller for conveyor belt operations via HTTP API."""
    
    def __init__(self, base_url, speed, initial_bucket='healthy'):
        self.base_url = base_url
        self.speed = speed
        self.current_bucket = initial_bucket  # Track which bucket is currently facing
        print(f"🎯 Initial bucket position: {self.current_bucket.upper()}")
    
    def start_conveyor(self):
        """Start the conveyor belt at configured speed."""
        try:
            url = f"{self.base_url}/continuous?speed={self.speed}"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ Conveyor started at speed {self.speed}")
                return True
            else:
                print(f"⚠️ Failed to start conveyor: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error starting conveyor: {e}")
            return False
    
    def stop_conveyor(self):
        """Stop the conveyor belt."""
        try:
            url = f"{self.base_url}/stop"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("⏸️  Conveyor stopped")
                return True
            else:
                print(f"⚠️ Failed to stop conveyor: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error stopping conveyor: {e}")
            return False
    
    def flip_bucket(self):
        """Flip the bucket to switch between healthy and anomaly collection."""
        try:
            url = f"{self.base_url}/flip"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                # Toggle current bucket state
                self.current_bucket = 'anomaly' if self.current_bucket == 'healthy' else 'healthy'
                print(f"🔄 Bucket flipped to: {self.current_bucket.upper()}")
                return True
            else:
                print(f"⚠️ Failed to flip bucket: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error flipping bucket: {e}")
            return False
    
    def ensure_bucket_position(self, needs_anomaly_bucket):
        """Ensure the correct bucket is facing based on detection results."""
        if needs_anomaly_bucket and self.current_bucket == 'healthy':
            # Need anomaly bucket but healthy is facing - flip it
            print("🚨 Defects detected! Switching to ANOMALY bucket...")
            self.flip_bucket()
        elif not needs_anomaly_bucket and self.current_bucket == 'anomaly':
            # Need healthy bucket but anomaly is facing - flip it
            print("✅ No defects! Switching to HEALTHY bucket...")
            self.flip_bucket()
        else:
            print(f"✓ Correct bucket already in position: {self.current_bucket.upper()}")


class FasterRCNNDefectDetector:
    """Wrapper for the Faster R-CNN defect detection model."""
    
    def __init__(self, weights_path, data_config_path=None, device=None):
        """Initialize the Faster R-CNN model."""
        print("🤖 Loading Faster R-CNN defect detection model...")
        
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"📍 Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Load class information from checkpoint
        self.num_classes = checkpoint['data']['NC']
        self.classes = checkpoint['data']['CLASSES']
        
        print(f"📋 Loaded {self.num_classes} classes: {self.classes}")
        
        # Build model
        try:
            build_model = create_model[checkpoint['model_name']]
        except:
            print("⚠️ Model name not found in checkpoint, using default fasterrcnn_resnet50_fpn_v2")
            build_model = create_model['fasterrcnn_resnet50_fpn_v2']
        
        self.model = build_model(num_classes=self.num_classes, coco_model=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()
        
        # Generate colors for each class
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        print("✅ Faster R-CNN model loaded successfully!")
    
    def preprocess_image(self, image, resize_to=640):
        """Preprocess image for model inference."""
        # Convert to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert back to 3 channels (grayscale in BGR format for consistency)
        img_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        # Resize image
        image_resized = resize(img_3ch, resize_to, square=False)
        
        # BGR to RGB
        img_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        img_transformed = infer_transforms(img_rgb)
        
        # Add batch dimension
        img_batch = torch.unsqueeze(img_transformed, 0)
        
        return img_batch, image_resized
    
    def predict(self, image, threshold=0.5):
        """Run inference on an image."""
        # Preprocess
        img_batch, image_resized = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_batch.to(self.device))
        
        # Move outputs to CPU
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
        # Parse detections
        detections = []
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            labels = outputs[0]['labels'].data.numpy()
            
            # Filter by threshold and ignored classes
            #print(zip(boxes, scores, labels))
            # print("✅ Detections found:")
            class_names = np.array(self.classes)
            print(list(zip(scores, class_names[labels])))
            for box, score, label in zip(boxes, scores, labels):
                if score >= threshold:
                    class_name = self.classes[label]
                    
                    # Skip if class is in the ignored list
                    if class_name in IGNORED_CLASSES:
                        print(f"   ⏭️  Ignoring detection: {class_name} (in ignored list)")
                        continue
                    
                    xmin, ymin, xmax, ymax = box
                    
                    # Scale boxes back to original image size
                    scale_x = image.shape[1] / image_resized.shape[1]
                    scale_y = image.shape[0] / image_resized.shape[0]
                    
                    xmin = int(xmin * scale_x)
                    ymin = int(ymin * scale_y)
                    xmax = int(xmax * scale_x)
                    ymax = int(ymax * scale_y)
                    
                    detections.append({
                        'bbox': (xmin, ymin, xmax, ymax),
                        'class': class_name,
                        'class_id': int(label),
                        'confidence': float(score)
                    })
        
        return detections


class RectangleTracker:
    """Track rectangles across frames using centroid-based tracking."""
    
    def __init__(self, max_distance=100, max_frames_lost=15):
        """Initialize tracker."""
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        self.tracked_objects = {}
        self.next_id = 0
        self.frame_count = 0
        self.detected_objects = []
        self.detection_pause_time = 0
        self.detected_object_id = None
        self.inspected_objects = set()
    
    @staticmethod
    def euclidean_distance(pt1, pt2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def update(self, rectangles, frame_shape=None):
        """Update tracker with new rectangle detections."""
        self.frame_count += 1
        detected_object_data = None
        
        if len(rectangles) == 0:
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]['lost_frames'] += 1
                if self.tracked_objects[obj_id]['lost_frames'] > self.max_frames_lost:
                    if obj_id in self.inspected_objects:
                        self.inspected_objects.remove(obj_id)
                    del self.tracked_objects[obj_id]
            return [], None
        
        if len(self.tracked_objects) == 0:
            for rect in rectangles:
                self.tracked_objects[self.next_id] = {
                    'centroid': rect['centroid'],
                    'box': rect['box'],
                    'rect': rect['rect'],
                    'lost_frames': 0,
                    'history': [rect['centroid']],
                    'consecutive_frames': 1,
                    'crossed_center': False,
                    'last_y': rect['centroid'][1]
                }
                self.next_id += 1
        else:
            num_detections = len(rectangles)
            num_tracked = len(self.tracked_objects)
            
            cost_matrix = np.zeros((num_detections, num_tracked))
            tracked_ids = list(self.tracked_objects.keys())
            
            for det_idx, rect in enumerate(rectangles):
                for track_idx, obj_id in enumerate(tracked_ids):
                    distance = self.euclidean_distance(rect['centroid'], 
                                                      self.tracked_objects[obj_id]['centroid'])
                    cost_matrix[det_idx, track_idx] = distance
            
            matched_detections = set()
            matched_tracked_indices = set()
            
            distances = []
            for det_idx in range(num_detections):
                for track_idx in range(num_tracked):
                    distances.append((cost_matrix[det_idx, track_idx], det_idx, track_idx))
            
            distances.sort()
            
            for distance, det_idx, track_idx in distances:
                if det_idx not in matched_detections and track_idx not in matched_tracked_indices:
                    if distance < self.max_distance:
                        obj_id = tracked_ids[track_idx]
                        current_centroid = rectangles[det_idx]['centroid']
                        previous_y = self.tracked_objects[obj_id]['last_y']
                        current_y = current_centroid[1]
                        
                        self.tracked_objects[obj_id]['centroid'] = current_centroid
                        self.tracked_objects[obj_id]['box'] = rectangles[det_idx]['box']
                        self.tracked_objects[obj_id]['rect'] = rectangles[det_idx]['rect']
                        self.tracked_objects[obj_id]['lost_frames'] = 0
                        self.tracked_objects[obj_id]['consecutive_frames'] += 1
                        self.tracked_objects[obj_id]['history'].append(current_centroid)
                        self.tracked_objects[obj_id]['last_y'] = current_y
                        
                        # Check if object crosses the center horizontal line (moving downward)
                        if frame_shape is not None and obj_id not in self.inspected_objects:
                            frame_height, frame_width = frame_shape
                            frame_center_y = frame_height / 2
                            
                            # Check if object crossed center line (was above, now below or at center)
                            if previous_y < frame_center_y and current_y >= frame_center_y:
                                print(f"🎯 Object ID {obj_id} crossed center line! (y: {previous_y:.0f} -> {current_y:.0f})")
                                
                                self.detected_objects.append({
                                    'id': obj_id,
                                    'frame': self.frame_count
                                })
                                self.detection_pause_time = time.time()
                                self.detected_object_id = obj_id
                                self.inspected_objects.add(obj_id)
                                
                                detected_object_data = {
                                    'id': obj_id,
                                    'centroid': current_centroid,
                                    'box': rectangles[det_idx]['box'],
                                    'rect': rectangles[det_idx]['rect'],
                                    'contour': rectangles[det_idx].get('contour')
                                }
                        
                        if len(self.tracked_objects[obj_id]['history']) > 30:
                            self.tracked_objects[obj_id]['history'].pop(0)
                        
                        matched_detections.add(det_idx)
                        matched_tracked_indices.add(track_idx)
            
            for det_idx, rect in enumerate(rectangles):
                if det_idx not in matched_detections:
                    self.tracked_objects[self.next_id] = {
                        'centroid': rect['centroid'],
                        'box': rect['box'],
                        'rect': rect['rect'],
                        'lost_frames': 0,
                        'history': [rect['centroid']],
                        'consecutive_frames': 1,
                        'crossed_center': False,
                        'last_y': rect['centroid'][1]
                    }
                    self.next_id += 1
            
            for track_idx, obj_id in enumerate(tracked_ids):
                if track_idx not in matched_tracked_indices:
                    self.tracked_objects[obj_id]['lost_frames'] += 1
                    self.tracked_objects[obj_id]['consecutive_frames'] = 0
                    if self.tracked_objects[obj_id]['lost_frames'] > self.max_frames_lost:
                        if obj_id in self.inspected_objects:
                            self.inspected_objects.remove(obj_id)
                        del self.tracked_objects[obj_id]
        
        output = []
        for obj_id, obj_data in self.tracked_objects.items():
            output.append({
                'id': obj_id,
                'centroid': obj_data['centroid'],
                'box': obj_data['box'],
                'rect': obj_data['rect'],
                'history': obj_data['history']
            })
        
        return output, detected_object_data


async def handle_connection(websocket, defect_detector, conveyor_controller):
    """Handle incoming WebSocket connections with conveyor belt control."""
    client_addr = websocket.remote_address
    print(f"✅ Client connected from {client_addr}")
    
    tracker = RectangleTracker(max_distance=100, max_frames_lost=15)
    frame_num = 0
    
    ai_results_cache = {}
    
    frozen_frame = None
    frozen_detections = None
    freeze_start_time = 0
    freeze_duration = 2.0
    is_frozen = False
    
    try:
        while True:
            try:
                # Check if we need to freeze and display results
                if is_frozen:
                    elapsed = time.time() - freeze_start_time
                    if elapsed < freeze_duration:
                        # Display frozen frame with results
                        if frozen_frame is not None:
                            display_frame = frozen_frame.copy()
                            h, w = display_frame.shape[:2]
                            
                            if frozen_detections is not None and len(frozen_detections) > 0:
                                for det in frozen_detections:
                                    xmin, ymin, xmax, ymax = det['bbox']
                                    color = tuple(map(int, defect_detector.colors[det['class_id']]))
                                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 3)
                                    
                                    label = f"{det['class']}: {det['confidence']:.2f}"
                                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                                    cv2.rectangle(display_frame,
                                                (xmin, ymin - label_size[1] - 10),
                                                (xmin + label_size[0], ymin),
                                                color, -1)
                                    cv2.putText(display_frame, label,
                                              (xmin, ymin - 5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                
                                status_text = f"DEFECTS FOUND: {len(frozen_detections)}"
                                status_bg_color = (0, 0, 255)
                                status_text_color = (255, 255, 255)
                            else:
                                status_text = "INSPECTION: PASS - NO DEFECTS"
                                status_bg_color = (0, 255, 0)
                                status_text_color = (0, 0, 0)
                            
                            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                            cv2.rectangle(display_frame,
                                        (10, 10),
                                        (status_size[0] + 30, status_size[1] + 40),
                                        status_bg_color, -1)
                            cv2.putText(display_frame, status_text,
                                      (20, status_size[1] + 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_text_color, 3)
                            
                            remaining = freeze_duration - elapsed
                            countdown_text = f"Resuming in: {remaining:.1f}s"
                            cv2.putText(display_frame, countdown_text,
                                      (20, h - 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                            
                            # Show bucket status
                            bucket_text = f"Bucket: {conveyor_controller.current_bucket.upper()}"
                            cv2.putText(display_frame, bucket_text,
                                      (20, h - 60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                            
                            cv2.imshow("Defect Inspection System", display_frame)
                            cv2.waitKey(30)
                        
                        image_data = await websocket.recv()
                        continue
                    else:
                        # Freeze period over, restart conveyor
                        print("✅ Unfreezing display and restarting conveyor...")
                        conveyor_controller.start_conveyor()
                        is_frozen = False
                        frozen_frame = None
                        frozen_detections = None
                        freeze_start_time = 0
                
                # Receive image data
                image_data = await websocket.recv()

                # Decode image
                frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # Convert to HSV and remove blue
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    lower_blue = np.array([90, 100, 50])
                    upper_blue = np.array([150, 255, 255])
                    mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    mask_inv = cv2.bitwise_not(mask)

                    # Morphological operations
                    kernel = np.ones((7, 7), np.uint8)
                    mask_closed = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
                    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)
                    mask_dilated = cv2.dilate(mask_opened, kernel, iterations=1)

                    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    segmented = cv2.bitwise_and(frame, frame, mask=mask_dilated)
                    current_detections = []

                    for contour in contours:
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        if (len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(contour) > 3000):
                            rect = cv2.minAreaRect(contour)
                            width = rect[1][0]
                            height = rect[1][1]
                            
                            if width <= 0 or height <= 0:
                                continue
                            
                            aspect_ratio = max(width, height) / min(width, height)
                            
                            if aspect_ratio > 3.0:
                                continue
                            
                            box = cv2.boxPoints(rect)
                            box = np.intp(box)
                            
                            M = cv2.moments(contour)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                            else:
                                cx, cy = int(rect[0][0]), int(rect[0][1])
                            
                            current_detections.append({
                                'centroid': (cx, cy),
                                'box': box,
                                'contour': contour,
                                'rect': rect
                            })

                    tracked_rectangles, detected_object = tracker.update(current_detections, frame_shape=frame.shape[:2])
                    
                    # If object crossed center line, stop conveyor and run inspection
                    if detected_object is not None:
                        print(f"🎯 Object ID {detected_object['id']} crossed center horizontal line!")
                        print("⏸️  STOPPING CONVEYOR...")
                        conveyor_controller.stop_conveyor()
                        
                        print("🤖 Running Faster R-CNN defect detection...")
                        
                        rect = detected_object['rect']
                        box = detected_object['box']
                        width = int(rect[1][0])
                        height = int(rect[1][1])
                        
                        if width > 0 and height > 0:
                            box_sorted = box[np.argsort(box[:,1]),:]
                            top_pts = box_sorted[:2,:]
                            bottom_pts = box_sorted[2:,:]
                            top_left, top_right = top_pts[np.argsort(top_pts[:,0]),:]
                            bottom_left, bottom_right = bottom_pts[np.argsort(bottom_pts[:,0]),:]
                            
                            src_pts = np.array([bottom_left, top_left, top_right, bottom_right], dtype="float32")
                            dst_pts = np.array([[0, height-1],
                                                [0, 0],
                                                [width-1, 0],
                                                [width-1, height-1]], dtype="float32")
                            
                            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                            warped = cv2.warpPerspective(frame, M, (width, height))
                            
                            # Crop 10% from all sides
                            h, w = warped.shape[:2]
                            crop_x = int(w * 0.1)
                            crop_y = int(h * 0.1)
                            cropped = warped[crop_y:h-crop_y, crop_x:w-crop_x]
                            
                            # Run detection
                            detections = defect_detector.predict(cropped, threshold=0.2)
                            
                            # Set bucket position based on results
                            has_defects = len(detections) > 0
                            conveyor_controller.ensure_bucket_position(has_defects)
                            
                            # Set freeze state
                            is_frozen = True
                            freeze_start_time = time.time()
                            frozen_frame = cropped.copy()
                            frozen_detections = detections
                            
                            ai_results_cache[detected_object['id']] = {
                                'detections': detections,
                                'cropped_image': cropped,
                                'box': box,
                                'centroid': detected_object['centroid']
                            }
                            
                            if detections:
                                print(f"✅ Found {len(detections)} defect(s):")
                                for det in detections:
                                    print(f"   - {det['class']}: {det['confidence']:.4f}")
                                
                                annotated_img = cropped.copy()
                                for det in detections:
                                    xmin, ymin, xmax, ymax = det['bbox']
                                    color = tuple(map(int, defect_detector.colors[det['class_id']]))
                                    cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), color, 2)
                                    label = f"{det['class']}: {det['confidence']:.2f}"
                                    cv2.putText(annotated_img, label, (xmin, ymin - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                filename = f"Defect_Detection/ID{detected_object['id']}_frame{frame_num}_DEFECTS.jpg"
                                os.makedirs("Defect_Detection", exist_ok=True)
                                cv2.imwrite(filename, annotated_img)
                                print(f"💾 Saved to {filename}")
                            else:
                                print("✅ No defects detected (PASS)")
                                filename = f"Defect_Detection/ID{detected_object['id']}_frame{frame_num}_PASS.jpg"
                                os.makedirs("Defect_Detection", exist_ok=True)
                                cv2.imwrite(filename, cropped)
                                print(f"💾 Saved to {filename}")
                    
                    # Draw tracked rectangles
                    for tracked in tracked_rectangles:
                        obj_id = tracked['id']
                        box = tracked['box']
                        centroid = tracked['centroid']
                        history = tracked['history']
                        
                        color = (int(255 * (obj_id % 10) / 10), 
                                int(255 * ((obj_id * 3) % 10) / 10), 
                                int(255 * ((obj_id * 7) % 10) / 10))
                        
                        cv2.polylines(segmented, [box], True, color, 2)
                        cv2.circle(segmented, centroid, 5, color, -1)
                        cv2.putText(segmented, f"ID: {obj_id}", 
                                  (centroid[0] - 15, centroid[1] - 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if len(history) > 1:
                            pts = np.array(history, dtype=np.int32)
                            cv2.polylines(segmented, [pts], False, color, 1)
                        
                        if obj_id in ai_results_cache:
                            ai_data = ai_results_cache[obj_id]
                            detections = ai_data['detections']
                            
                            if detections:
                                defect_summary = f"Defects: {len(detections)}"
                                text_size = cv2.getTextSize(defect_summary, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                text_x = centroid[0] - text_size[0] // 2
                                text_y = centroid[1] - 30
                                
                                cv2.rectangle(segmented, 
                                            (text_x - 5, text_y - text_size[1] - 5), 
                                            (text_x + text_size[0] + 5, text_y + 5),
                                            (0, 0, 255), -1)
                                cv2.putText(segmented, defect_summary,
                                          (text_x, text_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            else:
                                text = "PASS"
                                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                text_x = centroid[0] - text_size[0] // 2
                                text_y = centroid[1] - 30
                                
                                cv2.rectangle(segmented, 
                                            (text_x - 5, text_y - text_size[1] - 5), 
                                            (text_x + text_size[0] + 5, text_y + 5),
                                            (0, 255, 0), -1)
                                cv2.putText(segmented, text,
                                          (text_x, text_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    cv2.imshow("Defect Inspection System", segmented)
                    cv2.waitKey(1)

                frame_num += 1

            except asyncio.CancelledError:
                print(f"❌ Connection cancelled.")
                break
            except websockets.exceptions.ConnectionClosed:
                print(f"❌ Connection closed.")
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                import traceback
                traceback.print_exc()
                break
    finally:
        print(f"👋 Client disconnected after {frame_num} frames.")
        cv2.destroyAllWindows()


def signal_handler(sig, frame, conveyor_controller):
    """Handle keyboard interrupt to stop conveyor."""
    print("\n🛑 Keyboard interrupt detected!")
    print("⏸️  Stopping conveyor...")
    conveyor_controller.stop_conveyor()
    print("✅ Conveyor stopped. Exiting...")
    sys.exit(0)


async def main():
    print("="*60)
    print("🚀 DEFECT INSPECTION SYSTEM")
    print("="*60)
    
    # Initialize conveyor controller
    conveyor_controller = ConveyorBeltController(
        base_url=CONVEYOR_BASE_URL,
        speed=CONVEYOR_SPEED,
        initial_bucket=INITIAL_BUCKET
    )
    
    # Setup signal handler for Ctrl+C
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, conveyor_controller))
    
    # Start conveyor at beginning
    print("\n🎬 Starting conveyor belt...")
    conveyor_controller.start_conveyor()
    
    print("\n📍 Listening on ws://0.0.0.0:8765")
    
    # Initialize detection model
    weights_path = "fasterrcnn-pytorch-training-pipeline/outputs/training/neu_defect_fasterrcnn_mobilenetv3_large_fpn_nomosaic_20e/best_model.pth"
     
    try:
        defect_detector = FasterRCNNDefectDetector(
            weights_path=weights_path,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        conveyor_controller.stop_conveyor()
        return
    
    # Create handler
    async def handler(websocket):
        await handle_connection(websocket, defect_detector, conveyor_controller)
    
    server = await websockets.serve(handler, "0.0.0.0", 8765, max_size=None)
    
    print("\n✅ System ready! Waiting for connections...")
    print("💡 Press Ctrl+C to stop conveyor and exit\n")
    print("="*60)
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        conveyor_controller.stop_conveyor()
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
