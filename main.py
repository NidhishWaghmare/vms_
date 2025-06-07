import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import os
import threading
import logging
import logging.handlers
import requests
import time
import gc
import concurrent.futures
import uuid
import json
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configuration
CROSSING_TOLERANCE = 10  # Pixels, tolerance for line crossing detection
MODEL_CONFIG = {
    "1": {"name": "vehicleInOut", "classes": [1, 2, 3, 5, 7]},  # Classes: 2=car, 5=bus, 7=truck
    "2": {"name": "anpr", "classes": [1, 2, 3, 5, 7]}
}
SPECIFIC_CAMERA_ID = "53b3850d-e0ef-4668-9fb5-12c980aac83d"  # Camera ID for swap directions
DEFAULT_API_IP = "192.168.1.29:8001"  # Fallback API IP
OUTPUT_API_ENDPOINT = "/api/v1/aiAnalytics/sendAnalyticsJson"
INPUT_API_ENDPOINT = "/api/v1/aiAnalytics/getCamerasForAnalytics"
ERROR_API_ENDPOINT = "/api/v1/aiAnalytics/reportError"
FETCH_INTERVAL = 60  # Fetch new camera data every hour (seconds)
MAX_THREADS = 4  # Limit concurrent camera processing threads
FRAME_SKIP = 5  # Process every 5th frame to reduce CPU usage
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT = 5  # Keep 5 log file backups
SCREENSHOT_TIME_WINDOW = 5  # Seconds to treat multiple detections as one

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "screenshots"
JSON_DIR = DATA_DIR / "logs"
ROI_CONFIG_DIR = DATA_DIR / "roi_configs"
VEHICLE_MODEL_PATH = BASE_DIR / "models" / "yolov8s.pt"
ANPR_MODEL_PATH = BASE_DIR / "models" / "ANPR.pt"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)
ROI_CONFIG_DIR.mkdir(exist_ok=True)

# Setup logging with rotation
log_handler = logging.handlers.RotatingFileHandler(
    JSON_DIR / 'analytics.log',
    maxBytes=LOG_MAX_SIZE,
    backupCount=LOG_BACKUP_COUNT
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        log_handler,
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get API IP from environment variable or use default
API_IP = os.environ.get('API_IP', DEFAULT_API_IP)
INPUT_API_URL = f"http://{API_IP}{INPUT_API_ENDPOINT}"
OUTPUT_API_URL = f"http://{API_IP}{OUTPUT_API_ENDPOINT}"
ERROR_API_URL = f"http://{API_IP}{ERROR_API_ENDPOINT}"
logger.info(f"Input API URL: {INPUT_API_URL}")
logger.info(f"Output API URL: {OUTPUT_API_URL}")
logger.info(f"Error API URL: {ERROR_API_URL}")

# Load YOLO models
try:
    if not VEHICLE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Vehicle model file not found at {VEHICLE_MODEL_PATH}")
    vehicle_model = YOLO(str(VEHICLE_MODEL_PATH))
except Exception as e:
    logger.error(f"Failed to load vehicle model: {e}")
    exit(1)

try:
    if not ANPR_MODEL_PATH.exists():
        raise FileNotFoundError(f"ANPR model file not found at {ANPR_MODEL_PATH}")
    anpr_model = YOLO(str(ANPR_MODEL_PATH))
except Exception as e:
    logger.error(f"Failed to load ANPR model: {e}")
    exit(1)

# Global variables
camera_states = {}  # {camera_id: {model_id: state}}
stop_event = threading.Event()
fetch_trigger = threading.Event()  # Event to trigger immediate fetch after error confirmation

def report_error(camera_id, error_message):
    """Send error details to the error API endpoint and wait for confirmation."""
    error_data = {
        "cameraId": camera_id,
        "errorMessage": str(error_message),
        "timestamp": datetime.now().strftime("%d %b %Y %H:%M:%S"),
        "errorId": str(uuid.uuid4())
    }
    try:
        headers = {"Authorization": "Bearer YOUR_TOKEN_HERE"}
        response = requests.post(ERROR_API_URL, json=error_data, headers=headers, timeout=5)
        if response.status_code == 200:
            try:
                response_json = response.json()
                if response_json.get("status") == "received":
                    logger.info(f"Camera {camera_id} - Error report confirmed by frontend. Triggering immediate fetch.")
                    fetch_trigger.set()
                else:
                    logger.warning(f"Camera {camera_id} - Error report sent but no 'received' confirmation: {response_json}")
            except ValueError:
                logger.error(f"Camera {camera_id} - Invalid JSON response from error API: {response.text}")
        else:
            logger.error(f"Camera {camera_id} - Failed to report error: Status {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        logger.error(f"Camera {camera_id} - Failed to report error to {ERROR_API_URL}: {e}")
    finally:
        gc.collect()

def save_roi_config(camera_id, roi_points):
    """Save ROI points to a JSON file for the camera."""
    roi_file = ROI_CONFIG_DIR / f"{camera_id}_roi.json"
    roi_data = {"roi_points": roi_points}
    try:
        with open(roi_file, 'w') as f:
            json.dump(roi_data, f)
        logger.info(f"Camera {camera_id} - Saved ROI points to {roi_file}")
    except Exception as e:
        logger.error(f"Camera {camera_id} - Failed to save ROI points: {e}")

def load_roi_config(camera_id):
    """Load ROI points from a JSON file for the camera, if available."""
    roi_file = ROI_CONFIG_DIR / f"{camera_id}_roi.json"
    if roi_file.exists():
        try:
            with open(roi_file, 'r') as f:
                roi_data = json.load(f)
                roi_points = roi_data.get("roi_points", [])
                if len(roi_points) == 2:
                    logger.info(f"Camera {camera_id} - Loaded ROI points from {roi_file}")
                    return roi_points
        except Exception as e:
            logger.error(f"Camera {camera_id} - Failed to load ROI points: {e}")
    return None

def initialize_camera_state(model_id, camera_id):
    """Initialize state for a camera based on model ID."""
    base_state = {
        "roi_points": [],
        "roi_selected": False,
        "track_states": {},  # Track ID to state mapping
        "cap": None,
        "width": None,
        "height": None,
        "line_params": None,
        "swap_directions": camera_id == SPECIFIC_CAMERA_ID
    }
    if model_id == "1":
        base_state.update({"enter_count": 0, "exit_count": 0})
    return base_state

def calculate_centroid(bbox):
    """Calculate the centroid of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def signed_distance(x, y, a, b, c):
    """Calculate signed distance from point (x, y) to line ax + by + c = 0."""
    denominator = np.sqrt(a**2 + b**2)
    if denominator == 0:
        return 0
    return (a * x + b * y + c) / denominator

def save_screenshot(frame, camera_id, timestamp, prefix):
    """Save a full frame screenshot and return its filename."""
    try:
        timestamp_str = timestamp.strftime("%d %b %Y %H:%M:%S").replace(" ", "_").replace(":", "_")
        filename = f"{prefix}{camera_id}_{timestamp_str}.jpg"
        filepath = OUTPUT_DIR / filename
        cv2.imwrite(str(filepath), frame)
        logger.info(f"Camera {camera_id} - Saved full frame screenshot: {filepath}")
        return filename
    except Exception as e:
        logger.error(f"Camera {camera_id} - Failed to save screenshot: {e}")
        return None
    finally:
        gc.collect()

def detect_number_plate(frame, bbox):
    """Detect number plate in the cropped vehicle region (for ANPR)."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        vehicle_img = frame[y1:y2, x1:x2]
       
        results = anpr_model(vehicle_img)
        detections = results[0].boxes.data.cpu().numpy()
       
        if len(detections) > 0:
            best_det = max(detections, key=lambda x: x[4])
            if best_det[4] > 0.6:
                plate_x1, plate_y1, plate_x2, plate_y2 = map(int, best_det[:4])
                plate_x1 += x1
                plate_y1 += y1
                plate_x2 += x1
                plate_y2 += y1
                return [plate_x1, plate_y1, plate_x2, plate_y2], vehicle_img[plate_y1-y1:plate_y2-y1, plate_x1-x1:plate_x2-x1]
        return None, None
    except Exception as e:
        logger.error(f"ANPR detection failed: {e}")
        return None, None
    finally:
        del vehicle_img, results, detections
        gc.collect()

def determine_crossing_anpr(prev_centroid, curr_centroid, a, b, c, frame, bbox, camera_id, camera_info, state, shared_state):
    """Determine if the car crosses the ROI line and log to JSON for ANPR entering events."""
    try:
        if prev_centroid is None or curr_centroid is None:
            return None

        prev_x, prev_y = prev_centroid
        curr_x, curr_y = curr_centroid
        prev_dist = signed_distance(prev_x, prev_y, a, b, c)
        curr_dist = signed_distance(curr_x, curr_y, a, b, c)

        timestamp = datetime.now()
        log_entry = None
       
        if not state["has_crossed"]:
            if prev_dist > CROSSING_TOLERANCE and curr_dist <= CROSSING_TOLERANCE:
                state["has_crossed"] = True
                # Check if an event was logged recently for this camera
                last_timestamp = shared_state["last_screenshot_timestamps"].get("anpr_enter")
                if last_timestamp is None or (timestamp - last_timestamp).total_seconds() > SCREENSHOT_TIME_WINDOW:
                    vehicle_filename = save_screenshot(frame, camera_id, timestamp, "enter_")
                    plate_filename = None
                    plate_bbox, _ = detect_number_plate(frame, bbox)
                    if plate_bbox:
                        plate_filename = save_screenshot(frame, camera_id, timestamp, "plate_enter_")
                    shared_state["last_screenshot_timestamps"]["anpr_enter"] = timestamp
                    logger.info(f"Camera {camera_id} - ANPR saved screenshot and created log for enter event")
                    
                    log_entry = {
                        "modelName": MODEL_CONFIG["2"]["name"],
                        "logData": {
                            "time": timestamp.strftime("%d %b %Y %H:%M:%S"),
                            "plateNumber": "unknown",
                            "screenShotPath": vehicle_filename,
                            "plateScreenShotPath": plate_filename,
                            "cameraId": camera_id,
                            "location": camera_info.get("location", "UNKNOWN")
                        }
                    }
                else:
                    logger.info(f"Camera {camera_id} - ANPR skipped screenshot and log for enter event (within {SCREENSHOT_TIME_WINDOW}s)")
        else:
            if abs(curr_dist) > CROSSING_TOLERANCE * 2:
                state["has_crossed"] = False
       
        return log_entry
    finally:
        gc.collect()

def determine_crossing_vehicle(tracks, a, b, c, frame, camera_id, camera_info, state, shared_state):
    """Determine if vehicle centroids cross the ROI line and classify as Entering/Exiting."""
    try:
        log_entry = None
        timestamp = datetime.now()
        
        for track in tracks:
            track_id = track.track_id
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
            centroid = calculate_centroid(bbox)
            
            # Initialize track state if not present
            if track_id not in state['track_states']:
                state['track_states'][track_id] = {
                    'prev_centroid': None,
                    'has_crossed': False
                }
            
            track_state = state['track_states'][track_id]
            curr_centroid = centroid
            prev_centroid = track_state['prev_centroid']
            
            if prev_centroid is None:
                track_state['prev_centroid'] = curr_centroid
                continue
            
            prev_dist = signed_distance(prev_centroid[0], prev_centroid[1], a, b, c)
            curr_dist = signed_distance(curr_centroid[0], curr_centroid[1], a, b, c)
            
            swap_directions = state["swap_directions"]
            event_type = None
            filename = None
            
            if not track_state['has_crossed']:
                # Left to right (entering)
                if prev_dist < -CROSSING_TOLERANCE and curr_dist >= -CROSSING_TOLERANCE:
                    state['enter_count'] += 1
                    track_state['has_crossed'] = True
                    event_type = "enter"
                    logger.info(f"Camera {camera_id} - Vehicle {track_id} entered. Total entering: {state['enter_count']}")
                    # Check if an event was logged recently for this camera
                    last_timestamp = shared_state["last_screenshot_timestamps"].get("vehicle_enter")
                    if last_timestamp is None or (timestamp - last_timestamp).total_seconds() > SCREENSHOT_TIME_WINDOW:
                        filename = save_screenshot(frame, camera_id, timestamp, "enter_")
                        shared_state["last_screenshot_timestamps"]["vehicle_enter"] = timestamp
                        logger.info(f"Camera {camera_id} - Saved screenshot and created log for enter event")
                    else:
                        logger.info(f"Camera {camera_id} - Skipped screenshot and log for enter event (within {SCREENSHOT_TIME_WINDOW}s)")
                        event_type = None  # Prevent log creation
                # Right to left (exiting)
                elif prev_dist > CROSSING_TOLERANCE and curr_dist <= CROSSING_TOLERANCE:
                    state['exit_count'] += 1
                    track_state['has_crossed'] = True
                    event_type = "exit"
                    logger.info(f"Camera {camera_id} - Vehicle {track_id} exited. Total exiting: {state['exit_count']}")
                    # Check if an event was logged recently for this camera
                    last_timestamp = shared_state["last_screenshot_timestamps"].get("vehicle_exit")
                    if last_timestamp is None or (timestamp - last_timestamp).total_seconds() > SCREENSHOT_TIME_WINDOW:
                        filename = save_screenshot(frame, camera_id, timestamp, "exit_")
                        shared_state["last_screenshot_timestamps"]["vehicle_exit"] = timestamp
                        logger.info(f"Camera {camera_id} - Saved screenshot and created log for exit event")
                    else:
                        logger.info(f"Camera {camera_id} - Skipped screenshot and log for exit event (within {SCREENSHOT_TIME_WINDOW}s)")
                        event_type = None  # Prevent log creation
            
            if event_type:
                log_entry = {
                    "modelName": MODEL_CONFIG["1"]["name"],
                    "logData": {
                        "time": timestamp.strftime("%d %b %Y %H:%M:%S"),
                        "eventType": event_type,
                        "screenShotPath": filename,
                        "cameraId": camera_id,
                        "location": camera_info.get("location", "UNKNOWN"),
                        "entryCount": state["enter_count"],
                        "exitCount": state["exit_count"]
                    }
                }
            
            if abs(curr_dist) > CROSSING_TOLERANCE * 2:
                track_state['has_crossed'] = False
            
            track_state['prev_centroid'] = curr_centroid
        
        return log_entry
    finally:
        gc.collect()

def append_to_json_log(log_entry, camera_id):
    """Send log entry to the output API endpoint with retries."""
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(OUTPUT_API_URL, json=log_entry, timeout=5)
            if response.status_code == 200:
                logger.info(f"Camera {camera_id} - Successfully sent log entry to {OUTPUT_API_URL}: {log_entry}")
                return True
            else:
                logger.error(f"Camera {camera_id} - Failed to send log entry: Status {response.status_code}, Response: {response.text}")
        except requests.RequestException as e:
            logger.error(f"Camera {camera_id} - Failed to send log entry to {OUTPUT_API_URL}: {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            retry_delay *= 2
    logger.error(f"Camera {camera_id} - Failed to send log entry after {max_retries} attempts")
    return False

def select_roi(camera_id, first_frame, width, height):
    """Load ROI configuration for a camera from saved JSON file."""
    state = camera_states[camera_id]["shared"]
    if first_frame is None or first_frame.size == 0 or first_frame.shape[0] == 0 or first_frame.shape[1] == 0:
        error_msg = f"Invalid first frame for camera {camera_id}"
        logger.error(error_msg)
        report_error(camera_id, error_msg)
        return False

    saved_roi = load_roi_config(camera_id)
    if saved_roi:
        state["roi_points"] = saved_roi
        state["roi_selected"] = True
        logger.info(f"Camera {camera_id} - Loaded saved ROI: {saved_roi}")
        return True
    else:
        error_msg = f"No saved ROI configuration found for camera {camera_id}. Please provide a valid ROI configuration file."
        logger.error(error_msg)
        report_error(camera_id, error_msg)
        return False

def process_camera(camera, model_ids):
    """Process video stream for a single camera for specified model IDs."""
    camera_id = camera["cameraId"]
    video_path = camera["rtspUrl"]
   
    logger.info(f"Processing camera {camera_id} with video URL: {video_path} for modelIds {model_ids}")
   
    # Initialize state for each model and shared state
    camera_states[camera_id] = {
        "shared": {
            "cap": None,
            "width": None,
            "height": None,
            "roi_points": [],
            "roi_selected": False,
            "line_params": None,
            "last_screenshot_timestamps": {
                "vehicle_enter": None,
                "vehicle_exit": None,
                "anpr_enter": None
            }
        }
    }
    for model_id in model_ids:
        camera_states[camera_id][model_id] = initialize_camera_state(model_id, camera_id)
   
    state = camera_states[camera_id]["shared"]
   
    # Initialize DeepSORT
    deepsort = DeepSort(max_age=10, nn_budget=100, override_track_class=None)
   
    # Load video
    cap = None
    try:
        if Path(video_path).exists():
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(video_path)
       
        if not cap.isOpened():
            error_msg = f"Failed to open stream for camera {camera_id}"
            logger.error(error_msg)
            report_error(camera_id, error_msg)
            return
       
        state["cap"] = cap
       
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        logger.info(f"Camera {camera_id} - Width: {width}, Height: {height}, FPS: {fps}")
       
        # Resize frame if resolution is low
        if width < 640 or height < 480:
            scale_factor = 2
            width = int(width * scale_factor)
            height = int(height * scale_factor)
       
        state["width"] = width
        state["height"] = height
       
        # Read first frame for ROI selection
        ret, first_frame = cap.read()
        if not ret or first_frame is None or first_frame.size == 0:
            error_msg = f"Failed to read first frame for camera {camera_id}"
            logger.error(error_msg)
            report_error(camera_id, error_msg)
            return
       
        first_frame = cv2.resize(first_frame, (width, height))
        logger.info(f"Camera {camera_id} - First frame shape: {first_frame.shape}")
       
        # Select ROI
        if not select_roi(camera_id, first_frame, width, height):
            error_msg = f"Failed to select ROI for camera {camera_id}"
            logger.error(error_msg)
            report_error(camera_id, error_msg)
            return
       
        # Define ROI line
        x1, y1 = state["roi_points"][0]
        x2, y2 = state["roi_points"][1]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        state["line_params"] = (a, b, c)
       
        # Process video frames
        frame_count = 0
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                error_msg = f"End of video or failed to read frame for camera {camera_id}"
                logger.info(error_msg)
                report_error(camera_id, error_msg)
                break
           
            frame = cv2.resize(frame, (width, height))
           
            # Skip frames to reduce CPU usage
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                del frame
                continue
           
            # Process each model
            for model_id in model_ids:
                model_state = camera_states[camera_id][model_id]
               
                # Crop to bottom 40% for inference
                bottom_40 = frame[int(height * 0.6):, :, :]
               
                # Perform vehicle detection on bottom 40%
                try:
                    results = vehicle_model(bottom_40, classes=MODEL_CONFIG[model_id]["classes"])
                    detections = results[0].boxes.data.cpu().numpy()
                    logger.info(f"Camera {camera_id} - Model {model_id} - Frame processed: {len(detections)} vehicle detections")
                except Exception as e:
                    error_msg = f"Vehicle detection failed for camera {camera_id}, model {model_id}: {e}"
                    logger.error(error_msg)
                    report_error(camera_id, error_msg)
                    continue
               
                # Prepare detections for DeepSORT, adjusting y-coordinates to full frame
                deepsort_detections = []
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if conf < 0.25:
                        continue
                    # Adjust y-coordinates to full frame
                    y1 += int(height * 0.6)
                    y2 += int(height * 0.6)
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
                    deepsort_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))
               
                # Update DeepSORT tracker
                tracks = deepsort.update_tracks(deepsort_detections, frame=frame)
               
                # Determine crossing based on model ID
                log_entry = None
                if model_id == "1":
                    log_entry = determine_crossing_vehicle(
                        tracks, a, b, c, frame, camera_id, camera, model_state, state
                    )
                elif model_id == "2":
                    # For ANPR, process only the first confirmed track
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        bbox = track.to_tlbr()
                        centroid = calculate_centroid(bbox)
                        log_entry = determine_crossing_anpr(
                            model_state["prev_centroid"], centroid, a, b, c, frame, bbox, camera_id, camera, model_state, state
                        )
                        model_state["prev_centroid"] = centroid
                        break
               
                if log_entry:
                    success = append_to_json_log(log_entry, camera_id)
                    if success:
                        logger.info(f"Camera {camera_id} - Model {model_id} - Logged event: {log_entry}")
                    else:
                        logger.error(f"Camera {camera_id} - Model {model_id} - Failed to log event: {log_entry}")
               
                # Free memory
                del results, detections, deepsort_detections, tracks
                gc.collect()
           
            del frame
            gc.collect()
   
    except Exception as e:
        error_msg = f"Unexpected error in camera {camera_id}: {e}"
        logger.error(error_msg)
        report_error(camera_id, error_msg)
    finally:
        if cap is not None:
            cap.release()
        if camera_id in camera_states:
            del camera_states[camera_id]
        gc.collect()
        logger.info(f"Camera {camera_id} - Processing stopped")

def fetch_cameras():
    """Fetch camera data from the input API with retries."""
    max_retries = 5
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching camera data from {INPUT_API_URL} (Attempt {attempt + 1}/{max_retries})")
            response = requests.get(INPUT_API_URL, timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to fetch camera data: Status {response.status_code}, Response: {response.text}")
            cameras = response.json()
            logger.info(f"Received camera data: {len(cameras)} cameras")
            return cameras
        except (requests.RequestException, RuntimeError) as e:
            error_msg = f"Failed to fetch camera data: {e}"
            logger.error(error_msg)
            report_error("system", error_msg)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("Max retries reached. Will retry on next fetch cycle.")
                return []
    return []

def main():
    """Run processing continuously, fetching camera data every hour or on error confirmation."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS)
    active_cameras = set()
   
    try:
        while not stop_event.is_set():
            if fetch_trigger.is_set():
                logger.info("Immediate fetch triggered due to error confirmation")
                fetch_trigger.clear()
                cameras = fetch_cameras()
            else:
                cameras = fetch_cameras()
                time.sleep(FETCH_INTERVAL)
           
            # Group cameras by camera ID and collect model IDs
            camera_models = {}
            for cam in cameras:
                camera_id = cam["cameraId"]
                if camera_id not in camera_models:
                    camera_models[camera_id] = {"camera": cam, "model_ids": []}
                for model in cam.get("aiModels", []):
                    model_id = model["modelId"]
                    if model_id in MODEL_CONFIG and model_id not in camera_models[camera_id]["model_ids"]:
                        camera_models[camera_id]["model_ids"].append(model_id)
           
            # Log camera and model information
            for camera_id, info in camera_models.items():
                logger.info(f"Camera {camera_id} has modelIds: {info['model_ids']}")
           
            # Clean up inactive cameras
            new_camera_ids = set(camera_models.keys())
            for cam_id in active_cameras - new_camera_ids:
                if cam_id in camera_states:
                    if camera_states[cam_id]["shared"]["cap"] is not None:
                        camera_states[cam_id]["shared"]["cap"].release()
                    del camera_states[cam_id]
                    logger.info(f"Stopped processing for camera {cam_id} (no longer in input)")
           
            active_cameras = new_camera_ids
            for camera_id, info in camera_models.items():
                if camera_id not in camera_states:
                    executor.submit(process_camera, info["camera"], info["model_ids"])
                    logger.info(f"Started processing for camera {camera_id} with modelIds {info['model_ids']}")
           
            gc.collect()
   
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping all processing.")
        stop_event.set()
        executor.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}")
        report_error("system", f"Critical error in main loop: {e}")
    finally:
        stop_event.set()
        executor.shutdown(wait=True)
        for cam_id, states in camera_states.items():
            if states["shared"]["cap"] is not None:
                states["shared"]["cap"].release()
        camera_states.clear()
        gc.collect()
        logger.info("Application shutdown complete.")

if __name__ == "__main__":
    main()
