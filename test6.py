import cv2
import numpy as np
from ultralytics import YOLO
import logging
import sys
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "yolov8l.pt"  # Path to YOLO model
CLASSES = [1, 2, 3, 5, 7]  # Vehicle classes: 1=motorcycle, 2=car, 3=motorcycle, 5=bus, 7=truck
CROSSING_TOLERANCE = 10  # Pixels, tolerance for line crossing detection
OUTPUT_VIDEO_PATH = "night_video.mp4"  # Output video file name
FRAME_SKIP = 5  # Process every 5th frame

# Global variables for ROI selection
roi_points = []
click_count = 0
roi_selected = False

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

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select ROI points in the bottom 40%."""
    global click_count, roi_points, roi_selected
    frame_height = param[0]
    if event == cv2.EVENT_LBUTTONDOWN and click_count < 2 and y > frame_height * 0.6:
        roi_points.append((x, y))
        click_count += 1
        logger.info(f"ROI point {click_count} selected: ({x}, {y})")
        if click_count == 2:
            roi_selected = True
            logger.info("ROI selection completed")

def select_roi(frame, window_name):
    """Allow user to manually select two ROI points by clicking on the bottom 40% of the frame."""
    frame_height = frame.shape[0]
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, param=(frame_height,))
    
    while not roi_selected:
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], int(frame_height * 0.6)), (0, 0, 0), -1)  # Black out upper 60%
        for i, point in enumerate(roi_points):
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"Point {i+1}", (point[0] + 10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(roi_points) == 2:
            cv2.line(display_frame, roi_points[0], roi_points[1], (0, 0, 255), 2)
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            logger.error("ROI selection cancelled by user")
            sys.exit(1)
    
    cv2.setMouseCallback(window_name, lambda *args: None)  # Disable mouse callback
    return roi_points

def determine_crossing_vehicle(tracks, a, b, c, state):
    """Determine if vehicle centroids cross the ROI line and update crossing count."""
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
        
        if not track_state['has_crossed']:
            if prev_dist > CROSSING_TOLERANCE and curr_dist <= CROSSING_TOLERANCE:
                state['crossing_count'] += 1
                track_state['has_crossed'] = True
                logger.info(f"Vehicle {track_id} crossed ROI line. Total crossings: {state['crossing_count']}")
            elif prev_dist < -CROSSING_TOLERANCE and curr_dist >= -CROSSING_TOLERANCE:
                state['crossing_count'] += 1
                track_state['has_crossed'] = True
                logger.info(f"Vehicle {track_id} crossed ROI line. Total crossings: {state['crossing_count']}")
        
        if abs(curr_dist) > CROSSING_TOLERANCE * 2:
            track_state['has_crossed'] = False
        
        track_state['prev_centroid'] = curr_centroid

def main(video_path):
    """Process video input, display vehicle crossing counts in real-time, and save output video."""
    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    # Initialize DeepSORT
    deepsort = DeepSort(max_age=10, nn_budget=100, override_track_class=None)

    # Initialize state
    state = {
        "crossing_count": 0,
        "track_states": {}
    }

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    logger.info(f"Video - Width: {width}, Height: {height}, FPS: {fps}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.error(f"Failed to initialize video writer for {OUTPUT_VIDEO_PATH}")
        cap.release()
        sys.exit(1)

    # Read first frame for ROI selection
    ret, first_frame = cap.read()
    if not ret:
        logger.error("Failed to read first frame")
        cap.release()
        out.release()
        sys.exit(1)

    # Resize frame if resolution is low
    if width < 640 or height < 480:
        scale_factor = 2
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        first_frame = cv2.resize(first_frame, (width, height))

    # Select ROI manually in the bottom 40%
    global roi_points, roi_selected
    roi_points = select_roi(first_frame, "Select ROI (Click 2 points in bottom 40%, press ESC to cancel)")
    if len(roi_points) != 2:
        logger.error("Invalid ROI points selected")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        sys.exit(1)

    # Calculate line parameters
    x1, y1 = roi_points[0]
    x2, y2 = roi_points[1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    # Calculate threshold lines for visualization (±CROSSING_TOLERANCE)
    norm = np.sqrt(a**2 + b**2)
    if norm == 0:
        logger.error("Invalid ROI line (zero length)")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        sys.exit(1)
    # Offset for threshold lines: c' = c ± CROSSING_TOLERANCE * norm
    c_pos = c + CROSSING_TOLERANCE * norm
    c_neg = c - CROSSING_TOLERANCE * norm
    # Compute points for threshold lines (solve for x at y=0 and y=height)
    def get_line_points(a, b, c, height, width):
        points = []
        if b != 0:
            x1 = int(-c / b)  # y=0
            x2 = int(-(c + a * height) / b)  # y=height
            points = [(x1, 0), (x2, height)]
        else:
            y1 = int(-c / a)  # x=0
            y2 = int(-(c + b * width) / a)  # x=width
            points = [(0, y1), (width, y2)]
        # Clip points to frame boundaries
        for i, (x, y) in enumerate(points):
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            points[i] = (x, y)
        return points

    pos_threshold_points = get_line_points(a, b, c_pos, height, width)
    neg_threshold_points = get_line_points(a, b, c_neg, height, width)
    logger.info(f"Positive threshold points: {pos_threshold_points}")
    logger.info(f"Negative threshold points: {neg_threshold_points}")

    # Initialize frame counter and last tracks
    frame_counter = 0
    last_tracks = []

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video or failed to read frame")
            break

        frame = cv2.resize(frame, (width, height))
        
        # Process only every FRAME_SKIP frame
        tracks = last_tracks  # Use last processed tracks by default
        if frame_counter % FRAME_SKIP == 0:
            # Crop to bottom 40%
            bottom_40 = frame[int(height * 0.6):, :, :]

            # Perform vehicle detection on bottom 40%
            try:
                results = model(bottom_40, classes=CLASSES)
                detections = results[0].boxes.data.cpu().numpy()
            except Exception as e:
                logger.error(f"Vehicle detection failed: {e}")
                frame_counter += 1
                continue

            # Adjust bounding box coordinates to full frame
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

            # Determine crossings
            determine_crossing_vehicle(tracks, a, b, c, state)

            # Update last tracks
            last_tracks = tracks

        # Draw bounding boxes, track IDs, and centroids
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw centroid
            centroid = calculate_centroid(bbox)
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)

        # Draw ROI line (red)
        cv2.line(frame, roi_points[0], roi_points[1], (0, 0, 255), 2)

        # Draw threshold lines (yellow, thicker for visibility)
        cv2.line(frame, pos_threshold_points[0], pos_threshold_points[1], (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(frame, neg_threshold_points[0], neg_threshold_points[1], (255, 255, 0), 2, cv2.LINE_AA)

        # Display crossing count
        cv2.putText(frame, f"Crossings: {state['crossing_count']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Write processed frame to output video
        out.write(frame)

        # Show frame
        cv2.imshow("Vehicle Crossing Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

        frame_counter += 1

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info(f"Output video saved as {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Crossing Counter")
    parser.add_argument("video_path", help="Path to the input video file")
    args = parser.parse_args()
    main(args.video_path)