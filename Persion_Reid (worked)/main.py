import os
import cv2
import torch
import numpy as np
import time
import argparse
from pathlib import Path

# Import our utility modules
from utils.detector import setup_detector
from utils.feature_extractor import setup_feature_extractor
from utils.matcher import match_person, update_features
from utils.visualization import draw_results

def process_video(input_video_path, output_video_path, detector, feature_extractor, detection_interval=5, similarity_threshold=0.7):
    """
    Process video for person re-identification
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to save output video
        detector: Person detector
        feature_extractor: Feature extractor
        detection_interval: Process every N frames
        similarity_threshold: Threshold for matching persons
    """
    print(f"Processing video from {input_video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Person database: person_id -> features
    person_features = {}
    next_person_id = 0
    
    frame_count = 0
    detections_count = 0
    
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Processing video on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every N frames to reduce computational load
        if frame_count % detection_interval == 0:
            # Detect people in the frame
            detection_results = detector.detect(frame)
            detections_count += len(detection_results)
            
            for bbox in detection_results:
                x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
                
                # Extract person image
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0 or person_img.shape[0] < 10 or person_img.shape[1] < 10:
                    continue
                
                # Extract features
                features = feature_extractor.extract(person_img)
                
                # Match with existing persons or create new ID
                person_id = match_person(features, person_features, threshold=similarity_threshold)
                if person_id is None:
                    person_id = next_person_id
                    person_features[person_id] = features
                    next_person_id += 1
                else:
                    # Update features for existing person (moving average)
                    person_features[person_id] = update_features(person_features[person_id], features)
                
                # Draw bounding box and ID
                frame = draw_results(frame, x1, y1, x2, y2, person_id)
        
        # Add frame number display
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed
            estimated_remaining = (total_frames - frame_count) / fps_processing
            print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.2f} fps, est. {estimated_remaining:.2f}s remaining)")
    
    # Release resources
    cap.release()
    out.release()
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"Video processing complete in {elapsed:.2f}s ({frame_count/elapsed:.2f} fps)")
    print(f"Output saved to {output_video_path}")
    print(f"Total detections: {detections_count}")
    print(f"Total unique persons: {next_person_id}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Person Re-identification System')
    parser.add_argument('--input', type=str, default='data/videos/sample.mp4', help='Input video path')
    parser.add_argument('--output', type=str, default='data/results/output_reid.mp4', help='Output video path')
    parser.add_argument('--interval', type=int, default=5, help='Detection interval (every N frames)')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold for matching')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Check if input video exists
    if not os.path.exists(args.input):
        print(f"Input video not found: {args.input}")
        return
    
    print("Setting up models...")
    
    # Set up person detector
    detector = setup_detector()
    
    # Set up feature extractor
    feature_extractor = setup_feature_extractor()
    print("Models loaded successfully")
    
    # Process video
    process_video(args.input, args.output, detector, feature_extractor, 
                  detection_interval=args.interval, similarity_threshold=args.threshold)

if __name__ == "__main__":
    main()
