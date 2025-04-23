import cv2
import numpy as np

def draw_results(frame, x1, y1, x2, y2, person_id, color=(0, 255, 0)):
    """
    Draw bounding box and person ID on the frame
    
    Args:
        frame: OpenCV image
        x1, y1, x2, y2: Bounding box coordinates
        person_id: Person ID to display
        color: Color for bounding box and ID text
        
    Returns:
        Frame with annotations
    """
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Create a black background for text for better visibility
    text_bg_size = cv2.getTextSize(f"ID: {person_id}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_bg_size[0], y1), (0, 0, 0), -1)
    
    # Draw person ID
    cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame