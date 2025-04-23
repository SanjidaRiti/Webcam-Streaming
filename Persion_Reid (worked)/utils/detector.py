# import cv2
# import numpy as np

# class SimulatedDetector:
#     """
#     A simple detector that simulates person detection for testing purposes.
#     In production, you would replace this with a real detector like YOLOv5.
#     """
#     def __init__(self):
#         print("Using simulated detector - replace with YOLOv5 for production use")
    
#     def detect(self, frame):
#         """
#         Simulate person detection
#         Returns list of bounding boxes: [x1, y1, x2, y2, confidence]
#         """
#         height, width = frame.shape[:2]
        
#         # Simulating 1-3 random detections
#         num_detections = np.random.randint(1, 4)
#         detections = []
        
#         for _ in range(num_detections):
#             # Generate random box (ensure it's not too small)
#             x1 = np.random.randint(0, width - 100)
#             y1 = np.random.randint(0, height - 200)
#             w = np.random.randint(100, min(200, width - x1))
#             h = np.random.randint(150, min(300, height - y1))
#             x2 = x1 + w
#             y2 = y1 + h
#             confidence = np.random.uniform(0.7, 0.98)
            
#             detections.append([x1, y1, x2, y2, confidence])
        
#         return detections

# class YOLOv5Detector:
#     """
#     YOLOv5 detector for person detection
#     """
#     def __init__(self):
#         try:
#             import torch
#             self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#             self.model.classes = [0]  # Person class only
#             print("YOLOv5 detector initialized successfully")
#         except Exception as e:
#             raise ImportError(f"Failed to initialize YOLOv5: {e}")
    
#     def detect(self, frame):
#         """
#         Detect people in frame using YOLOv5
#         Returns list of bounding boxes: [x1, y1, x2, y2, confidence]
#         """
#         results = self.model(frame)
        
#         # Extract person detections
#         detections = []
#         for *box, conf, cls in results.xyxy[0].cpu().numpy():
#             if cls == 0:  # Person class
#                 detections.append([*box, conf])
                
#         return detections

# def setup_detector():
#     """
#     Set up and return the appropriate detector
#     Returns YOLOv5 if available, otherwise a simulated detector
#     """
#     try:
#         # Try to set up YOLOv5 detector
#         detector = YOLOv5Detector()
#         return detector
#     except ImportError:
#         print("YOLOv5 not available, using simulated detector")
#         return SimulatedDetector()

import cv2
import numpy as np

class SimulatedDetector:
    """
    A simple detector that simulates person detection for testing purposes.
    """
    def __init__(self):
        print("Using simulated detector for testing")
    
    def detect(self, frame):
        """
        Simulate person detection
        Returns list of bounding boxes: [x1, y1, x2, y2, confidence]
        """
        height, width = frame.shape[:2]
        
        # Simulating 1-3 random detections
        num_detections = np.random.randint(1, 4)
        detections = []
        
        for _ in range(num_detections):
            # Generate random box (ensure it's not too small)
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 200)
            w = np.random.randint(100, min(200, width - x1))
            h = np.random.randint(150, min(300, height - y1))
            x2 = x1 + w
            y2 = y1 + h
            confidence = np.random.uniform(0.7, 0.98)
            
            detections.append([x1, y1, x2, y2, confidence])
        
        return detections

class OpenCVHOGDetector:
    """
    Person detector using OpenCV's HOG detector
    """
    def __init__(self):
        # Initialize OpenCV's HOG detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("OpenCV HOG person detector initialized")
    
    def detect(self, frame):
        """
        Detect people in frame using HOG detector
        Returns list of bounding boxes: [x1, y1, x2, y2, confidence]
        """
        # Resize frame for faster detection
        height, width = frame.shape[:2]
        
        # Don't resize if image is already small
        if width > 640:
            scale = 640.0 / width
            frame_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        else:
            frame_resized = frame
            scale = 1.0
        
        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            frame_resized, 
            winStride=(8, 8),
            padding=(16, 16), 
            scale=1.05
        )
        
        # Convert to format [x1, y1, x2, y2, confidence]
        detections = []
        for (x, y, w, h), confidence in zip(boxes, weights):
            # Scale back to original image size
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + w) / scale)
            y2 = int((y + h) / scale)
            
            detections.append([x1, y1, x2, y2, float(confidence)])
        
        return detections

class HaarCascadeDetector:
    """
    Alternative person detector using OpenCV's Haar Cascade classifier
    """
    def __init__(self):
        # Initialize cascade classifier
        try:
            # Try to load the full body cascade
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            print("OpenCV Haar Cascade person detector initialized")
        except Exception as e:
            raise ImportError(f"Failed to initialize Haar Cascade detector: {e}")
    
    def detect(self, frame):
        """
        Detect people in frame using Haar Cascade
        Returns list of bounding boxes: [x1, y1, x2, y2, confidence]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect people
        bodies = self.body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 80)
        )
        
        # Convert to format [x1, y1, x2, y2, confidence]
        detections = []
        for (x, y, w, h) in bodies:
            # Add a placeholder confidence (1.0)
            detections.append([x, y, x+w, y+h, 1.0])
        
        return detections

def setup_detector():
    """
    Set up and return the appropriate detector
    """
    try:
        # Try OpenCV's HOG detector first (more accurate)
        detector = OpenCVHOGDetector()
        return detector
    except Exception as e:
        print(f"HOG detector initialization failed: {e}")
        
        try:
            # Try Haar Cascade as a backup
            detector = HaarCascadeDetector()
            return detector
        except Exception as e:
            print(f"Haar Cascade detector initialization failed: {e}")
            print("Falling back to simulated detector")
            return SimulatedDetector()