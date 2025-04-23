import cv2
import torch
import numpy as np

class OSNetExtractor:
    """
    Feature extractor using OSNet from torchreid
    """
    def __init__(self):
        """Initialize the OSNet feature extractor"""
        try:
            # Try importing directly from the torchreid package
            import torchreid
            from torchreid.utils import FeatureExtractor as TorchreidExtractor
            
            # Initialize feature extractor
            self.extractor = TorchreidExtractor(
                model_name='osnet_x1_0',
                model_path='',  # Empty path will use default weights
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print(f"OSNet feature extractor initialized on {'GPU' if torch.cuda.is_available() else 'CPU'}")
        except ImportError:
            # Alternative import path
            try:
                import torchreid
                # Check if the structure matches the deep-person-reid GitHub repo
                from torchreid.reid.utils.feature_extractor import FeatureExtractor as TorchreidExtractor
                
                self.extractor = TorchreidExtractor(
                    model_name='osnet_x1_0',
                    model_path='',
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print(f"OSNet feature extractor initialized (alternative import) on {'GPU' if torch.cuda.is_available() else 'CPU'}")
            except ImportError as e:
                raise ImportError(f"Failed to initialize OSNet extractor: {e}")
    
    def extract(self, image):
        """
        Extract features from person image
        
        Args:
            image: OpenCV image in BGR format
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Preprocess image
            img = cv2.resize(image, (128, 256))  # Common size for person re-id
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract features using OSNet
            features = self.extractor(img)
            return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

def setup_feature_extractor():
    """Set up and return feature extractor"""
    return OSNetExtractor()