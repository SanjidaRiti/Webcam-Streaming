import numpy as np
import torch

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two feature vectors
    
    Args:
        vec1, vec2: Feature vectors
        
    Returns:
        Similarity score between 0 and 1 (higher is more similar)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def match_person(features, person_features, threshold=0.7):
    """
    Match a person with existing database
    
    Args:
        features: Feature vector of the current person
        person_features: Dictionary of person_id -> feature vectors
        threshold: Matching threshold (default 0.7)
        
    Returns:
        Matched person ID or None if no match
    """
    if features is None or len(person_features) == 0:
        return None
    
    # Convert features to numpy array if needed
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    # Reshape if needed
    query_features = features.reshape(1, -1) if features.ndim == 1 else features
    
    best_match_id = None
    best_match_similarity = -1
    
    # Compare with each person in database
    for person_id, stored_features in person_features.items():
        if stored_features is None:
            continue
        
        # Reshape if needed
        stored = stored_features.reshape(1, -1) if stored_features.ndim == 1 else stored_features
        
        # Calculate similarity (cosine similarity)
        similarity = cosine_similarity(query_features.flatten(), stored.flatten())
        
        if similarity > threshold and similarity > best_match_similarity:
            best_match_similarity = similarity
            best_match_id = person_id
    
    return best_match_id

def update_features(old_features, new_features, alpha=0.7):
    """
    Update features for a person using moving average
    
    Args:
        old_features: Previous feature vector
        new_features: New feature vector
        alpha: Weight for old features (default 0.7)
        
    Returns:
        Updated feature vector
    """
    if old_features is None:
        return new_features
    if new_features is None:
        return old_features
    
    # Simple moving average
    updated = alpha * old_features + (1 - alpha) * new_features
    return updated