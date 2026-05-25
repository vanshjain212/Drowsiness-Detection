from utils import euclidean

def calculate_ear(landmark, indices, width, height):
    """Calculates the Eye Aspect Ratio."""
    points = [(landmark[idx].x * width, landmark[idx].y * height) for idx in indices]
    p1, p2, p3, p4, p5, p6 = points
    
    v1 = euclidean(p2, p6)
    v2 = euclidean(p3, p5)
    v3 = euclidean(p1, p4)
    return (v1 + v2) / (2.0 * v3)

def calculate_mar(landmark, indices, width, height):
    """Calculates the Mouth Aspect Ratio using inner lips."""
    points = [(landmark[idx].x * width, landmark[idx].y * height) for idx in indices]
    p1, p2, p3, p4, p5, p6, p7, p8 = points
    
    v1 = euclidean(p2, p8)
    v2 = euclidean(p3, p7)
    v3 = euclidean(p4, p6)
    horizontal = euclidean(p1, p5)
    
    if horizontal == 0: return 0.0
    return (v1 + v2 + v3) / (2.0 * horizontal)