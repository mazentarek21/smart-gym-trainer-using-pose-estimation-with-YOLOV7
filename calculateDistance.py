import math
import cv2
import numpy as np
def find_distance(kpts, p1, p2)-> int:
    
    coord = []
    no_kpt = len(kpts) // 3
    for i in range(no_kpt):
        cx, cy, conf = kpts[3*i : 3*i+3]
        coord.append([i, cx, cy, conf])

    points = (p1, p2)

    if all(p in range(no_kpt) for p in points):
        x1, y1 = coord[p1][1:3]
        x2, y2 = coord[p2][1:3]
        # calculate the Euclidean distance between the two keypoints
        distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        pixels_to_cm = 0.1  # assume 1 pixel = 0.1 cm
        distance_cm = distance_pixels * pixels_to_cm
        
        return distance_cm
        
        
#x -> shoulders distance
#y -> elbows distance
#z-> wrist distance

def ensure_distance(x,y,z):
    if (y and z) < x+10:
        return True
    elif(y and z) > x+10:
        return False
              
    