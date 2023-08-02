import math
import numpy as np
def findangle(kpts, p1, p2, p3)-> int:
    
    coord = []
    no_kpt = len(kpts) // 3
    for i in range(no_kpt):
        cx, cy, conf = kpts[3*i : 3*i+3]
        coord.append([i, cx, cy, conf])

    points = (p1, p2, p3)

    if all(p in range(no_kpt) for p in points):
        x1, y1 = coord[p1][1:3]
        x2, y2 = coord[p2][1:3]
        x3, y3 = coord[p3][1:3]
        

        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        
        if angle > 180.0:
            angle = 360 - angle
            
        if p1==5 and p2==7 and p3 == 9: 
            ALE=int(abs(angle))
            return ALE
        
        if p1==6 and p2==8 and p3 == 10:
            ARE=int(abs(angle))
            return ARE
        
        if p1==7 and p2==5 and p3 == 11:
            ALS=int(abs(angle))
            return ALS
        
        if p1==8 and p2==6 and p3 == 12:
            ARS=int(abs(angle))
            return ARS
        
        if p1==5 and p2==11 and p3 == 13:
            ALH=int(abs(angle))
            return ALH
        
        if p1==11 and p2==13 and p3 == 15:
            ALK=int(abs(angle))
            return ALK
        
        if p1==6 and p2==12 and p3 == 14:
            ARH=int(abs(angle))
            return ARH