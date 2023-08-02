import cv2
def blue_background(im0,flag):
    #blue top rectangle
    if flag ==True:
        cv2.rectangle(im0, (0,0), (1280,115), (245,117,16), -1)

def ankle_dist_warning(im0,flag):
    if flag ==True:
        cv2.putText(im0, "please keep your foot wider", (48, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)  
 
def left_shoulder_warning(im0,flag):
    if flag ==True:
        cv2.putText(im0, "please close your left shoulder", (48, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)
        
def right_shoulder_warning(im0,flag):
    if flag ==True:
        cv2.putText(im0, "please close your right shoulder", (640, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)

def counters_and_stages_left(im0, flag, stage_left, counter_left):
    if flag ==True:
        # drawing stage and counter - left
        cv2.putText(im0, f"Stage: {stage_left}", (48, 48), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 3,cv2.LINE_AA)
        cv2.putText(im0, f"Reps: {str(counter_left)}", (48, 96), cv2.FONT_HERSHEY_SIMPLEX, 2,(225, 225, 225), 3, cv2.LINE_AA)
        
def counters_and_stages_right(im0, flag,  stage_right,counter_right):
    if flag ==True:
        # drawing stage and counter - left
        cv2.putText(im0, f"Stage: {stage_right}", (640, 48), cv2.FONT_HERSHEY_SIMPLEX, 2, (225, 225, 225), 3,cv2.LINE_AA)
        cv2.putText(im0, f"Reps: {str(counter_right)}", (640, 96), cv2.FONT_HERSHEY_SIMPLEX, 2,(225, 225, 225), 3, cv2.LINE_AA)

def pu_back_warning(im0,flag):
    if flag == True:
        cv2.putText(im0, "Keep your back straight.", (640, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)

def pu_wrist_dist_warning(im0,flag):
    if flag == True:
        cv2.putText(im0, "keep hands at the same level as shoulders ", (48, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)
       
def speed_warning(im0,flag):
    if flag == True:
        cv2.putText(im0, "make your hip and knee angle changes in the same speed.", (48, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)
        
def wrist_dist(im0,flag):
    if flag == True:
        cv2.putText(im0, "please keep distance between wrists is the same as shoulders.", (48, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)
        