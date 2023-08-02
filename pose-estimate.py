import math
import cv2
import time
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt,plot_labels
import calculateAngle
import calculateDistance
import GUIs
from pygame import mixer
import seaborn as sns
@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="bi.mp4", device='cpu' ,view_img=True, save_conf=False,line_thickness = 3,hide_labels=True, hide_conf=True):
    
    ##  count system   ##
    stage_left = None
    counter_left = 0
    stage_right = None
    counter_right = 0
    
    ##  points system   ##
    angle_max: int = 150
    angle_min: int = 30
    threshold: int = 35

    # Create a list to store the rate of change of hip and knee angle.
    rate_of_change_list = []
    ##  points system   ##
    score=0

    
    ## points system v2.0 ##
    left_arm_score = 0
    right_arm_score = 0
    
    ##  GUI Flag   ##
    gui_flag =True
    
    ##  Voice over Flag   ##
    sound_flag_cong=True
    sound_flag_one_count_left=True
    sound_flag_good_job=True
    
    device = select_device(opt.device) 

    model = attempt_load(poseweights, map_location=device) 
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names 
    
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source))    
        
    else :
        cap = cv2.VideoCapture(source)    
   
    if (cap.isOpened() == False):  
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    else:
        
        frame_width = 1280 #int(cap.get(3)) #1280  #get video frame width
        frame_height = 720 #int(cap.get(4)) #720 #get video frame height
        
        
        
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] 
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))
        while(cap.isOpened): 


            
            ret, frame = cap.read()  
            if ret: 
                orig_image = frame 
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) 
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  
                #image = image.float() 
                start_time = time.time() 
            
                with torch.no_grad():  #get predictions
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                            0.25,   # Conf. Threshold.
                                            0.65, # IoU Threshold.
                                            nc=model.yaml['nc'], # Number of classes.
                                            nkpt=model.yaml['nkpt'], # Number of keypoints.
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)
                
                im0 = image[0].permute(1, 2, 0) * 255 
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) 
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
                for i, pose in enumerate(output_data):  
                
                    if len(output_data):  
                        for c in pose[:, 5].unique(): 
                            n = (pose[:, 5] == c).sum()  
                            
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])):               
                            
                            
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            #find and print distance
                            shoulder_dist=calculateDistance.find_distance(kpts,5,6) #shoulder keypoints distance
                            eldow_dist=calculateDistance.find_distance(kpts,7,8) #elbow keypoints distance
                            wrist_dist=calculateDistance.find_distance(kpts,9,10)#wrist keypoints distance
                            
                            distance =calculateDistance.ensure_distance(shoulder_dist,eldow_dist,wrist_dist)

                            
                            #blue top rectangle
                            GUIs.blue_background(im0,gui_flag)
                            
                            ######################################################################################################################
                            #                                           Biceps                                                                   #
                            ######################################################################################################################
                            # elbow joints
                            angle_left=calculateAngle.findangle(kpts, 5, 7, 9)
                            angle_right=calculateAngle.findangle(kpts, 6, 8, 10)
                            # shoulder joints
                            arm_left=calculateAngle.findangle(kpts, 7, 5, 11)
                            arm_right=calculateAngle.findangle(kpts, 8, 6, 12)
                            #cv2.putText(im0, f"Angle: {angle_left}", (48, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)
                            #cv2.putText(im0, f"Angle: {angle_right}", (500, 144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3,cv2.LINE_AA)    
                            if wrist_dist <= shoulder_dist+12 and wrist_dist >= shoulder_dist-12 :
                                              
                                if (arm_left is not None and arm_left < threshold):
                                    if angle_left > angle_max:
                                        stage_left = 'down'
                                    if angle_left < angle_min and stage_left == 'down':
                                        stage_left = 'up'
                                        counter_left += 1
                                else:
                                    stage_left = "skip"
                                    GUIs.left_shoulder_warning(im0,gui_flag)


                                if (arm_right is not None and arm_right < threshold):
                                    if angle_right > angle_max:
                                        stage_right = 'down'
                                    if angle_right < angle_min and stage_right == 'down':
                                        stage_right = 'up'
                                        counter_right += 1


                                else:
                                    stage_right = "skip"
                                    GUIs.right_shoulder_warning(im0,gui_flag)
                            else:
                                stage_right = "skip"
                                stage_left = "skip"
                                GUIs.wrist_dist(im0,gui_flag)
                                                   
                            GUIs.counters_and_stages_left(im0, gui_flag, stage_left,counter_left)
                            GUIs.counters_and_stages_right(im0, gui_flag, stage_right,counter_right)
                            
                            ######################################################################################################################
                            ######################################################################################################################
                            
                            ######################################################################################################################
                            #                                           Squats                                                                   #
                            ######################################################################################################################
                            """
                            left_hip=calculateAngle.findangle(kpts, 5, 11, 13)
                            left_knee=calculateAngle.findangle(kpts, 11, 13, 15)
                            ankle_dist=calculateDistance.find_distance(kpts,15,16) #elbow keypoints distance

                            difference = left_hip - left_knee
                            # Calculate the rate of change between hip angle and knee angle.
                            rate_of_change = difference / 0.5
                            # Add the rate of change of hip and knee angle to the list.
                            rate_of_change_list.append(rate_of_change)
                            if ankle_dist <= shoulder_dist+10 and ankle_dist >= shoulder_dist:
                                if left_hip > 160:
                                    stage_left = 'up'
                                        
                                if left_hip < 50 and stage_left == 'up' and left_knee <90:
                                    if ankle_dist <= shoulder_dist+20:
                                        counter_left += 1
                                        stage_left = 'down'
                                        score+=1
                            else:
                                stage_left = "skip"
                                GUIs.ankle_dist_warning(im0,gui_flag)
                                score -= 1
                            #if rate_of_change > 50:
                            #    stage_left="skip"
                            #    GUIs.speed_warning(im0,gui_flag)
                            GUIs.counters_and_stages_left(im0, gui_flag, stage_left,counter_left)
                            """
                            
                            ######################################################################################################################
                            ######################################################################################################################
                            
                            ######################################################################################################################
                            #                                           push ups                                                                 #
                            ######################################################################################################################
                            '''
                            # those angles should be between 175 and 180 "to keep body straight" #
                            left_hip=calculateAngle.findangle(kpts, 5, 11, 13)
                            right_hip=calculateAngle.findangle(kpts, 6, 12, 14)
                            
                            #those angles are for shoulders and should be between 42 and 47 #
                            arm_left=calculateAngle.findangle(kpts, 7, 5, 11)
                            arm_right=calculateAngle.findangle(kpts, 8, 6, 12)
                            
                            #those angles are for the elbow and should incline as much as the trainer can #
                            angle_left=calculateAngle.findangle(kpts, 5, 7, 9)
                            angle_right=calculateAngle.findangle(kpts, 6, 8, 10)
                            
                            # counting and showing warning messages
                            if (left_hip  >160 ):
                                if wrist_dist <= shoulder_dist+20 and wrist_dist >= shoulder_dist:
                                    if angle_left > 150 :
                                        stage_left = 'up'
                                    if angle_left < 50 and stage_left == 'up':
                                        stage_left = 'down'
                                        counter_left += 1

                                else:
                                    stage_left = "skip"
                                    GUIs.pu_wrist_dist_warning(im0,gui_flag)
 
                            elif (left_hip  <170):
                                stage_left = "skip"
                                GUIs.pu_back_warning(im0,gui_flag)

                            GUIs.counters_and_stages_left(im0, gui_flag, stage_left,counter_left)
                            '''
                            ######################################################################################################################
                            ######################################################################################################################                            
                            
                            
                            
                            
                            ######################################################################################################################
                            #                                           Voice Over                                                               #
                            ######################################################################################################################
                            if counter_left == 10 and sound_flag_cong==True:
                                gui_flag=False
                                mixer.init()
                                sound = mixer.Sound('audios/goodjob.wav')
                                sound_flag_cong = False
                                sound.play()
                            
                            if counter_left == 5 and sound_flag_good_job==True:
                                mixer.init()
                                sound2 = mixer.Sound('audios/KeepGoing.wav')
                                sound_flag_good_job = False
                                sound2.play()
                            
                            if counter_left == 9 and sound_flag_one_count_left==True:
                                mixer.init()
                                sound2 = mixer.Sound('audios/OneCountLeft.wav')
                                sound_flag_one_count_left = False
                                sound2.play()
                                
                            ##############################################################################################
                            ##############################################################################################
                            
                            ######################################################################################################################
                            #                                           reset counter                                                            #
                            ######################################################################################################################
                            if counter_left == 10 or counter_right ==10:
                                cv2.putText(im0, "congrates you have done the exercise", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,cv2.LINE_AA)
                                key1 = cv2.waitKey(1)
                                if key == ord('r'):
                                    gui_flag = True
                                    sound_flag_good_job = True
                                    sound_flag_cong = True
                                    sound_flag_one_count_left = True
                                    stage_left = None
                                    counter_left = 0
                                    stage_right = None
                                    counter_right = 0
                            ##############################################################################################
                            ##############################################################################################
                            
                            
                            temp_var = names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}'
                            label = None if opt.hide_labels else temp_var
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                        line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts, steps=3,
                                        orig_shape=im0.shape[:2])
                        

                # Stream results
                if view_img:
                    cv2.imshow("Smart Gym Trainer", im0)
                    cv2.waitKey(1)  # 1 millisecond
                    
                out.write(im0)  #writing the video frame
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            else:
                break 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='video/0 for webcam') 
    parser.add_argument('--device', type=str, default='0', help='cpu/0,1,2,3(gpu)')   
    parser.add_argument('--view-img', default='0',action='store_true', help='display results') 
    parser.add_argument('--save-conf', default=True,action='store_true', help='save confidences in --save-txt labels') 
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels') 
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    return opt

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
