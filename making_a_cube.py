import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import time
t= time.time()

base = python.BaseOptions(model_asset_path= "hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options= base,
    num_hands= 2,
    min_tracking_confidence= 0.7,
    min_hand_presence_confidence= 0.7,
    min_hand_detection_confidence= 0.7
)

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
rectangle_coordinates = [0,0,0,0]
previous_x = 0
previous_y = 0
offset = 80
angle = 0
rotation_speed = 0
prev_x = 0
draw_rectangle = False
while cap.isOpened():
    r,fps = cap.read()
    if r == True:
        fps = cv2.resize(fps,(1080,860))
        fps = cv2.flip(fps,1)

        rgb_fps = cv2.cvtColor(fps,cv2.COLOR_BGR2RGB)
        mp_fps = mp.Image(mp.ImageFormat.SRGB,data= rgb_fps)
        detection_result = detector.detect(mp_fps)

        if detection_result.hand_landmarks:
            for landmarks in detection_result.hand_landmarks:
                for landmark in landmarks:
                    x = int(landmark.x * fps.shape[1])
                    y = int(landmark.y * fps.shape[0])

                    cv2.circle(fps,(x,y),8,(0,0,255),-1)
                    cv2.circle(fps,(x,y),12,(0,0,255),1)
                    
                    connections = mp.solutions.hands.HAND_CONNECTIONS

                    for connection in connections:
                        start = connection[0]
                        end = connection[1]
                        
                        start_pt = landmarks[start]
                        end_pt = landmarks[end]

                        pt1 = (int(start_pt.x * fps.shape[1]),int(start_pt.y * fps.shape[0]))
                        pt2 = (int(end_pt.x * fps.shape[1]),int(end_pt.y * fps.shape[0]))

                        cv2.line(fps,pt1,pt2,(255,255,255),3,)
                    
            if len(detection_result.hand_landmarks) >= 1:
                pinches = []
                for hand in detection_result.hand_landmarks:
                    x_thumb, y_thumb = int(hand[4].x * fps.shape[1]), int(hand[4].y * fps.shape[0])
                    x_index, y_index = int(hand[8].x * fps.shape[1]), int(hand[8].y * fps.shape[0])
                    dist = math.sqrt(pow((x_thumb - x_index), 2) + pow((y_thumb - y_index), 2))
                    pinches.append((dist < 30, x_thumb, y_thumb))

                if len(pinches) == 2 and pinches[0][0] and pinches[1][0]:
                    x1, y1 = pinches[0][1], pinches[0][2]
                    x4, y4 = pinches[1][1], pinches[1][2]
                    
                    rectangle_coordinates = [min(x1, x4), min(y1, y4), max(x1, x4), max(y1, y4)]
                    saved = rectangle_coordinates
                    draw_rectangle = True
                    previous_x = 0 
                else:
                    moved = False
                    for is_pinch, finger_x, finger_y in pinches:
                        if is_pinch and draw_rectangle:
                            if rectangle_coordinates[0] < finger_x < rectangle_coordinates[2] and rectangle_coordinates[1] < finger_y < rectangle_coordinates[3]:
                                if previous_x != 0 and previous_y != 0:
                                    dx = finger_x - previous_x
                                    dy = finger_y - previous_y

                                    rectangle_coordinates[0] += dx
                                    rectangle_coordinates[1] += dy
                                    rectangle_coordinates[2] += dx
                                    rectangle_coordinates[3] += dy

                                previous_x = finger_x
                                previous_y = finger_y
                                moved = True
                                break 
                    if not moved:
                        previous_x = 0
                        previous_y = 0
        else:
            previous_x = 0
            previous_y = 0

        if draw_rectangle:
            cube_centre_x = (rectangle_coordinates[0] + rectangle_coordinates[2])//2
            cube_centre_y = (rectangle_coordinates[1] + rectangle_coordinates[3])//2
            cube_size = (rectangle_coordinates[2] - rectangle_coordinates[0]) // 2

            if len(detection_result.hand_landmarks) > 0:
                hand1 = detection_result.hand_landmarks[0]
                fing_x = int(hand1[8].x * fps.shape[1]) 
                fing_y = int(hand1[8].y * fps.shape[0])
                
                if prev_x != 0:
                    velocity = fing_x - prev_x
                    if rectangle_coordinates[0] < fing_x < rectangle_coordinates[2] and rectangle_coordinates[1] < fing_y < rectangle_coordinates[3]:
                        if abs(velocity) > 5: 
                            rotation_speed = velocity * 0.005
                prev_x = fing_x
            else:
                prev_x = 0

            angle += rotation_speed
            rotation_speed *= 0.95 

            points = []
            for x, y in [(-cube_size, -cube_size),( cube_size, -cube_size),( cube_size,  cube_size),(-cube_size,  cube_size)]:
                rotated_x = x * math.cos(angle) - y * math.sin(angle)
                rotated_y = x * math.sin(angle) + y * math.cos(angle) 

                screen_x = int(rotated_x + cube_centre_x)
                screen_y = int(rotated_y + cube_centre_y)
                points.append((screen_x,screen_y))

            points_back = [(p[0] + offset, p[1] - offset) for p in points]

            for i in range(4):
                cv2.line(fps, points[i], points[(i+1)%4], (0,0,255), 3)

            for i in range(4):
                cv2.line(fps, points_back[i], points_back[(i+1)%4], (0,0,255), 3)

            for i in range(4):
                cv2.line(fps, points[i], points_back[i], (0,0,255), 3)
                
        cv2.imshow("window",fps)
        if cv2.waitKey(25) & 0xff == ord('q'):
            break
    else:
        break
detector.close()
cv2.destroyAllWindows()
cap.release() 