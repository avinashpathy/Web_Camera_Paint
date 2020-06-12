import numpy as np
import cv2
from collections import deque
yLower = np.array([20, 100, 100])
yUpper = np.array([30,255,255])



bp = [deque(maxlen=512)]
gp = [deque(maxlen=512)]
rp = [deque(maxlen=512)]
yp = [deque(maxlen=512)]

bi,gi,ri,yi = 0,0,0,0
colors = [ (0, 255, 0), (0, 0, 255)] 
colorIndex = 0
pWindow = np.zeros((600,800,3)) 

pWindow = cv2.rectangle(pWindow, (40,1), (140,65), (255,255,255), 2)
pWindow = cv2.rectangle(pWindow, (300,1), (400,65), colors[0], -1)
pWindow = cv2.rectangle(pWindow, (605,1), (700,65), colors[1], -1)


cv2.putText(pWindow, "ERASE", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)
cv2.putText(pWindow, "GREEN", (310, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)
cv2.putText(pWindow, "RED", (630, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)


cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not ret:
        break

    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    frame = cv2.rectangle(frame, (275,1), (370,65), colors[0], -1)
    frame = cv2.rectangle(frame, (500,1), (600,65), colors[1], -1)
    cv2.putText(frame, "ERASE", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(frame, "RED", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)

    
    kernel = np.ones((5,5),np.uint8)
    b_Mask = cv2.inRange(hsv,yLower,yUpper)
    b_Mask = cv2.erode(b_Mask,kernel,iterations = 1)
    b_Mask = cv2.morphologyEx(b_Mask,cv2.MORPH_OPEN,kernel)
    b_Mask = cv2.dilate(b_Mask, kernel, iterations=1)

    (cnts, _) = cv2.findContours(b_Mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(cnts) > 0:
    	
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
       
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        if center[1] <= 65:
            if 40 <= center[0] <= 140:
                pWindow[67:,:,:] = 0
            elif 275 <= center[0] <= 370:
                    colorIndex = 0 
            elif 500 <= center[0] <= 600:
                    colorIndex = 1
        else :
            if colorIndex == 0:
                gp[gi].appendleft(center)
            elif colorIndex == 1:
                rp[ri].appendleft(center)
            
    
    else:
        gp.append(deque(maxlen=512))
        gi += 1
        rp.append(deque(maxlen=512))
        ri += 1
        

    points = [ gp, rp]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(pWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    
    cv2.imshow("VideoTrack", frame)
    cv2.imshow("Paint", pWindow)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.DestroyAllWindows()
        
