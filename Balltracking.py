import cv2
import numpy as np

cap = cv2.VideoCapture('volleyball_match.mp4')
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
while True:
    
    ret,frame = cap.read()
    if not ret:
        break
    image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    lower_yellow = np.array([75, 133, 0], dtype=np.uint8)
    upper_yellow = np.array([255, 255, 129], dtype=np.uint8)
    yellow_mask = cv2.inRange(image_rgb, lower_yellow, upper_yellow)

    image_yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    yuv_lower = np.array([51, 65, 139])
    yuv_upper = np.array([227, 102, 182])
    yuv_mask = cv2.inRange(image_yuv, yuv_lower, yuv_upper)
    result = cv2.bitwise_or(yellow_mask,yuv_mask)

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    combine =  cv2.bitwise_and(result,fgmask)
    combine = cv2.morphologyEx(combine, cv2.MORPH_OPEN, kernel)

    combine = cv2.cvtColor(result,cv2.COLOR_BAYER_BG2GRAY)

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        if len(approx)>12:
            ((x, y), radius) = cv2.minEnclosingCircle(approx)

            if radius < 10 and radius > 1 and y<210:
                cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 2)
                cv2.putText(frame, 'ball', (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),2)
            
            
               
    cv2.imshow('Foreground',fgmask)
    cv2.imshow('combine',combine)
    cv2.imshow('Track',frame)
    cv2.imshow('Mask',result)
    
    key = cv2.waitKey(10) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
    