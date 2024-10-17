import cv2 as cv2
import numpy as np

#cap = cv2.VideoCapture("volleyball_match.mp4")


def nothing():	
	pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LR", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LG", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LB", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UR", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UG", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UB", "Tracking", 255, 255, nothing)

detector = cv2.createBackgroundSubtractorKNN()

while True:
	frame = cv2.imread("Calibration_Image.png")
 
	RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
	lower_R = cv2.getTrackbarPos("LR", "Tracking")
	lower_G = cv2.getTrackbarPos("LG", "Tracking")
	lower_B = cv2.getTrackbarPos("LB", "Tracking")
  
	upper_R = cv2.getTrackbarPos("UR", "Tracking")
	upper_G = cv2.getTrackbarPos("UG", "Tracking")
	upper_B = cv2.getTrackbarPos("UB", "Tracking")

	lower = np.array([lower_R, lower_G, lower_B])
	upper = np.array([upper_R, upper_G, upper_B])

	mask = cv2.inRange(RGB, lower, upper)
	result = cv2.bitwise_and(frame, frame, mask=mask)
	
	cv2.imshow("frame", frame)	
	cv2.imshow("res", result)
 
	key = cv2.waitKey(5) & 0xFF
	if key == ord('q'):
		break

cv2.destroyAllWindows()