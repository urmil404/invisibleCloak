import cv2
import numpy as np
import time

print("==============>Hello Guys,This My Invisible Cloak Python Program.\n\nYou should ready for get Invisible\n\nKeep Red thing in your hand. <==============")

fourCC = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourCC, 20, (200, 200))

cap = cv2.VideoCapture(0)
time.sleep(3)

background = 0
count = 0

for i in range(60):  # capturing the bkgrd 60 times, and overlapping to make one
    ret, background = cap.read()

background = np.flip(background, axis=1)

while(cap.isOpened()):
    ret, img = cap.read()
    if not(ret):
        break
    count += 1
    img = np.flip(img, axis=1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (35, 35), 0)

    lower = np.array([0, 120, 70])
    upper = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([100, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1+mask2

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)
    img[np.where(mask == 255)] = background[np.where(mask == 255)]
    result1 = cv2.bitwise_and(img, img, mask=mask2)
    result2 = cv2.bitwise_and(background, background, mask=mask1)

    finalOutput = cv2.addWeighted(result1, 1, result2, 1, 0)
    out.write(finalOutput)

    cv2.imshow("Invisible Clock", finalOutput)

    # if you hold q for 1 second, the loop breaks
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
