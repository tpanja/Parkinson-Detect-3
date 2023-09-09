print(1)
import cv2
print(2)
import numpy as np

#camera = cv2.VideoCapture(1)
#_, image = camera.read()

print(3)
image = cv2.imread('/Users/tanaypanja/Downloads/IMG_1217 2.JPG')
print(4)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# Find Canny edges
corners = cv2.goodFeaturesToTrack(gray, 27, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(image, (x, y), 4, 200, -1)

cv2.imshow(image, 'Corners')
cv2.waitKey(0)
