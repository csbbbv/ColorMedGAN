import cv2

img = cv2.imread('conference_images/110_orig.png')

gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 4)
dst = cv2.convertScaleAbs(gray_lap)
cv2.imwrite('testlpls.png',dst)