import numpy as np
import cv2
file1 = np.load('y2.npy')*255
file2 = np.load('segft2.npy')*255
file3 = np.load('segft2_sigmoid.npy')*255
file4 = np.load('origin.npy')*255
file2 = file2.astype(np.uint8)
file3 = file3.astype(np.uint8)
file1 = file1.astype(np.uint8)
file4 = file4.astype(np.uint8)

img1 = file1 + file2
img2 = file1 + file3
cv2.imwrite('check_fakecolor/input.jpg',file4[0][1])
for i in range(128):
    im_color1 =  cv2.applyColorMap(file2[0][i], cv2.COLORMAP_JET)
    im_color2 =  cv2.applyColorMap(file3[0][i], cv2.COLORMAP_JET)
    cv2.imwrite('check_fakecolor/y2{}.jpg'.format(i),im_color1)
    cv2.imwrite('check_fakecolor/y2{}_sigmoid.jpg'.format(i),im_color2)
    
print()