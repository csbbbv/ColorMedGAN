import cv2
import glob

root = '/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/mapillary_vistas_v2_part/training/v2.0/instances'
img_list = glob.glob(root+'/*.png')

for imgpath in img_list:
    img = cv2.imread(imgpath)
    
