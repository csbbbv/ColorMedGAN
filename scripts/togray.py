import cv2
import glob,os

img_dir = 'data/thyroid_nodule_test/medical/val/benign/'
target_dir = 'data/thyroid_nodule/val/benign'
save_dir = 'data/thyroid_nodule_test/medical/val/benign2'

img_list = os.listdir(img_dir)
target_list =  os.listdir(target_dir)
img_list.sort(key=lambda x:int(x.split('pred')[1].split('.jpg')[0]))
target_list.sort()
for idx,img_path in enumerate(img_list):
    name = target_list[idx]#.split('/')[-1]
    img = cv2.imread(img_dir+img_path)
    cv2.imwrite(save_dir+'/'+name,img)