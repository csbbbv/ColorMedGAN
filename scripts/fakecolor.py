import cv2
# img = cv2.imread('./p2.png',2)
import glob,os
# if __name__ == '__main__':
img_dir = r"data/BG/A/train"
save_dir = r"data/BG/A/color"

# dirs = ['yes','no','pred']

# for dire in dirs:
# files = os.path.join(img_dir)
img_lists = glob.glob(img_dir+'/*.jpg')
for img in img_lists:
    im_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
    save_path = os.path.join(save_dir,img.split('/')[-1])
    cv2.imwrite(save_path, im_color)