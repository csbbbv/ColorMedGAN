import glob,os,cv2
import numpy as np
def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map
palette = [[0], [1], [2],[3],[4]] 
palette2 = [[0], [1], [2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],[31],[32],[33],[34],[35]]
gt = '/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/oasis1/label4/'
savepath = '/media/user/8f28289d-aa45-46ab-b9e4-a7d4d9f08b7e/shaobo/oasis1/label4_onehot/'
mask_list = glob.glob(gt+'*.png')
cnt = 0
for mask in mask_list:
    mask_name = mask.split('/')[-1].replace('.png','.npy')
    im = cv2.imread(mask,0)
    im = np.expand_dims(im,axis=-1)
    gt_onehot = mask_to_onehot(im, palette)
    # print(gt_onehot)
    if os.path.exists(os.path.join(savepath,mask_name)):
        continue
    np.save(os.path.join(savepath,mask_name),gt_onehot)
    cnt += 1
    print('save: {0}'.format(cnt))