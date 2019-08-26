import pickle
import random

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import im_size, pickle_file, num_train, device
from data_gen import data_transforms
from utils import ensure_folder, draw_bboxes2, draw_bboxes, sort_four_dot, cut_and_adjust_img, get_iou
import os

img_num = 64

transformer = data_transforms['valid']
test_dir = 'test_img'


def visual_img(model):
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = [item for item in data]
    samples = random.sample(samples, img_num)
    imgs = torch.zeros([img_num, 3, im_size, im_size], dtype=torch.float)
    ensure_folder('images')
    origin_pts = []
    for i in range(img_num):
        sample = samples[i]
        fullpath = sample['fullpath']
        raw = cv.imread(fullpath)
        raw = cv.resize(raw, (im_size, im_size))
        img = raw[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        imgs[i] = img

        cv.imwrite('images/{}_img.jpg'.format(i), raw)
        # print(sample['pts'])
        raw = draw_bboxes2(raw, sample['pts'], thick=3)
        origin_pts.append(sample['pts'])
        cv.imwrite('images/{}_true.jpg'.format(i), raw)

    with torch.no_grad():
        outputs = model(imgs.to(device))

    iou_sum = 0
    for i in range(img_num):
        output = outputs[i].cpu().numpy()
        output = output * im_size
        # print('output: ' + str(output))
        # print('output.shape: ' + str(output.shape))

        img = cv.imread('images/{}_img.jpg'.format(i))
        # print(output)
        img = draw_bboxes2(img, output, thick=3)
        iou_sum += get_iou(origin_pts[i], output)
        cv.imwrite('images/{}_out.jpg'.format(i), img)
    return iou_sum / img_num


def cut_img(model, file, realoutput=None):
    fullpath = os.path.join(test_dir, file)
    print(fullpath)
    img_origin = cv.imdecode(np.fromfile(fullpath, dtype=np.uint8), cv.IMREAD_COLOR)

    w, h, c = img_origin.shape
    img = cv.resize(img_origin, (im_size, im_size))
    img = img[..., ::-1]  # RGB
    img = transforms.ToPILImage()(img)
    img = transformer(img)

    with torch.no_grad():
        output = model(torch.unsqueeze(img, 0).to(device))
    output = output.reshape(4, -1)
    output = output.cpu().numpy()
    output = output * [h, w]
    output = output.reshape(-1)
    output = sort_four_dot(output)


    img = draw_bboxes2(img_origin, output)
    cv.imwrite('{}_out.jpg'.format(test_dir + '/image/' + file), img)
    img2 = cut_and_adjust_img(img, output)
    cv.imwrite('{}_adjust.jpg'.format(test_dir + '/image/' + file), img2)

    if realdots is not None:
        roi = get_iou(output, realoutput)
        print(roi)
        img0 = draw_bboxes2(img_origin, realoutput, 'g')
        cv.imwrite('{}_real.jpg'.format(test_dir + '/image/' + file), img0)
        return roi


if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with open('data/test.pkl', 'rb') as f:
        testimg = pickle.load(f)
        realdots = testimg[0]

    files = os.listdir(test_dir)
    rois = {}
    for file in files:
        if not 'images' in file and file in realdots:
            realoutput = realdots[file]
            realoutput = sort_four_dot(realoutput)
            rois[file] = (cut_img(model, file, realoutput))
    print(rois)
    print(len(rois))
    rois0 = [roi for roi in rois if rois[roi] > 0.95]
    print(len(rois0))
    print(rois0)
    rois1 = [roi for roi in rois if rois[roi] > 0.9]
    print(len(rois1))
    rois2 = [roi for roi in rois if rois[roi] > 0.8]
    print(len(rois2))
    rois3 = [roi for roi in rois if rois[roi] > 0.7]
    print(len(rois3))
