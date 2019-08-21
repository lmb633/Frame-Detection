import pickle
import random

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import im_size, pickle_file, num_train, device
from data_gen import data_transforms
from utils import ensure_folder, draw_bboxes2, draw_bboxes, sort_four_dot,cut_and_adjust_img
import os

img_num = 32

transformer = data_transforms['valid']


def visual_img(model):
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = [item for item in data]
    samples = random.sample(samples, img_num)
    imgs = torch.zeros([img_num, 3, im_size, im_size], dtype=torch.float)
    ensure_folder('images')

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
        raw = draw_bboxes2(raw, sample['pts'])
        cv.imwrite('images/{}_true.jpg'.format(i), raw)

    with torch.no_grad():
        outputs = model(imgs.to(device))

    for i in range(img_num):
        output = outputs[i].cpu().numpy()
        output = output * im_size
        # print('output: ' + str(output))
        # print('output.shape: ' + str(output.shape))

        img = cv.imread('images/{}_img.jpg'.format(i))
        # print(output)
        img = draw_bboxes2(img, output)
        cv.imwrite('images/{}_out.jpg'.format(i), img)


def visual_img1(model):
    files = os.listdir('real_screen')
    samples = random.sample(files, img_num)
    imgs = torch.zeros([img_num, 3, im_size, im_size], dtype=torch.float)
    ensure_folder('real_screen/images')

    for i in range(img_num):
        sample = samples[i]
        fullpath = os.path.join('real_screen', sample)
        if not 'images' in fullpath:
            print(fullpath)
            raw = cv.imread(fullpath)
            raw = cv.resize(raw, (im_size, im_size))
            img = raw[..., ::-1]  # RGB
            img = transforms.ToPILImage()(img)
            img = transformer(img)
            imgs[i] = img

            # cv.imwrite('real_screen/images/{}_img.jpg'.format(i), raw)

    for i in range(img_num):
        with torch.no_grad():
            output = model(torch.unsqueeze(imgs[i], 0).to(device))
        print(output)
        output = torch.squeeze(output)
        print(output)
        output = output.cpu().numpy()
        output = output * im_size
        # print('output: ' + str(output))
        # print('output.shape: ' + str(output.shape))

        img = cv.imread('real_screen/images/{}_img.jpg'.format(i))
        # print(output)
        img = draw_bboxes2(img, output)
        cv.imwrite('real_screen/images/{}_out.jpg'.format(i), img)


def cut_img(model, imgpath):
    img_origin = cv.imread(imgpath)
    print(img_origin.shape)
    w, h, c = img_origin.shape
    img = cv.resize(img_origin, (im_size, im_size))
    img = img[..., ::-1]  # RGB
    img = transforms.ToPILImage()(img)
    img = transformer(img)

    with torch.no_grad():
        output = model(torch.unsqueeze(img, 0).to(device))
    print(output)
    output = output.reshape(4, -1)
    print(output)
    output = output.cpu().numpy()
    output = output * [h, w]
    output = output.reshape(-1)

    # img = draw_bboxes2(img_origin, output)
    # cv.imwrite('real_screen/{}_out.jpg'.format(fullpath), img)




if __name__ == "__main__":
    # checkpoint = 'BEST_checkpoint.tar'
    # checkpoint = torch.load(checkpoint)
    # model = checkpoint['model']
    # model = model.to(device)
    # model.eval()
    # visual_img1(model)
    # # transformer = data_transforms['valid']
    #
    # files = os.listdir('real_screen')
    # for file in files:
    #     fullpath = os.path.join('real_screen', file)
    #     # fullpath = 'real_screen/test2.jpg'
    #     if not 'images' in fullpath:
    #         cut_img(model, fullpath)

    # visual_img1(model)
    fullpath = 'test_img/test2.jpg'
    # cut_img(None, fullpath)
    img_origin = cv.imread(fullpath)
    print(img_origin.shape)
    w, h, c = img_origin.shape
    output = np.array([0.7690, 0.0856, 0.2605, 0.1825, 0.2752, 0.7799, 0.7802, 0.8301])
    output = output.reshape(4, -1)
    print(output)
    output = output * [h, w]
    print(output)

    output = output.reshape(-1)
    output = sort_four_dot(output)

    img = draw_bboxes2(img_origin, output)
    # img = cv.resize(img_origin, (1024, 700))
    cv.imwrite('out_test.png', img)
    print(output)
    img2 = cut_and_adjust_img(img,output)
    cv.imshow('result', img2)
    cv.waitKey(0)
