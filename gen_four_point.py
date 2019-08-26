import cv2 as cv
import imutils
import numpy as np
import pickle
import os
from utils import sort_four_dot
import json
import csv


def fit_quad(filename):
    image = cv.imread(filename)
    image = cv.resize(image, (224, 224))
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 11, 17, 17)
    edged = cv.Canny(gray, 30, 200)

    cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:10]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.015 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    return image, screenCnt


def gen_points_from_image():
    data = []
    files = os.listdir("images2")
    back_files = []
    screen_files = []
    for file in files:
        if 'back' in file and not file.endswith('meta'):
            back_files.append(file)
        if 'screen' in file and not file.endswith('meta'):
            screen_files.append(file)
    for file in back_files:
        filename = 'images2/{}'.format(file)
        print(filename)
        image, screenCnt = fit_quad(filename)
        # cv.imwrite('images2/img_{}.jpg'.format(i), image)

        if screenCnt is not None:
            cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
            cv.imwrite('images2/out_{}'.format(file).replace('back', ''), image)
            screenCnt = np.squeeze(screenCnt, 1).reshape((8,))
            print(screenCnt)
            screenCnt = sort_four_dot(screenCnt)
            print(screenCnt)
            data.append({'fullpath': filename.replace('back', 'screen'), 'pts': screenCnt})
    with open('data/data.pkl', 'wb') as file:
        pickle.dump(data, file)


def gen_points_from_csv():
    with open('data/via_region_data.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        header = next(lines)
        print(header)
        img2points0 = {}
        img2points1 = {}
        for line in lines:
            imgname = line[0]
            regin_id = line[4]
            points = json.loads(line[5])
            if 'name' in points:
                print(imgname, regin_id, points)
                if regin_id == '0':
                    img2points0[imgname] = np.array([points['all_points_x'][0:4], points['all_points_y'][0:4]]).T.reshape(-1)
                else:
                    img2points1[imgname] = np.array([points['all_points_x'][0:4], points['all_points_y'][0:4]]).T.reshape(-1)
                print(img2points0[imgname])
        with open('data/test.pkl', 'wb') as file:
            pickle.dump([img2points0, img2points1], file)


if __name__ == '__main__':
    # gen_points_from_image()
    gen_points_from_csv()
