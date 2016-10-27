#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import sys
import math
import random
import numpy
import shutil

random.seed()


def prepare(output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    os.mkdir(output_dir + os.sep + "negative/")
    os.mkdir(output_dir + os.sep + "positive/")


def parse_label(label_str):
    return map(int, label_str.split(","))


def parse_mblabel_line(line):
    segs = line.split(" ")
    pic = segs[0]
    labels = [parse_label(seg) for seg in segs[1:]]
    return pic, labels


def parse_mblabel_file(file_):
    ret = {}
    for line in open(file_):
        pic, labels = parse_mblabel_line(line)
        ret[pic] = labels
    return ret


def contains(rect, point):
    x, y, w, h = rect
    px, py = point
    return(px > x and px < x + w and py > y and py < y + h)


def distance_from_center(rect, point):
    x, y, w, h = rect
    px, py = point
    return math.sqrt((px - (x + w / 2)) * (px - (x + w / 2)) +
                     (py - (y + h / 2)) * (py - (y + h / 2)))


def shrink_rect(rect, ratio=4):
    x, y, w, h = rect
    return (x / ratio, y / ratio, w / ratio, h / ratio)


def output_p(range_rect, rects):
    p = []
    x, y, w, h = range_rect
    for y_ in range(y, y + h):
        for x_ in range(x, x + w):
            pvalue = -1
            for rect in rects:
                rx, ry, rw, rh = rect
                max_distance_in_rect = math.sqrt(
                    rw / 2 * rw / 2 + rh / 2 * rh / 2)
                if not contains((rx, ry, rw, rh), (x_, y_)):
                    continue
                else:
                    dfc = distance_from_center((rx, ry, rw, rh), (x_, y_))
                    pvalue = (max_distance_in_rect - dfc) / \
                        max_distance_in_rect
            p.append(max(0, pvalue))
    return p


def output_loc(range_rect, rects, img_size):
    ih, iw = img_size
    dx0 = []
    dy0 = []
    dx1 = []
    dy1 = []
    x, y, w, h = range_rect
    for y_ in range(y, y + h):
        for x_ in range(x, x + w):
            dx0_v = -1
            dy0_v = -1
            dx1_v = -1
            dy1_v = -1
            for rect in rects:
                rx, ry, rw, rh = rect
                if not contains((rx, ry, rw, rh), (x_, y_)):
                    continue
                else:
                    dx0_v = (x_ - float(rx)) / iw
                    dy0_v = (y_ - float(ry)) / ih
                    dx1_v = (float(rx) + rw - x_) / iw
                    dy1_v = (float(ry) + rh - y_) / ih
            dx0.append(max(0, dx0_v))
            dy0.append(max(0, dy0_v))
            dx1.append(max(0, dx1_v))
            dy1.append(max(0, dy1_v))
    return dx0, dy0, dx1, dy1


def sample_one_rect(rect, sample_size, samples_per_label, img_size, rects, img):
    ret = []
    x, y, w, h = rect
    sw, sh = sample_size
    ih, iw = img_size
    min_x = max(x - sw / 2 + 1, 0)
    min_y = max(y - sh / 2 + 1, 0)
    max_x = min(x + w - sw / 2, iw - sw)
    max_y = min(y + h - sh / 2, ih - sh)
    # print min_x, max_x
    # print x, w, sw, iw
    if (min_x + sw > iw) or (min_y + sh > ih) or max_x < 0:
        return None
    for i in range(samples_per_label):

        x0 = random.randint(min_x, max_x)
        y0 = random.randint(min_y, max_y)
        x1 = x0 + sw
        y1 = y0 + sh

        p = output_p((x0, y0, sw, sh), rects)

        dx0_, dy0_, dx1_, dy1_ = output_loc((x0, y0, sw, sh), rects, img_size)

        map_list = [p, dx0_, dy0_, dx1_, dy1_]
        # cv2.imshow("Pic", img[y0 * 4:y0 * 4 +
        #                      sh * 4, x0 * 4:x0 * 4 + sw * 4])
        pa = numpy.array(p)
        pa = pa.reshape(60, 60)
        #cv2.imshow("P", pa)
        # print dx0_
        # for i in map_list:
        #     pa = numpy.array(i)
        #     pa = pa.reshape(60, 60)
        #     # print id(dx0_)
        #    cv2.imshow(str(id(i)), pa)
        # cv2.waitKey(1)
        # TODO Hard code...
        ret.append(((x0 * 4, y0 * 4, sw * 4, sh * 4), p))
    return ret


def get_size(pic_path):
    img = cv2.imread(pic_path)
    return img.shape[0:2]


def sample_positive(labels, output, sample_size,
                    samples_per_label, img_size, data_dir):
    positives = {}
    i = 0
    for pic, rects in labels.items():
        i += 1
        sys.stderr.write(str(i) + "\n")
        sys.stderr.flush()
        img = cv2.imread(data_dir + os.sep + pic)
        for rect in rects:
            samples = sample_one_rect(rect, sample_size,
                                      samples_per_label,
                                      img_size,
                                      rects, img)
            if samples is not None:
                for (rect, p) in samples:
                    print " ".join([pic, ",".join(map(str, rect)), ",".join(map(str, p))])
        sys.stdout.flush()


def shrink_labels(labels, ratio=4):
    shrinked_labels = {}
    for k, rects in labels.items():
        shrinked_rects = []
        for rect in rects:
            shrinked_rects.append(shrink_rect(rect))
        shrinked_labels[k] = shrinked_rects
    return shrinked_labels


def main(labels_file, data_dir, output_dir):
    sample_size = (60, 60)
    samples_per_label = 10
    prepare(output_dir)
    labels = parse_mblabel_file(labels_file)
    labels = shrink_labels(labels)
    # print labels.keys()
    img_size = get_size(data_dir + os.sep + labels.keys()[0])
    h, w = img_size
    img_size = (h / 4, w / 4)
    sample_positive(labels, output_dir, sample_size,
                    samples_per_label, img_size, data_dir)


if __name__ == '__main__':
    main("/home/qin/car_rear/正样本人工标注/uniq_labels.txt",
         "/home/qin/car_rear/正样本人工标注/all/", "/tmp/ffff/")
