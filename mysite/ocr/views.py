"""
@graduation project
@School of Computer Science and Engineering, Beihang University
@author  : hyc
@File    : views.py
@contact : hyc2026@yeah.net
"""
import os
import cv2
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from datetime import datetime
from django.shortcuts import redirect, reverse

import sys
import shutil
from mysite.settings import BASE_DIR
# Create your views here.
sys.path.append(os.path.dirname(__file__) + os.sep + '../../')
print(sys.path)
from tools.det_eval import InferanImage
from crnn.tools.infer.predict_rec import rec

def home(request):
    if request.method == "POST":
        print(request.POST)

    full_path = os.path.abspath(os.path.join(os.getcwd(), "result", "result_img"))
    if not os.path.exists(full_path):
        imgurl = "/static/images/eg.png"
    else:
        if not os.listdir(full_path):
            imgurl = "/static/images/eg.png"
        else:
            imgurl = "/pic/" + str(os.listdir(full_path)[0])
        p = os.path.abspath("")
        resu = []
        try:
            with open(os.path.join(p, "result", "res.txt"), "r") as f:
                for line in f:
                    resu.append(line)
        except:
            pass
    return render(request, "home.html", locals())


def doc(request):
    return render(request, "doc.html")


def config(request):
    return render(request, "config.html")


def db(request):
    return render(request, "db.html")


def pse(request):
    return render(request, "pse.html")


def pan(request):
    return render(request, "pan.html")


def get_rotate_crop_image(img, points):
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def rotate_crop_image(name):
    path = os.path.abspath("result")
    files = os.listdir(os.path.join(path, "result_txt"))
    print(files)
    print(os.path.join(path, "result_img", name))
    img = cv2.imread(os.path.join(path, "static", name))
    gt = open(os.path.join(path, "result_txt", files[0]), 'r').readlines()
    points = [np.array([np.float32(i) for i in s.split(",")]).reshape(4, 2) for s in gt]
    for i in range(0, len(points)):
        rimg = get_rotate_crop_image(img, points[i])
        cv2.imwrite(os.path.join(path, "rec_img", "img" + str(i) + ".jpg"), rimg)


def upload(request):
    if request.method == 'GET':
        return render(request, "home.html")
    else:
        pic = request.FILES.get('avator')
        full_path = os.path.abspath(os.path.join(os.getcwd(), "result"))

        if os.path.exists(full_path):  # 判断路径是否存在
            shutil.rmtree(full_path)
        os.makedirs(full_path)  # 创建此路径
        os.makedirs(os.path.join(full_path, "static"))
        with open(os.path.join(full_path, "static", pic.name), 'wb') as f:
            for c in pic.chunks():  # 相当于切片
                f.write(c)
        InferanImage()
        path = os.path.abspath("result")
        if not os.path.exists(os.path.join(path, "rec_img")):
            os.mkdir(os.path.join(path, "rec_img"))
        rotate_crop_image(pic.name)
        rec_files = os.listdir(os.path.join(path, "rec_img"))
        if len(rec_files) > 0:
            rec()
        p = os.path.abspath("")
        resu = []

        full_path = os.path.abspath(os.path.join(os.getcwd(), "result", "result_img"))
        if not os.path.exists(full_path):
            imgurl = "/static/images/eg.png"
        else:
            try:
                with open(os.path.join(p, "result", "res.txt"), "r") as f:
                    for line in f:
                        resu.append(line)
            except:
                pass
            imgurl = "/pic/" + str(os.listdir(full_path)[0])
        return render(request, "home.html", locals())