"""
@graduation project
@School of Computer Science and Engineering, Beihang University
@author  : hyc
@File    : urls.py
@contact : hyc2026@yeah.net
"""

"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from django.views.static import serve
from ocr.views import home, doc, config, db, pse, pan, upload

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('doc/', doc),
    path('config/', config),
    path('doc/db/', db),
    path('doc/pse/', pse),
    path('doc/pan/', pan),
    path('upload/', upload), # 上传图片
    re_path(r'^pic/(?P<path>.*)$', serve, {'document_root': '/raid/heyc-s21/pytorchOCR/result/result_img'})
]
