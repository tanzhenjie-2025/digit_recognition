from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),          # 首页
    path('upload/', views.upload, name='upload'), # 上传页面
    path('predict/', views.predict, name='predict'), # 识别接口
]