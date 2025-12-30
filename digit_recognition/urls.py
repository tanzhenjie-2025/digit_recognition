from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('digit_app.urls')),  # 挂载应用路由
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)  # 配置媒体文件访问