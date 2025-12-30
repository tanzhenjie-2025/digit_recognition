import os
import uuid
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 非交互式后端，避免绘图报错
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from PIL import Image, ImageOps, ImageFilter
import cv2  # 新增：用OpenCV做形态学操作（需安装：pip install opencv-python）
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse

# 初始化路径
MODEL_PATH = os.path.join(settings.STATICFILES_DIRS[0], 'model/digit_rf.model')
IMAGE_DIR = os.path.join(settings.STATICFILES_DIRS[0], 'images')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# 训练模型（进一步优化参数）
def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"数据集样本总数：{len(X)}，数字分布：{np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 超参数优化：提升对印刷体特征的匹配度
    rf_model = RandomForestClassifier(
        n_estimators=300,        # 更多树提升泛化
        max_depth=20,            # 适配复杂特征
        min_samples_split=2,     # 保留细节特征
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'  # 平衡类别权重
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型训练完成，测试集准确率：{accuracy:.4f}")

    dump((rf_model, accuracy), MODEL_PATH)
    plot_digits_sample(digits)
    plot_accuracy(accuracy)

    return rf_model, accuracy


# 数据集样本可视化（新增：显示数据集原始8x8数字，供参考）
def plot_digits_sample(digits):
    plt.figure(figsize=(10, 6))
    # 显示前10个数据集数字（参考样式）
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(digits.images[i], cmap='gray')
        plt.title(f'Digit: {digits.target[i]}')
        plt.axis('off')
    plt.suptitle('参考：数据集原始8x8印刷体数字样式', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'digits_sample.png'))
    plt.close()


def plot_accuracy(accuracy):
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='skyblue')
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title(f'Model Test Accuracy: {accuracy:.4f}')
    plt.savefig(os.path.join(IMAGE_DIR, 'accuracy.png'))
    plt.close()


def load_model():
    if os.path.exists(MODEL_PATH):
        rf_model, accuracy = load(MODEL_PATH)
    else:
        rf_model, accuracy = train_model()
    return rf_model, accuracy


# 图片预处理（核心：贴合数据集印刷体特征）
def preprocess_image(image_path):
    try:
        # 1. 打开图片并转为灰度
        img = Image.open(image_path).convert('L')
        # 2. 缩放为100x100（中间步骤，方便形态学操作）
        img = img.resize((100, 100), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)

        # 3. 形态学操作：让粗体手写数字变纤细，接近印刷体
        # 腐蚀操作（减少笔画粗细）
        kernel = np.ones((2, 2), np.uint8)
        img_array = cv2.erode(img_array, kernel, iterations=1)
        # 二值化（匹配数据集的高对比度）
        _, img_binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)

        # 4. 裁剪为正方形+中心化
        img = Image.fromarray(img_binary)
        min_dim = min(img.size)
        img = ImageOps.fit(img, (min_dim, min_dim), Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        # 5. 缩放为8x8（最终匹配数据集尺寸）
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)

        # 6. 灰度拉伸+匹配数据集格式（0=白，16=黑）
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
        img_normalized = (255 - img_array) / 255 * 16
        img_normalized = np.clip(img_normalized, 0, 16).astype(np.int8)

        # 7. 展平特征
        img_flat = img_normalized.flatten()

        # 调试信息
        print(f"预处理后特征（前10个像素）：{img_flat[:10]}")
        print(f"特征最大值/最小值：{img_flat.max()} / {img_flat.min()}")
        # 保存8x8调试图
        debug_img = Image.fromarray(img_normalized.reshape(8, 8).astype(np.uint8))
        debug_img_path = os.path.join(settings.MEDIA_ROOT, f"debug_{uuid.uuid4()}.png")
        debug_img.save(debug_img_path)
        print(f"预处理后8x8图像已保存：{debug_img_path}")

        return img_flat
    except Exception as e:
        raise ValueError(f"图片预处理失败：{str(e)}（请按参考样式制作PNG图片）")


# 首页视图
def index(request):
    model, accuracy = load_model()
    context = {
        'accuracy': accuracy,
        'sample_img': '/static/images/digits_sample.png',  # 显示数据集参考样式
        'accuracy_img': '/static/images/accuracy.png'
    }
    return render(request, 'digit_app/index.html', context)


# 上传页面视图
def upload(request):
    if request.method == 'GET':
        return render(request, 'digit_app/upload.html')
    return redirect('index')


# 预测视图
def predict(request):
    if request.method != 'POST':
        return redirect('upload')

    if 'digit_image' not in request.FILES:
        return render(request, 'digit_app/upload.html', {'error': '请上传图片文件！'})

    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

    # 保存上传的图片
    image_file = request.FILES['digit_image']
    try:
        file_ext = image_file.name.split('.')[-1].lower()
        if file_ext not in ['png']:
            return render(request, 'digit_app/upload.html', {'error': '仅支持PNG格式图片！'})
        unique_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(settings.MEDIA_ROOT, unique_filename)

        with open(image_path, 'wb+') as f:
            for chunk in image_file.chunks():
                f.write(chunk)
    except Exception as e:
        return render(request, 'digit_app/upload.html', {'error': f'文件保存失败：{str(e)}'})

    try:
        model, accuracy = load_model()
        img_flat = preprocess_image(image_path)
        pred = model.predict([img_flat])[0]
        pred_proba = model.predict_proba([img_flat])[0]
        prob_info = {i: f"{p:.4f}" for i, p in enumerate(pred_proba)}
        print(f"各数字预测概率：{prob_info}")
        print(f"最终识别结果：{pred}")

        context = {
            'prediction': pred,
            'accuracy': accuracy,
            'uploaded_image': os.path.join(settings.MEDIA_URL, unique_filename),
            'accuracy_img': '/static/images/accuracy.png'
        }
        return render(request, 'digit_app/result.html', context)
    except ValueError as e:
        return render(request, 'digit_app/upload.html', {'error': str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'digit_app/upload.html', {'error': f'识别失败：{str(e)}'})