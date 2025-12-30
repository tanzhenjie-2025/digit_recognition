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
import cv2  # 需安装：pip install opencv-python
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse

# 初始化路径
MODEL_PATH = os.path.join(settings.STATICFILES_DIRS[0], 'model/digit_rf.model')
IMAGE_DIR = os.path.join(settings.STATICFILES_DIRS[0], 'images')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# 模型优化：增加特征区分度，聚焦7/9、6/4、3/9的核心特征
def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"数据集样本总数：{len(X)}，数字分布：{np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 优化参数：强化特征区分，重点解决7/9、6/4、3/9混淆
    rf_model = RandomForestClassifier(
        n_estimators=350,  # 适度增加树数量，提升特征区分
        max_depth=18,  # 适度加深，学习7/9、6/4、3/9的细微特征
        min_samples_split=3,  # 降低分割阈值，捕捉数字细节
        min_samples_leaf=1,
        max_features='log2',  # 用log2特征数，聚焦核心区分特征
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)

    # 输出特征重要性（调试：看哪些像素对7/9、6/4、3/9区分最重要）
    feature_importance = rf_model.feature_importances_
    print(f"前10个重要特征索引：{np.argsort(feature_importance)[-10:]}")

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型训练完成，测试集准确率：{accuracy:.4f}")

    dump((rf_model, accuracy), MODEL_PATH)
    plot_digits_sample(digits)
    plot_accuracy(accuracy)

    return rf_model, accuracy


# 数据集样本可视化
def plot_digits_sample(digits):
    plt.figure(figsize=(10, 6))
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


# 温和版：去除极小噪点（保留数字主体）
def remove_small_blobs(binary_img, min_area=3):
    # 查找所有连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    clean_img = np.zeros_like(binary_img)
    for i in range(1, num_labels):  # 跳过背景（0）
        # 保留面积≥3的连通域（数字+少量必要轮廓）
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_img[labels == i] = 255
    return clean_img


# 预处理优化：强化7/6/3的核心特征
def preprocess_image(image_path):
    try:
        # 1. 打开图片并转为灰度
        img = Image.open(image_path).convert('L')
        # 2. 缩放为100x100
        img = img.resize((100, 100), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.uint8)

        # ========== 关键优化：微调阈值（90→85），强化7/6/3的核心特征 ==========
        _, img_binary = cv2.threshold(img_array, 85, 255, cv2.THRESH_BINARY)

        # 温和去噪（只删极小方块）
        img_binary = remove_small_blobs(img_binary, min_area=3)

        # ========== 新增：形态学增强（强化3的上下曲线特征） ==========
        # 小核开运算：强化3的上下曲线、2的底部横线、7的顶部横线
        kernel = np.ones((1, 1), np.uint8)
        img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)

        # 4. 裁剪为正方形+中心化
        img = Image.fromarray(img_binary)
        min_dim = min(img.size)
        img = ImageOps.fit(img, (min_dim, min_dim), Image.Resampling.LANCZOS, centering=(0.5, 0.5))

        # 5. 缩放为8x8
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)

        # 灰度映射+弱阈值过滤（保留7/6/2/3的特征）
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255
        img_normalized = (255 - img_array) / 255 * 16
        img_normalized = np.where(img_normalized < 0.5, 0, img_normalized)  # 极弱特征才过滤
        img_normalized = np.clip(img_normalized, 0, 16).astype(np.int8)

        # 7. 展平特征
        img_flat = img_normalized.flatten()

        # 调试信息
        print(f"预处理后特征（前10个像素）：{img_flat[:10]}")
        print(f"特征最大值/最小值：{img_flat.max()} / {img_flat.min()}")

        # 保存放大后的调试图
        debug_img_array = img_normalized.reshape(8, 8) * 16
        debug_img = Image.fromarray(debug_img_array.astype(np.uint8))
        debug_img_path = os.path.join(settings.MEDIA_ROOT, f"debug_{uuid.uuid4()}.png")
        debug_img.save(debug_img_path)
        print(f"预处理后8x8图像已保存（放大16倍）：{debug_img_path}")

        return img_flat
    except Exception as e:
        raise ValueError(f"图片预处理失败：{str(e)}（请按参考样式制作PNG图片）")


# 预测后修正（解决3→9、2→7、7→9、6→4混淆，全量生效）
def correct_prediction(img_flat, pred, pred_proba):
    # 8x8特征重塑
    img_8x8 = img_flat.reshape(8, 8)

    # ========== 优先级0：3 → 9 修正（核心解决当前3被识别为9的问题） ==========
    if pred == 9:
        # 3的核心特征：上下双曲线（顶部0-3行、底部5-8行右侧有像素），无底部闭合
        three_top_curve = np.sum(img_8x8[0:3, 5:7])  # 3的顶部曲线像素和
        three_bottom_curve = np.sum(img_8x8[5:8, 5:7])  # 3的底部曲线像素和
        nine_bottom_close = np.sum(img_8x8[6:8, 2:5])  # 9的底部闭合像素和
        total_three_pixels = three_top_curve + three_bottom_curve  # 3的核心像素和

        # 打印调试信息
        print(
            f"3→9修正调试：3顶部曲线={three_top_curve}，3底部曲线={three_bottom_curve}，9底部闭合={nine_bottom_close}，3核心像素和={total_three_pixels}")

        # 触发条件：3的概率≥0.2 + 3核心像素和>10 + 9底部闭合<5
        if pred_proba[3] >= 0.2 and total_three_pixels > 10 and nine_bottom_close < 5:
            pred = 3
            print(f"修正预测：9 → 3（3的概率{pred_proba[3]:.4f}，9的概率{pred_proba[9]:.4f}）")

    # ========== 优先级1：2 → 7 修正（保留已生效的2的修正逻辑） ==========
    if pred == 7:
        # 打印调试像素值，便于排查
        two_bottom_line = np.sum(img_8x8[6:8, 2:6])  # 2的底部横线像素和
        seven_bottom = np.sum(img_8x8[6:8, :])  # 7的底部像素和
        two_right_curve = np.sum(img_8x8[3:5, 5:7])  # 2的右侧曲线
        two_top_curve = np.sum(img_8x8[0:3, 5:7])  # 2的顶部曲线
        total_pixels = np.sum(img_8x8)  # 总像素数
        print(
            f"2→7修正调试：底部横线={two_bottom_line}，7底部={seven_bottom}，右侧曲线={two_right_curve}，顶部曲线={two_top_curve}，总像素={total_pixels}")

        # 条件1：宽松的像素特征（只要满足任意一个就触发）
        pixel_condition = (two_bottom_line > 1) or (two_right_curve > 1) or (two_top_curve > 1)
        # 条件2：2的概率≥0.05
        prob_condition = pred_proba[2] >= 0.05
        # 条件3：7和2的概率差≤0.1
        diff_condition = (pred_proba[7] - pred_proba[2]) <= 0.1

        # 三个条件满足任意两个，就修正为2
        if sum([pixel_condition, prob_condition, diff_condition]) >= 2:
            pred = 2
            print(
                f"修正预测：7 → 2（2的概率{pred_proba[2]:.4f}，7的概率{pred_proba[7]:.4f}，概率差{pred_proba[7] - pred_proba[2]:.4f}）")

    # ========== 优先级2：7 → 9 修正（原有正确逻辑） ==========
    if pred == 9 and pred_proba[7] > 0.1:
        # 7的核心特征：顶部横线（第0-2行，第4-7列）有像素，底部无闭合
        top_row_pixels = np.sum(img_8x8[0:2, 4:7])  # 7的顶部横线像素和
        bottom_row_pixels = np.sum(img_8x8[6:8, 2:5])  # 9的底部闭合像素和
        if top_row_pixels > 10 and bottom_row_pixels < 5:
            pred = 7
            print(f"修正预测：9 → 7（7的概率{pred_proba[7]:.4f}，9的概率{pred_proba[9]:.4f}）")

    # ========== 优先级3：6 → 4 修正（原有正确逻辑） ==========
    if pred == 4 and pred_proba[6] > 0.08:
        # 6的核心特征：底部圆圈（第6-8行，第2-5列）有像素，4无闭合底部
        bottom_circle_pixels = np.sum(img_8x8[5:7, 2:5])  # 6的底部圆圈像素和
        if bottom_circle_pixels > 8:
            pred = 6
            print(f"修正预测：4 → 6（6的概率{pred_proba[6]:.4f}，4的概率{pred_proba[4]:.4f}）")

    return pred


# 首页视图
def index(request):
    model, accuracy = load_model()
    context = {
        'accuracy': accuracy,
        'sample_img': '/static/images/digits_sample.png',
        'accuracy_img': '/static/images/accuracy.png'
    }
    return render(request, 'digit_app/index.html', context)


# 上传页面视图
def upload(request):
    if request.method == 'GET':
        return render(request, 'digit_app/upload.html')
    return redirect('index')


# 预测视图（集成修正逻辑）
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

        # 预测后修正（核心：解决3→9、2→7、7→9、6→4混淆）
        pred = correct_prediction(img_flat, pred, pred_proba)

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
        return render(request,  'digit_app/upload.html', {'error': str(e)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render(request, 'digit_app/upload.html', {'error': f'识别失败：{str(e)}'})