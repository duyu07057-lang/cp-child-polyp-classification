"""Generate the teachable CP-CHILD fusion notebook from versioned cell sources."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    md(r"""
    # 儿童结肠镜息肉分类：手工特征 + CNN 特征 + SVM

    **讲解主线：数据清理 → 双路特征提取 → 特征融合 → SVM 训练 → 测试与跨设备评价。**

    本笔记本只展示一种方法。A、B 数据集分别训练与测试；`Polyp=1` 表示息肉，`Non-Polyp=0` 表示非息肉（可能包含其他病变）。

    | 步骤 | 输入与输出 |
    |---|---|
    | 数据准备 | 审计图像 → 排除训练重复副本 → 固定训练、验证、测试划分 |
    | 手工特征 | 颜色 38 维 + 纹理 102 维 = **140 维** |
    | CNN 特征 | 预训练 ResNet18 的全局平均池化输出 = **512 维** |
    | 特征融合 | 分块标准化、按维数加权、拼接为 **652 维** |
    | 分类与评价 | 加权 SVM → 验证集选参和阈值 → 官方测试、去重测试、跨设备测试 |

    **运行方式：**在克隆后的项目目录中打开本文件，选择已安装项目依赖的 Python 3.10 内核，然后选择“全部运行”。首次配置和 CPU/GPU 版本安装步骤见项目根目录的 `README.md`。

    请先按 README 获取数据、运行审计并生成本地特征缓存。笔记本复用本地缓存，重新训练 SVM 并展示测试结果；新产物保存到独立时间戳目录。测试集此前可能已被查看过，结果不是新收集的独立验证。
    """),
    md(r"""
    ## 1. 环境与路径

    仅使用项目目录中的数据、审计清单和 ResNet18 权重。要强制从原图重新计算全部特征，可将 `USE_FEATURE_CACHE` 改为 `False`；这样会耗费更多时间。若原始图片发生变化，应先重新执行项目的数据审计。
    """),
    code(r"""
    from pathlib import Path
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor
    import hashlib, json, shutil, sys, time

    import numpy as np
    import pandas as pd
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    from IPython.display import display
    from PIL import Image
    from skimage.color import rgb2gray, rgb2hsv, rgb2lab
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
        confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
        roc_auc_score, roc_curve)
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import joblib
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    def find_project():
        here = Path.cwd().resolve()
        for candidate in [here, *here.parents]:
            if (candidate / 'analysis_cp_child' / 'image_manifest.csv').exists() and (candidate / 'configs' / 'cp_child.json').exists():
                return candidate.resolve()
        raise FileNotFoundError('未找到项目目录，请从克隆后的项目根目录启动 Jupyter。')

    ROOT = find_project()
    DATA_ROOT = ROOT / '6.Colonoscopy_CP_CHILD'
    AUDIT = ROOT / 'analysis_cp_child'
    EXPERIMENT = ROOT / 'experiments' / 'reproduced'
    RUN_DIR = ROOT / 'notebook_outputs' / ('fusion_svm_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    USE_FEATURE_CACHE = True
    RECHECK_ALL_IMAGE_HASHES = False
    SEED = 42
    BATCH_SIZE = 48
    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print('项目目录：', ROOT)
    print('本次 Notebook 输出：', RUN_DIR)
    print('Python：', sys.executable)
    print('PyTorch：', torch.__version__, '；计算设备：', torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU')
    """),
    md(r"""
    ## 2. 数据审计、去重与固定划分

    数据审计清单记录每张 JPG 的标签、文件哈希和解码后 RGB 像素哈希。以下代码重新计算去重关系并重建划分：

    1. 测试集保持官方划分；如果训练图像与测试图像的像素完全相同，就从训练输入排除。
    2. 对剩余训练图像中的像素重复组，只保留一张。
    3. 将感知哈希近重复候选保守地合并为相似图像组，再按类别和组划分训练/验证集。候选图片不会因此被删除。

    文件哈希只比较 JPEG 文件内容；像素哈希比较实际解码后的画面。没有患者编号时，相似组不能代表患者组。
    """),
    code(r"""
    raw = pd.read_csv(AUDIT / 'image_manifest.csv', keep_default_na=False).sort_values('path').reset_index(drop=True)
    raw['image_index'] = np.arange(len(raw))
    raw['target'] = raw['target'].astype(int)
    near_pairs = pd.read_csv(AUDIT / 'near_duplicate_candidates.csv', keep_default_na=False)
    assert len(raw) > 0 and raw.path.is_unique and (raw.error == '').all()
    print('图片总数：', len(raw), '；近重复候选对数：', len(near_pairs))
    print('标签映射：', {'Non-Polyp': 0, 'Polyp': 1})
    display(raw.groupby(['dataset', 'split', 'label']).size().unstack(fill_value=0))
    """),
    md(r"""
    ### 2.1 查看一组训练/测试重复图像

    使用 `width × height + RGB 解码像素` 的 SHA-256 与已有审计值进行比对。下面默认复核两张示例图，若设置 `RECHECK_ALL_IMAGE_HASHES=True`，则对全部图片逐张复核。
    """),
    code(r"""
    def image_hashes(relative_path):
        path = (DATA_ROOT / relative_path).resolve()
        if not path.is_relative_to(DATA_ROOT.resolve()):
            raise ValueError('图片路径超出数据目录')
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source:
            image = source.convert('RGB')
        prefix = f'{image.width}x{image.height}:RGB:'.encode()
        pixel_hash = hashlib.sha256(prefix + image.tobytes()).hexdigest()
        return file_hash, pixel_hash

    overlaps = pd.read_csv(AUDIT / 'train_test_pixel_overlap.csv')
    example = overlaps.iloc[0]
    first, second = image_hashes(example.path_1), image_hashes(example.path_2)
    assert first[1] == second[1] == example.pixel_sha256
    if RECHECK_ALL_IMAGE_HASHES:
        with ThreadPoolExecutor(max_workers=4) as pool:
            checked = list(pool.map(image_hashes, raw.path))
        assert all(f == row.file_sha256 and p == row.pixel_sha256
                   for (f, p), row in zip(checked, raw.itertuples(index=False)))
        print('已完成全部原图哈希核对。')
    print('跨训练/测试完全重复组数：', len(overlaps))
    print('示例：', example.path_1, '\n     ', example.path_2)
    print('解码像素相同：', first[1] == second[1])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, path, title in zip(axes, [example.path_1, example.path_2], ['训练集图片', '测试集图片']):
        with Image.open(DATA_ROOT / path) as im:
            ax.imshow(im.convert('RGB'))
        ax.set_title(title); ax.axis('off')
    plt.tight_layout(); plt.show()
    """),
    code(r"""
    class UnionFind:
        def __init__(self, n):
            self.parents = list(range(n))
        def find(self, i):
            while self.parents[i] != i:
                self.parents[i] = self.parents[self.parents[i]]
                i = self.parents[i]
            return i
        def union(self, a, b):
            a, b = self.find(a), self.find(b)
            self.parents[max(a, b)] = min(a, b)

    groups = UnionFind(len(raw))
    for indices in raw.groupby('pixel_sha256').indices.values():
        for index in indices[1:]:
            groups.union(int(indices[0]), int(index))
    path_to_index = dict(zip(raw.path, raw.image_index))
    for pair in near_pairs.itertuples(index=False):
        groups.union(path_to_index[pair.path_1], path_to_index[pair.path_2])
    raw['similarity_group'] = [f'similarity_{groups.find(i):05d}' for i in range(len(raw))]

    raw['phase'] = 'excluded'
    test_mask = raw['split'].eq('Test')
    train_mask = raw['split'].eq('Train')
    raw.loc[test_mask, 'phase'] = 'test'
    test_hashes = set(raw.loc[test_mask, 'pixel_sha256'])
    overlap_train = train_mask & raw.pixel_sha256.isin(test_hashes)
    eligible = raw.loc[train_mask & ~overlap_train]
    repeated_train = eligible.duplicated('pixel_sha256', keep='first')
    retained = eligible.loc[~repeated_train]
    print('排除与测试图像重叠的训练图片：', int(overlap_train.sum()))
    print('排除训练集内部重复副本：', int(repeated_train.sum()))

    # 测试集中保留官方全部图片，同时标记唯一像素口径。
    raw['unique_test'] = False
    unique_test = raw.loc[test_mask].drop_duplicates('pixel_sha256', keep='first')
    raw.loc[unique_test.index, 'unique_test'] = True
    for dataset in sorted(raw.dataset.unique()):
        pool = retained.loc[retained.dataset == dataset]
        split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_local, val_local = list(split.split(pool.path, pool.target, groups=pool.similarity_group))[0]
        raw.loc[pool.index[train_local], 'phase'] = 'train'
        raw.loc[pool.index[val_local], 'phase'] = 'validation'

    usable = raw.loc[raw.phase.isin(['train', 'validation', 'test'])]
    for field in ['pixel_sha256', 'similarity_group']:
        group_sets = {phase: set(usable.loc[usable.phase == phase, field]) for phase in ['train','validation','test']}
        assert not group_sets['train'] & group_sets['validation']
        assert not group_sets['train'] & group_sets['test']
        assert not group_sets['validation'] & group_sets['test']

    # 与先前已锁定的实验划分逐图核对；任何不一致即停止。
    existing = pd.read_csv(EXPERIMENT / 'split_manifest.csv', keep_default_na=False)
    check_fields = ['phase','similarity_group','unique_test','file_sha256','pixel_sha256','target','image_index']
    comparison = raw[['path'] + check_fields].merge(
        existing[['path'] + check_fields], on='path', suffixes=('_new','_existing'), validate='one_to_one')
    for field in check_fields:
        assert comparison[field + '_new'].astype(str).equals(comparison[field + '_existing'].astype(str))
    raw[['path','dataset','label','target','image_index','phase','similarity_group','pixel_sha256','unique_test']].to_csv(
        RUN_DIR / 'split_manifest.csv', index=False, encoding='utf-8-sig')
    print('官方测试图像：', int(test_mask.sum()), '；唯一像素测试图像：', int(raw.unique_test.sum()))
    display(raw.groupby(['dataset','phase','label']).size().unstack(fill_value=0))
    """),
    md(r"""
    ## 3. 手工特征：颜色 + 纹理，共 140 维

    - **颜色 38 维**：HSV 三通道直方图（16 + 8 + 8）与 Lab 三通道的均值、标准差（6）。
    - **纹理 102 维**：三个尺度的 uniform LBP 直方图（10 + 18 + 26 = 54），以及 2 个距离、4 个方向、6 项灰度共生矩阵统计（48）。

    手工特征使用完整的 256×256 RGB 图像；无需病灶框或分割掩膜。
    """),
    code(r"""
    HAND_DIM, CNN_DIM, FUSED_DIM = 140, 512, 652

    def extract_hand_features(image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert('RGB')
        image = image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        hsv, lab = rgb2hsv(rgb), rgb2lab(rgb)
        values = []
        for channel, bins in enumerate((16, 8, 8)):
            histogram = np.histogram(hsv[:, :, channel], bins=bins, range=(0, 1))[0].astype(float)
            values.extend(histogram / histogram.sum())
        values.extend(lab.mean(axis=(0, 1)))
        values.extend(lab.std(axis=(0, 1)))
        gray = np.rint(np.clip(rgb2gray(rgb), 0, 1) * 255).astype(np.uint8)
        for radius, points in ((1, 8), (2, 16), (3, 24)):
            lbp = local_binary_pattern(gray, points, radius, method='uniform')
            histogram = np.bincount(lbp.astype(int).ravel(), minlength=points + 2)
            values.extend(histogram / histogram.sum())
        glcm = graycomatrix(gray // 8, distances=[1,3], angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                            levels=32, symmetric=True, normed=True)
        for prop in ('contrast','dissimilarity','homogeneity','energy','correlation','ASM'):
            values.extend(graycoprops(glcm, prop).ravel())
        result = np.asarray(values, dtype=np.float32)
        assert result.shape == (HAND_DIM,) and np.isfinite(result).all()
        return result

    sample = raw.loc[(raw.dataset == 'CP-CHILD-A') & (raw.phase == 'train') & (raw.target == 1)].iloc[0]
    with Image.open(DATA_ROOT / sample.path) as source:
        sample_image = source.convert('RGB')
    sample_hand = extract_hand_features(sample_image)
    gray = np.rint(rgb2gray(np.asarray(sample_image, dtype=np.float32) / 255.0) * 255).astype(np.uint8)
    lbp_view = local_binary_pattern(gray, 8, 1, method='uniform')
    fig, axes = plt.subplots(1,3,figsize=(11,3))
    axes[0].imshow(sample_image); axes[0].set_title('息肉图像'); axes[0].axis('off')
    axes[1].imshow(lbp_view, cmap='gray'); axes[1].set_title('LBP 局部纹理'); axes[1].axis('off')
    axes[2].bar(np.arange(16), sample_hand[:16], color='#3274a1'); axes[2].set_title('HSV 色调直方图'); axes[2].set_xlabel('区间'); axes[2].set_ylabel('频率')
    plt.tight_layout(); plt.show()
    print('手工特征维数：', sample_hand.shape[0], '= 38 颜色 + 102 纹理')
    """),
    md(r"""
    ## 4. CNN 特征：冻结 ResNet18，输出 512 维

    把完整图像双线性缩放到 224×224，并使用 ImageNet 的均值、标准差进行归一化。加载预训练 ResNet18，冻结全部参数；去掉最后的 1,000 类分类层，保留全局平均池化输出。这个方法的 SVM 训练不更新 CNN 权重。
    """),
    code(r"""
    pretrained_path = EXPERIMENT / 'models' / 'frozen_resnet18.pt'
    if pretrained_path.exists():
        network = resnet18(weights=None)
        network.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        weight_source = str(pretrained_path)
    else:
        torch.hub.set_dir(str(ROOT / '.cache' / 'torch'))
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        weight_source = 'torchvision 官方 ImageNet1K 预训练权重'
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    cnn_extractor = torch.nn.Sequential(*list(network.children())[:-1]).to(DEVICE).eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
    std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]

    def cnn_input(image):
        image = image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32).transpose(2,0,1) / 255.0
        return (torch.from_numpy(rgb.copy()) - mean) / std

    def extract_cnn_features(images):
        with torch.inference_mode():
            batch = torch.stack([cnn_input(image) for image in images]).to(DEVICE)
            return cnn_extractor(batch).flatten(1).cpu().numpy().astype(np.float32)

    sample_cnn = extract_cnn_features([sample_image])[0]
    print('权重来源：', weight_source)
    print('CNN 特征维数：', len(sample_cnn))
    print('特征提取器可训练参数数量：', sum(p.numel() for p in cnn_extractor.parameters() if p.requires_grad))
    """),
]

cells.extend([
    md(r"""
    ## 5. 为全部图片提取并对齐两路特征

    特征矩阵的每一行对应固定清单中的一张图片，行号为 `image_index`。默认读取本地生成的缓存，并检查缓存的协议指纹、矩阵维度及样本特征；切换 `USE_FEATURE_CACHE=False` 可以使用上面的函数从原图重新计算。

    原始图、验证图和测试图均使用同一套固定 CNN。该方法不对训练图像进行随机增强；只在训练集中拟合后续的标准化器和 SVM。
    """),
    code(r"""
    source_protocol = json.loads((EXPERIMENT / 'protocol.json').read_text(encoding='utf-8'))
    source_lock_path = EXPERIMENT / 'selection_lock.json'
    source_lock = json.loads(source_lock_path.read_text(encoding='utf-8')) if source_lock_path.exists() else None
    if pretrained_path.exists() and source_lock:
        pretrained_hash = hashlib.sha256(pretrained_path.read_bytes()).hexdigest()
        assert pretrained_hash == source_lock['feature_backbone_sha256'], '预训练权重与锁定记录不一致'

    feature_dir = EXPERIMENT / 'features'
    if USE_FEATURE_CACHE:
        for meta_file in ['hand_features.json', 'cnn_features.json']:
            metadata = json.loads((feature_dir / meta_file).read_text(encoding='utf-8'))
            assert metadata['fingerprint'] == source_protocol['fingerprint'], '特征缓存与数据划分不匹配'
        hand_features = np.load(feature_dir / 'hand_features.npy', mmap_mode='r')
        cnn_features = np.load(feature_dir / 'cnn_features.npy', mmap_mode='r')
        print('已复用本地生成的特征缓存。')
    else:
        all_paths = [DATA_ROOT / path for path in raw.path]
        hand_features = np.empty((len(raw), HAND_DIM), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, values in enumerate(pool.map(extract_hand_features, all_paths)):
                hand_features[index] = values
                if (index + 1) % 1000 == 0:
                    print('手工特征：', index + 1, '/', len(raw))
        cnn_features = np.empty((len(raw), CNN_DIM), dtype=np.float32)
        def open_image(path):
            with Image.open(path) as image:
                return image.convert('RGB')
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(raw), BATCH_SIZE):
                images = list(pool.map(open_image, all_paths[start:start + BATCH_SIZE]))
                cnn_features[start:start + len(images)] = extract_cnn_features(images)
                if (start // BATCH_SIZE) % 20 == 0:
                    print('CNN 特征：', min(start + len(images), len(raw)), '/', len(raw))
        np.save(RUN_DIR / 'hand_features.npy', hand_features)
        np.save(RUN_DIR / 'cnn_features.npy', cnn_features)

    assert hand_features.shape == (len(raw), HAND_DIM) and cnn_features.shape == (len(raw), CNN_DIM)
    assert np.isfinite(hand_features).all() and np.isfinite(cnn_features).all()
    sample_index = int(sample.image_index)
    hand_difference = float(np.max(np.abs(sample_hand - hand_features[sample_index])))
    # 使用相同批量复核；单张 GPU 推理可能因计算方式有微小浮点差异。
    batch_start = (sample_index // BATCH_SIZE) * BATCH_SIZE
    batch_ids = list(range(batch_start, min(batch_start + BATCH_SIZE, len(raw))))
    check_images = []
    for index in batch_ids:
        with Image.open(DATA_ROOT / raw.iloc[index].path) as image:
            check_images.append(image.convert('RGB'))
    check_cnn = extract_cnn_features(check_images)
    cnn_difference = float(np.max(np.abs(check_cnn - cnn_features[batch_ids])))
    cnn_relative_difference = float(np.linalg.norm(check_cnn - cnn_features[batch_ids]) /
                                    max(np.linalg.norm(cnn_features[batch_ids]), 1e-12))
    single_cnn_difference = float(np.max(np.abs(sample_cnn - cnn_features[sample_index])))
    assert hand_difference < 2e-3 and cnn_relative_difference < 1e-2, '样本特征与缓存偏差过大，请检查环境与原始图像'
    print('手工特征矩阵：', hand_features.shape, '；CNN 特征矩阵：', cnn_features.shape)
    print('从原图重新计算的最大差异：手工单图 =', round(hand_difference, 7), '，CNN 相同批量 =', round(cnn_difference, 7))
    print('CNN 批量相对误差：', round(cnn_relative_difference, 7))
    print('CNN 单图与批量的浮点差异：', round(single_cnn_difference, 7))
    """),
    md(r"""
    ## 6. 特征融合与 SVM 分类

    对两路特征分别标准化后，分别乘以 $1/\sqrt{140}$ 和 $1/\sqrt{512}$，再拼接为 652 维向量。标准化和加权都放在 `Pipeline` 中，随 SVM 一起只在训练集上拟合。

    分别在 A 和 B 上训练。候选包括线性核、RBF 核，`C` 取 0.1、1、10，并比较一组不加类别权重的 RBF。**用验证集 AP 选参数，AP 相同时参考 F1；分类阈值取验证集 F1 最大时的阈值。**测试集不参与这些选择。
    """),
    code(r"""
    fused_features = np.concatenate([hand_features, cnn_features], axis=1)
    assert fused_features.shape == (len(raw), FUSED_DIM)
    targets = raw.target.to_numpy(dtype=int)
    print('融合特征维数：', HAND_DIM, '+', CNN_DIM, '=', fused_features.shape[1])

    def make_fusion_svm(parameters):
        transformer = ColumnTransformer(
            [('hand', StandardScaler(), slice(0, HAND_DIM)),
             ('cnn', StandardScaler(), slice(HAND_DIM, HAND_DIM + CNN_DIM))],
            transformer_weights={'hand': 1 / np.sqrt(HAND_DIM), 'cnn': 1 / np.sqrt(CNN_DIM)})
        return Pipeline([('scaler', transformer), ('classifier', SVC(cache_size=384, **parameters))])

    def choose_threshold(labels, scores):
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
        best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
        return float(thresholds[best])

    def calculate_metrics(labels, scores, threshold):
        labels = np.asarray(labels, dtype=int)
        scores = np.asarray(scores, dtype=float)
        predicted = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0,1]).ravel()
        return dict(n=len(labels), threshold=float(threshold), accuracy=float(accuracy_score(labels,predicted)),
            precision=float(precision_score(labels,predicted,zero_division=0)),
            recall=float(recall_score(labels,predicted,zero_division=0)),
            specificity=float(tn / (tn + fp)), f1=float(f1_score(labels,predicted,zero_division=0)),
            balanced_accuracy=float(balanced_accuracy_score(labels,predicted)),
            auroc=float(roc_auc_score(labels,scores)), ap=float(average_precision_score(labels,scores)),
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

    candidates = [dict(C=c, kernel=kernel, gamma='scale', class_weight='balanced')
                  for kernel in ('linear','rbf') for c in (0.1,1.0,10.0)]
    candidates.append(dict(C=1.0, kernel='rbf', gamma='scale', class_weight=None))
    print('每个子集的候选参数组数：', len(candidates))
    """),
    code(r"""
    selected_models = {}
    all_trials = []
    for dataset in ('CP-CHILD-A', 'CP-CHILD-B'):
        train_ids = np.flatnonzero((raw.dataset == dataset) & (raw.phase == 'train'))
        val_ids = np.flatnonzero((raw.dataset == dataset) & (raw.phase == 'validation'))
        X_train, y_train = fused_features[train_ids], targets[train_ids]
        X_val, y_val = fused_features[val_ids], targets[val_ids]
        best = None
        for parameters in candidates:
            model = make_fusion_svm(parameters)
            model.fit(X_train, y_train)
            val_score = model.decision_function(X_val)
            threshold = choose_threshold(y_val, val_score)
            validation = calculate_metrics(y_val, val_score, threshold)
            all_trials.append(dict(dataset=dataset, **parameters, validation_ap=validation['ap'],
                                   validation_f1=validation['f1'], threshold=threshold))
            rank = (validation['ap'], validation['f1'])
            if best is None or rank > best['rank']:
                best = dict(rank=rank, estimator=model, parameters=parameters,
                            threshold=threshold, validation=validation)
        # 标准化器的均值必须来自训练集，验证/测试数据不参与 fit。
        fitted = best['estimator'].named_steps['scaler']
        assert np.allclose(fitted.named_transformers_['hand'].mean_, X_train[:,:HAND_DIM].mean(axis=0,dtype=np.float64), atol=1e-7)
        assert np.allclose(fitted.named_transformers_['cnn'].mean_, X_train[:,HAND_DIM:].mean(axis=0,dtype=np.float64), atol=1e-7)
        selected_models[dataset] = best
        model_file = RUN_DIR / f'{dataset}__fusion_svm.joblib'
        joblib.dump(dict(estimator=best['estimator'], parameters=best['parameters'],
                         threshold=best['threshold'], dataset=dataset, feature_dimensions=[140,512],
                         score_type='decision_function', source_fingerprint=source_protocol['fingerprint']), model_file, compress=3)
        print(f"{dataset}：训练 {len(train_ids)} 张，验证 {len(val_ids)} 张；选定 {best['parameters']}；验证 AP={best['validation']['ap']:.4f}")
    trials = pd.DataFrame(all_trials)
    trials.to_csv(RUN_DIR / 'validation_trials.csv', index=False, encoding='utf-8-sig')
    display(trials[['dataset','kernel','C','class_weight','validation_ap','validation_f1']])
    """),
    md(r"""
    ## 7. 锁定选择后评价测试集

    先将两个训练好的 SVM、参数和验证集阈值保存并写入锁定记录，然后才计算测试集指标。**Accuracy** 是整体准确率；**Recall** 是息肉召回率；**Specificity** 是非息肉特异度；**F1** 平衡息肉精确率与召回率。SVM 的决策分数不经过概率校准，不能直接解释为发病概率。
    """),
    code(r"""
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    lock = {'source_fingerprint': source_protocol['fingerprint'],
            'selection': 'validation AP, then validation F1; threshold maximizes validation F1',
            'test_used_for_selection_in_this_run': False,
            'created_at': datetime.now().astimezone().isoformat(), 'models': {}}
    for dataset, best in selected_models.items():
        model_file = RUN_DIR / f'{dataset}__fusion_svm.joblib'
        lock['models'][dataset] = dict(parameters=best['parameters'], threshold=best['threshold'],
                                       model_sha256=sha256(model_file))
    (RUN_DIR / 'selection_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2),encoding='utf-8')
    print('已锁定两个模型和验证集阈值，下面开始计算测试指标。')
    display(pd.DataFrame([{'dataset':d, 'kernel':b['parameters']['kernel'], 'C':b['parameters']['C'],
                           'threshold':b['threshold'], 'validation_AP':b['validation']['ap']}
                          for d,b in selected_models.items()]))
    """),
    code(r"""
    test_rows, predictions = [], {}
    def evaluate(model_dataset, test_dataset, experiment):
        ids = np.flatnonzero((raw.dataset == test_dataset) & (raw.phase == 'test'))
        chosen = selected_models[model_dataset]
        assert sha256(RUN_DIR / f'{model_dataset}__fusion_svm.joblib') == lock['models'][model_dataset]['model_sha256']
        scores = chosen['estimator'].decision_function(fused_features[ids])
        y = targets[ids]
        threshold = chosen['threshold']
        pred = raw.iloc[ids][['path','dataset','target','unique_test']].copy()
        pred['score'] = scores
        pred['prediction'] = (scores >= threshold).astype(int)
        tag = f'{model_dataset}_to_{test_dataset}'
        pred.to_csv(RUN_DIR / f'{tag}_predictions.csv', index=False, encoding='utf-8-sig')
        predictions[tag] = pred
        for policy in ('official','unique_pixels'):
            mask = np.ones(len(ids), dtype=bool) if policy == 'official' else raw.iloc[ids].unique_test.to_numpy(dtype=bool)
            result = calculate_metrics(y[mask], scores[mask], threshold)
            test_rows.append(dict(experiment=experiment, train_dataset=model_dataset, test_dataset=test_dataset,
                                  test_policy=policy, **result))
    evaluate('CP-CHILD-A','CP-CHILD-A','same_device')
    evaluate('CP-CHILD-B','CP-CHILD-B','same_device')
    # 直接使用 A 的训练模型和 A 验证阈值，不使用 B 数据重新训练该模型。
    evaluate('CP-CHILD-A','CP-CHILD-B','cross_device')
    results = pd.DataFrame(test_rows)
    results.to_csv(RUN_DIR / 'fusion_test_metrics.csv', index=False, encoding='utf-8-sig')
    columns = ['experiment','train_dataset','test_dataset','test_policy','n','accuracy','recall','specificity','f1','auroc','ap']
    display(results[columns].round(4))
    """),
    code(r"""
    # 若已执行完整实验，可对照同一划分和缓存下的测试结果。
    metrics_path = EXPERIMENT / 'metrics.csv'
    if metrics_path.exists():
        original = pd.read_csv(metrics_path)
        reference = original[original.model.eq('M5_fusion_svm')].copy()
        reference['experiment'] = reference['experiment'].replace({'within_device':'same_device','cross_device':'cross_device'})
        comparison = results.merge(reference, on=['experiment','train_dataset','test_dataset','test_policy'],
                                   suffixes=('_notebook','_original'), validate='one_to_one')
        print('与本地完整实验的测试指标对照；不同平台可能有轻微数值差异。')
        display(comparison[['experiment','test_dataset','test_policy','f1_notebook','f1_original','ap','average_precision']].round(6))
    display(results.loc[results.test_policy.eq('official'),
            ['experiment','train_dataset','test_dataset','accuracy','recall','specificity','f1','ap']].round(4))
    """),
    md(r"""
    ## 8. 混淆矩阵与 ROC / PR 曲线

    混淆矩阵中，左上为正确识别的非息肉（TN），右上为误报（FP），左下为漏检（FN），右下为正确识别的息肉（TP）。ROC 曲线使用 AUROC，PR 曲线使用 Average Precision（AP）。
    """),
    code(r"""
    names = ['CP-CHILD-A_to_CP-CHILD-A', 'CP-CHILD-B_to_CP-CHILD-B', 'CP-CHILD-A_to_CP-CHILD-B']
    labels = ['A → A', 'B → B', 'A → B（跨设备）']
    fig, axes = plt.subplots(1,3,figsize=(12,3.7))
    for ax, tag, label in zip(axes, names, labels):
        pred = predictions[tag]
        matrix = confusion_matrix(pred.target, pred.prediction, labels=[0,1])
        ax.imshow(matrix, cmap='Blues')
        for (i,j), count in np.ndenumerate(matrix):
            ax.text(j,i,str(count),ha='center',va='center',fontsize=15,
                    color='white' if count > matrix.max() * .6 else '#17324d')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['非息肉','息肉'],yticklabels=['非息肉','息肉'],
               xlabel='预测类别',ylabel='真实类别',title=label)
    plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for tag, label in zip(names, labels):
        pred = predictions[tag]
        fpr,tpr,_ = roc_curve(pred.target,pred.score)
        precision,recall,_ = precision_recall_curve(pred.target,pred.score)
        axes[0].plot(fpr,tpr,label=f'{label} AUROC={roc_auc_score(pred.target,pred.score):.3f}')
        axes[1].plot(recall,precision,label=f'{label} AP={average_precision_score(pred.target,pred.score):.3f}')
    axes[0].plot([0,1],[0,1],'--',color='gray',linewidth=.8)
    axes[0].set(xlabel='假阳性率',ylabel='息肉召回率',title='ROC 曲线',xlim=(0,1),ylim=(0,1.02))
    axes[1].set(xlabel='息肉召回率',ylabel='精确率',title='PR 曲线',xlim=(0,1),ylim=(0,1.02))
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=9,loc='lower left')
    plt.tight_layout(); plt.show()
    """),
    md(r"""
    ## 9. 从原始图片进行一次预测

    单张图片依次经过手工特征提取、冻结 CNN 特征提取、融合标准化与 SVM 决策。模型分数越大，越倾向于判为息肉；实际分类使用该子集在验证集选定的阈值。
    """),
    code(r"""
    def predict_image(image_path, dataset):
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        hand = extract_hand_features(image)
        cnn = extract_cnn_features([image])[0]
        vector = np.concatenate([hand, cnn]).reshape(1,-1)
        chosen = selected_models[dataset]
        score = float(chosen['estimator'].decision_function(vector)[0])
        prediction = int(score >= chosen['threshold'])
        return dict(dataset=dataset, path=str(image_path), score=score, threshold=chosen['threshold'],
                    prediction=prediction, label='Polyp（息肉）' if prediction else 'Non-Polyp（非息肉）')

    example_row = raw.loc[(raw.dataset == 'CP-CHILD-A') & (raw.phase == 'test') & (raw.target == 1)].iloc[0]
    example_prediction = predict_image(DATA_ROOT / example_row.path, 'CP-CHILD-A')
    expected = predictions['CP-CHILD-A_to_CP-CHILD-A'].loc[lambda d: d.path == example_row.path].iloc[0]
    assert example_prediction['prediction'] == int(expected.prediction)
    assert np.isclose(example_prediction['score'], float(expected.score), atol=1e-2, rtol=5e-3)
    display(pd.DataFrame([example_prediction]))
    print('与测试阶段缓存特征得到的预测一致。')
    """),
    md(r"""
    ## 10. 结果与讲解提示

    - 同设备和跨设备的测试结果以第 7 节本次运行的表格为准。
    - 数据没有患者或视频对应表，因此当前只能进行图像级评价；高分不能证明患者级独立或临床有效性。
    - `RUN_DIR` 内保存本次 SVM 模型、固定划分、验证选参、测试预测和指标；它们不覆盖本地完整实验。

    **汇报时可用一句话概括：**手工特征描述颜色和纹理，冻结的预训练 CNN 提取高层视觉特征，拼接后交给加权 SVM 判断是否存在息肉。
    """),
])

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python (CP-CHILD GPU)', 'language': 'python', 'name': 'cpchild-gpu'},
    'language_info': {'name': 'python', 'version': '3.10'},
})
destination = Path(__file__).resolve().parent / '手工与CNN特征融合_SVM全过程.ipynb'
nbf.write(notebook, destination)
print('Created:', destination)
