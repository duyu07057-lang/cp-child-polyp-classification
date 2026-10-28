# CP-CHILD 儿童结肠镜息肉分类

本项目使用手工图像特征、预训练 ResNet18 特征和支持向量机（SVM）进行图像级二分类，也提供传统模型和 CNN 微调的完整实验代码。[Jupyter Notebook](notebooks/手工与CNN特征融合_SVM全过程.ipynb) 按数据处理、特征提取、融合、训练和评价顺序讲解融合方法。

`Polyp=1` 表示息肉，`Non-Polyp=0` 表示非息肉；非息肉图像也可能包含其他病变。数据没有患者或视频对应关系，实验结果只能作为图像级结果，不能证明患者级独立或临床有效性。

## 一、获取项目和数据

```bash
git clone https://github.com/duyu07057-lang/cp-child-polyp-classification.git
cd cp-child-polyp-classification
```

仓库仅发布代码和复现说明，**不包含数据集图像、图像审计结果、特征缓存、训练模型或实验结果**。请按照数据发布方的访问方式自行获取 [CP-CHILD 数据集](https://doi.org/10.6084/m9.figshare.12554042)，并遵循数据发布方的许可与引用要求。数据获取链接能否访问取决于发布方；本项目不代为分发数据。

解压后，将数据集放在项目根目录，保持以下目录结构：

```text
6.Colonoscopy_CP_CHILD/
├── CP-CHILD-A/
│   ├── Train/{Non-Polyp,Polyp}/
│   └── Test/{Non-Polyp,Polyp}/
└── CP-CHILD-B/
    ├── Train/{Non-Polyp,Polyp}/
    └── Test/{Non-Polyp,Polyp}/
```

## 二、创建环境

以下以 Python 3.10 和 Conda 为例，Windows PowerShell 和 Linux 终端均可使用：

```bash
conda create -n cpchild python=3.10 -y
conda activate cpchild
```

安装 PyTorch：支持 CUDA 12.1 的 NVIDIA 显卡可使用下面的 GPU 构建：

```bash
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
```

没有 NVIDIA 显卡时，使用 CPU 构建：

```bash
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu
```

根据系统、显卡和驱动，也可以参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择兼容的版本。然后安装其余依赖：

```bash
python -m pip install -r requirements-reproduce.txt
python -c "import torch, sklearn, skimage; print('torch:', torch.__version__, 'CUDA可用:', torch.cuda.is_available())"
```

Linux 图表中文如显示为方框，可安装 Noto CJK 等中文字体。

## 三、准备数据和特征

在项目根目录依次执行以下命令。审计会在本地生成图像清单、哈希、重复图像检查结果及样本图；`prepare` 固定数据划分；后两步提取手工特征和 CNN 特征：

```bash
python analysis_cp_child/audit_dataset.py
python run_experiment.py --stage prepare
python run_experiment.py --stage hand
python run_experiment.py --stage cnn-features
```

首次提取 CNN 特征时，torchvision 会下载 ImageNet 预训练 ResNet18 权重，需要网络连接。GPU 通常能明显缩短 CNN 特征提取时间；CPU 也可运行，但耗时较长。生成的审计文件位于 `analysis_cp_child/`，实验文件位于 `experiments/reproduced/`；这些本地生成文件不纳入 Git。

## 四、运行融合方法 Notebook

完成上面的数据准备后，为 Jupyter 注册环境并启动：

```bash
python -m ipykernel install --user --name cpchild-gpu --display-name "Python (CP-CHILD)"
jupyter notebook
```

打开 `notebooks/手工与CNN特征融合_SVM全过程.ipynb`，选择 **Python (CP-CHILD)** 内核并点击“全部运行”。Notebook 默认读取上一步生成的本地特征缓存，重新训练 A、B 两套融合 SVM，展示测试与跨设备结果；每次运行都会在 `notebook_outputs/` 创建独立目录。

要在 Notebook 内从图像重新计算全部特征，将第一段代码中的 `USE_FEATURE_CACHE` 改为 `False`；这仍需要先完成数据审计和 `prepare`，并准备预训练权重。CPU/GPU 和依赖版本差异可能造成轻微数值变化。

## 五、运行完整 Python 实验

```bash
python -m pytest tests -q
python run_experiment.py --stage all
```

完整实验包含划分、传统模型、融合模型、CNN 微调、评价和报告。它会复用同一配置下已经生成的中间结果，但 CNN 微调和中间激活缓存仍可能耗费较长时间及数 GB 磁盘空间。默认输出目录为 `experiments/reproduced/`，请预留至少 4 GB 空间；中断后可用相同命令恢复。

也可只在准备好两路特征后训练融合模型：

```bash
python run_experiment.py --stage hybrid
```

以自行训练得到的 A 集模型预测一张图片：

```bash
python -m cpchild.predict --model "experiments/reproduced/models/CP-CHILD-A__M5_fusion_svm.joblib" --images "6.Colonoscopy_CP_CHILD/CP-CHILD-A/Test/Polyp/1 (1).jpg"
```

程序输出类别、SVM 决策分数和验证集确定的阈值。决策分数并非经过校准的风险概率；只加载可信来源的模型文件。

## 六、方法和文件说明

融合方法先检查训练与测试图像的重复情况，并按相似组固定训练与验证划分；接着提取 HSV/Lab 颜色和 LBP/GLCM 纹理特征，以及冻结 ResNet18 的 CNN 特征。两路特征分别标准化和加权后拼接，交给 SVM；超参数和决策阈值在验证集选择，最后报告官方测试、去重测试及跨设备结果。

| 路径 | 说明 |
|---|---|
| `analysis_cp_child/audit_dataset.py` | 数据审计及去重检查脚本 |
| `cpchild/` | 特征提取、训练、评价、预测和报告代码 |
| `configs/cp_child.json` | 实验配置 |
| `notebooks/` | 融合方法 Notebook 及生成脚本 |
| `tests/` | 数据划分、训练与评价相关测试 |
| `run_experiment.py` | 命令行实验入口 |

数据出处：Wang 和 Tian 的 [CP-CHILD 数据集](https://doi.org/10.6084/m9.figshare.12554042)；相关研究见 [Wang 等，2020](https://doi.org/10.1186/s12880-020-00482-3)。本项目仅用于研究和教学，不直接用于医疗诊断。
