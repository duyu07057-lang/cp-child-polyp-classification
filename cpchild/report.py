"""Export test metrics, comparison plots and a Chinese report without retuning."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import precision_recall_curve, roc_curve

from .data import save_json
from .fonts import image_font


DISPLAY = {
    "M0_majority": "M0 多数类",
    "M1_hand_svm": "M1 手工特征+SVM",
    "M2_hand_rf": "M2 手工特征+随机森林",
    "M3_cnn_svm": "M3 CNN特征+SVM",
    "M4_cnn_finetune": "M4 CNN微调",
    "M5_fusion_svm": "M5 特征融合+SVM",
    "A1_color_svm": "A1 仅颜色+SVM",
    "A2_texture_svm": "A2 仅纹理+SVM",
}


def percent(value):
    return f"{float(value):.2%}"


def markdown_table(frame, columns, labels=None):
    labels = labels or columns
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in ("recall", "specificity", "precision", "f1", "accuracy", "balanced_accuracy", "auroc", "average_precision"):
                value = percent(value)
            elif column == "model":
                value = DISPLAY[value]
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def configure_plots():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "figure.dpi": 130, "savefig.dpi": 180,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def comparison_plots(cfg, metrics, figures):
    output = Path(cfg["output_dir"])
    main = [method for method in cfg["models"] if method.startswith("M")]
    for dataset in ("CP-CHILD-A", "CP-CHILD-B"):
        selected = metrics[(metrics.experiment == "within_device") & (metrics.test_dataset == dataset) &
                           (metrics.test_policy == "official")].set_index("model")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for method in main:
            predictions = pd.read_csv(output / "predictions" / f"{dataset}__{method}__test.csv")
            fpr, tpr, _ = roc_curve(predictions.target, predictions.score)
            precision, recall, _ = precision_recall_curve(predictions.target, predictions.score)
            axes[0].plot(fpr, tpr, label=f"{DISPLAY[method]}  {selected.loc[method,'auroc']:.3f}", linewidth=1.8)
            axes[1].plot(recall, precision, label=f"{DISPLAY[method]}  {selected.loc[method,'average_precision']:.3f}", linewidth=1.8)
        axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=.8)
        axes[1].axhline(float(predictions.target.mean()), ls="--", color="gray", linewidth=.8)
        axes[0].set(xlabel="假阳性率", ylabel="真阳性率", title=f"{dataset} ROC（图例为 AUROC）", xlim=(0, 1), ylim=(0, 1.02))
        axes[1].set(xlabel="息肉召回率", ylabel="精确率", title=f"{dataset} PR（图例为 AP）", xlim=(0, 1), ylim=(0, 1.02))
        for axis in axes:
            axis.legend(fontsize=8, loc="lower left")
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_curves.png")
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for axis, method in zip(axes.flat, main):
            row = selected.loc[method]
            matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
            axis.imshow(matrix, cmap="Blues")
            for (i, j), count in np.ndenumerate(matrix):
                axis.text(j, i, str(count), ha="center", va="center", fontsize=14,
                          color="white" if count > matrix.max() * .6 else "#12304a")
            axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["非息肉", "息肉"],
                     yticklabels=["非息肉", "息肉"], xlabel="预测", ylabel="真实", title=DISPLAY[method])
        fig.suptitle(f"{dataset} 官方测试集混淆矩阵；阈值均来自验证集")
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_confusions.png")
        plt.close(fig)
        display = selected.loc[main, ["recall", "specificity", "f1", "average_precision"]]
        display.index = [DISPLAY[index] for index in display.index]
        axis = display.rename(columns={"recall": "召回率", "specificity": "特异度", "f1": "F1", "average_precision": "AP"}).plot.bar(figsize=(12, 5), ylim=(0, 1.05), rot=15, colormap="viridis")
        axis.set(title=f"{dataset} 分类对照", ylabel="指标值", xlabel="")
        axis.legend(loc="lower right")
        axis.figure.tight_layout()
        axis.figure.savefig(figures / f"{dataset}_comparison.png")
        plt.close(axis.figure)


def error_gallery(cfg, dataset, method, destination):
    path = Path(cfg["output_dir"]) / "predictions" / f"{dataset}__{method}__test__errors.csv"
    errors = pd.read_csv(path)
    font = image_font(15)
    title_font = image_font(22, bold=True)
    examples = pd.concat([errors[errors.error_type == "false_negative"].sort_values("score").head(4),
                          errors[errors.error_type == "false_positive"].sort_values("score", ascending=False).head(4)])
    count = len(examples)
    canvas = Image.new("RGB", (1120, 70 + max(1, (count + 3) // 4) * 320), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), f"{dataset} 验证集选定模型：{DISPLAY[method]}｜测试错误样本", font=title_font, fill="#17324d")
    if not count:
        draw.text((20, 100), "该模型在此测试集未出现误分类；这不代表真实场景不会出错。", font=font, fill="#222222")
    for i, (_, row) in enumerate(examples.iterrows()):
        x, y = (i % 4) * 280 + 12, 70 + (i // 4) * 320
        with Image.open(Path(cfg["dataset_root"]) / row.path) as image:
            canvas.paste(ImageOps.contain(image.convert("RGB"), (248, 248)), (x, y))
        text = f"{'漏检 FN' if row.error_type == 'false_negative' else '误报 FP'} | 分数 {row.score:.4f}\n阈值 {row.threshold:.4f}\n{Path(row.path).name}"
        draw.multiline_text((x, y + 250), text, font=font, fill="#222222")
    canvas.save(destination)


def verify_saved_inference(cfg, frame, output, log=print):
    from .predict import Predictor

    records = []
    for dataset in ("CP-CHILD-A", "CP-CHILD-B"):
        for method in ("M1_hand_svm", "M3_cnn_svm", "M4_cnn_finetune", "M5_fusion_svm"):
            suffix = ".pt" if method == "M4_cnn_finetune" else ".joblib"
            predictor = Predictor(output / "models" / f"{dataset}__{method}{suffix}")
            log(f"Saved-model inference check: {dataset} {method} device={predictor.device}")
            predictions = pd.read_csv(output / "predictions" / f"{dataset}__{method}__test.csv")
            # One sample from each class, selected by manifest order, never by outcome.
            selected = predictions.groupby("target", sort=True).head(1)
            paths = [Path(cfg["dataset_root"]) / path for path in selected.path]
            inferred = predictor.predict(paths)
            for (_, row), result in zip(selected.iterrows(), inferred):
                delta = abs(float(row.score) - result["score"])
                if delta > 2e-3 or result["prediction"] != int(row.prediction):
                    raise ValueError(f"Saved-model inference mismatch: {dataset} {method} delta={delta}")
                records.append(dict(dataset=dataset, model=method, path=row.path,
                                    cached_score=float(row.score), reloaded_score=result["score"],
                                    absolute_error=delta, label_matches=True))
            # Release this model before allocating the next one.
            del predictor
    pd.DataFrame(records).to_csv(output / "inference_roundtrip_checks.csv", index=False, encoding="utf-8-sig")
    log(f"Passed {len(records)} saved-model image inference roundtrip checks")


def build_report(cfg, frame, protocol, log=print):
    output = Path(cfg["output_dir"])
    metrics = pd.read_csv(output / "metrics.csv")
    lock = json.loads((output / "selection_lock.json").read_text(encoding="utf-8"))
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    configure_plots()
    comparison_plots(cfg, metrics, figures)
    verify_saved_inference(cfg, frame, output, log)
    selections = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "selection").glob("CP-CHILD-*__*.json")) if "cnn_history" not in path.name]
    validation = pd.DataFrame([dict(dataset=item["dataset"], model=item["method"], **item["validation"]) for item in selections])
    validation.to_csv(output / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    sections = ["# CP-CHILD 分类实验结果", "",
        "本报告对应固定协议 `protocol.json`。所有模型、参数、检查点和决策阈值先在训练/验证部分确定，并写入 `selection_lock.json`，之后才计算测试指标。未按测试结果再次调参。", "",
        "## 数据与划分", "",
        "原始 9,500 张图像保持不变。训练输入排除 27 张与测试内容相同的副本，以及 71 张训练内部完全重复副本；8,002 张开发图像进一步划分为训练 6,398 张、验证 1,604 张。保守地把 74 对感知哈希候选并入相似图像组，未根据感知哈希删除图像。", "",
        "官方测试集为 1,400 张；唯一像素口径为 1,376 张。两种口径都提前声明，不能根据分数择优采用。无患者或视频对应表，相似组不是患者组；下列结果是图像级评价，不能确认患者级独立，也不构成临床有效性结论。", "",
        "| 子集 | 划分 | 非息肉 | 息肉 | 总数 |", "|---|---|---:|---:|---:|"]
    for (dataset, phase), members in frame[frame.phase != "excluded"].groupby(["dataset", "phase"]):
        sections.append(f"| {dataset} | {phase} | {sum(members.target==0)} | {sum(members.target==1)} | {len(members)} |")
    sections += ["", "## 官方测试集结果", "",
        "Polyp=1 为阳性。Recall 表示息肉敏感度；Specificity 表示非息肉特异度。AP 使用 scikit-learn 的 Average Precision 定义。SVM 以决策分数计算 AUROC/AP，未把其分数当成概率。", ""]
    winners = {}
    for dataset in ("CP-CHILD-A", "CP-CHILD-B"):
        subset = metrics[(metrics.experiment == "within_device") & (metrics.test_dataset == dataset) & (metrics.test_policy == "official")]
        sections += [f"### {dataset}", "", markdown_table(subset, ["model", "accuracy", "recall", "specificity", "f1", "auroc", "average_precision"],
            ["方法", "Accuracy", "Recall", "Specificity", "F1", "AUROC", "AP"]), ""]
        choices = validation[(validation.dataset == dataset) & validation.model.str.startswith("M") & (validation.model != "M0_majority")]
        chosen = choices.sort_values(["average_precision", "f1"], ascending=False, kind="stable").iloc[0]
        winners[dataset] = chosen.model
        test_row = subset[subset.model == chosen.model].iloc[0]
        sections += [f"按验证集 AP/F1 规则选定 **{DISPLAY[chosen.model]}**，其验证 AP={chosen.average_precision:.4f}；对应官方测试 Recall={test_row.recall:.4f}、Specificity={test_row.specificity:.4f}、F1={test_row.f1:.4f}。此处的推荐依据是验证表现，不是测试表中的最高分。", "",
            f"![ROC和PR](figures/{dataset}_curves.png)", "",
            f"![混淆矩阵](figures/{dataset}_confusions.png)", ""]
        error_gallery(cfg, dataset, chosen.model, figures / f"{dataset}_errors.jpg")
        from .explain import occlusion_gallery
        occlusion_gallery(cfg, dataset, chosen.model, figures / f"{dataset}_occlusion.jpg")
    sections += ["## 测试像素去重敏感性分析", ""]
    unique = metrics[(metrics.experiment == "within_device") & (metrics.test_policy == "unique_pixels")]
    sections += [markdown_table(unique, ["test_dataset", "model", "n", "recall", "specificity", "f1", "average_precision"],
        ["子集", "方法", "样本数", "Recall", "Specificity", "F1", "AP"]), "",
        "## 跨设备迁移", ""]
    cross = metrics[(metrics.experiment == "cross_device") & (metrics.test_policy == "official")]
    sections += [f"在查看 B/Test 评分前，按 A/Validation 选定 `{lock['cross_device_selected_model']}`，将其 A 训练模型和 A 验证阈值直接用于 B/Test。该迁移模型未使用 B/Train 或 B/Validation。A/B 同属同一医院，且患者对应关系未知，结果只能解释为跨设备测试，不能称为独立多中心外部验证。", "",
        markdown_table(cross, ["model", "n", "recall", "specificity", "f1", "auroc", "average_precision"],
                       ["A训练模型", "B测试样本", "Recall", "Specificity", "F1", "AUROC", "AP"]), "",
        "## 实施细节与解释范围", "",
        "- 手工特征 140 维：38 维颜色特征（HSV 直方图、Lab 均值/标准差），102 维纹理特征（多尺度 LBP、GLCM）。另有仅颜色/仅纹理两组消融。",
        "- M3 使用 ImageNet ResNet18 的 512 维 GAP 特征；M5 使用与 M3 完全相同的 CNN 特征，拼接手工特征后分块标准化并按维度归一加权。",
        "- M4 冻结 conv1 至 layer3，先训练分类头，再微调 layer4 和分类头，BatchNorm 运行统计固定。训练时在原图和一份固定轻度增强视图之间采样；没有裁剪，验证/测试没有随机增强。",
        "- 冻结前端激活以 float16 缓存，训练后端为 float32。此缓存是确定性计算优化，不是在全数据上微调；仅使用训练部分更新网络权重。",
        "- SVM 比较线性/RBF 核和预先声明的 C、类别权重；随机森林比较预先声明的叶节点样本量。标准化在训练部分拟合。",
        "- 各方法使用同一分组划分。超参数、早停和阈值均基于同一固定验证集；本轮不是嵌套交叉验证，验证分数不能当作无偏最终性能。",
        "- 当前是一轮固定种子 42 的对照实验。尚未做多种子稳定性和患者级置信区间；不能把很小的模型分差当作统计显著优势。",
        "- 正类标注含息肉，非息肉可能包含其他病变；本项目不输出息肉病理分型、良恶性或定位精度。", "",
        "## 复核与交付", "",
        "- `split_manifest.csv`：逐图固定划分、相似组、排除原因和测试去重标志。",
        "- `selection/*.json`、`*.trials.csv`：参数搜索、验证指标、选择结果及 CNN 训练曲线。",
        "- `selection_lock.json`：测试前锁定的模型哈希、阈值和跨设备候选。",
        "- `models/`：可重载的传统/混合模型、CNN 检查点及共享预训练特征网络。",
        "- `predictions/`：每张图像的分数、预测及 FN/FP 错误清单。",
        "- `metrics.csv`、`validation_metrics.csv`：全部评价表。",
        "- `inference_roundtrip_checks.csv`：16 项从原图重新读取模型预测的复核，覆盖两子集、四种方法和两类标签。",
        "- `figures/`：对照图、ROC/PR、混淆矩阵和误分类样本。",
        "- `environment*.json`、`requirements-lock.txt`、`logs/`：依赖版本、运行日志和状态。", "",
        "误分类图仅展示模型错误，不做医学重新诊断。`figures/*_occlusion.jpg` 针对每个验证集选定模型自己的最终评分执行 4×4 平均色遮挡分析；SVM 热图来自 SVM 自己的评分。热图表示对扰动的敏感性，不是病灶分割或定位真值。", ""]
    path = output / "实验结果报告.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    save_json(output / "recommended_models.json", {dataset: dict(model=method, selection="validation AP then F1") for dataset, method in winners.items()})
    log(f"Wrote report: {path}")
