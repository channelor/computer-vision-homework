# Home1: ResNet-18 CIFAR-10 图像分类实验

本实验使用 CIFAR-10 数据集完成 ResNet-18 图像分类任务，包括从零训练和预训练模型微调两部分。实验结果、训练曲线和对比分析见 `reports/experiment_report.md`。

## 实验内容

1. 基于 ResNet-18 设计适用于 CIFAR-10 的网络结构，并从零开始训练。
2. 加载 ImageNet 预训练 ResNet-18，在 CIFAR-10 上进行微调。
3. 对比两种方法的 Loss、Accuracy、收敛速度和最终测试准确率。

## 实验结果

| 方法 | 最终测试准确率 | 最佳测试准确率 |
| --- | ---: | ---: |
| 从零训练 ResNet-18 | 95.00% | 95.14% |
| 预训练 ResNet-18 微调 | 93.24% | 93.58% |

从零训练模型在 CIFAR-10 官方测试集上的最终准确率为 95.00%，超过 85% 的目标要求。

## 结果分析

从零训练模型最终准确率更高，主要原因是网络结构针对 CIFAR-10 的 32x32 小图像做了调整：首层卷积由原始 ResNet-18 的大卷积改为 3x3 卷积，并去掉了最大池化层。这使得网络能够保留更多细节信息，更好地适应小图像分类。

预训练微调模型的优势体现在收敛速度上。它加载了 ImageNet 预训练特征，第 1 个 epoch 已达到较高测试准确率，说明通用视觉特征可以迁移到 CIFAR-10 数据集。虽然最终准确率略低于从零训练，但收敛更快、训练更稳定。

总体来看，预训练微调适合快速获得较好结果；从零训练在训练时间充足、模型结构针对数据集适配的情况下，最终性能更好。

## 目录结构

```text
home1/
  README.md
  requirements.txt
  提交清单.md
  src/
    models.py
    train_from_scratch.py
    finetune_pretrained.py
    summarize_results.py
  reports/
    experiment_report.md
  outputs/
    results_summary.md
    from_scratch/
      training_log.csv
      loss_curve.png
      acc_curve.png
    finetune/
      training_log.csv
      loss_curve.png
      acc_curve.png
```

## 运行方法

安装依赖：

```bash
pip install -r requirements.txt
```

从零训练 ResNet-18：

```bash
python src/train_from_scratch.py \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1 \
  --num-workers 2 \
  --output-dir outputs/from_scratch
```

微调预训练 ResNet-18：

```bash
python src/finetune_pretrained.py \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --num-workers 2 \
  --output-dir outputs/finetune
```

汇总实验结果：

```bash
python src/summarize_results.py
```

## 输出文件

- `outputs/from_scratch/training_log.csv`：从零训练日志。
- `outputs/from_scratch/loss_curve.png`：从零训练 Loss 曲线。
- `outputs/from_scratch/acc_curve.png`：从零训练 Accuracy 曲线。
- `outputs/finetune/training_log.csv`：预训练微调日志。
- `outputs/finetune/loss_curve.png`：预训练微调 Loss 曲线。
- `outputs/finetune/acc_curve.png`：预训练微调 Accuracy 曲线。
- `outputs/results_summary.md`：最终测试准确率汇总。
