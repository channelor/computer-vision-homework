# Home2: Vision Transformer 与 DeiT

本实验围绕 Vision Transformer 完成图像分类实验和论文阅读，包括 ViT 从零训练、预训练 ViT 微调、位置编码消融以及 DeiT 论文分析。

## 实验内容

1. 从零实现一个简化版 Vision Transformer，并在 MNIST 上端到端训练至收敛。
2. 使用 `timm` 加载预训练 ViT，在 CIFAR-10 上进行微调。
3. 对比不同微调策略，包括全参数微调和 linear probe。
4. 去掉 ViT 的位置编码，分析位置编码对分类效果的影响。
5. 阅读 DeiT 论文，分析知识蒸馏如何降低 ViT 对大规模数据的依赖。

## 实验结果

| 实验 | 最终测试准确率 | 最佳测试准确率 |
| --- | ---: | ---: |
| ViT 从零训练 | 98.25% | 98.27% |
| 预训练 ViT 全参数微调 | 95.77% | 96.18% |
| 预训练 ViT linear probe | 81.58% | 81.85% |
| 有位置编码 | 98.25% | 98.27% |
| 无位置编码 | 96.21% | 96.50% |

结果表明，简化 ViT 在 MNIST 上可以完成端到端训练并收敛；位置编码能够提升模型对图像空间结构的建模能力；预训练 ViT 的全参数微调效果明显优于只使用固定特征的 linear probe。

## 目录结构

```text
home2/
  README.md
  requirements.txt
  材料索引.md
  src/
    vit.py
    train_vit.py
    finetune_timm_vit.py
    linear_probe_timm_vit.py
    run_finetune_strategies.py
    position_encoding_ablation.py
    summarize_results.py
  reports/
    vit_experiment_report.md
    deit_reading_report.md
  outputs/
    results_summary.md
    vit_from_scratch/
      training_log.csv
      loss_curve.png
      acc_curve.png
    timm_finetune/
      training_log.csv
      loss_curve.png
      acc_curve.png
    timm_linear_probe/
      training_log.csv
      loss_curve.png
      acc_curve.png
      run_config.txt
    position_ablation/
      with_pos_embed/
        training_log.csv
        loss_curve.png
        acc_curve.png
      without_pos_embed/
        training_log.csv
        loss_curve.png
        acc_curve.png
```

## 运行方法

安装依赖：

```bash
pip install -r requirements.txt
```

从零训练 ViT：

```bash
python src/train_vit.py \
  --dataset MNIST \
  --epochs 30 \
  --batch-size 128 \
  --lr 3e-4 \
  --output-dir outputs/vit_from_scratch
```

位置编码消融：

```bash
python src/position_encoding_ablation.py \
  --dataset MNIST \
  --epochs 30 \
  --batch-size 128 \
  --lr 3e-4
```

预训练 ViT 全参数微调：

```bash
python src/finetune_timm_vit.py \
  --dataset CIFAR10 \
  --model-name vit_tiny_patch16_224 \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-4 \
  --finetune-strategy full \
  --output-dir outputs/timm_finetune
```

预训练 ViT linear probe：

```bash
python src/linear_probe_timm_vit.py \
  --dataset CIFAR10 \
  --model-name vit_tiny_patch16_224 \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-3 \
  --output-dir outputs/timm_linear_probe
```

汇总结果：

```bash
python src/summarize_results.py
```

## 输出说明

- `training_log.csv` 记录每个 epoch 的训练和测试 loss/accuracy。
- `loss_curve.png` 展示训练集和测试集 loss 变化。
- `acc_curve.png` 展示训练集和测试集 accuracy 变化。
- `outputs/results_summary.md` 汇总最终准确率、最佳准确率和最佳 epoch。
- `reports/vit_experiment_report.md` 给出实验设置、曲线分析和结果讨论。
- `reports/deit_reading_report.md` 给出 DeiT 论文精读分析。
