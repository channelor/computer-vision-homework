# ViT 实验报告

## 1. 任务目标

本实验围绕 Vision Transformer 完成三个部分：第一，从零实现一个简化版 ViT，并在 MNIST 上进行端到端训练；第二，使用 `timm` 加载预训练 ViT，在 CIFAR-10 上进行微调；第三，通过去掉位置编码的消融实验，分析位置编码对分类效果的影响。

## 2. 数据集

从零训练和位置编码消融使用 MNIST 数据集。MNIST 图像尺寸为 28x28，类别数为 10，训练集 60,000 张，测试集 10,000 张。该数据集难度较低，适合验证 ViT 的完整训练流程是否正确。

预训练 ViT 微调使用 CIFAR-10 数据集。CIFAR-10 图像尺寸为 32x32，类别数为 10，训练集 50,000 张，测试集 10,000 张。由于 `vit_tiny_patch16_224` 的预训练输入尺寸为 224x224，微调时将 CIFAR-10 图像 resize 到 224x224，并使用 ImageNet 均值和方差进行标准化。

## 3. 模型结构

从零训练使用简化版 ViT：

| 配置 | 数值 |
| --- | ---: |
| 输入图像尺寸 | 28x28 |
| Patch size | 7 |
| Patch 数量 | 16 |
| Embedding dimension | 128 |
| Transformer encoder 层数 | 4 |
| Attention heads | 4 |
| MLP ratio | 4 |
| Class token | 使用 |
| 位置编码 | 默认使用 |

模型流程为：图像首先通过卷积式 patch embedding 转换为 patch token；然后拼接一个 class token；如果启用位置编码，则将可学习位置向量加到 token 序列上；最后输入 Transformer encoder，取 class token 对应输出进行分类。

## 4. 实验设置

| 实验 | 数据集 | Epoch | Batch size | Optimizer | Learning rate | Weight decay |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| ViT 从零训练 | MNIST | 30 | 128 | AdamW | 3e-4 | 0.05 |
| 位置编码消融：有位置编码 | MNIST | 30 | 128 | AdamW | 3e-4 | 0.05 |
| 位置编码消融：无位置编码 | MNIST | 30 | 128 | AdamW | 3e-4 | 0.05 |
| 预训练 ViT 全参数微调 | CIFAR-10 | 30 | 64 | AdamW | 1e-4 | 0.05 |
| 预训练 ViT linear probe | CIFAR-10 | 100 | 256 | AdamW | 1e-3 | 0.01 |

## 5. 实验结果

| 实验 | 最终测试准确率 | 最佳测试准确率 | 最佳 Epoch | 备注 |
| --- | ---: | ---: | ---: | --- |
| ViT 从零训练 | 98.25% | 98.27% | 25 | MNIST，端到端训练 |
| timm 预训练 ViT 全参数微调 | 95.77% | 96.18% | 11 | CIFAR-10，全模型更新 |
| timm 预训练 ViT linear probe | 81.58% | 81.85% | 95 | CIFAR-10，冻结 backbone |
| 有位置编码 | 98.25% | 98.27% | 25 | 消融 baseline |
| 无位置编码 | 96.21% | 96.50% | 28 | 去掉位置编码 |

结果汇总见：`outputs/results_summary.md`。

## 6. 从零训练 ViT 分析

从零训练 ViT 在 MNIST 上最终测试准确率为 98.25%，最佳测试准确率为 98.27%。训练曲线显示，前几个 epoch 准确率迅速提升，随后进入缓慢提升阶段。最后 5 个 epoch 的测试准确率只在约 0.26 个百分点范围内波动，说明模型已经基本收敛。

Loss 曲线也符合收敛特征：训练 loss 持续下降，测试 loss 前期快速下降，后期在较低水平小幅波动。训练准确率最终为 99.44%，测试准确率为 98.25%，泛化差距约 1.19 个百分点，没有出现严重过拟合。

相关曲线：

- `outputs/vit_from_scratch/loss_curve.png`
- `outputs/vit_from_scratch/acc_curve.png`

## 7. 位置编码消融分析

有位置编码时，模型最终测试准确率为 98.25%，最佳测试准确率为 98.27%。去掉位置编码后，最终测试准确率下降到 96.21%，最佳测试准确率下降到 96.50%。最佳准确率下降约 1.77 个百分点，最终准确率下降约 2.04 个百分点。

这说明位置编码对 ViT 是有效的。Transformer 自注意力本身对输入 token 的排列顺序不敏感，如果没有位置编码，模型只能根据 patch 内容判断类别，难以显式利用图像中 patch 的空间位置关系。MNIST 虽然任务较简单，模型在无位置编码时仍然可以学习到较高准确率，但泛化性能明显低于有位置编码的版本。

相关曲线：

- `outputs/position_ablation/with_pos_embed/acc_curve.png`
- `outputs/position_ablation/without_pos_embed/acc_curve.png`

## 8. 预训练 ViT 微调分析

全参数微调的最佳测试准确率为 96.18%，出现在第 11 个 epoch；最终测试准确率为 95.77%。第 1 个 epoch 测试准确率已经达到 95.15%，说明预训练 ViT 已经具备较强的通用视觉特征，迁移到 CIFAR-10 后能够很快获得较好结果。

不过，全参数微调后期测试准确率没有持续提升，而是在 95% 左右波动。训练准确率最终达到 99.42%，测试准确率为 95.77%，泛化差距约 3.65 个百分点，说明模型后期有一定过拟合趋势。实际使用时应选择最佳 epoch 的模型，而不是只看最后一个 epoch。

Linear probe 策略冻结 ViT backbone，只训练线性分类头，最终测试准确率为 81.58%，最佳测试准确率为 81.85%。该结果明显低于全参数微调，说明仅使用固定预训练特征不足以充分适配 CIFAR-10。全参数微调可以调整 backbone 表征，因此性能上限更高；linear probe 的优势是训练参数少、收敛稳定，更适合快速评估预训练特征的可迁移性。

相关曲线：

- `outputs/timm_finetune/acc_curve.png`
- `outputs/timm_linear_probe/acc_curve.png`

## 9. 总结

本实验验证了 ViT 的端到端训练流程，并通过消融实验说明位置编码对图像分类任务有明显帮助。简化 ViT 在 MNIST 上训练至收敛，最终测试准确率达到 98.25%。位置编码消融表明，去掉位置编码后模型仍能训练，但准确率下降约 2 个百分点。预训练 ViT 在 CIFAR-10 上全参数微调效果明显优于 linear probe，说明目标数据集上的参数更新对于提升迁移效果非常重要。
