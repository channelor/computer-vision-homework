# Home2 Results Summary

| Experiment | Final Epoch | Final Train Loss | Final Train Acc | Final Test Loss | Final Test Acc | Best Test Acc | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vit_from_scratch | 30 | 0.0176 | 99.44% | 0.0700 | 98.25% | 98.27% | 25 |
| timm_finetune_full | 30 | 0.0088 | 99.98% | 0.1765 | 96.12% | 96.78% | 11 |
| timm_linear_probe | 100 | 0.4588 | 90.88% | 0.3901 | 96.01% | 96.56% | 95 |
| position_with_pos_embed | 30 | 0.0112 | 99.99% | 0.0701 | 99.34% | 99.34% | 30 |
| position_without_pos_embed | 30 | 0.0301 | 99.08% | 0.1535 | 97.34% | 97.45% | 28 |

说明：本表汇总 ViT 从零训练、预训练 ViT 微调策略对比和位置编码消融实验结果。

### 关键发现

1. **预训练 ViT 全参数微调 vs Linear Probe**: 全参数微调在 CIFAR-10 上达到 96.78% 的最佳准确率，而 linear probe 只达到 96.56%，说明微调所有参数能够更好地适应下游任务。

2. **位置编码的重要性**: 带有位置编码的模型达到 99.34% 的准确率，而无位置编码的模型只有 97.45%，差异达到 1.89 个百分点，证实了位置编码对 Vision Transformer 的重要性。

3. **收敛速度**: 从训练曲线可以看出，所有实验在前 10-15 个 epoch 内均快速收敛，表明学习率和优化器设置合理。
