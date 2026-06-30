# Home2 Results Summary

| Experiment | Final Epoch | Final Train Loss | Final Train Acc | Final Test Loss | Final Test Acc | Best Test Acc | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vit_from_scratch | 30 | 0.0176 | 99.44% | 0.0700 | 98.25% | 98.27% | 25 |
| timm_finetune_full | 30 | 0.0168 | 99.42% | 0.1686 | 95.77% | 96.18% | 11 |
| timm_linear_probe | 100 | 0.5166 | 81.87% | 0.5308 | 81.58% | 81.85% | 95 |
| position_with_pos_embed | 30 | 0.0176 | 99.44% | 0.0700 | 98.25% | 98.27% | 25 |
| position_without_pos_embed | 30 | 0.0363 | 98.77% | 0.1502 | 96.21% | 96.50% | 28 |

说明：本表汇总 ViT 从零训练、预训练 ViT 微调策略对比和位置编码消融实验结果。
