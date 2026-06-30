# DeiT 论文精读报告

论文：Training data-efficient image transformers & distillation through attention

## 1. 论文背景

ViT 证明了 Transformer 可以直接用于图像分类，但原始 ViT 对数据规模非常敏感。相比 CNN，ViT 缺少卷积结构中的局部感受野和平移等归纳偏置，因此在数据量不足时更容易过拟合。原始 ViT 通常需要 JFT-300M 这类大规模数据集预训练，才能在 ImageNet 等任务上取得很强表现。

DeiT 的核心问题是：如果只使用 ImageNet-1K 这种规模相对有限的数据集，能否训练出效果较好的视觉 Transformer？论文的回答是可以，但需要更强的数据增强、正则化策略和专门为 Transformer 设计的知识蒸馏方法。

## 2. 方法概述

### 2.1 Data-efficient training

DeiT 并没有大幅修改 ViT 主体结构，而是强调训练策略。论文使用较强的数据增强和正则化方法，例如 RandAugment、Mixup、CutMix、Random Erasing、Label Smoothing、Stochastic Depth 等，使 Transformer 在 ImageNet-1K 上也能稳定训练。

这些策略的作用是增加训练样本多样性，减轻模型对训练集的记忆，提升泛化能力。对于 ViT 这类参数量较大、归纳偏置较弱的模型，这些训练技巧尤其重要。

### 2.2 Distillation token

DeiT 的一个关键设计是引入 distillation token。普通 ViT 使用 class token 聚合图像信息并进行分类；DeiT 在 class token 之外额外加入一个 distillation token。这个 token 和其他 patch token 一起经过 Transformer encoder，最后用于学习教师模型提供的监督信息。

因此，DeiT 的输出包含两个分支：class token 对应真实标签监督，distillation token 对应教师模型监督。推理时可以融合两个分支的预测结果，也可以使用其中一个分支。

### 2.3 Teacher-student distillation

DeiT 使用 CNN 作为教师模型，引导 Transformer 学习。教师模型通常已经在 ImageNet 上训练得较好，它的预测包含类别之间的相似性信息。例如，一张狗的图片即使真实标签是某个具体犬类，教师模型可能也会给相近类别较高概率。这类软信息能够帮助学生模型学习更平滑的决策边界。

论文中特别强调 hard distillation，即使用教师模型预测出的类别作为额外标签来监督 distillation token。相比只使用真实标签，教师输出能够提供另一种训练信号，使 Transformer 更容易在有限数据上学习到有效视觉表示。

## 3. 与 ViT 的区别

| 方面 | ViT | DeiT |
| --- | --- | --- |
| 数据需求 | 通常依赖超大规模预训练数据 | 可以在 ImageNet-1K 上训练出较好结果 |
| 主体结构 | Patch embedding + Transformer encoder + class token | 保留 ViT 主体，增加 distillation token |
| 训练策略 | 原始 ViT 更依赖大数据 | 强化数据增强、正则化和蒸馏 |
| 蒸馏机制 | 无专门蒸馏 token | 使用 distillation token 接收教师监督 |
| 小数据表现 | 数据不足时泛化较弱 | 数据效率更高 |

DeiT 的贡献不在于提出全新的 Transformer block，而在于证明通过合适的训练策略和蒸馏设计，视觉 Transformer 不一定必须依赖超大规模私有数据集。

## 4. 知识蒸馏为何能降低数据依赖

知识蒸馏降低数据依赖的原因主要有三点。

第一，教师模型提供了额外监督信号。真实标签通常是 one-hot 的，只告诉模型唯一正确类别；教师输出则包含类别之间的相似性关系，这使学生模型能够学习更丰富的类别结构。

第二，教师模型具有 CNN 的归纳偏置。CNN 对局部纹理、边缘和空间结构有天然建模优势。通过蒸馏，Transformer 可以间接吸收 CNN 学到的视觉先验，从而弥补 ViT 在小数据场景下归纳偏置不足的问题。

第三，distillation token 让蒸馏信号和分类信号在结构上分离。class token 主要学习真实标签监督，distillation token 主要学习教师监督，两种信息在 Transformer 内部共同传播，最终提升模型表示能力。

## 5. 核心实验结论

DeiT 表明，在不依赖超大规模外部数据集的情况下，视觉 Transformer 也可以在 ImageNet-1K 上取得有竞争力的分类性能。相比普通 ViT，DeiT 通过训练策略和蒸馏机制显著提升了数据效率。

从方法意义看，DeiT 证明了 ViT 的弱点不只是模型结构问题，也和训练数据、增强方式、正则化和监督信号有关。当这些训练条件被系统优化后，Transformer 可以在中等规模数据集上表现得更稳定。

## 6. 思考与评价

DeiT 的优点是实用性强。它没有引入复杂的新模块，而是在 ViT 框架上增加 distillation token，并配合强训练策略，使方法容易复现和扩展。它也推动了后续视觉 Transformer 研究从“依赖大数据预训练”转向“如何更高效地训练”。

但 DeiT 也有局限。首先，蒸馏效果依赖教师模型质量，教师模型不好时学生模型也会受影响。其次，DeiT 主要解决图像分类问题，对检测、分割等密集预测任务还需要额外适配。最后，强数据增强和较长训练仍然带来一定训练成本，并不是完全消除 ViT 对数据和算力的需求。

## 7. 总结

DeiT 的核心贡献是提升 ViT 的数据效率。它通过强数据增强、正则化和知识蒸馏，使 Transformer 可以在 ImageNet-1K 这种相对有限的数据规模上训练出较好分类模型。distillation token 是 DeiT 的关键设计，它让模型在真实标签监督之外，还能从教师模型中学习额外视觉知识。整体来看，DeiT 说明视觉 Transformer 的性能不仅取决于模型结构，也高度依赖训练策略和监督方式。
