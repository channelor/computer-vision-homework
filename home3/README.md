# Home3: CLIP、SAM 与多模态系统设计

本作业包含三个部分：CLIP 零样本分类、SAM 交互式分割分析，以及结合 SAM、CLIP、LLM 的“看图说话切东西”助手设计。

## 任务内容

1. 使用 CLIP 对 10 类图像进行零样本分类，并统计准确率和失败案例。
2. 使用或分析 SAM 的点提示、框提示分割效果，比较两种交互方式的差异。
3. 设计一个结合 SAM、CLIP、LLM 的多模态助手，说明系统流程、模块职责和难点。

## 目录结构

```text
home3/
  README.md
  requirements.txt
  src/
    clip_zero_shot.py
    sam_prompt_demo.py
  inputs/
    apple.jpg
    apple_source.txt
  reports/
    clip_report.md
    sam_report.md
    system_design.md
  outputs/
    clip_zero_shot/
    sam_demo/
```

## 1. CLIP 零样本分类

推荐使用 CIFAR-10 测试集完成 10 类零样本分类。CIFAR-10 正好包含 10 个类别，适合截图中的“任选 10 类图片”要求。

运行命令：

```bash
cd /home/robot/Downloads/Task/home3

/home/robot/anaconda3/envs/mamba/bin/python src/clip_zero_shot.py \
  --dataset CIFAR10 \
  --model-name ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --batch-size 128 \
  --num-workers 0 \
  --output-dir outputs/clip_zero_shot
```

输出文件：

```text
outputs/clip_zero_shot/
  summary.md
  prompts.csv
  confusion_matrix.csv
  per_class_accuracy.csv
  top_confusions.csv
  examples.csv
```

文件含义：

- `summary.md`：总体准确率、表现最好/最差类别、主要混淆类别。
- `prompts.csv`：每个类别使用的文本提示词。
- `confusion_matrix.csv`：10 类混淆矩阵。
- `per_class_accuracy.csv`：每类准确率。
- `top_confusions.csv`：最常见错误分类。
- `examples.csv`：部分预测样例和置信度。

如果只想使用单个 prompt，而不是 prompt ensemble：

```bash
/home/robot/anaconda3/envs/mamba/bin/python src/clip_zero_shot.py \
  --dataset CIFAR10 \
  --single-prompt \
  --batch-size 128 \
  --num-workers 0 \
  --output-dir outputs/clip_zero_shot_single_prompt
```

## 2. SAM 交互式分割

当前代码提供本地 SAM 演示脚本。本次实验使用 `inputs/apple.jpg` 作为示例图像，分别测试点提示和框提示。

如果需要重新运行 SAM，请先准备 `sam_vit_b_01ec64.pth` 权重文件，并放到：

```text
checkpoints/sam_vit_b_01ec64.pth
```

安装依赖：

```bash
pip install segment-anything opencv-python matplotlib
```

点提示示例：

```bash
cd /home/robot/Downloads/Task/home3

/home/robot/anaconda3/envs/mamba/bin/python src/sam_prompt_demo.py \
  --image inputs/apple.jpg \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --point 1550 1250 \
  --multimask \
  --output-dir outputs/sam_demo/point_prompt
```

框提示示例：

```bash
/home/robot/anaconda3/envs/mamba/bin/python src/sam_prompt_demo.py \
  --image inputs/apple.jpg \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --box 520 260 2500 2210 \
  --multimask \
  --output-dir outputs/sam_demo/box_prompt
```

点提示 + 负点提示示例：

```bash
/home/robot/anaconda3/envs/mamba/bin/python src/sam_prompt_demo.py \
  --image inputs/apple.jpg \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --point 1550 1250 \
  --negative-point 200 200 \
  --multimask \
  --output-dir outputs/sam_demo/point_with_negative
```

SAM 输出文件：

```text
outputs/sam_demo/.../
  mask_0.png
  mask_1.png
  mask_2.png
  mask_metrics.csv
  prompt_config.txt
```

如果没有本地 checkpoint，可以使用官方 Demo 或 HuggingFace Demo 体验点提示和框提示，然后把观察结果写入 `reports/sam_report.md`。

本次已经保存的 SAM 输出包括：

```text
outputs/sam_demo/point_prompt/
  mask_0.png
  mask_1.png
  mask_2.png
  mask_metrics.csv
  prompt_config.txt

outputs/sam_demo/box_prompt/
  mask_0.png
  mask_1.png
  mask_2.png
  mask_metrics.csv
  prompt_config.txt
```

## 3. 报告撰写顺序

建议按以下顺序完成：

1. 先运行 CLIP 零样本分类，拿到 `outputs/clip_zero_shot/summary.md` 和混淆矩阵。
2. 根据 CLIP 输出填写 `reports/clip_report.md`。
3. 体验或运行 SAM，填写 `reports/sam_report.md`。
4. 完善 `reports/system_design.md`，说明 SAM、CLIP、LLM 如何串联。
5. 最后清理目录，只保留代码、报告、示例图片和正式输出。

## 4. 重点分析方向

CLIP 部分重点分析：

- 总体零样本准确率。
- 哪些类别最容易识别。
- 哪些类别最容易混淆。
- CIFAR-10 低分辨率图片对 CLIP 的影响。
- prompt ensemble 是否比单 prompt 更稳定。

SAM 部分重点分析：

- 点提示操作简单，但可能存在歧义。
- 框提示范围更明确，通常对复杂背景更稳定。
- SAM 是分割模型，不是计数模型；密集苹果计数需要结合检测、实例分割后处理或人工校正。

系统设计部分重点分析：

- SAM 负责分割候选区域。
- CLIP 负责判断区域与文本类别是否匹配。
- LLM 负责理解用户指令、组织流程和生成解释。
- 多模型串联系统需要处理误差累积、类别歧义、密集小目标和交互纠错。
