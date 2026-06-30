# CLIP Zero-Shot Classification Summary

- Dataset: CIFAR10
- Test samples: 10000
- Model: ViT-B-32
- Pretrained weights: laion2b_s34b_b79k
- Prompt mode: prompt ensemble
- Accuracy: 0.9338

## Best Classes

- horse: 0.9890 (989/1000)
- truck: 0.9820 (982/1000)
- automobile: 0.9760 (976/1000)

## Worst Classes

- deer: 0.8980 (898/1000)
- bird: 0.8960 (896/1000)
- cat: 0.8540 (854/1000)

## Top Confusions

- cat -> dog: 95
- deer -> horse: 75
- dog -> cat: 42
- bird -> deer: 31
- bird -> dog: 27
