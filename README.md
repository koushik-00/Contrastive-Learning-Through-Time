# CLTT: Contrastive Learning with Temporal Triplets

Self-supervised pre-training of a video encoder on a single unlabeled video, evaluated via linear probe on UCF-101.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<repo-name>/blob/main/CLTT_notebook.ipynb)

## Objective

The goal was to see how much a video encoder could learn from a single unlabeled video using contrastive learning. Two clips sampled from nearby timestamps are treated as a positive pair — the model learns to bring them close in embedding space and push apart clips from distant timestamps. After pre-training, the encoder is frozen and a linear classifier is trained on UCF-101 to measure how well the representations transfer to a standard action recognition benchmark.

## Setup

```
pip install torch torchvision yt-dlp decord av tqdm matplotlib
```

The notebook mounts Google Drive and expects the following structure:

```
MyDrive/CLTT_Soccer/
├── frames/
├── checkpoints/
└── ucf101/
```

UCF-101 should be placed in `ucf101/` as `archive.zip`. Download it from the [UCF-101 website](https://www.crcv.ucf.edu/data/UCF101.php).

**Expected runtime:** Frame extraction ~5 min, pre-training ~40 min/epoch, feature extraction ~2 hrs. Plan for 3-4 hours total on an A100.

## Pre-training video

A 3-hour soccer broadcast ([YouTube](https://www.youtube.com/watch?v=gTmR5PMIIjs)). The notebook downloads it automatically if not already on Drive.

## How to run

Run all cells in `CLTT_notebook.ipynb` in order. The notebook handles:
1. Video download and frame extraction
2. Model definition and pre-training
3. Feature extraction on UCF-101
4. Linear probe evaluation

## Model

- **Backbone:** R(2+1)D-18
- **Projection head:** Linear(512, 512) → BatchNorm → ReLU → Linear(512, 128)
- **Loss:** NT-Xent (temperature = 0.07)
- **Optimizer:** AdamW (lr = 3e-4, weight decay = 1e-4)
- **Scheduler:** CosineAnnealingLR
- **Hardware:** Google Colab A100 GPU

## Implementation details

**Clip sampling:**
- Frames extracted at 5 fps, scaled to 128×128
- Each clip is 8 consecutive frames (~1.6 seconds of video)
- Positive pairs: two clips whose start frames are within 2 seconds of each other (`pos_window=2s`)
- Minimum offset between the two clips: 0.8 seconds, so they don't overlap

**Augmentations (applied independently to each clip):**
- RandomCrop to 112×112
- RandomHorizontalFlip
- ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
- Normalized with Kinetics-400 stats (mean=[0.432, 0.395, 0.376], std=[0.228, 0.221, 0.217])

**Linear probe:**
- Encoder frozen after pre-training
- Single Linear(512, 101) head trained for 20 epochs
- AdamW (lr = 1e-3, weight decay = 1e-4), batch size 256
- Train features: 209,582 clips — Test features: 81,743 clips

## Results

**Pre-training** (2 epochs, 55,075 frames at 5 fps, 128×128):

| Epoch | Avg NT-Xent Loss |
|-------|-----------------|
| 1     | 0.6132          |
| 2     | 0.2293          |

![Loss curve](loss_curve.png)

**Linear probe on UCF-101** (frozen encoder, 101 classes):

| Epoch | Test Accuracy |
|-------|--------------|
| 5     | 22.17%       |
| 10    | 22.09%       |
| 15    | 22.39%       |
| 20    | 23.04%       |

## Interpretation

The loss dropped from 0.6132 to 0.2293 over two epochs, a good sign that the model is separating temporally distant clips from nearby ones.

The linear probe hit 23.04% on UCF-101. Random chance across 101 classes is ~1%, so the model is picking up something real. Fully supervised R(2+1)D-18 on the same benchmark reaches 75-85%, so the gap is large.

Pre-training on one soccer video and testing on 101 diverse action classes is a hard transfer problem — most of UCF-101 looks nothing like soccer footage. Two epochs doesn't give the encoder much to work with either. The result that stands out is that any transfer happened at all. Representations learned from a single unlabeled video generalized well enough to support above-chance classification on a benchmark the model never saw.

To push accuracy further, more pre-training epochs or a more diverse video set would be the first things to try.
