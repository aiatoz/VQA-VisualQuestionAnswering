# VQA - Late Fusion (BERT + ViT), with and without LoRA

In 2024, I implemented a Visual Question Answering (VQA) system as part of my Visual Recognition project. Because it had a couple of issues, I wanted to refine the work. This led to VQAv2, which serves as the direct successor to that first version.

## Structure

```
VQA-LateFusion/
├── vqa/ #helper functions
│   ├── config.py
│   ├── dataPrep.py # download, sample, join, write CSVs
│   ├── dataProc.py # answer space, label mapping, FusionDataProcessor
│   ├── model.py # VQAClass, LoRA targets, save/load
│   ├── metrics.py
│   └── train.py # TrainingArguments builder, train wrapper
├── notebooks/
│   ├── 01_DataPrep.ipynb
│   ├── 02_Train.ipynb
│   ├── 03_Inference.ipynb
│   └── 04_Report.ipynb
├── Dataset/
│   ├── rawZips/  extracted/  images/  train/  val/  test/  answerSpace.json
├── outputs/
│   ├── woLoRA/  frozenEncoders/  withLoRA/  logs/
└── report/
    ├── figures/  results.md
```

## Setup

Its better to create a new environment

```powershell
cd "...directory\VQA-LateFusion"
conda create -n vqaProject python=3.11 -y
conda activate vqaProject

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
jupyter lab
```

## Disk space

`01_DataPrep` needs ~40 GB free while it runs, but only ~800 MB survives:

| Stage | Size |
|---|---|
| `rawZips/` (train2014 13 GB + val2014 6 GB + json) | ~19 GB |
| `extracted/` (both COCO folders unpacked) | ~19 GB |
| `Dataset/images/` (the sampled subset) | ~800 MB |
| **Peak** | **~39 GB** |
| **After `cleanupRaw(confirm=True)`** | **~800 MB** |

The last cells of `01_DataPrep` call `P.diskUsage()` and `P.cleanupRaw()`. Cleanup refuses to run
unless all three CSVs exist and every image they reference is on disk, and it is a dry run until you
pass `confirm=True`.

## Running order

1. **01_DataPrep** - downloads ~19 GB of COCO images the first time, then samples `C.TOTAL_IMAGES`
   (default 5000) and writes the three CSVs plus `answerSpace.json`. Rerunnable, everything is skipped
   if it already exists. Ends with the cleanup cells above.
2. **02_Train** - set the flags at the top, run, repeat for each mode
3. **03_Inference** - point at a run, get test VQA accuracy and the breakdown.
4. **04_Report** - collects whatever has been trained into one table and figure set.