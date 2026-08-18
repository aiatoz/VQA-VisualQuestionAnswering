#--------------------------------------------------------------------------------------------<Data Processing>-----------||
# Answer space + label mapping + the collator that feeds BERT and ViT
import json
import torch
import numpy as np
import pandas as pd

from collections import Counter
from dataclasses import dataclass
from typing import List
from PIL import Image

from datasets import load_dataset
from transformers import AutoTokenizer, AutoImageProcessor

from vqa import config as C


#--------------------------------------------------------------------------------------------<Answer Space>--------------||
def buildAnswerSpace(topK=None, save=True):
    # top-K most frequent answers in the TRAIN split, sorted for determinism, <unk> pinned at index 0
    topK = topK or C.TOP_K_ANSWERS

    frame  = pd.read_csv(C.csvPath("train"))
    counts = Counter(frame["answer_val"].astype(str))

    kept = [ans for ans, _ in counts.most_common(topK)]
    kept = sorted(kept)
    space = [C.UNK_TOKEN] + kept

    total    = sum(counts.values())
    covered  = sum(counts[a] for a in kept)
    print(f"Distinct answers in train : {len(counts)}")
    print(f"Answer space size         : {len(space)} (top-{topK} + {C.UNK_TOKEN})")
    print(f"Coverage                  : {100 * covered / total:.2f}% of train answers")

    if save:
        payload = {
            "answers"  : space,
            "topK"     : topK,
            "unkToken" : C.UNK_TOKEN,
            "unkIndex" : 0,
            "builtFrom": "train",
        }
        with open(C.ANSWER_FILE, "w") as file:
            json.dump(payload, file, indent=2)
        print(f"Saved -> {C.ANSWER_FILE}")

    return space

def loadAnswerSpace():
    #----------------------------------------------------------------------------------------<Load saved space>----------||
    with open(C.ANSWER_FILE, "r") as file:
        payload = json.load(file)
    return payload["answers"]


def answerToLabel(space):
    # look up dict
    return {ans: idx for idx, ans in enumerate(space)}


#--------------------------------------------------------------------------------------------<Dataset Loading>-----------||
def loadSplits(space, subsetN=None):
    # Loads train + val as a DatasetDict and attaches integer labels.
    subsetN = subsetN if subsetN is not None else C.SUBSET_N
    lookup  = answerToLabel(space)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(C.csvPath("train")),
            "test" : str(C.csvPath("val")), # its the val split
        },
    )

    dataset = dataset.map(
        lambda batch: {"label": [lookup.get(str(ans), 0) for ans in batch["answer_val"]]},
        batched=True,
    )

    if subsetN:
        dataset["train"] = dataset["train"].shuffle(seed=C.SEED).select(range(min(subsetN, len(dataset["train"]))))
        dataset["test"]  = dataset["test"].select(range(min(subsetN // 4, len(dataset["test"]))))
        print(f"Subset mode : train={len(dataset['train'])}, val={len(dataset['test'])}")

    #----------------------------------------------------------------------------------------<Label health check>--------||
    for split in dataset:
        labels = dataset[split]["label"]
        unk    = sum(1 for l in labels if l == 0)
        print(f"{split:6} : {len(labels)} rows | {100 * unk / len(labels):.2f}% mapped to {C.UNK_TOKEN}")

    return dataset


def loadEvalSplit(space, split="test"):
    #----------------------------------------------------------------------------------------<Held out split>------------||
    lookup  = answerToLabel(space)
    dataset = load_dataset("csv", data_files={"eval": str(C.csvPath(split))})
    dataset = dataset.map(
        lambda batch: {"label": [lookup.get(str(ans), 0) for ans in batch["answer_val"]]},
        batched=True,
    )
    return dataset


#--------------------------------------------------------------------------------------------<Fusion Data Processor>-----||
@dataclass
class FusionDataProcessor:
    tokenizer: AutoTokenizer #txt
    preprocessor: AutoImageProcessor  # imgs
    imageDir: str = None   # defaults to C.activeImageDir() at call time

    def __post_init__(self):
        if self.imageDir is None:
            self.imageDir = str(C.activeImageDir())

        # normalisation constants
        mean = getattr(self.preprocessor, "image_mean", [0.5, 0.5, 0.5])
        std  = getattr(self.preprocessor, "image_std",  [0.5, 0.5, 0.5])
        self.imgMean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.imgStd  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)

    # Extract data from imgs
    def processImgs(self, images: List[str]):
        '''
        The HF slow image processor upcasts every image to float64 before rescaling,
        which is 4x the memory of float32 and happens in every dataloader worker at once.
        Since resizeImages() already wrote 224x224 files, the decode,resize, rescale, normalize chain reduces to a 
        uint8 -> float32 tensor op
        '''
        
        if C.FAST_IMAGE_PATH:
            return {"pixel_values": self.fastImgs(images)}

        loaded = [
            Image.open(f"{self.imageDir}/{name}").convert("RGB")
            for name in images
        ]
        processed = self.preprocessor(images=loaded, return_tensors="pt")
        for img in loaded:
            img.close()
        return {
            "pixel_values": processed["pixel_values"],   # no squeeze, keeps the batch dim at size 1
        }

    # Lean image path
    def fastImgs(self, images: List[str]):
        size  = C.IMAGE_SIZE_DIR or 224
        frames = []

        for name in images:
            with Image.open(f"{self.imageDir}/{name}") as img:
                img = img.convert("RGB")
                if img.size != (size, size):
                    img = img.resize((size, size), Image.BILINEAR)
                frames.append(np.asarray(img, dtype=np.uint8))

        # uint8 -> float32, never float64
        batch = torch.from_numpy(np.stack(frames))          # B, H, W, C
        batch = batch.permute(0, 3, 1, 2).float().div_(255) # B, C, H, W in [0, 1]
        batch = (batch - self.imgMean) / self.imgStd        # same normalisation as the HF processor

        return batch

    # Extract data from text
    def processText(self, texts: List[str]):
        encoded = self.tokenizer(
            text=[str(t) for t in texts],
            padding="max_length",
            max_length=C.MAX_TOKENS,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids"      : encoded["input_ids"],
            "token_type_ids" : encoded["token_type_ids"],
            "attention_mask" : encoded["attention_mask"],
        }

    def __call__(self, rawBatch):
        asDict = isinstance(rawBatch, dict)
        return {
            **self.processText(
                rawBatch["question_val"] if asDict else [i["question_val"] for i in rawBatch]
            ),
            **self.processImgs(
                rawBatch["image_name"] if asDict else [i["image_name"] for i in rawBatch]
            ),
            "labels": torch.tensor(
                rawBatch["label"] if asDict else [i["label"] for i in rawBatch],
                dtype=torch.int64,
            ),
        }


def buildProcessor(textModel=C.TEXT_MODEL, imageModel=C.IMAGE_MODEL):
    #----------------------------------------------------------------------------------------<Tokenizer + Extractor>-----||
    tokenizer = AutoTokenizer.from_pretrained(textModel)

    #----------------------------------------------------------------------------------------<Fast processor>------------||
    # use_fast uses the torchvision backend, noticeably quicker than the PIL path.
    # Older transformers builds do not accept the kwarg, hence the fallback.
    try:
        preprocessor = AutoImageProcessor.from_pretrained(imageModel, use_fast=True)
    except TypeError:
        preprocessor = AutoImageProcessor.from_pretrained(imageModel)

    processor = FusionDataProcessor(tokenizer=tokenizer, preprocessor=preprocessor)
    print(f"Collator reading images from : {processor.imageDir}")
    print(f"Image path                   : {'lean float32' if C.FAST_IMAGE_PATH else type(preprocessor).__name__}")
    if not C.FAST_IMAGE_PATH and "Fast" not in type(preprocessor).__name__:
        print("  ^ slow processor, it upcasts to float64. Set C.FAST_IMAGE_PATH = True.")
    return processor
