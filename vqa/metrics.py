#--------------------------------------------------------------------------------------------<Performance Tracking>------||
#The official VQA accuracy plus the sklearn metrics from the old notebook

import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from vqa import config as C


#--------------------------------------------------------------------------------------------<Answer Normalisation>------||
CONTRACTIONS = {"aint": "ain't", "dont": "don't", "isnt": "isn't", "wont": "won't", "cant": "can't"}
ARTICLES     = {"a", "an", "the"}

def normaliseAnswer(answer):
    # Light version of the official VQA eval normalisation, enough for consistent scoring.
    answer = str(answer).lower().strip().replace("\n", " ").replace("\t", " ")
    answer = "".join(ch for ch in answer if ch.isalnum() or ch in " '")
    tokens = [CONTRACTIONS.get(t, t) for t in answer.split() if t not in ARTICLES]
    return " ".join(tokens)


#--------------------------------------------------------------------------------------------<VQA Soft Accuracy>---------||
def vqaScore(prediction, humanAnswers):
    # Official metric : an answer scores 1.0 only if at least 3 of the 10 annotators gave it
    prediction = normaliseAnswer(prediction)
    if prediction == normaliseAnswer(C.UNK_TOKEN) or prediction == "":
        return 0.0
    matches = sum(1 for a in humanAnswers if normaliseAnswer(a) == prediction)
    return min(matches / 3.0, 1.0)


def batchVqaScore(predictions, answerLists):
    return float(np.mean([vqaScore(p, a) for p, a in zip(predictions, answerLists)]))


#--------------------------------------------------------------------------------------------<Metric Store>--------------||
def freshMetrics():
    return {"vqa": [], "acc": [], "f1": [], "prec": [], "recall": [], "steps": 0}


#--------------------------------------------------------------------------------------------<Evaluation metrices>-------||
def makeEvalPerformance(answerSpace: List[str], evalAnswers: List[List[str]] = None,
                        metricsData: Dict = None, useWandb: bool = False):
    
    '''Returns a compute_metrics function for the HF Trainer.
    evalAnswers is the list of 10 human answers per eval row, in dataset order, used for VQA accuracy.
    Trainer evaluates sequentially so the ordering lines up'''

    metricsData = metricsData if metricsData is not None else freshMetrics()

    def evalPerformance(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_tuple
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(axis=-1)

        #--------------------------------------------------------------------------------<Sklearn metrics>---------------||
        present   = sorted(set(labels.tolist()) | set(preds.tolist()))
        accuracy  = accuracy_score(labels, preds)
        f1        = f1_score(labels, preds, average="macro", labels=present, zero_division=0)
        precision = precision_score(labels, preds, average="macro", labels=present, zero_division=0)
        recall    = recall_score(labels, preds, average="macro", labels=present, zero_division=0)

        #--------------------------------------------------------------------------------<VQA soft accuracy>-------------||
        if evalAnswers is not None and len(evalAnswers) >= len(preds):
            words = [answerSpace[p] for p in preds]
            vqa   = batchVqaScore(words, evalAnswers[:len(preds)])
        else:
            vqa = float("nan")

        metricsData["vqa"].append(vqa)
        metricsData["acc"].append(accuracy)
        metricsData["f1"].append(f1)
        metricsData["prec"].append(precision)
        metricsData["recall"].append(recall)
        metricsData["steps"] += 1

        if useWandb:
            import wandb
            wandb.log({"vqa": vqa, "acc": accuracy, "f1": f1, "prec": precision, "recall": recall})

        return {
            "vqa"    : vqa,
            "acc"    : accuracy,
            "f1"     : f1,
            "prec"   : precision,
            "recall" : recall,
        }

    return evalPerformance, metricsData


#--------------------------------------------------------------------------------------------<Plot Graphs>---------------||
def pltDiag(metricDict, title=None, savePath=None):
    X = range(1, metricDict["steps"] + 1)

    figure, axis = plt.subplots(2, 3, figsize=(16, 8))

    panels = [
        ("vqa",    "VQA Accuracy", axis[0, 0]),
        ("acc",    "Exact Match",  axis[0, 1]),
        ("f1",     "f1 Score",     axis[0, 2]),
        ("prec",   "Precision",    axis[1, 0]),
        ("recall", "Recall",       axis[1, 1]),
    ]

    for key, name, ax in panels:
        ax.plot(X, metricDict[key], marker="o", markersize=3)
        ax.set_title(name)
        ax.set_xlabel("eval step")
        ax.grid(alpha=0.3)

    axis[1, 2].axis("off")
    if title:
        figure.suptitle(title)

    plt.tight_layout()
    if savePath:
        plt.savefig(savePath, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {savePath}")
    plt.show()


#--------------------------------------------------------------------------------------------<Persist Metrics>-----------||
def saveMetrics(metricDict, path):
    with open(path, "w") as file:
        json.dump(metricDict, file, indent=2)
    print(f"Metrics saved -> {path}")


def loadMetrics(path):
    with open(path, "r") as file:
        return json.load(file)
