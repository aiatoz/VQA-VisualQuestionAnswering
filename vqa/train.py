#--------------------------------------------------------------------------------------------<Training Helpers>----------||
# TrainingArguments builder + the train/eval wrapper used by 02_Train.ipynb

import json
import time
import inspect

from copy import deepcopy
from transformers import TrainingArguments, Trainer

from vqa import config as C


#--------------------------------------------------------------------------------------------<Build Args>----------------||
def buildArgs(outDir, epochs=None, batchSize=None, learningRate=None, weightDecay=None,
              evalSteps=None, **overrides):

    epochs       = epochs       if epochs       is not None else C.EPOCHS
    batchSize    = batchSize    if batchSize    is not None else C.BATCH_SIZE
    learningRate = learningRate if learningRate is not None else C.LEARNING_RATE
    weightDecay  = weightDecay  if weightDecay  is not None else C.WEIGHT_DECAY
    evalSteps    = evalSteps    if evalSteps    is not None else C.EVAL_STEPS

    kwargs = dict(
        output_dir                  = str(outDir),
        seed                        = C.SEED,
        eval_strategy               = "steps",
        eval_steps                  = evalSteps,
        logging_strategy            = "steps",
        logging_steps               = evalSteps,
        save_strategy               = "steps",
        save_steps                  = evalSteps,
        save_total_limit            = 2,
        metric_for_best_model       = "vqa",
        greater_is_better           = True,
        load_best_model_at_end      = True,
        per_device_train_batch_size = batchSize,
        per_device_eval_batch_size  = batchSize,
        gradient_accumulation_steps = C.GRAD_ACCUM,
        num_train_epochs            = epochs,
        learning_rate               = learningRate,
        weight_decay                = weightDecay,
        warmup_ratio                = 0.06,
        fp16                        = C.FP16,
        dataloader_num_workers      = C.NUM_WORKERS,
        dataloader_pin_memory       = C.PIN_MEMORY,
        dataloader_persistent_workers = C.PERSISTENT_WORKERS and C.NUM_WORKERS > 0,
        dataloader_prefetch_factor  = C.PREFETCH_FACTOR if C.NUM_WORKERS > 0 else None,
        remove_unused_columns       = False,
        label_names                 = ["labels"],
        report_to                   = "none",
        logging_dir                 = str(C.LOG_DIR),
    )
    kwargs.update(overrides)

    #----------------------------------------------------------------------------------------<Version guard>-------------||
    # Older transformers still call it evaluation_strategy
    valid = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in valid and "evaluation_strategy" in valid:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    dropped = [k for k in list(kwargs) if k not in valid]
    for key in dropped:
        kwargs.pop(key)
    if dropped:
        print(f"[args] not supported by this transformers version, dropped : {dropped}")

    return TrainingArguments(**kwargs)


#--------------------------------------------------------------------------------------------<Train + Evaluate>----------||
def trainModel(dataset, args, fdProcessor, vqaModel, evalPerformance):

    trainer = Trainer(
        model           = vqaModel,
        args            = args,
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["test"],
        data_collator   = fdProcessor,
        compute_metrics = evalPerformance,
    )

    startTime    = time.time()
    trainMetrics = trainer.train()
    evalMetrics  = trainer.evaluate()
    TTT          = time.time() - startTime

    print(f"\n{trainMetrics}\n")
    print(f"{evalMetrics}\n")
    print("Time taken for training :", time.strftime("%H hrs %M mins %S seconds", time.gmtime(TTT)))

    return trainer, trainMetrics, evalMetrics, TTT


#--------------------------------------------------------------------------------------------<Pipeline Benchmark>---------||
def benchmarkPipeline(dataset, fdProcessor, vqaModel, batchSize=None, nBatches=8):
    import torch

    batchSize = batchSize or C.BATCH_SIZE
    data      = dataset["train"]

    dataMs, stepMs = [], []
    optimiser = torch.optim.AdamW([p for p in vqaModel.parameters() if p.requires_grad], lr=1e-5)

    for n in range(nBatches):
        rows = [data[i] for i in range(n * batchSize, (n + 1) * batchSize)]

        #------------------------------------------------------------------------------------<Collator>-------------------||
        start = time.time()
        batch = fdProcessor(rows)
        dataMs.append(1000 * (time.time() - start))

        #------------------------------------------------------------------------------------<Forward + backward>---------||
        batch = {k: v.to(C.DEVICE, non_blocking=True) for k, v in batch.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        out = vqaModel(**batch)
        out["loss"].backward()
        optimiser.step()
        optimiser.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stepMs.append(1000 * (time.time() - start))

    #----------------------------------------------------------------------------------------<Report>---------------------||
    avgData = sum(dataMs[1:]) / len(dataMs[1:])
    avgStep = sum(stepMs[1:]) / len(stepMs[1:])

    print(f"data prep   : {avgData:8.1f} ms / batch")
    print(f"gpu step    : {avgStep:8.1f} ms / batch")
    print(f"ratio       : {avgData / avgStep:8.2f}x")

    workers = max(C.NUM_WORKERS, 1)
    print(f"\nWith {workers} worker(s), estimated GPU utilisation : {100 * min(1.0, avgStep / max(avgData / workers, avgStep)):.0f}%")

    if avgData > 2 * avgStep:
        print("\nData bound. The GPU is waiting on image decoding.")
        print("  1. run P.resizeImages() once, in 01_DataPrep")
        print("  2. raise C.NUM_WORKERS")
    else:
        print("\nCompute bound, the pipeline is keeping up.")

    return avgData, avgStep


#--------------------------------------------------------------------------------------------<Run Record>------------------||
def saveRun(outDir, name, trainMetrics, evalMetrics, TTT, extra=None):
    # One json per run so 04_Report.ipynb can build the comparison table without rerunning anything.
    outDir.mkdir(parents=True, exist_ok=True)
    record = {
        "name"        : name,
        "trainRuntime": TTT,
        "trainMetrics": trainMetrics.metrics if hasattr(trainMetrics, "metrics") else dict(trainMetrics),
        "evalMetrics" : dict(evalMetrics),
        "config"      : {
            "epochs"       : C.EPOCHS,
            "batchSize"    : C.BATCH_SIZE,
            "learningRate" : C.LEARNING_RATE,
            "weightDecay"  : C.WEIGHT_DECAY,
            "topK"         : C.TOP_K_ANSWERS,
            "totalImages"  : C.TOTAL_IMAGES,
        },
    }
    if extra:
        record.update(extra)

    path = outDir / "runRecord.json"
    with open(path, "w") as file:
        json.dump(record, file, indent=2, default=str)
    print(f"Run record saved -> {path}")
    return path
