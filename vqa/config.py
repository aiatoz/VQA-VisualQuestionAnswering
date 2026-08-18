#--------------------------------------------------------------------------------------------<Central Config>------------||

import os
import torch
from pathlib import Path

ROOT = Path(os.environ.get("VQA_ROOT", Path(__file__).resolve().parent.parent)) #chnage if dataset is kept separately

#--------------------------------------------------------------------------------------------<Dataset Paths>-------------||
DATA_DIR    = ROOT / "Dataset"
RAWZIP_DIR  = DATA_DIR / "rawZips"      #-------------------| Downloaded VQAv2 / COCO zips
EXTRACT_DIR = DATA_DIR / "extracted"    #-------------------| Unzipped jsons and full COCO image folders
IMAGE_DIR   = DATA_DIR / "images"       #-------------------| Flat folder, only the sampled images
IMAGE_SIZE_DIR = 224                    #-------------------| Read pre resized images from images224/, None = originals

def activeImageDir():
    # The folder the collator actually reads from. Falls back to the originals
    # if resizeImages() isnt used
    if IMAGE_SIZE_DIR:
        resized = DATA_DIR / f"images{IMAGE_SIZE_DIR}"
        if resized.exists() and any(resized.glob("*.jpg")):
            return resized
    return IMAGE_DIR
ANSWER_FILE = DATA_DIR / "answerSpace.json"

SPLIT_DIR = {
    "train" : DATA_DIR / "train",
    "val"   : DATA_DIR / "val",
    "test"  : DATA_DIR / "test",
}

def csvPath(split):
    return SPLIT_DIR[split] / f"{split}QstnAns.csv"

#--------------------------------------------------------------------------------------------<Output Paths>--------------||
OUT_DIR    = ROOT / "outputs"
LOG_DIR    = OUT_DIR / "logs"
FIGURE_DIR = ROOT / "report" / "figures"

def runDir(useLoRA=False, frozen=False):
    # One folder per training mode, keeps checkpoints from stomping on each other
    if useLoRA:
        name = "withLoRA"
    elif frozen:
        name = "frozenEncoders"
    else:
        name = "woLoRA"
    return OUT_DIR / name

#--------------------------------------------------------------------------------------------<Source URLs>---------------||
# Crucial. This has the dataset URLs
URLS = {
    "trainImg"  : "http://images.cocodataset.org/zips/train2014.zip",
    "valImg"    : "http://images.cocodataset.org/zips/val2014.zip",
    "trainQstn" : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "valQstn"   : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "trainAnnot": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "valAnnot"  : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}


COCO_SPLIT = {"train": "train2014", "val": "val2014", "test": "val2014"}

#--------------------------------------------------------------------------------------------<Sampling Plan>-------------||

TOTAL_IMAGES = 5000
SPLIT_RATIO  = {"train": 0.73, "val": 0.135, "test": 0.135} #3650 / 675 / 675 plan

def imageBudget(split):
    if TOTAL_IMAGES is None:
        return None                        #-----------------| None => takes all
    return int(TOTAL_IMAGES * SPLIT_RATIO[split])

#--------------------------------------------------------------------------------------------<Answer Space>--------------||
TOP_K_ANSWERS = 1000       #--------------------------------| Top-K most frequent answers from TRAIN only
UNK_TOKEN     = "<unk>"    #--------------------------------| Everything else lands here, excluded from scoring

#--------------------------------------------------------------------------------------------<Model Config>--------------||
TEXT_MODEL  = "bert-base-uncased"
IMAGE_MODEL = "google/vit-base-patch16-224-in21k"

FUSION_DIM  = 512
DROPOUT     = 0.5
MAX_TOKENS  = 24

#--------------------------------------------------------------------------------------------<Training Config>-----------||
SEED             = 12345
SUBSET_N         = None    #--------------------------------| Set to e.g. 2000 for a quick smoke run, None = full split
BATCH_SIZE       = 32
EPOCHS           = 5
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 1e-4
GRAD_ACCUM       = 2
# Worker processes each hold a full copy of the interpreter, and evaluation spawns its own set
NUM_WORKERS      = 4       #--------------------------------| Total live workers can be 2x this during eval
PIN_MEMORY       = True    #--------------------------------| Faster host to device copies
PREFETCH_FACTOR  = 2       #--------------------------------| Batches queued per worker
PERSISTENT_WORKERS = False #--------------------------------| True keeps train workers alive through eval, doubling RAM
FAST_IMAGE_PATH  = True    #--------------------------------| Skip the HF processor, uint8 -> float32 directly
EVAL_STEPS       = 250
FP16             = torch.cuda.is_available()

#--------------------------------------------------------------------------------------------<LoRA Config>---------------||
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05

#--------------------------------------------------------------------------------------------<Device>--------------------||
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def makeDirs():
    #----------------------------------------------------------------------------------------<Create folder tree>--------||
    folders = [DATA_DIR, RAWZIP_DIR, EXTRACT_DIR, IMAGE_DIR, OUT_DIR, LOG_DIR, FIGURE_DIR] + list(SPLIT_DIR.values())
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    print(f"Folder tree ready under : {ROOT}")


def checkEnv():
    #----------------------------------------------------------------------------------------<Version sanity>-----------||
    # Env sanity check
    # Very useful. Earlier I was stuck at this point for a really long time
    import sys
    import importlib

    print(f"python   : {sys.version.split()[0]}")
    print(f"executable : {sys.executable}")
    if "envs" not in sys.executable.replace("\\", "/"):
        print("  ^ this looks like a BASE conda install, not a project env.")
        print("    If you meant to use a named env, register its kernel :")
        print("      python -m ipykernel install --user --name <env> --display-name 'Python (<env>)'")
        print("    then Kernel > Change Kernel in Jupyter.")
    print()

    expected = {
        "torch"        : (2, 1),
        "transformers" : (4, 44),
        "datasets"     : (2, 19),
        "peft"         : (0, 11),
    }
    ceilings = {"transformers": 5, "datasets": 4, "peft": 1}

    ok = True
    for name, (major, minor) in expected.items():
        try:
            mod = importlib.import_module(name)
        except ImportError:
            print(f"{name:14} {'-':12} MISSING")
            ok = False
            continue
        except Exception as err:
            #----------------------------------------------------------------------------| broken install, not a missing one
            print(f"{name:14} {'-':12} BROKEN : {type(err).__name__} : {err}")
            ok = False
            continue

        version = getattr(mod, "__version__", "?")
        parts   = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
        flag    = "ok"

        if not parts:
            flag = "version unreadable, install may be damaged"
            ok = False
        elif parts < (major, minor):
            flag = f"TOO OLD, need >= {major}.{minor}"
            ok = False
        elif name in ceilings and parts[0] >= ceilings[name]:
            flag = f"TOO NEW, this project targets {name} < {ceilings[name]}.0"
            ok = False

        print(f"{name:14} {version:12} {flag}")

    #----------------------------------------------------------------------------------------<GPU check>---------------||
    try:
        import torch as _torch
        build = _torch.__version__
        if "+cpu" in build:
            print(f"\ntorch build    : {build}  <-- CPU ONLY, no GPU training")
            print("  pip uninstall torch -y")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            ok = False
        else:
            print(f"\ntorch build    : {build}")

        print(f"cuda available : {_torch.cuda.is_available()}")
        if _torch.cuda.is_available():
            print(f"gpu            : {_torch.cuda.get_device_name(0)}")
            print(f"vram           : {_torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as err:
        print(f"\ntorch check failed : {err}")
        ok = False

    print("\nEnvironment looks fine" if ok else "\nFix the flagged items before running anything")
    return ok


def showConfig():
    #----------------------------------------------------------------------------------------<Print active config>-------||
    print(f"Root           : {ROOT}")
    print(f"Device         : {DEVICE}")
    print(f"Total images   : {TOTAL_IMAGES}  (train {imageBudget('train')} / val {imageBudget('val')} / test {imageBudget('test')})")
    print(f"Answer space   : top-{TOP_K_ANSWERS} + {UNK_TOKEN}")
    print(f"Batch / Epochs : {BATCH_SIZE} / {EPOCHS}")
    print(f"fp16           : {FP16}")
