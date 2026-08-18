#--------------------------------------------------------------------------------------------<Data Preparation>----------||
# Download -> extract -> sample images -> join questions and annotations -> write one CSV per split

import json
import shutil
import random
import pandas as pd

from collections import Counter
from urllib.request import urlretrieve
from zipfile import ZipFile

from vqa import config as C


#--------------------------------------------------------------------------------------------<Download and Extract>------||
def splitsBuilt():
    # True once all three CSVs exist and every image they reference is on disk.
    if not all(C.csvPath(s).exists() for s in ["train", "val", "test"]):
        return False

    onDisk = set(p.name for p in C.IMAGE_DIR.glob("*.jpg"))
    if not onDisk:
        return False

    for split in ["train", "val", "test"]:
        needed = set(pd.read_csv(C.csvPath(split))["image_name"])
        if needed - onDisk:
            return False
    return True


def fetchZip(key, extract=True, force=False):
    # Downloads a zip into rawZips/ and extracts into extracted/. Skips whatever already exists.
    '''cleanupRaw() deletes the image zips on purpose. Without this check, rerunning the
    notebook top to bottom afterwards would redownload 19 GB that is no longer needed.'''

    url  = C.URLS[key]
    dest = C.RAWZIP_DIR / f"{key}.zip"

    if not force and not dest.exists() and splitsBuilt():
        print(f"[skip] {key} : splits are already built and {dest.name} was cleaned up.")
        return None

    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
    else:
        print(f"[get ] {key} <- {url}")
        urlretrieve(url, dest)
        print(f"[done] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

    if not extract:
        return dest

    marker = C.EXTRACT_DIR / key
    if marker.exists():
        print(f"[skip] {key} already extracted")
    else:
        with ZipFile(dest, "r") as z:
            z.extractall(path=marker)
        print(f"[done] extracted to {marker}")

    return marker


def findFile(root, pattern):
    #----------------------------------------------------------------------------------------<Locate a file by glob>-----||
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"'{pattern}' not found under {root}")
    return hits[0]


#--------------------------------------------------------------------------------------------<Filename Helper>-----------||
def imgName(imgID, cocoSplit):
    # Padding to address the variable length of the images
    return f"COCO_{cocoSplit}_{str(imgID).zfill(12)}.jpg"


#--------------------------------------------------------------------------------------------<Image Sampling>------------||
def sampleImages(split, seed=C.SEED):
    # Picks the image ids for one split.
    cocoSplit = C.COCO_SPLIT[split]
    folder    = C.EXTRACT_DIR / ("trainImg" if cocoSplit == "train2014" else "valImg") / cocoSplit
    allImgs = sorted(p.name for p in folder.glob("*.jpg"))
    print(f"{cocoSplit} holds {len(allImgs)} images")

    budget = C.imageBudget(split)
    if budget is None:
        picked = allImgs if split != "test" else []
    else:
        rng = random.Random(seed)
        pool = allImgs[:]
        rng.shuffle(pool)

        if split == "train":
            picked = pool[:budget]
        elif split == "val":
            picked = pool[:budget]
        else:
            picked = pool[C.imageBudget("val"):C.imageBudget("val") + budget]

    print(f"{split} : sampled {len(picked)} images")
    return set(picked), folder


#--------------------------------------------------------------------------------------------<Question + Answer Join>----||
def buildSplit(split, copyImages=True, seed=C.SEED, force=False):
    # Produces Dataset/<split>/<split>QstnAns.csv and copies the sampled images into Dataset/images
    if not force and C.csvPath(split).exists():
        sourceGone = not (C.EXTRACT_DIR / ("trainImg" if C.COCO_SPLIT[split] == "train2014" else "valImg")).exists()
        if sourceGone:
            frame = pd.read_csv(C.csvPath(split))
            print(f"[skip] {split} : CSV exists ({len(frame)} rows) and the COCO source folder was cleaned up.")
            print(f"       Reusing {C.csvPath(split).name}. Pass force=True to rebuild, that needs a re-download.")
            return frame

    cocoSplit = C.COCO_SPLIT[split]
    srcKey    = "train" if cocoSplit == "train2014" else "val"

    picked, imgFolder = sampleImages(split, seed)

    qFile = findFile(C.EXTRACT_DIR / f"{srcKey}Qstn",  "*_questions.json")
    aFile = findFile(C.EXTRACT_DIR / f"{srcKey}Annot", "*_annotations.json")

    #----------------------------------------------------------------------------------------<Index annotations>--------||
    # Lookup dict instead of scanning all annotations per question
    with open(aFile, "r") as file:
        annotations = json.load(file)["annotations"]
    annotByQid = {a["question_id"]: a for a in annotations}
    print(f"Indexed {len(annotByQid)} annotations")

    with open(qFile, "r") as file:
        questions = json.load(file)["questions"]
    print(f"Loaded {len(questions)} questions")

    #----------------------------------------------------------------------------------------<Collect matching rows>----||
    rows = []
    for qT in questions:
        fname = imgName(qT["image_id"], cocoSplit)
        if fname not in picked:
            continue

        annot = annotByQid.get(qT["question_id"])
        if annot is None:
            continue # question with no annotation, skip

        answers = [a["answer"] for a in annot["answers"]]
        rows.append({
            "image_id"      : qT["image_id"],
            "image_name"    : fname,
            "question_id"   : qT["question_id"],
            "question_val"  : qT["question"],
            "answer_val"    : annot["multiple_choice_answer"],
            "answers"       : json.dumps(answers),
            "question_type" : annot["question_type"],
            "answer_type"   : annot["answer_type"],
        })

    frame = pd.DataFrame(rows)
    C.SPLIT_DIR[split].mkdir(parents=True, exist_ok=True)
    frame.to_csv(C.csvPath(split), index=False)
    print(f"{split} : {len(frame)} questions over {frame['image_name'].nunique()} images -> {C.csvPath(split).name}")

    #----------------------------------------------------------------------------------------<Copy images across>-------||
    if copyImages:
        C.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        used = set(frame["image_name"])
        copied = 0
        for fname in used:
            target = C.IMAGE_DIR / fname
            if not target.exists():
                shutil.copy(imgFolder / fname, target)
                copied += 1
        print(f"{split} : copied {copied} new images ({len(used)} used)")

    return frame


#--------------------------------------------------------------------------------------------<Pre resize Images>--------||
def resizeImages(size=224, quality=95, workers=8):
    '''Decoding a 640x480 JPEG and resizing it to 224 costs ~15-25 ms on CPU. Done inside the
    collator that is paid for every image, every epoch, and it starves the GPU.
    Doing it once here cuts per batch data prep by roughly 5-10x.

    ViT resizes to a square 224x224 anyway, so pre-resizing to the same square changes nothing
    about what the model sees'''

    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    target = C.IMAGE_DIR.parent / f"images{size}"
    target.mkdir(parents=True, exist_ok=True)

    sources = sorted(C.IMAGE_DIR.glob("*.jpg"))
    todo    = [p for p in sources if not (target / p.name).exists()]
    print(f"{len(sources)} images, {len(todo)} still to resize -> {target}")

    if not todo:
        print("Nothing to do")
        return target

    def shrink(path):
        with Image.open(path) as img:
            img = img.convert("RGB").resize((size, size), Image.BILINEAR)
            img.save(target / path.name, "JPEG", quality=quality)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(shrink, todo):
            done += 1
            if done % 500 == 0:
                print(f"  {done} / {len(todo)}")

    before = sum(p.stat().st_size for p in sources)
    after  = sum(p.stat().st_size for p in target.glob("*.jpg"))
    print(f"Done. {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB")
    print(f"Set C.IMAGE_SIZE_DIR = {size} to make the collator read from here.")

    return target


#--------------------------------------------------------------------------------------------<Disk Cleanup>-------------||
def diskUsage():
    def folderSize(folder):
        if not folder.exists():
            return 0
        return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())

    targets = {
        "rawZips"   : C.RAWZIP_DIR,
        "extracted" : C.EXTRACT_DIR,
        "images"    : C.IMAGE_DIR,
        "csv+json"  : C.DATA_DIR,
    }

    print(f"{'folder':12} {'size':>10}")
    for name, folder in targets.items():
        print(f"{name:12} {folderSize(folder) / 1e9:>9.2f} GB")


def cleanupRaw(dropZips=True, dropExtracted=True, confirm=False):
    # Deletes the COCO zips and the unpacked folders once the splits are built
    # Only the sampled images in Dataset/images and the CSVs are needed from here on
    for split in ["train", "val", "test"]:
        if not C.csvPath(split).exists():
            raise RuntimeError(f"{split} CSV missing, refusing to clean up. Build the splits first.")

    onDisk = set(p.name for p in C.IMAGE_DIR.glob("*.jpg"))
    needed = set()
    for split in ["train", "val", "test"]:
        needed |= set(pd.read_csv(C.csvPath(split))["image_name"])

    missing = needed - onDisk
    if missing:
        raise RuntimeError(f"{len(missing)} images referenced by the CSVs are not in {C.IMAGE_DIR}, refusing to clean up.")

    print(f"All {len(needed)} referenced images present in {C.IMAGE_DIR}")

    targets = []
    if dropZips:
        targets += [p for p in C.RAWZIP_DIR.glob("*.zip") if "Img" in p.stem]   #----| keep the small json zips
    if dropExtracted:
        targets += [C.EXTRACT_DIR / key for key in ["trainImg", "valImg"] if (C.EXTRACT_DIR / key).exists()]

    freed = 0
    for target in targets:
        size = target.stat().st_size if target.is_file() else sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
        freed += size
        print(f"  {'[would drop]' if not confirm else '[dropping]  '} {target.name:14} {size / 1e9:.2f} GB")

    if not confirm:
        print(f"\nWould free {freed / 1e9:.2f} GB. Rerun with confirm=True to actually delete.")
        return

    for target in targets:
        if target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    print(f"\nFreed {freed / 1e9:.2f} GB")


#--------------------------------------------------------------------------------------------<Sanity Checks>-------------||
def checkSplits():
    frames = {s: pd.read_csv(C.csvPath(s)) for s in ["train", "val", "test"]}
    imgs   = {s: set(f["image_name"]) for s, f in frames.items()}

    print("\n- Split sizes")
    for split, frame in frames.items():
        print(f"{split:6} : {len(frame):6} questions | {frame['image_name'].nunique():5} images")

    print("\n- Image overlap (must all be 0)")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"{a} n {b} : {len(imgs[a] & imgs[b])}")

    print("\n- Files on disk")
    onDisk = set(p.name for p in C.IMAGE_DIR.glob("*.jpg"))
    for split in frames:
        missing = imgs[split] - onDisk
        print(f"{split:6} : {len(missing)} missing")

    print("\n- Answer coverage (train)")
    counts = Counter(frames["train"]["answer_val"])
    total  = sum(counts.values())
    topK   = sum(c for _, c in counts.most_common(C.TOP_K_ANSWERS))
    print(f"distinct answers : {len(counts)}")
    print(f"top-{C.TOP_K_ANSWERS} covers  : {100 * topK / total:.2f}% of answer mass")

    return frames
