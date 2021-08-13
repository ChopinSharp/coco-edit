# COCO-Edit Dataset
Code to sample visual edit dataset using COCO image-caption pairs.

Steps:
1. Clone this repository.
```
git clone --recursive git@github.com:ChopinSharp/coco-edit.git
```
2. Setup coco-caption submodule following [this](coco_caption/README.md).
1. Setup [environment](environment.yml).
1. Calculate image-caption similarity.
```
python img_cap_sim.py
```
5. Sample negative captions for editing.
```
python build_cocoedit_mp.py --start <from> --end <to> --split <split> 2>/dev/null
```