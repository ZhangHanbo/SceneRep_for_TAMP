# scenerep

## Installation
```bash
conda create -n scenerep python=3.11
conda activate scenerep
cd scenerep
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```
Download the checkpoints [OWLv2 CLIP B/16 ST/FT ens](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b) to `~/scenerep/rosbag2dataset/owl`.
Modify "checkpoint_path" in `~/anaconda3/envs/scenerep/lib/python3.11/site-packages/scenic/projects/owl_vit/configs/owl_v2_clip_b16.py`.

Then download the checkpoints [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to `~/scenerep/rosbag2dataset/sam`.

(Optional) Install MobileSAM for realtime application: [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

## Data Processing
1. Prepare ROS bag
Bag should include: RGB-D images, TF, End-effector pose

2. Convert ROS bag to dataset
```bash
cd ~/scenerep
python rosbag2dataset/icp_amcl.py
python rosbag2dataset/rosbag2dataset_5hz.py [dataname.bag]
```

3. Run OWL-ViT object scoring & SAM segmentation
```bash
python rosbag2dataset/owl/owl_object_scores.py [dataname]
python rosbag2dataset/sam/sam.py [dataname]
```

## Demo

1. Download demo data from huggingface： [Mobile_Manipulation_on_Fetch](https://huggingface.co/datasets/wby031018/Mobile_Manipulation_on_Fetch)
2. Edit file path in config file: configs/demo.yaml
3. Run:
```bash
python data_demo.py --config configs/demo.yaml
```

## Running Realtime Application with ROS
1. Check config file: configs/realtime_app.yaml
2. launch detection server: 
```bash
cd det_pipeline
python det_server.py
```
3. Open another window and run realtime application:
```bash
python realtime_app.py
```

## Further Reading

- `docs/DISCUSSION.md` — architectural rationale (two-tier hierarchy, base-frame fusion, composed observation noise).
- `docs/PLAN.md` — implementation roadmap with current status mapping each task to its source file.
- `docs/improvements.md` — original EKF + factor-graph design notes.
- `docs/khronos_lessons.md` — design patterns borrowed from Khronos.
- `docs/survey_and_analysis.md` — comparative survey vs TSDF++/MidFusion/ConceptGraphs/etc.
- `det_pipeline/README_detect_objects.md` — `detect_objects_on_image()` API reference.