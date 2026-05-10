# Dynamic Scene Graph

Object-centric, dynamic 3D scene-graph representation for robotic manipulation.
Given a stream of RGB-D frames, SLAM poses, and gripper proprioception, the
package tracks every object's world-frame pose, geometry, existence, and
pairwise spatial relations across grasp / lift / drop interactions.

## Quickstart

```bash
git clone git@github.com:ZhangHanbo/dynamic_scene_graph.git
cd dynamic_scene_graph

conda create -n dynamic_scene_graph python=3.11
conda activate dynamic_scene_graph
pip install -r requirements.txt
pip install -e .

python demo/run_demo.py
```

`demo/run_demo.py` extracts the bundled `demo/apple_in_the_tray.zip`
(a 37-frame slice of a real Fetch trajectory) into the repo root and
runs the EKF tracker over it. Expected output: one line per frame
listing each tracked object's world-frame position and Bernoulli
existence probability, e.g.

```
frame 0488  objects: 0:apple@[-0.12, 0.41, 0.78] r=0.97, 1:tray@[-0.05, 0.39, 0.74] r=0.99
…
done.
```
