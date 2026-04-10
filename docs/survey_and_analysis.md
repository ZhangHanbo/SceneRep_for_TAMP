# Object-Centric 3D Scene Representation for Robotic Manipulation: A Comparative Survey and Analysis

## 1. Introduction

This survey examines the landscape of 3D scene representation systems designed for or applicable to robotic manipulation, with particular focus on object-level SLAM, dynamic scene reconstruction during manipulation, and semantic scene graphs. The analysis is centered around a specific system (hereafter "SceneRep") that combines VLM-based detection (OWL-ViT + SAM), per-object TSDF volumes with a background TSDF, manipulation-phase-aware tracking (ICP during grasping/placing, end-effector propagation during movement), camera pose refinement via ICP against accumulated object clouds, and a geometric semantic scene graph with on/in/under/contain relations propagated during manipulation.

We compare this against the major systems in the field, identify what is genuinely novel, and then provide a detailed analysis of algorithmic improvements that could tightly couple the currently loosely-connected components.

---

## 2. Detailed Comparison with Existing Systems

### 2.1 TSDF++ (Grinvald et al., RA-L 2021)

**Core idea.** TSDF++ extends classical TSDF-based volumetric mapping to object-level reconstruction. It maintains a single volumetric grid where each voxel stores not only the truncated signed distance and color but also an *object instance label*. When a new object is detected (via Mask R-CNN), its voxels are "carved out" of the background volume and assigned to a new per-object sub-volume. Objects can be moved, and their voxels travel with them.

**Mathematical formulation.** The TSDF update follows the standard running weighted average:

$$D_i(x) = \frac{W_{i-1}(x) \cdot D_{i-1}(x) + w_i(x) \cdot d_i(x)}{W_{i-1}(x) + w_i(x)}$$

where $d_i(x)$ is the new truncated signed distance observation, $w_i(x)$ is its weight (often depth-dependent), and $W_i(x)$ is the accumulated weight. TSDF++ augments each voxel with an instance ID $l(x) \in \{0, 1, \ldots, K\}$ and resolves conflicts when multiple instance masks overlap the same voxel via a priority scheme.

**Comparison with SceneRep.**
- TSDF++ uses a *single global voxel grid* with per-voxel instance labels; SceneRep uses *separate TSDF volumes per object* plus a background TSDF. The separate-volume approach is more flexible for object manipulation (each object can be independently transformed without re-indexing voxels) but wastes memory on overlapping spatial regions.
- TSDF++ does not model manipulation phases or track objects during grasping.
- TSDF++ does not maintain a semantic scene graph.
- Camera pose in TSDF++ comes from an external SLAM system; SceneRep refines camera poses via ICP against accumulated object clouds.

### 2.2 Co-Fusion (Rünz & Agapito, ICRA 2017)

**Core idea.** Co-Fusion reconstructs multiple independently moving objects in real-time by extending surfel-based dense SLAM. It uses motion segmentation (optical flow + geometric residuals) to detect independently moving regions, then tracks each object with a separate surfel model and a per-object SE(3) motion model.

**Mathematical formulation.** The camera pose $\xi_c$ and each object pose $\xi_k$ are estimated by minimizing a photometric + geometric energy:

$$E(\xi) = \sum_p \left[ \lambda_{ph} \cdot (I_t(p) - I_{t-1}(\pi(\xi, \pi^{-1}(p, d_t(p)))))^2 + \lambda_{geo} \cdot (d_t(p) - [\pi(\xi, \pi^{-1}(p, d_t(p)))]_z)^2 \right]$$

**Comparison with SceneRep.**
- Co-Fusion uses *surfel maps* rather than TSDF volumes.
- Co-Fusion detects motion via geometric residuals; SceneRep uses a state machine based on gripper width. Co-Fusion generalizes to arbitrary moving objects but is noisier; SceneRep's approach is manipulation-specific and more robust for that domain.
- Co-Fusion performs joint camera + object pose optimization via direct image alignment; SceneRep does separate ICP-based pose refinement. Co-Fusion's formulation is more principled.
- Co-Fusion has no semantic understanding or scene graph.

### 2.3 MaskFusion (Rünz et al., ISMAR 2018)

**Core idea.** MaskFusion extends Co-Fusion by replacing motion-based segmentation with Mask R-CNN instance segmentation, enabling the system to track and reconstruct objects even when static.

**Comparison with SceneRep.**
- MaskFusion is the closest architectural ancestor to SceneRep's detection-then-track pipeline. Key difference: surfels (MaskFusion) vs. TSDF volumes (SceneRep).
- MaskFusion uses direct photometric + geometric tracking; SceneRep uses ICP during specific phases.
- SceneRep's use of OWL-ViT + SAM instead of Mask R-CNN gives open-vocabulary detection capability, which is a significant practical advantage.

### 2.4 MidFusion (Xu et al., RA-L 2019)

**Core idea.** MidFusion combines object-level SLAM with octree-based TSDF for each object, and a **joint factor graph** for camera and object pose estimation.

**Mathematical formulation.** The factor graph jointly optimizes:

$$\arg\min_{\{T_c^t\}, \{T_o^k\}} \sum_t \|T_c^t \ominus f_{odom}(T_c^{t-1})\|_{\Sigma_{odom}}^2 + \sum_{t,k} \|h(T_c^t, T_o^k) - z_{t,k}\|_{\Sigma_{obs}}^2$$

**Comparison with SceneRep.**
- MidFusion's factor graph formulation is the key differentiator: it *jointly* optimizes camera and object poses. SceneRep first refines camera pose via ICP, then separately updates object poses.
- MidFusion does not have manipulation awareness or a scene graph.

### 2.5 ConceptGraphs (Gu et al., RSS 2024)

**Core idea.** ConceptGraphs builds open-vocabulary 3D scene graphs by associating CLIP/LLM features with 3D object segments.

**Mathematical formulation.** Object association uses a combined geometric-semantic score:

$$s(o_i, d_j) = \alpha \cdot \text{sim}(f_{o_i}, f_{d_j}) + (1 - \alpha) \cdot \exp\left(-\frac{\|c_{o_i} - c_{d_j}\|^2}{2\sigma^2}\right)$$

**Comparison with SceneRep.**
- ConceptGraphs focuses on *semantic scene graph construction* from **static** scenes; it does not track dynamic manipulation. SceneRep's scene graph is manipulation-aware and propagates state changes.
- ConceptGraphs stores *point clouds* per object rather than TSDF volumes — no mesh extraction.
- ConceptGraphs uses *CLIP features* for rich language grounding; SceneRep uses OWL-ViT scores.
- ConceptGraphs' edge relations are determined by LLM reasoning; SceneRep computes geometric relations from 3D bounding box overlap.
- ConceptGraphs assumes known camera poses; SceneRep refines poses.

### 2.6 BundleSDF (Wen et al., CVPR 2023)

**Core idea.** BundleSDF performs simultaneous 6-DoF tracking and 3D reconstruction of unknown objects from monocular RGBD video. Uses a pose graph with keyframes and a neural implicit surface (neural SDF).

**Mathematical formulation.** Neural SDF trained online:

$$\mathcal{L}_{SDF} = \sum_x |f_\theta(x) - \hat{d}(x)| + \lambda_{eik} \|\|\nabla f_\theta(x)\| - 1\|^2$$

**Comparison with SceneRep.**
- BundleSDF focuses on a *single object*; SceneRep manages a *multi-object scene*. They are complementary.
- BundleSDF's pose graph + bundle adjustment is more sophisticated than SceneRep's frame-by-frame ICP.
- BundleSDF does not consider manipulation or scene graphs.

### 2.7 FoundationPose (Wen et al., CVPR 2024)

**Core idea.** Unified framework for 6-DoF pose estimation and tracking. Works with either CAD model or reference images. Uses render-and-compare with learned pose scoring and refinement networks.

**Comparison with SceneRep.**
- FoundationPose is a *pose estimation* system, not a scene representation system. Could serve as a drop-in replacement for SceneRep's ICP-based pose tracking.
- One could use SceneRep's reconstructed mesh as input to FoundationPose for subsequent tracking.

### 2.8 vMAP (Kong et al., CVPR 2023)

**Core idea.** Represents each object as a small neural network (MLP) that maps 3D coordinates to color and density (per-object NeRF). Camera pose and object poses are jointly optimized via **differentiable rendering**.

**Mathematical formulation:**

$$\mathcal{L} = \sum_t \sum_r \left[ \|\hat{C}(r; \{T_c^t, T_o^k, \theta_k\}) - C(r)\|^2 + \lambda \|\hat{D}(r) - D(r)\|^2 \right]$$

**Comparison with SceneRep.**
- vMAP achieves *joint optimization* of camera pose, object poses, and object shape through differentiable rendering. This is the tight coupling SceneRep lacks.
- vMAP's neural implicit representation provides smooth surfaces but is slower than TSDF.
- vMAP does not model manipulation dynamics or scene graphs.

### 2.9 NICE-SLAM / iMAP

Scene-level neural SLAM systems. Not object-level. Differentiable rendering enables joint camera pose + map optimization but cannot handle dynamic objects.

### 2.10 Other Relevant Systems

- **NodeSLAM (Sucar et al., 3DV 2020):** Per-node neural implicit functions. Related to per-object approach but no semantics.
- **Panoptic Multi-TSDFs (Schmid et al., RA-L 2022):** Extends TSDF++ to panoptic segmentation.
- **SplaTAM (Keetha et al., CVPR 2024):** 3D Gaussian Splatting SLAM. Fast but no object-level decomposition.
- **GaussianGrasper (Zheng et al., ICRA 2024):** Uses 3DGS with language features for grasp planning. Shows 3DGS as manipulation-ready representation.
- **OK-Robot (Liu et al., RSS 2024):** Complete open-knowledge manipulation system. Simpler scene representation but full system integration.

---

## 3. What Makes SceneRep Unique

### 3.1 Manipulation-Phase-Aware Tracking

No other system implements a *manipulation-phase-aware state machine* that selects different tracking strategies per phase:
- **Grasping/Placing**: ICP-based tracking (untrusted transition model — contact is unpredictable)
- **Holding/Movement**: EE-propagation (trusted rigid attachment via $T_{oe}$)
- **Idle**: Standard TSDF fusion

This is a principled design choice grounded in the physics of manipulation: during grasping, the relative transform between gripper and object is uncertain; during movement, rigid attachment is reliable.

### 3.2 Dynamic Geometric Scene Graph with State Propagation

SceneRep maintains a scene graph with four geometric relation types computed from 3D bounding box overlap. When a parent object is moved, its *child objects* are propagated via stored relative transforms. This is not found in ConceptGraphs, OVSG, or other scene graph systems which treat the graph as a static semantic structure.

### 3.3 Camera Pose Refinement via Object-Level ICP

Uses objects as landmarks for camera pose correction — merges point clouds from all visible (non-held) objects and aligns via ICP. Distinct from feature-based SLAM, MidFusion's factor graph, and vMAP's differentiable rendering.

### 3.4 The Integration Itself

No single existing system combines *all* of: open-vocabulary detection, per-object TSDF, manipulation-phase-aware tracking, camera pose refinement against object clouds, and a dynamic scene graph with state propagation.

---

## 4. Algorithmic Improvements for Tight Coupling

The current system is a *pipeline architecture*: each component passes point estimates forward without propagating uncertainty or enabling joint optimization. This section provides concrete formulations for tighter coupling.

### 4.1 Joint Factor Graph Optimization

**Current:** Sequential — camera pose → object pose → scene graph.

**Proposed:** A factor graph (GTSAM/Ceres) jointly optimizing:

1. **Odometry factors** between consecutive camera poses:
$$f_{odom}(T_c^t, T_c^{t+1}) = \|T_c^{t-1} T_c^{t+1} \ominus \Delta T_{odom}\|_\Sigma^2$$

2. **Object observation factors:**
$$f_{obs}(T_c^t, T_o^k) = \sum_{p \in \mathcal{P}_k^t} d(\pi(T_c^{t-1} T_o^k, p), \text{obs}_p)^2$$

3. **Manipulation constraint factors** (during holding, tight covariance):
$$f_{manip}(T_o^k, T_{ee}^t) = \|T_{ee}^{t-1} T_o^k - T_{oe}^k\|_\Sigma^2$$

4. **Scene graph constraint factors** (e.g., "A ON B" constrains vertical relation):
$$f_{graph}(T_o^i, T_o^j) = \max(0, z_j^{top} - z_i^{bottom} + \epsilon)^2$$

This allows the scene graph relation "cup ON table" to constrain the cup's z-coordinate, which in turn affects the camera pose estimate through the observation factor.

### 4.2 Differentiable Rendering for End-to-End Optimization

Replace TSDF with a differentiable representation and optimize all parameters jointly via rendering loss.

**Neural SDF approach (à la vMAP):** Each object $k$ has MLP $f_{\theta_k}$ mapping position to SDF + color. Gradients flow from photometric loss through the renderer into object MLP parameters, object poses, and camera poses simultaneously.

**3D Gaussian Splatting approach:** Each object is a set of Gaussians. Splatting is differentiable and much faster (~30 Hz vs ~2 Hz). Per-object 3DGS with dynamic composition is an open research problem.

**Practical recommendation:** Hybrid — use TSDF for fast online fusion, periodically distill into a differentiable representation for batch pose optimization.

### 4.3 Probabilistic / Bayesian Formulation

**Current:** Deterministic poses + boolean `pose_uncertain` flag.

**Proposed:** Bayesian pose tracking via EKF on $\mathfrak{se}(3)$:

$$p(T_o^k | z_{1:t}) \propto p(z_t | T_o^k) \cdot p(T_o^k | z_{1:t-1})$$

With manipulation-phase-dependent process noise:
- **Holding:** $p(T_o^k | T_{ee}^t) = \mathcal{N}(T_{ee}^t \cdot T_{oe}^k, \Sigma_{tight})$
- **Grasping/Placing:** $p(T_o^k | T_o^{k,t-1}) = \mathcal{N}(T_o^{k,t-1}, \Sigma_{drift})$

**Bayesian label assignment:**
$$p(\ell = k | d_{1:t}) = \frac{p(d_t | \ell = k) \cdot p(\ell = k | d_{1:t-1})}{\sum_j p(d_t | \ell = j) \cdot p(\ell = j | d_{1:t-1})}$$

This connects to the alpha_robot uncertainty calibration work — calibrated detection scores become proper likelihoods.

**Scene graph as probabilistic graphical model:**
$$p(r_{ij} | T_o^i, T_o^j) = \text{softmax}(\phi(T_o^i, T_o^j))$$

### 4.4 Active Perception / Next-Best-View

**Information-theoretic view planning:**
$$v^* = \arg\max_v I(X; Z_v | Z_{1:t})$$

For TSDF, approximate by counting unknown voxels weighted by visibility:
$$I_v \approx \sum_{x \in \text{unknown}} w(x, v) \cdot H(D(x))$$

where $H(D(x))$ is entropy estimated from weight $W(x)$.

**Manipulation-aware active perception:** Plan views that reduce uncertainty in manipulation-relevant quantities — grasp quality, placement feasibility, scene graph verification.

### 4.5 Manipulation-Aware SLAM

**Action-conditioned motion model:**
$$p(T_o^{k,t+1} | T_o^{k,t}, a_t) = \begin{cases} \delta(T_{ee}^{t+1} T_{oe}^k) & \text{if } a_t = \text{hold}(k) \\ \mathcal{N}(T_o^{k,t}, \Sigma_{static}) & \text{if } a_t \text{ doesn't involve } k \\ p_{physics}(T_o^{k,t+1} | T_o^{k,t}, a_t) & \text{if } a_t = \text{push}(k) \end{cases}$$

**Manipulation as observation:** A successful grasp provides a precise pose observation. A failed grasp tells us the object was not where expected. Both are informative factors.

### 4.6 Uncertainty in the Representation Itself

**TSDF-level uncertainty:** Track per-voxel variance:
$$\sigma^2_D(x) = \frac{\sum_i w_i (d_i(x) - D(x))^2}{\sum_i w_i}$$

High-variance regions near planned grasp contacts should trigger re-observation.

**Propagation to downstream tasks:** Uncertain object poses → larger collision-checking margins, uncertain scene graph relations → conservative manipulation planning.

---

## 5. Synthesis: Prioritized Roadmap

| Priority | Improvement | Effort | Impact | What it enables |
|----------|-------------|--------|--------|----------------|
| 1 | Probabilistic pose + label tracking (EKF on SE(3)) | Medium | High | Principled phase-switching, uncertainty-aware planning, connects to alpha_robot calibration |
| 2 | Factor graph for joint camera-object optimization | High | High | Replaces sequential pipeline with jointly optimal solution, scene graph as constraints |
| 3 | Active perception (information-theoretic view planning) | Medium | Medium | Reduces reconstruction uncertainty, leverages idle time between manipulation |
| 4 | Neural representation hybrid (TSDF + neural SDF distillation) | High | Medium | Surface completion, differentiable optimization, better grasp planning |
| 5 | Manipulation-aware predictive tracking | Low | Medium | Handles occlusion during manipulation, physics-based prediction |

---

## 6. Conclusion

SceneRep occupies a unique position: it is the only system that combines open-vocabulary object-level TSDF reconstruction with manipulation-phase-aware tracking and a dynamic geometric scene graph with state propagation. Its closest relatives are TSDF++ (multi-object TSDF, no manipulation awareness), MaskFusion/Co-Fusion (dynamic multi-object tracking, no scene graph), and ConceptGraphs (semantic scene graph, no dynamic tracking).

The main limitation is the **loose coupling** between components. Each module operates independently, passing point estimates forward without propagating uncertainty or enabling joint optimization. The factor graph formulation (Priority 2) and probabilistic tracking (Priority 1) offer the highest impact-to-effort ratio for tightening this coupling.

The field is moving toward neural implicit/explicit representations (NeRF, 3DGS) and foundation-model-driven perception. SceneRep's per-object architecture is well-positioned to incorporate these: TSDF can be swapped for neural SDF or 3DGS per-object, and the VLM detection pipeline is already foundation-model-based.

---

## References

1. Grinvald et al., "TSDF++: A Multi-Object Formulation for Dynamic Object Tracking and Reconstruction," RA-L 2021.
2. Rünz & Agapito, "Co-Fusion: Real-time Segmentation, Tracking and Fusion of Multiple Objects," ICRA 2017.
3. Rünz et al., "MaskFusion: Real-Time Recognition, Tracking and Reconstruction of Multiple Moving Objects," ISMAR 2018.
4. Xu et al., "MidFusion: Octree-based Object-Level Multi-Instance Dynamic SLAM," RA-L 2019.
5. Gu et al., "ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning," RSS 2024.
6. Wen et al., "BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects," CVPR 2023.
7. Wen et al., "FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects," CVPR 2024.
8. Kong et al., "vMAP: Vectorised Object Mapping for Neural Field SLAM," CVPR 2023.
9. Zhu et al., "NICE-SLAM: Neural Implicit Scalable Encoding for SLAM," CVPR 2022.
10. Sucar et al., "iMAP: Implicit Mapping and Positioning in Real-Time," ICCV 2021.
11. Keetha et al., "SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM," CVPR 2024.
12. Matsuki et al., "Gaussian Splatting SLAM," CVPR 2024.
13. Schmid et al., "Panoptic Multi-TSDFs," RA-L 2022.
14. Liu et al., "OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics," RSS 2024.
15. Zheng et al., "GaussianGrasper: 3D Language Gaussian Splatting for Open-vocabulary Robotic Grasping," ICRA 2024.
16. Sucar et al., "NodeSLAM: Neural Object Descriptors for Multi-View Shape Reconstruction," 3DV 2020.
