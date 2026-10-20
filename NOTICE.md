# Third-party code and models

| Component | Source | License | How it is used here |
|---|---|---|---|
| DSTA-SLR | https://github.com/hulianyuyy/DSTA-SLR | CC BY-NC 4.0 | Adapted copy in `scripts/models/dsta_slr.py` (modifications listed in its header). Non-commercial use only. |
| SignBart | https://github.com/TinhNguyen2312/SignBart | none stated | Not copied. The architecture is imported at run time from a local clone (`SIGNBART_DIR`). |
| Uni-Sign | https://github.com/ZechengLi19/Uni-Sign | code: none stated; weights (Hugging Face): CC BY-NC 4.0 | Not copied. Cloned at run time at the pinned commit; the weights are downloaded from Hugging Face. |
| SHuBERT | https://github.com/ShesterG/SHuBERT and its Hugging Face Space and weights | GitHub: none stated; Space and weights: MIT | Not copied. Downloaded at run time at the pinned revisions. |
| rtmlib, MMPose | https://github.com/Tau-J/rtmlib, https://github.com/open-mmlab/mmpose | Apache-2.0 | Dependencies. |
| MediaPipe | https://github.com/google-ai-edge/mediapipe | Apache-2.0 | Dependency (version 0.10.14). |
| DINOv2 | https://github.com/facebookresearch/dinov2 | Apache-2.0 | Image encoder used by SHuBERT. |

The SIBI dataset is not included. It is available from its owner under a data use agreement for non-commercial research.
