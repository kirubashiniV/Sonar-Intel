# SW-Net Second-Stage Structural Verification & Semantic Segmentation Design

**Document ID:** `DOC-SWNET-DESIGN-2026.09`  
**Role:** Person 2 — ML / Model Architecture & Verification Lead  
**Component:** Second-Stage Candidate Verification (`ml/verification/`)  
**Status:** **INVESTIGATED & SKELETON INTERFACE PREPARED**  
**Related Components:** YOLOv8s Candidate Search (`ml/inference/drishti_detector.py`), Unified Dataset (`data/dataset_v1.0/`)  

---

## 1. Executive Summary & Research Context

This document establishes the technical design and architectural interface for the **SW-Net second-stage verification component** within **SONAR-INTEL**.

In side-scan sonar (SSS) object detection, standard rectangular object detectors (e.g., YOLOv8s / YOLO11N) excel at rapid candidate search across massive hydrographic surveys, but suffer from false positive susceptibility on seafloor clutter (rocky outcrops, sandwave ridges, rectangular seafloor depressions) that visually mimic target dimensions.

To resolve this limitation, SONAR-INTEL adopts a **Two-Stage Cascade Architecture**:
```
Full SSS Waterfall Tile (640x640)
           │
           ▼
Stage 1: YOLO Candidate Detector (SEARCH)
           │  └─ Outputs candidate bounding boxes [x1, y1, x2, y2, conf, class_id]
           ▼
Candidate ROI Extraction (ROIExtractor)
           │  └─ Extracts localized crop with acoustic context margins
           ▼
Stage 2: SW-Net Semantic Verifier (VERIFICATION)
           │  └─ Direction-aware acoustic segmentation & morphological physical priors
           ▼
Structured Verification Result (VerificationResult)
           │  └─ [is_verified, seg_conf, mask_area, orientation, elongation, compactness]
           ▼
(Downstream Stage 3: Multi-Evidence Fusion & Geolocation — Deferred)
```

> [!NOTE]
> **Attribution & Research Provenance:**  
> **SW-Net** is an integration and reuse of the research published by **Dai & He (2026)**: *"SW-Net: A Lightweight Direction-Aware Semantic Segmentation Network for Shipwrecks in Side-Scan Sonar Imagery" (Remote Sensing / MDPI)*. It is **not** a SONAR-INTEL invention.

---

## 2. Why YOLO Alone is Insufficient

While YOLOv8s provides high-throughput candidate proposals across sliding-window tiles, single-stage bounding-box detection exhibits fundamental acoustic limitations:

1. **Rectangular Bounding Box Ambiguity:** Side-scan sonar targets (shipwrecks, linear pipelines, debris) are irregular, non-axis-aligned polygons. An axis-aligned rectangular box frequently contains $>60\%$ ambient seabed background, diluting acoustic contrast.
2. **Geological Clutter False Alarms:** Natural seafloor formations (glacial drop-offs, orthogonal fault lines, boulder fields) produce rectangular echo-shadow profiles that trigger high detector confidence despite lacking man-made structural continuity.
3. **No Acoustic Shadow Geometry Verification:** YOLO bounding boxes do not verify whether an acoustic highlight is physically paired with a downstream acoustic shadow of corresponding height and orientation relative to towfish nadir.
4. **Fragmentation Vulnerability:** Large shipwrecks spanning several tiles are often broken into disconnected bounding boxes without global structural coherence.

---

## 3. Why SW-Net is a Second-Stage Verifier

SW-Net resolves single-stage ambiguities by evaluating the fine-grained physical structure of proposed candidates:

1. **Directional Physical Priors:** SW-Net incorporates a **Directional Filter Bank** and **Directional Attention Mechanism**, dynamically filtering acoustic backscatter along target structural axes (e.g. ship hull keels, pipeline axes) while attenuating isotropic seabed speckle.
2. **Multi-Scale Context Integration:** Refined skip connections fuse high-level semantic shape with granular acoustic texture.
3. **Computational Lightweightness:** With only **4.01 Million parameters**, SW-Net is compact enough for localized execution on embedded Autonomous Underwater Vehicle (AUV) edge compute.

---

## 4. Why SW-Net Runs Only on Candidate ROIs (Search vs. Verification)

Running dense pixel-level semantic segmentation across continuous side-scan sonar waterfall surveys (often $2000 \times 20,000$ pixels per survey line) is computationally prohibitive and generates immense false-positive segmentation noise on empty seabed.

| Pipeline Dimension | Stage 1: YOLO Search | Stage 2: SW-Net Verification |
| :--- | :--- | :--- |
| **Operational Role** | High-speed global candidate search | Targeted structural verification |
| **Input Spatial Domain** | Full $640\times 640$ sliding-window tiles | Localized $32\text{px}\text{--}256\text{px}$ candidate bounding box ROIs |
| **Output Representation** | Bounding boxes $[x_1, y_1, x_2, y_2, \text{conf}, \text{cls}]$ | Pixel mask + physical morphology metrics |
| **Inference Frequency** | Every survey tile ($100\%$ of survey data) | Only proposed candidates ($<2\%$ of survey data) |
| **Compute Overhead** | Moderate batch inference | Sub-millisecond localized ROI pass |

---

## 5. Segmentation Evidence & Morphological Physical Priors

The SW-Net verification interface extracts structured physical priors:

```
Candidate Crop ──> [SW-Net / Contour Module] ──> Binary Mask (255=Target)
                                                        │
         ┌──────────────────┬───────────────────────────┴───────────────────────────┬──────────────────┐
         ▼                  ▼                                                       ▼                  ▼
   [Mask Area]     [PCA Orientation]                                           [Elongation]      [Compactness]
Total target px   Major axis angle $\theta$                            Major / Minor Axis      $4\pi A / P^2$
vs box area       relative to track                                    Ratio ($\ge 1.0$)       Isoperimetric score
```

1. **Mask-to-Box Area Ratio ($\text{Area}_{\text{mask}} / \text{Area}_{\text{bbox}}$):** Identifies underfilled spurious detections ($<0.05$) or diffuse non-target regions.
2. **Principal Orientation Angle ($\theta \in [-90^\circ, +90^\circ]$):** Computed via second-order central image moments of the inertia tensor:
   $$\theta = \frac{1}{2} \arctan\left( \frac{2\mu_{11}}{\mu_{20} - \mu_{02}} \right)$$
3. **Elongation Ratio ($\lambda_1 / \lambda_2$):** Quantifies target eccentricity from inertia tensor eigenvalues. Distinguishes elongated linear pipelines and keels from circular noise clusters.
4. **Isoperimetric Compactness Score ($C \in [0.0, 1.0]$):**
   $$C = \frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2}$$
5. **Acoustic Highlight / Shadow Contrast Ratio:** Mean intensity of segmented foreground vs. local ambient background.

---

## 6. Dataset Compatibility & Segmentation Ground-Truth Audit

We conducted a forensic audit of `data/dataset_v1.0/` and raw source datasets to establish whether sufficient segmentation ground truth exists to train SW-Net:

| Source Dataset | Ground-Truth Mask Availability | Ground-Truth Type | Usable for SW-Net Training? | Technical Notes |
| :--- | :---: | :---: | :---: | :--- |
| **AI4Shipwrecks** | **YES (286 Swaths)** | **REAL GROUND TRUTH** | **YES — PRIMARY SEGMENTATION BENCHMARK** | Raw 1-channel PNG binary masks in `data/raw/AI4Shipwrecks/*/labels/`. High-quality archaeological labels. |
| **SubPipe** | **NO** | NO SEGMENTATION LABEL | **NO (Bounding Boxes Only)** | Annotations in `dataset_v1.0` are 2D bounding boxes. Mask training would require manual polygon labeling. |
| **MILCO-NOMBO** | **NO** | NO SEGMENTATION LABEL | **NO (Bounding Boxes Only)** | Synthetic/real cylindrical point/box targets only. |
| **GhostVision (Crab Pots)**| **NO** | NO SEGMENTATION LABEL | **NO (Bounding Boxes Only)** | 2D bounding boxes around trap targets. |
| **GhostNetZero / DRISHTI** | **NO** | NO SEGMENTATION LABEL | **NO (Bounding Boxes Only)** | Synthetic bounding-box debris dataset. |

### Ground-Truth vs. Pseudo-Labeling Verdict:
- **Genuine Ground Truth:** Exists **strictly for Class 2 (`shipwreck`)** from AI4Shipwrecks.
- **Other Classes:** Currently have **zero genuine pixel-level segmentation ground truth**.
- **Engineering Rule:** Under no circumstances should bounding boxes be pseudo-filled or manufactured as fake ground truth.

---

## 7. Integration Interface Architecture

The integration interface is implemented in `ml/verification/`:

### 7.1 `ROIExtractor` (`ml/verification/roi_extractor.py`)
Extracts candidate crops with configurable context padding (`context_margin=0.15`) and image boundary clamping:
```python
extractor = ROIExtractor(context_margin=0.15, min_crop_size=32)
roi: ExtractedROI = extractor.extract_roi(image=tile_img, bbox=[x1, y1, x2, y2], metadata={"class_name": "shipwreck"})
```

### 7.2 `SWNetVerifier` & `VerificationResult` (`ml/verification/swnet_verifier.py`)
Model-agnostic interface returning rich verification metrics:
```python
verifier = SWNetVerifier(model_path="ml/models/swnet/best_swnet.pt")
result: VerificationResult = verifier.verify_roi(roi)

# Structured Output Fields:
# result.is_verified: bool
# result.verification_confidence: float
# result.mask_area_px: int
# result.mask_to_box_ratio: float
# result.orientation_deg: float
# result.elongation: float
# result.compactness: float
# result.highlight_mean: float
# result.shadow_mean: float
# result.contrast_ratio: float
# result.status: str
```

---

## 8. What Cannot Currently Be Claimed

To preserve strict scientific honesty:
1. ❌ We **cannot claim** that SW-Net is currently trained across all 5 canonical classes (only shipwreck ground truth exists).
2. ❌ We **cannot claim** that SW-Net weights are already trained and finalized in `ml/models/` (pre-trained YOLO `best_detector.pt` exists, but SW-Net `.pt` weights require dedicated training on AI4Shipwrecks raw masks).
3. ❌ We **cannot claim** full multi-class segmentation without acquiring or annotating polygonal masks for pipelines, nets, and crab pots.

---

## 9. Blockers & Exact Next Steps for Person 2

### Blockers:
- SW-Net model weights (`.pt`) for shipwreck segmentation must be trained from `data/raw/AI4Shipwrecks/*/labels/*.png`.
- Multi-class segmentation for non-shipwreck classes requires polygon ground truth or must rely on the structural heuristic contour fallback.

### Next Implementation Steps (Post-Approval):
1. **Train SW-Net Baseline:** Train the 4.01M parameter SW-Net architecture on AI4Shipwrecks raw binary masks using the site-aware split.
2. **Wire into Inference Service:** Integrate `ROIExtractor` and `SWNetVerifier` into `backend/app/services/inference_service.py` downstream of `deduplicate_detections()`.
3. **Implement Evidence Fusion:** Combine YOLO detection confidence ($C_{\text{yolo}}$), SW-Net verification score ($C_{\text{swnet}}$), and acoustic context metrics into the final contact priority score.
