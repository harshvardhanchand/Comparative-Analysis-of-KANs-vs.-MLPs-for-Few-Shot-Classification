KAN vs. MLP: A Comparative Analysis for Few-Shot Transfer Learning

This project benchmarks KANs (PyKAN B-spline and FastKAN RBF) against a parameter-matched MLP for few-shot image classification on frozen backbones. Using seed-averaged results and an accuracy–complexity Pareto view, we find a clear boundary: in frozen-feature transfer with scarce labels, the MLP outperforms KANs, marking a practical limit for when KANs help.

Dataset:CIFAR-10, 32×32, label budgets 1% (500), 2% (1k), 5% (2.5k), stratified per class.

Backbone: Frozen ResNet-18 pretrained on ImageNet (from torchvision).

Methodology (fairness controls)
Parameter-matched MLP vs KAN heads (hidden width tuned to match params).
Identical optimizer/schedule: AdamW + cosine; same label smoothing.
Frozen backbone; feature normalization parity; adaptive ECE; multiple seeds.
Fixed grids by regime: PyKAN grid=3–5, FastKAN num_grids=6–8.
Practical takeaway
On frozen high-level features with ≤1% labels, a parameter-matched MLP head is simpler, more accurate, and better calibrated than KAN.

Results:
## KAN vs. MLP for Few-Shot CIFAR-10: Key Findings  
*(Frozen ResNet-18 backbone, 5 seeds, ≤ 5 % labels)*

| Label % | Metric (mean ± std) | **MLP** (267 k params) | **PyKAN** (269 k, grid=3) | **FastKAN** (268 k, grid=5) |
|---------|---------------------|------------------------|---------------------------|-----------------------------|
| **0.5 %** | **Accuracy**        | **71.7 %** ± 1.1       | 70.7 % ± 1.4              | 69.5 % ± 2.1                |
|           | **ECE (cal)**       | **0.076** ± 0.047      | 0.076 ± 0.032             | 0.093 ± 0.028               |
|           | **Brier (cal)**     | **0.406** ± 0.006      | 0.414 ± 0.011             | 0.434 ± 0.012               |
| **1 %** | **Accuracy**        | **76.1 %** ± 0.7       | 75.8 % ± 0.6              | 73.7 % ± 1.5                |
|           | **ECE (cal)**       | **0.057** ± 0.025      | 0.090 ± 0.026             | 0.099 ± 0.016               |
| **2 %** | **Accuracy**        | **77.0 %** ± 0.5       | 77.0 % ± 0.5              | 75.5 % ± 0.8                |
|           | **ECE (cal)**       | **0.056** ± 0.019      | 0.083 ± 0.015             | 0.089 ± 0.014               |
| **5 %** | **Accuracy**        | **81.0 %** ± 0.7       | 80.8 % ± 0.5              | 79.9 % ± 0.6                |
|           | **ECE (cal)**       | **0.078** ± 0.006      | 0.104 ± 0.009             | 0.095 ± 0.003               |

### Take-aways
- **Accuracy**: Plain MLP consistently **≥ PyKAN** and **> FastKAN** across every label budget.  
- **Calibration**: MLP achieves **lower ECE & Brier scores** after temperature scaling.  
n.  
- **Efficiency**: All models have ~ 267–269 k trainable parameters—differences are **architectural**, not capacity.

> TL;DR: In the few-shot regime with a frozen backbone, a simple MLP outperforms both spline-based KAN variants on accuracy **and** calibration.
