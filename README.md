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
