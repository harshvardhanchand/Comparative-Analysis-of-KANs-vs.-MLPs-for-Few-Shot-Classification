KAN vs. MLP: A Comparative Analysis for Few-Shot Transfer Learning
This project conducts a rigorous, seed-averaged comparative analysis of the novel Kolmogorov-Arnold Network (KAN) (B spline and rbf variant) architecture against a traditional Multi-Layer Perceptron (MLP) for few-shot image classification. The experiment investigates the accuracy-vs-complexity trade-off using a Pareto frontier analysis and identifies a key performance boundary for KANs in a common transfer learning scenario.
Dataset:CIFAR-10, 32×32, label budgets 1% (500), 2% (1k), 5% (2.5k), stratified per class.
Backbone: Frozen ResNet-18 pretrained on ImageNet (from torchvision).

