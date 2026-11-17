# PyGX: Python Graph XAI

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3123/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0+-red.svg)](https://pytorch-geometric.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PyGX** is an easily extendible toolkit designed to select, assess, and explain Deep Graph Networks (DGNs). It serves as a comprehensive framework for building explainable DGNs and benchmarking graph explainers, specifically in the context of graph classification tasks.

The package leverages the concept of **explanatory patterns** and supports a wide variety of datasets. Built on top of [Ray](https://www.ray.io/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), PyGX ensures efficient resource management, allowing for parallel training during grid search and parallelized computation of explanations. Additionally, it includes a built-in drawing module to directly visualize computed explanations.

## 🚀 Key Features

* **Efficient Parallelization:** Utilizes Ray to manage computational resources (CPU/GPU) for parallel grid searches and explanation generation.
* **Extensive Model Zoo:** Supports a wide range of convolutional and recursive graph layers.
* **Built-in Visualization:** Tools to draw and inspect graph explanations directly.
* **Benchmarking Ready:** Includes synthetic and chemistry-based datasets with ground-truth explanations.

---

## 🧠 Supported Architectures

PyGX allows for the construction of models using state-of-the-art convolutional and recursive layers.

| Layer Type | Reference |
| :--- | :--- |
| **GCN** | [Kipf et al., (2017)](https://arxiv.org/abs/1609.02907) |
| **GraphConv** | [Morris et al., (2019)](https://arxiv.org/abs/1810.02244) |
| **GIN** | [Xu et al., (2019)](https://arxiv.org/abs/1810.00826) |
| **PNA** | [Corso et al., (2020)](https://arxiv.org/abs/2004.05718) |
| **ChebConv** | [Defferrard et al., (2016)](https://arxiv.org/abs/1606.09375) |
| **ARMA** | [Bianchi et al., (2019)](https://arxiv.org/abs/1901.01343) |
| **GESN** | [Gallicchio et al., (2010)](https://ieeexplore.ieee.org/document/5596796) |

---

## 🔍 Explainability Methods

The library integrates several popular explainers to interpret model predictions.

| Explainer | Reference |
| :--- | :--- |
| **GNNExplainer** | [Ying et al., (2019)](https://arxiv.org/abs/1903.03894) |
| **Integrated Gradients** | [Sundararajan et al., (2017)](https://arxiv.org/abs/1703.01365) |
| **Gradient Method** | [Baldassarre et al., (2019)](https://arxiv.org/abs/1905.13686) |
| **CAM** | [Pope et al., (2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) |
| **Random Explainer** | Baseline for benchmarking |

---

## 💾 Datasets

PyGX includes datasets specifically curated for graph classification with ground truth explanations.

| Synthetic Datasets | Reference | Repository|
| :--- | :--- | :--- |
| **BA2Grid** | [Longa et al., (2025)](https://dl.acm.org/doi/10.1145/3696444)* | [Repo](https://github.com/AntonioLonga/Explaining-the-Explainers-in-Graph-Neural-Networks) |
| **GridHouse** | [Longa et al., (2025)](https://dl.acm.org/doi/10.1145/3696444)* | [Repo](https://github.com/AntonioLonga/Explaining-the-Explainers-in-Graph-Neural-Networks) |
| **HouseColors** | [Longa et al., (2025)](https://dl.acm.org/doi/10.1145/3696444) | [Repo](https://github.com/AntonioLonga/Explaining-the-Explainers-in-Graph-Neural-Networks) |
| **BA2Motif** | [Luo et al., (2020)](https://arxiv.org/abs/2011.04573)*| [Repo](https://github.com/flyingdoog/PGExplainer) |

*here extended with counterexamples

| Chemistry Datasets | Reference | Repository|
| :--- | :--- | :--- |
| **OpenGraphXAI** | [Fontanesi et al., (2025)](https://arxiv.org/abs/2505.12437) | [Repo](https://github.com/OpenGraphXAI/benchmarks) |
| **Mutagenicity** | [Agarwal et al., (2023)](https://www.nature.com/articles/s41597-023-01974-x) | [Repo](https://github.com/mims-harvard/GraphXAI) |
| **Benzene** | [Agarwal et al., (2023)](https://www.nature.com/articles/s41597-023-01974-x) | [Repo](https://github.com/mims-harvard/GraphXAI) |
| **Alkane Carbonyl** | [Agarwal et al., (2023)](https://www.nature.com/articles/s41597-023-01974-x) | [Repo](https://github.com/mims-harvard/GraphXAI) |
| **Fluoride Carbonyl** | [Agarwal et al., (2023)](https://www.nature.com/articles/s41597-023-01974-x) | [Repo](https://github.com/mims-harvard/GraphXAI) |

---

## 🛠 Installation

**Requirements:**
* Linux OS
* Python 3.12.3
* CPU or NVIDIA GPU

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## 💻 Usage

PyGX manages grid searches and explanation parameters via YAML configuration files located in `config_files/` and `config_expl/`. Additionally, computational resources, such as the number of CPUs, GPUs, and the maximum number of concurrent processes, can be configured directly through the command line.

### 1. Training & Grid Search

**For Graph Echo State Networks (GESN):**
To perform a grid search with GESN, modify `config_file/grid_GESN.yaml` as needed, then execute:

```bash
python3 main.py trainGraphConv --GESN --name GESN --cuda --exp_config ./config_file/grid_GESN.yaml
```
**For Convolutional Layers (GCN, GIN, etc.):**
To perform a grid search with convolutional layers, modify `config_file/grid_Conv.yaml` as needed, then execute:
```bash
python3 main.py trainGraphConv --name Conv --cuda --exp_config ./config_file/grid_Conv.yaml
```
### 2. Generating Explanations

Explaining every trained model can be computationally expensive. We recommend the following workflow:

1.  **Initial Scan:** Explain all models using the optimized and fast **CAM** explainer during the initial training phase.
2.  **Deep Dive:** Select a specific model and compute detailed explanations using other methods by modifying the `config_expl/expl.yaml` file.

To execute the explanation process for a selected model:
```bash
python3 main.py explainModels --cuda --exp_config ./config_expl/expl.yaml
````
> **⚠️ Note:** The `explainModels` option is currently not available for GESN models.

## 📊 Examples & Visualization

To help users understand the tool's output and leverage its features effectively, we provide a pre-trained model and pre-computed explanations.

Please refer to the **Jupyter Notebooks** included in the repository, which demonstrate:
* How to load and inspect grid search results.
* How to use the built-in drawing tools to visualize explanations directly on graphs.
* How to access additional explanations generated via the `explainModels` command.

## 📚 References & Publications

This code forms the technical basis for the following research papers:

### Related Publication
> M. Fontanesi, A. Micheli, and M. Podda. **XAI and bias of deep graph networks** *Proceedings of the 32nd European Symposium on Artificial Neural Networks (ESANN)* 2024.
> [DOI: 10.14428/esann/2024.es2024-85](https://doi.org/10.14428/esann/2024.es2024-85)

> M. Fontanesi, A. Micheli, and M. Podda. **Relating explanations with the inductive biases of deep graph networks** *AIxIA 2024 – Advances in Artificial Intelligence*. Springer Nature Switzerland.
> [DOI: 10.1007/978-3-031-80607-0_14](https://doi.org/10.1007/978-3-031-80607-0_14)

> M. Fontanesi, A. Micheli, M. Podda, and D. Tortorella. **Analyzing Explanations of Deep Graph Networks Through Node Centrality and Connectivity** *Discovery Science, LNCS Vol 15243*. Springer Nature Switzerland, 2025.
> [DOI: 10.1007/978-3-031-78977-9_19](https://doi.org/10.1007/978-3-031-78977-9_19)

> M. Fontanesi, A. Micheli, M. Podda, and D. Tortorella. **Bridging XAI and spectral analysis to investigate the inductive biases of deep graph networks** *Machine Learning, 114(11):257*, 2025.
> [DOI: 10.1007/s10994-025-06884-0](https://doi.org/10.1007/s10994-025-06884-0)

> D. Tortorella, M. Fontanesi, A. Micheli, and M. Podda.   **An empirical investigation of shortcuts in graph learning** *International Workshop on Graph-Based Representations in Pattern Recognition. GbRPR 2025. Lecture Notes in Computer Science, vol 15727*. Springer, 2025 Cham.
> [DOI: 10.1007/978-3-031-94139-9_14](https://doi.org/10.1007/978-3-031-94139-9_14)

> M. Fontanesi. **Explainability in Deep Graph Networks: Inductive Bias Analysis and New Benchmarks**, Ph.D. Thesis, University of Pisa, 2026.
### Citation

If you find this code useful in your research, please cite the following article:

```bibtex
@article{fontanesi2025bridging,
  title={Bridging XAI and spectral analysis to investigate the inductive biases of deep graph networks},
  author={Fontanesi, Michele and Micheli, Alessio and Podda, Marco and Tortorella, Domenico},
  journal={Machine Learning},
  volume={114},
  number={11},
  pages={257},
  year={2025},
  publisher={Springer},
  doi={https://doi.org/10.1007/s10994-025-06884-0}
}
```
