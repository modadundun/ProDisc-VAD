# ProDisc-VAD: An Efficient System for Weakly-Supervised Anomaly Detection in Video Surveillance Applications

[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?logo=github)](https://github.com/modadundun/ProDisc-VAD)

## Introduction

This repository contains the official PyTorch implementation for **ProDisc-VAD**, an efficient framework for **Weakly-Supervised Video Anomaly Detection (WS-VAD)** in surveillance applications. The method addresses the label ambiguity inherent in Multiple Instance Learning (MIL) for WS-VAD, which stems from the sparsity of anomalous events and hinders the learning of discriminative features when only video-level labels are available.

## Abstract 

Weakly-supervised video anomaly detection (WS-VAD) using Multiple Instance Learning (MIL) suffers from label ambiguity, hindering discriminative feature learning. We propose ProDisc-VAD, an efficient framework tackling this via two synergistic components. The Prototype Interaction Layer (PIL) provides controlled normality modeling using a small set of learnable prototypes, establishing a robust baseline without being overwhelmed by dominant normal data. The Pseudo-Instance Discriminative Enhancement (PIDE) loss boosts separability by applying targeted contrastive learning exclusively to the most reliable extreme-scoring instances (highest/lowest scores). ProDisc-VAD achieves strong AUCS (97.98% ShanghaiTech, 87.12% UCF-Crime) using only 0.4M parameters, over 800x fewer than recent ViT-based methods like VadCLIP, demonstrating exceptional efficiency alongside state-of-the-art performance. Code is available at https://github.com/modadundun/ProDisc-VAD.

## Framework

[框架.pdf](https://github.com/user-attachments/files/20028254/default.pdf)

## Key Features

* **Prototype Interaction Layer (PIL):** Efficiently integrates normality context using a limited set of learnable prototypes and an attention mechanism, balancing normality capture with model simplicity and robustness.
* **Pseudo-Instance Discriminative Enhancement (PIDE) Loss:** Employs targeted contrastive learning on extreme-scoring instances, leveraging the most reliable pseudo-labels under weak supervision to enhance feature separability and mitigate noise amplification.
* **Efficiency:** Extremely lightweight model (0.4M parameters) with fast inference speed, offering a great balance between performance and computational cost.

Our codebase mainly refers to AR-net. We greatly appreciate their excellent contribution with nicely organized code!
## Results

ProDisc-VAD demonstrates strong performance on standard WS-VAD benchmarks:

| Dataset     | ProDisc-VAD (AUC %) |
| :-----------  | :------------------ | 
| ShanghaiTech   |97.98 |
| UCF-Crime     | 87.12   |


| Method        | Params (G) | Test Time (s) | Model Size (MB) |
| :------------ | :--------- | :------------ | :-------------- |
| ProDisc-VAD | 0.0004 |0.0009 | 1.7 |

## Dependencies

* Python
* PyTorch
* NumPy
* Matplotlib

## Dataset Preparation


1.  **Datasets:**
2.  ShanghaiTech：
https://drive.google.com/drive/folders/1LUQg9J8olBwNqEnymfCEC6CkLsjaBgKO?usp=sharing

3.  UCF-Crime：
https://drive.google.com/drive/folders/1B0p22_efg_26-491AAYLWdzLPc5edHSQ?usp=drive_link
4.  **Directory Structure:**
    * Place the datasets under a root directory (default is `../dataset`, modify via `--dataset_path` in `options.py`).
    * Each dataset directory should contain:
        * `train_split_10crop.txt`: List of training video segments (10 crops per video). 
        * `test_split_10crop.txt`: List of testing video segments (10 crops per video).
        * `GT/`: Contains Ground Truth files (e.g., `video_label_10crop.pickle`, `frame_label.pickle` depending on the dataset).
    * **Important:** Modify the `self.feature_path` variable inside `Dataset11_10_10crop_txt_ucf.py` and `Dataset11_10_10crop_txt_sh.py` to point to the actual directory where your **feature files (in .npy format)** are stored.

## Usage

### 1. Configure Parameters


Main configuration arguments are defined in `options.py` and can be overridden via command-line arguments. Key arguments include:
* `--dataset_name`: Choose the dataset ('shanghaitech' or 'ucf-crime').
* `--dataset_path`: Path to the dataset root directory.
* `--feature_size`: Input feature dimension (512 for ViT-B/16).
* `--lr`: Learning rate (default: 0.0005).
* `--batch_size`: Training batch size (default: 1, but samples are concatenated internally).
* `--sample_size`: Number of instances sampled per MIL bag (default: 30).
* `--max_epoch`: Number of training epochs (default: 100).
* `--model_name`: Name used for saving checkpoints and results (default: 'ProDisc_VAD').
* `--device`: GPU ID to use (default: 0).
* `--pretrained_ckpt`: Optional path to a pretrained model checkpoint.

### 2. Training


Run the `main.py` script to start training.

```bash


## Project Structure
├── dataset/
│   ├── shanghaitech/
│   │   ├── test_split_10crop.txt      # ShanghaiTech test set video list 
│   │   └── train_split_10crop.txt     # ShanghaiTech train set video list 
│   └── ucf-crime/
│       ├── test_split_10crop.txt      # UCF-Crime test set video list 
│       └── train_split_10crop.txt     # UCF-Crime train set video list 
├── Dataset_sh.py      # Dataset loader for ShanghaiTech
├── Dataset_ucf.py     # Dataset loader for UCF-Crime
├── eval.py               # Evaluation script (calculates AUC)
├── main.py                            # Main script to run training
├── options.py                         # Script defining command-line arguments/options
├── ProDisc_VAD.py                     # Core ProDisc-VAD model implementation (PIL, PIDE)
├── test.py                            # Script for testing the model
├── train.py                      # Script defining the training loop and losses
└── utils.py                           # Utility functions (feature processing, plotting etc.)

# Train on UCF-Crime (modify other arguments as needed)
python main.py --dataset_name ucf-crime --feature_size 512

# Train on ShanghaiTech (modify other arguments as needed)
python main.py --dataset_name shanghaitech --feature_size 512

```

## Citation

Thank you very much for your interest in our work! If you find this paper and the accompanying code helpful in your research, we would be truly grateful if you would consider citing our work. Your citation is a great encouragement and support for us to continue our research and development in this area.

**Paper:**

Tao Zhu, Qi Yu, Xinru Dong, Shiyu Li, Yue Liu, Jinlong Jiang, and Lei Shu. (2025). ProDisc-VAD: An Efficient System for Weakly-Supervised Anomaly Detection in Video Surveillance Applications. *arXiv preprint arXiv:2505.02179*.

Available at: [https://arxiv.org/abs/2505.02179](https://arxiv.org/abs/2505.02179)

**BibTeX Entry:**

```bibtex
@misc{zhu2025prodiscvadefficientweaklysupervisedanomaly,
      title={ProDisc-VAD: An Efficient System for Weakly-Supervised Anomaly Detection in Video Surveillance Applications}, 
      author={Tao Zhu and Qi Yu and Xinru Dong and Shiyu Li and Yue Liu and Jinlong Jiang and Lei Shu},
      year={2025},
      eprint={2505.02179},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2505.02179](https://arxiv.org/abs/2505.02179)},
}
