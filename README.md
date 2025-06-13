# paper***

[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?logo=github)](https://github.com/modadundun/ProDisc-VAD)

## Important Notice: Regarding Preprint Withdrawal and Future Plans

Dear Visitors and Users,

The preprint paper titled "[***]," associated with this code repository and originally available on arXiv , has been **withdrawn due to conflicts with our ongoing patent application (based on the professional advice of intellectual property legal counsel)**, to avoid potential risks posed by its public disclosure to the patent application.

Consequently, the abstract and PDF file of the paper previously available in this repository have been removed.

We are currently actively addressing matters related to the patent application. After completing the necessary intellectual property protection steps, **we plan to re-release a revised version of this research or related scholarly work around July 10, 2025.** Please look for updates around that time.

The codebase within this repository remains available for academic research and reference.

We apologize for any inconvenience or confusion this may cause and appreciate your understanding and support.

Should you have any questions regarding the code, or wish to be notified when the paper is re-released, please feel free to reach out via GitHub Issues or other established contact methods.

## original README（without paper）
Our codebase mainly refers to AR-net. We greatly appreciate their excellent contribution with nicely organized code!
## Results

ProDisc-VAD demonstrates strong performance on standard WS-VAD benchmarks:

| Dataset     | ProDisc-VAD (AUC %) |
| :-----------  | :------------------ | 
| ShanghaiTech   |97.98 |
| UCF-Crime     | 87.12   |


| Method        | Params (G) | Test Time (s) | Model Size (MB) |
| :------------ | :--------- | :------------ | :-------------- |
| MIST | 0.003 |0.25 | 48.5 |
| RTFM | 0.02 |0.14 | 94.3 |
| S3R | 0.05 |0.16 | 310.7 |
| VadCLIP | 0.35 |0.27 | 619.1 |
| **ProDisc-VAD** | **0.0004** |**0.0009** | **1.7** |
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

Available at: [https://arxiv.org/***)

**BibTeX Entry:**

```bibtex
@misc{
      url={[https://arxiv.org/abs/***](https://arxiv.org/abs/***)},
}
