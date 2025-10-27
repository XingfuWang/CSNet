# Cholesky-Space-for-Brain-Computer-Interfaces

![overall framework](overall_framework.png)

This repository is the official implementation for the paper submission titled:  
> "Cholesky Space for Brain–Computer Interfaces"


## 1. Environment Setup

### 1.1 Requirements
The following dependencies are required to run the code:
- Python 3.10
- PyTorch 2.2.2 + CUDA 12.1
- numpy == 1.24.4
- pandas == 2.2.3
- matplotlib == 3.9.0
- scikit-learn == 1.5.0
- moabb == 1.2.0
- braindecode == 0.8.1


### 1.2 Installation Steps
We recommend installing **PyTorch first manually** (to ensure CUDA compatibility) before other dependencies:

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```


## 2. Datasets

Currently, the code supports the following dataset:
- **BCIC IV 2a (BNCI 2014-001)** (~743MB)


### Important Note
✅ **No manual download required!**  
The raw dataset will be **automatically downloaded from the official source** during the first run of the script.


## 3. How to Run

### 3.1 Default Run
To train all models on all supported datasets with default hyperparameters, simply execute:
```bash
python main.py
```

This command will:
1. Automatically download and preprocess the dataset.
2. Train all baseline/models in the pipeline.
3. Save training logs and results to the specified directory (see `main.py` for details).


### 3.2 Custom Run (Specific Datasets/Models)
To train on **specific datasets** or use **specific models**, modify `main.py` as follows:
1. Open `main.py` and locate the two key lists (e.g., `DATASETS` and `MODELS`).
2. Uncomment the lines corresponding to your target dataset/model.
3. Re-run the script:
   ```bash
   python main.py
   ```

## 4. Citation

If you find this code or our work helpful for your research, please consider giving this repository a ⭐ **Star** and citing our paper:

```bibtex

@ARTICLE{10922209,
  author={Wang, Xingfu and Qi, Wenxia and Yang, Wenjie and Wang, Wei},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Cholesky Space for Brain–Computer Interfaces}, 
  year={2025},
  volume={36},
  number={8},
  pages={15424-15435},
  keywords={Electroencephalography;Manifolds;Brain modeling;Decoding;Feature extraction;Motors;Emotion recognition;Covariance matrices;Computational efficiency;Vectors;Brain–computer interface (BCI);Cholesky space;electroencephalogram (EEG);Riemannian manifold},
  doi={10.1109/TNNLS.2025.3542801}}
