# Cholesky-Space-for-Brain-Computer-Interfaces
This package accompanies the submission titled:

    "Cholesky Space for Brain–Computer Interfaces"

========================
1. Environment Setup
========================

Required environment:

- Python 3.10
- PyTorch 2.2.2 + CUDA 12.1
- numpy==1.24.4
- pandas==2.2.3
- matplotlib==3.9.0
- scikit-learn==1.5.0
- moabb==1.2.0
- braindecode==0.8.1

We recommend installing PyTorch manually before the other dependencies:

    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

Then install all other packages via:

    pip install -r requirements.txt

========================
2. Datasets
========================
- BCIC IV 2a (BNCI 2014-001, ~743MB)

Important:  
You do NOT need to download any dataset manually.  
On the first run, the script will automatically download the raw data from the internet.

========================
3. How to Run
========================

To train models on all supported datasets using default settings, simply run:

    python main.py

This will:
- Automatically download and preprocess the data
- Train all models and save results.

Customizing dataset or model 
To run specific datasets or models, open `main.py` and edit the following two lists by simply uncommenting the corresponding lines. 😄
