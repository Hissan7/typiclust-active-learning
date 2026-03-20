![Logo](logos/KCL.png)

# Typiclust Active Learning

This is an active learning problem regarding TPCRP (TypiClust variant), coded using CIFAR-10.


## Overview
This project implements an active learning pipeline on CIFAR-10 inspired by the TPCRP / TypiClust method from Hacohen et al. (2022).

The goal is to reproduce the idea of selecting **representative samples** using:
- Self-supervised representation learning (SimCLR)
- Clustering in embedding space
- Typicality-based sample selection

The project also compares:
1. Original TPCRP-style selection  
2. A modified centrality-aware TPCRP  
3. A random baseline  

---

## What the Program Does

The pipeline follows these steps:

### 1. Load Dataset
- CIFAR-10 training set used as unlabeled pool  
- CIFAR-10 test set used for evaluation  

### 2. Train Representation Model
- SimCLR-style model trained from scratch  
- Learns feature embeddings without labels  

### 3. Extract Embeddings
- Use encoder to generate feature vectors  
- Embeddings represent images in feature space  

### 4. Cluster Embeddings
- K-means clustering applied to embeddings  
- Groups similar data points  

### 5. Compute Typicality
- Typicality = inverse average distance to nearest neighbours  
- Identifies representative points in dense regions  

### 6. Select Samples
- TPCRP: most typical per cluster  
- Modified TPCRP: centrality-aware selection  
- Random: baseline selection  

### 7. Train Classifier
- Train supervised model on selected samples only  

### 8. Evaluate
- Test accuracy measured on CIFAR-10 test set  
- Results averaged over multiple runs  

---

## What This Project Reproduces

This project reproduces the **core TPCRP idea**:
- Learn representations using SimCLR  
- Cluster the embedding space  
- Select representative samples using typicality  

It follows the structure described in the paper:
- Representation learning → clustering → typicality → selection  

---

## Repository Structure

src/
main.py
clustering.py
selector.py
random_selector.py
train_classifier.py
test_loader.py
simclr/
train_simclr.py
extract_embeddings.py

results/
models/
notebooks/
report/

---

## Algorithms Implemented

### 1. TPCRP (Baseline)
- Cluster embeddings  
- Select most typical samples  

### 2. Centrality-Aware TPCRP (Modified)
- Improves selection by incorporating centrality  
- Coursework Task 3 modification  

### 3. Random Baseline
- Uniform random sampling  
- Used for comparison  

---

## Expected Outputs

When running the program:

### Console Output
- Dataset loading  
- SimCLR training (or loading saved model)  
- Embedding extraction  
- Algorithm execution logs  
- Training progress (loss and accuracy)  
- Final comparison of accuracies  

### Saved Files

results/tpcrp_indices.npy
results/weighted_tpcrp_indices.npy
results/random_indices.npy
results/bar_plot.png
results/line_plot.png

### Metrics
- Test accuracy for each method  
- Mean and standard deviation across runs  

---

## Requirements

- Python 3.11  
- PyTorch  
- torchvision  
- NumPy  
- scikit-learn  
- matplotlib  

---

## Installation and Setup

After cloning the repository : 

### 1. Create a virtual environment 

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install all dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the main script 

```bash
python src/main.py
```


