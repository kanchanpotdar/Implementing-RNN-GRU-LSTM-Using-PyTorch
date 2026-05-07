# Name-to-Language Classification with RNN, LSTM, and GRU

A PyTorch implementation comparing three recurrent neural network architectures — **Vanilla RNN**, **LSTM**, and **GRU** — on the task of classifying names by their language of origin.

Inspired by [Jake Tae's PyTorch RNN tutorial](https://jaketae.github.io/study/pytorch-rnn/).

---

## Project Overview

Given a person's name as input, the model predicts which language (nationality) the name originates from. Each character in the name is processed sequentially as a one-hot encoded vector, and the final hidden state is used to classify the language.

**Example predictions:**
- `"Mike"` → English
- `"Qin"` → Chinese
- `"Fernando"` → Spanish/Portuguese

---

## Dataset

The project uses the **Names Dataset** from the PyTorch RNN tutorial, containing names from 18 different languages stored as `.txt` files under `./data/data/names/`.

Each file is named after the language (e.g., `English.txt`, `Chinese.txt`) and contains one name per line.

**Preprocessing:**
- Unicode characters are normalized using `unidecode`
- Names are converted to one-hot tensors of shape `(seq_len, 1, num_letters)`
- An 90/10 train/test stratified split is applied

---

## Project Structure

```
├── utils.py          # Data loading, preprocessing, tensor conversion
├── rnn.py            # Vanilla RNN model, training, and evaluation
├── lstm.py           # LSTM model with GPU support
├── gru.py            # GRU model
├── test.py           # PyTorch/CUDA environment check
├── rnn_loss.png      # RNN training loss curve
├── lstm_loss.png     # LSTM training loss curve
├── gru_loss.png      # GRU training loss curve
└── data/
    └── data/
        └── names/    # Language name files (.txt)
```

---

## Model Architectures

### Vanilla RNN (`rnn.py`)

A manually implemented RNN cell without using `nn.RNN`.

```
Input (one-hot char) + Previous Hidden State
        ↓
    nn.Linear → tanh → New Hidden State
    nn.Linear → Output logits
```

- Single-layer, single hidden state
- Output taken at the **last timestep**
- No GPU support (CPU only)

### LSTM (`lstm.py`)

Uses PyTorch's built-in `nn.LSTM` with support for multiple layers and GPU acceleration.

```
Input sequence → nn.LSTM (2 layers) → Final timestep output → nn.Linear → Logits
```

- Maintains both **hidden state** and **cell state**
- 2-layer stacked LSTM
- Full **CUDA GPU support**

### GRU (`gru.py`)

Uses PyTorch's built-in `nn.GRU`, a simpler alternative to LSTM.

```
Input sequence → nn.GRU (2 layers) → Final timestep output → nn.Linear → Logits
```

- Single **hidden state** (no cell state)
- 2-layer stacked GRU
- CPU only (no explicit device mapping)

---

## Hyperparameters

| Parameter      | Value  |
|----------------|--------|
| Hidden Size    | 256    |
| Learning Rate  | 0.0005 |
| Epochs         | 15     |
| Optimizer      | Adam   |
| Loss Function  | CrossEntropyLoss |
| Gradient Clip  | 1.0    |
| LSTM/GRU Layers| 2      |

---

## Training Loss Curves

All three models were trained for 15 epochs (~265,000 steps on the full dataset).

| RNN | LSTM | GRU |
|-----|------|-----|
| ![RNN Loss](rnn_loss.png) | ![LSTM Loss](lstm_loss.png) | ![GRU Loss](gru_loss.png) |

**Observations:**
- The **RNN** loss shows an increasing trend over training, suggesting the model struggles to generalize and may be suffering from the vanishing/exploding gradient problem
- The **LSTM** loss stabilizes and shows a slight downward trend, benefiting from its gating mechanisms
- The **GRU** loss behaves similarly to LSTM but with slightly higher variance, consistent with its simpler architecture

---

## Results

| Model | Accuracy |
|-------|----------|
| Vanilla RNN | 73.00% |
| LSTM | 83.00% |
| GRU | 84.00% |

The gated architectures (LSTM and GRU) significantly outperform the vanilla RNN by **+10-11%**, demonstrating the importance of gating mechanisms for capturing long-range dependencies in sequential data. GRU edges out LSTM by 1% while using fewer parameters, making it the most efficient model for this task.

---

## Requirements

```
torch
matplotlib
scikit-learn
unidecode
```

Install dependencies:

```bash
pip install torch matplotlib scikit-learn unidecode
```

---

## Usage

**Check your environment:**
```bash
python test.py
```

**Train and evaluate each model:**
```bash
python rnn.py
python lstm.py
python gru.py
```

Each script will:
1. Train the model for 15 epochs
2. Print loss every 3000 steps
3. Evaluate on the test set and print accuracy
4. Run sample predictions (`Mike`, `Qin`, `Fernando`)
5. Save the model weights (`.pth`) and loss plot (`.png`)

---

## Key Differences Between Models

| Feature | Vanilla RNN | LSTM | GRU |
|--------|-------------|------|-----|
| Implementation | Manual (`nn.Linear`) | `nn.LSTM` | `nn.GRU` |
| Hidden States | 1 (hidden) | 2 (hidden + cell) | 1 (hidden) |
| Layers | 1 | 2 | 2 |
| GPU Support | No | Yes | No |
| Parameters | Fewest | Most | Medium |
| Gradient Flow | Weakest | Best | Good |

---

## References

- [Jake Tae – PyTorch RNN Tutorial](https://jaketae.github.io/study/pytorch-rnn/)
- [PyTorch Documentation – RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [PyTorch Documentation – LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch Documentation – GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Understanding LSTM Networks – Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)