# Name-to-Language Classification with RNN, LSTM, and GRU

A PyTorch experiment comparing three recurrent architectures on **name-to-language classification**. Each name is read one character at a time; the model predicts one of 18 language labels. This is a prediction from spelling patterns in the supplied dataset, not a determination of a person's nationality.

Inspired by [Jake Tae's PyTorch RNN tutorial](https://jaketae.github.io/study/pytorch-rnn/).

## Why compare these models?

A name is a sequence: letter combinations and endings can carry information about its language label. One-hot character vectors preserve the exact observed character without requiring pretrained embeddings. A recurrent model updates a state after each letter, allowing the final state to summarize earlier letters for classification.

The experiment places a manually written vanilla RNN beside PyTorch's built-in LSTM and GRU. A vanilla RNN repeatedly transforms its previous hidden state; during backpropagation through many steps, gradients can become small or large. LSTM and GRU use learned gates to control what information is retained or replaced, which can help information flow across longer sequences. LSTM maintains both hidden and cell states; GRU uses a hidden state with fewer gate types. Those are architectural motivations, not proof that gating caused any particular score in this run.

## Data and encoding

[`data/data/names/`](data/data/names/) contains 18 text files, one language label per filename and one name per line. There are 20,074 lines across those files before preprocessing. [`utils.py`](utils.py) transliterates names with `unidecode`, skips names containing characters outside its fixed vocabulary, and converts each remaining name to a tensor of shape `(name length, 1, vocabulary size)`. The vocabulary consists of ASCII uppercase and lowercase letters plus a few spaces and punctuation marks.

The code makes a stratified **90/10 train/test split** after encoding. It does not pass a split seed to `train_test_split`. It sets a PyTorch seed but does not fix Python's shuffle or file enumeration order, so exact reruns are not guaranteed. The number of accepted names and the test-set count are not recorded in the repository.

## Architectures and training

| Script | Implementation | State and classifier | Device handling |
| --- | --- | --- | --- |
| [`rnn.py`](rnn.py) | Manual single-layer RNN cell with `tanh` | Concatenates the current character and previous hidden state; its final step produces language logits | CPU tensors |
| [`lstm.py`](lstm.py) | Two-layer `nn.LSTM` | Classifies the final timestep output; hidden and cell states | CUDA when available, otherwise CPU |
| [`gru.py`](gru.py) | Two-layer `nn.GRU` | Classifies the final timestep output; hidden state | CPU tensors |

All three scripts use hidden size 256, Adam with learning rate 0.0005, cross-entropy loss, gradient clipping at norm 1, and 15 epochs. They shuffle the training examples each epoch and update weights after each name. Gradient clipping limits large updates; it does not eliminate vanishing gradients. The RNN has one layer while LSTM and GRU have two, so this is an **architecture comparison**, not a controlled test of gating alone.

For a sequence of character vectors `x₁ … xₜ`, the basic idea is to update a state `hₜ` from `xₜ` and `hₜ₋₁`, then use the final output to predict a language. Cross-entropy rewards higher scores for the correct label. The gated models add learned controls over the state update, which may help retain useful early letters while processing later ones.

## Results and interpretation

The previous README reported **73% RNN, 83% LSTM, and 84% GRU accuracy**. The scripts print accuracy when run, but the repository does not commit a test log, prediction file, split indices, or metric table tying those three numbers to a particular run. Treat them as **historically reported values**, not verified or freshly reproduced results. The one-point GRU–LSTM gap is too small to establish a reliable winner without repeated runs and uncertainty estimates.

The committed training-loss plots are available for [RNN](Results/rnn_loss.png), [LSTM](Results/lstm_loss.png), and [GRU](Results/gru_loss.png). They show noisy **per-name training losses**, not validation curves. The RNN plot has large spikes and an upward-looking envelope; the gated-model plots also have spikes. These images alone cannot identify vanishing or exploding gradients, diagnose overfitting, or explain the reported accuracy gaps. A stronger explanation would require repeated seeded splits, matched depth/parameter budgets, validation curves, and error analysis by language and name length.

The dataset is imbalanced across languages, so overall accuracy may hide weak performance on smaller classes. No per-language evaluation is committed. There is also no measured runtime, memory, or parameter-count comparison; the reported 1-point GRU advantage does not establish that it is the most efficient model here.

## Run the experiment

From the repository root, in a Python environment with PyTorch, scikit-learn, Unidecode, and Matplotlib installed:

```bash
python -m pip install torch scikit-learn Unidecode matplotlib
python rnn.py
python lstm.py
python gru.py
```

The dataset is already present at the path expected by `utils.py`. Each script trains for 15 epochs, prints test accuracy and predictions for `Mike`, `Qin`, and `Fernando`, and writes a `.pth` state dictionary and a loss `.png` **to the current working directory**. Those printed example predictions depend on the trained run; they are not guaranteed language labels. Training is long because each name is processed separately.

[`Models/`](Models/) and [`Results/`](Results/) contain previously committed checkpoints and plots; the scripts do **not** save new artifacts into those directories. The committed `requirements.txt` is UTF-16 LE and includes a CUDA-specific PyTorch pin, so a plain `pip install -r requirements.txt` may fail on other platforms. The direct package command above reflects imports and has not been verified as a fresh-environment install. [`test.py`](test.py) calls `torch.cuda.get_device_name(0)` unconditionally and may fail on a CPU-only machine; it is not required for training.

## Reproducibility limits

The saved `.pth` files are model weights without the corresponding split, label order, package versions, or evaluation record. Language labels are assigned in filesystem enumeration order, making an independently reconstructed label map potentially different. The scripts do not load the committed checkpoints. To make the comparison reproducible, record the exact label mapping and split indices, seed all random sources, save evaluation outputs, and run multiple trials.

## References

- [PyTorch tutorial: Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [PyTorch `nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) and [`nn.GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
