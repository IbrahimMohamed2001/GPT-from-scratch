# GPT Text Generation with PyTorch

This project implements a GPT (Generative Pre-trained Transformer) model using PyTorch to generate text based on Shakespeare's writings. The project includes the following files:

- `train.py`: Script for training the GPT model.
- `model.py`: Definition of the GPT model and its components.
- `dataset.txt`: Raw text file containing approximately 1 million characters of Shakespeare's writings.

## Requirements

- Python 3.6+
- PyTorch
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/IbrahimMohamed2001/GPT-from-scratch.git
    cd GPT-from-scratch
    ```

2. Install the required packages:
    ```sh
    pip install torch
    ```

## Usage

### Training the Model

To train the model, run the `train.py` script:
```sh
python train.py
```


This script will:
- Load and preprocess the dataset from `dataset.txt`.
- Define and initialize the GPT model.
- Train the model using the specified hyperparameters.
- Print the training and validation loss at regular intervals.

### Generating Text

After training, the script will generate text using the trained model and print it to the console.

## Model Architecture

The GPT model is defined in `model.py` and includes the following components:
- `MultiHeadAttention`: Implements multiple heads of self-attention in parallel.
- `FFN`: Feed-forward neural network.
- `DecoderBlock`: A single block of the decoder, consisting of self-attention and feed-forward layers.
- `GPT`: The main GPT model, which stacks multiple `DecoderBlock` layers.

## Hyperparameters

The following hyperparameters are used in the training script (`train.py`):
- `vocab_size`: Size of the vocabulary.
- `batch_size`: Number of samples per batch.
- `seq_len`: Length of the input sequences.
- `n_embed`: Dimensionality of the embeddings.
- `n_heads`: Number of attention heads.
- `n_layers`: Number of decoder layers.
- `dropout`: Dropout rate.
- `max_iters`: Maximum number of training iterations.
- `learning_rate`: Learning rate for the optimizer.
- `eval_iters`: Number of iterations for evaluation.
- `eval_interval`: Interval for evaluation during training.

## Results

The training script will output the training and validation loss at regular intervals. After training, it will generate a sample of text based on the trained model.
