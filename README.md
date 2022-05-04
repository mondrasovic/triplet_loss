# Triplet Loss and Sample Mining Strategies

**PyTorch** implementation of ***triplet* loss function** with various **mining strategies** [[1](https://arxiv.org/abs/1703.07737)].

## Usage

### Importing required packages and loss functions

```python
import torch

from triplet_loss import BatchAllTripletLoss, SemiHardTripletLoss
```

### Instantiating objects

```python
batch_all_criterion = BatchAllTripletLoss()
batch_hard_criterion = SemiHardTripletLoss()
```

### Creating embedding vectors

```python
n_items = 10  # Number of samples.
emb_dim = 128  # Embedding dimension.

embs = torch.randn((n_items, emb_dim))  # Random embedding vectors.
labels = torch.randint(0, n_items, (n_items,))  # Random 0-indexed integer labels.
```

### Loss computation

```python
ba_loss = batch_all_criterion(embs, labels)
bh_loss = batch_hard_criterion(embs, labels)
```

## References

* [[1](https://arxiv.org/abs/1703.07737)] Hermans, Alexander, Lucas Beyer, and Bastian Leibe. "**In defense of the triplet loss for person re-identification**." arXiv preprint arXiv:1703.07737 (2017).