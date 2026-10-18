# regularization_labs
# Regularization Lab: Character-level RNN on Tiny Shakespeare (from scratch)

Build a character-level RNN **from scratch** (forward pass + BPTT backward pass, no ML frameworks)
and study three regularization techniques:

- **Dropout** on the hidden state
- **L2 weight decay**
- **Early stopping**

combined with **Batch Normalization** on the hidden pre-activation, and compared across three optimizers:

- **Adam**
- **RMSprop**
- **Adagrad**