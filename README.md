
#  Recursive Semantic Refinement Network (RSR-Net)

## Project Goal
The **Recursive Semantic Refinement Network (RSR-Net)** is an innovative approach to abstractive summarization. Instead of generating text word-by-word, RSR-Net is designed to iteratively refine a fixed-size semantic embedding of a summary until it converges to the desired output state, which is semantically close to the ground-truth summary. This technique leverages principles from Deep Equilibrium Models (DEQ) and Recurrent Neural Networks (RNNs).

The project uses the standard **CNN/DailyMail** dataset and **BART-base embeddings** for all inputs and targets.

---

## 💡 Core Methodology: Iterative Embedding Correction

RSR-Net treats the summarization task as a **regression problem in a high-dimensional semantic space**.

1.  **Input Preparation:** The source article ($\mathbf{x}$) and the target summary ($\mathbf{y}_{\text{true}}$) are converted into fixed-size **BART-base encoder embeddings** (768 dimensions).
2.  **Recursion:** The core model iteratively consumes the document context ($\mathbf{x}$) and refines the current summary state ($\mathbf{y}$) and an internal latent state ($\mathbf{z}$).
3.  **Refinement:** After multiple recursive steps, the final state ($\mathbf{y}_{\text{hat}}$) represents the network's best prediction of the target summary embedding.
4.  **Loss:** Training minimizes the distance (using **Mean Squared Error**) between the predicted summary embedding (**y_hat**) and the ground-truth summary embedding (**y_true_emb.**).

---

## 🧠 Model Architecture: `RecursionModel`

The `RecursionModel` is a simple feed-forward network at its core, designed for recurrent application:

| State | Role | Dimension (Adjusted) |
| :--- | :--- | :--- |
| **Input ($\mathbf{x}$)** | **Document Context** (Fixed for all steps) | 768 (BART $\text{d\_model}$) |
| **State ($\mathbf{y}$)** | **Current Summary Embedding** (Refined output) | 768 (BART $\text{d\_model}$) |
| **Latent ($\mathbf{z}$)** | **Internal Memory** (Accumulates context) | 64 ($\text{latent\_dim}$) |
| **Combined Input** | $\text{torch.cat}([\mathbf{x}, \mathbf{y}, \mathbf{z}])$ | $768 + 768 + 64$ |

The network's output consists of a refined summary state ($\mathbf{y}_{\text{out}}$), an auxiliary output ($\mathbf{y}_{\text{aux}}$), and a new latent state ($\mathbf{z}_{\text{new}}$).

## 🔄 Recursive Training Mechanism

The training process uses two nested recursive functions that implement a form of **Deep Supervision** to stabilize training and ensure convergence.

### 1. `latent_recursion(x, y, z, net, n=4)`

This is the **inner loop** that runs the core network $n$ times. It quickly pushes the state towards a stable point for the current input:
$$\mathbf{y}_{t+1}, \mathbf{z}_{t+1} = \text{net}(\mathbf{x}, \mathbf{y}_t, \mathbf{z}_t)$$
The output $\mathbf{y}$ and $\mathbf{z}$ from the final step are then passed to the outer loop.

### 2. `deep_recursion(x, y, z, net, n=4, T=3)`

This is the **outer loop** that enables stable gradient propagation:
* **Preconditioning ($\text{T}-1$ steps):** The states $\mathbf{y}$ and $\mathbf{z}$ are updated repeatedly using `latent_recursion` while detaching the gradient ($\text{with torch.no\_grad}()$). This stabilizes the initial state for the final, critical step.
* **Final Step (1 step):** The network runs `latent_recursion` one last time **with the gradient enabled**, producing the final predicted embedding $\mathbf{y}_{\text{hat}}$ and a confidence score $\mathbf{q}_{\text{hat}}$.
* **State Update:** The resulting $\mathbf{y}$ and $\mathbf{z}$ are **detached** before being passed back to the main training loop, ensuring the model trains on sequential segments of the refinement process.

### Loss Function

The combined loss function for semantic refinement is:
$$
\text{Loss} = \text{MSE}(\mathbf{y}_{\text{hat}}, \mathbf{y}_{\text{true\_emb}}) + 0.1 \times \text{BCE}(\mathbf{q}_{\text{hat}}, \mathbf{1})
$$

* **$\text{MSE Loss}$:** Measures the distance between the predicted summary embedding ($\mathbf{y}_{\text{hat}}$) and the target summary embedding ($\mathbf{y}_{\text{true\_emb}}$).
* **$\text{Auxiliary Loss}$:** A **Binary Cross-Entropy (BCE)** loss applied to the sigmoid confidence $\mathbf{q}_{\text{hat}}$, which encourages the network to be highly confident (score $\to 1$) in its final prediction.

---

## 🚀 Setup and Execution

### Requirements

```bash
!pip install transformers datasets torch
```

### Data Setup

The project requires the CNN/DailyMail dataset files. The provided code includes steps to download and extract these files:

```bash
!wget -c https://huggingface.co/datasets/ccdv/cnn_dailymail/resolve/main/cnn_stories.tgz
!wget -c https://huggingface.co/datasets/ccdv/cnn_dailymail/resolve/main/dailymail_stories.tgz
!mkdir -p ./cnn_dailymail
!tar -xvzf cnn_stories.tgz -C ./cnn_dailymail
!tar -xvzf dailymail_stories.tgz -C ./cnn_dailymail
```

### Training

The `train_refinement` function handles the entire training process for one batch, including embedding conversion, dynamic model adjustment, and the recursive loop.

```python
# Initialization (as seen in the code)
input_dim = bart_model.config.d_model  # 768
hidden_dim = 128
latent_dim = 64
num_classes = 768 # dynamically adjusted
net = RecursionModel(input_dim, hidden_dim, latent_dim, num_classes).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Training loop execution
num_epochs = 2
for epoch in range(num_epochs):
    # ... loop over dataloader and call train_refinement(...)
```
