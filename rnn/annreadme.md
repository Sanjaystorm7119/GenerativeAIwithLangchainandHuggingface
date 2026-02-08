# Deep Learning Interview Notes: ANN, RNN, CNN

This document provides a structured overview of Artificial Neural Networks (ANN), Recurrent Neural Networks (RNN), and Convolutional Neural Networks (CNN) with a focus on interview preparation.

---

## 1. Artificial Neural Networks (ANN) & Feedforward Neural Networks (FNN)

Artificial Neural Networks are the backbone of deep learning, inspired by the structure of the human brain.

### Key Concepts

- **Perceptron**: The simplest form of a neural network, consisting of a single layer with weights, bias, and an activation function.
- **Multilayer Perceptron (MLP)**: A feedforward neural network with one or more hidden layers between the input and output layers.
- **Activation Functions**:
  - **Sigmoid**: Outputs between 0 and 1. Used in binary classification.
  - **ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$. Solves vanishing gradient problem.
  - **Tanh**: Outputs between -1 and 1. Zero-centered.
  - **Softmax**: Used in the output layer for multi-class classification to provide probabilities.

### Weight Initialization

- **Xavier/Glorot Initialization**: Good for Sigmoid/Tanh activations.
- **He Initialization**: Best for ReLU and its variants.

---

## 2. Recurrent Neural Networks (RNN)

RNNs are designed for sequential data (time-series, text, audio) where the output at a certain time step depends on previous inputs.

### Forward Propagation in RNN

For a single time step $t$:

1. **Hidden State Update**:
   $$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
   _Where $h_t$ is the current hidden state, $h_{t-1}$ is the previous hidden state, $x_t$ is the current input.\_
2. **Output Calculation**:
   $$y_t = \sigma(W_{hy}h_t + b_y)$$

### The Chain Rule: The Engine of Backpropagation

In neural networks, the **Chain Rule** is used to calculate the gradient of the loss function with respect to any weight.

#### 1. Chain Rule in ANN (Standard)

To find the impact of a weight $w$ on the loss $L$:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \text{output}} \times \frac{\partial \text{output}}{\partial \text{activation}} \times \frac{\partial \text{activation}}{\partial w}$$

#### 2. Chain Rule in RNN (Backpropagation Through Time - BPTT)

In RNNs, because the current state $h_t$ depends on $h_{t-1}$, the chain rule must backtrack through all time steps:
$$\frac{\partial L_t}{\partial W} = \sum_{k=1}^{t} \left( \frac{\partial L_t}{\partial h_t} \times \frac{\partial h_t}{\partial h_k} \times \frac{\partial h_k}{\partial W} \right)$$
The term $\frac{\partial h_t}{\partial h_k}$ is a product of gradients: $\prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}}$.

- **Vanishing Gradient**: Occurs if the product of these gradients is $< 1$ over many steps.
- **Exploding Gradient**: Occurs if the product is $> 1$.

### Vanishing & Exploding Gradients (The Main Challenges)

- **Vanishing Gradient**: Gradients become exponentially small during backpropagation through time (BPTT).
  - **Result**: The weights for early time steps are barely updated, making the model "forget" long-term dependencies.
  - **Fix**: Use LSTMs/GRUs (which have gates) or ReLU activation.
- **Exploding Gradient**: Gradients become exponentially large, leading to unstable weight updates and `NaN` values.
  - **Fix**: **Gradient Clipping** (capping gradients at a maximum value).

### Other Core Problems

1.  **Lack of Parallelism**: RNNs process data word-by-word (sequentially). Each step depends on the previous hidden state $h_{t-1}$.
    - **Result**: They cannot leverage GPU parallel processing efficiently, making training slow for long sequences.
2.  **Information Bottleneck**: The fixed-size hidden state $h_t$ must compress the entire history of the sequence. As the sequence grows, it becomes harder to retain precise details.

## 3. Convolutional Neural Networks (CNN)

### Core Operations

- **Convolution**: Applying filters (kernels) to the input image to extract features like edges, textures, etc.
- **Padding**: Adding zeros around the borders to maintain spatial dimensions.
- **Stride**: The number of pixels the filter moves across the image.
- **Pooling Layers**:
  - **Max Pooling**: Retains the maximum value in a window (reduces spatial size while keeping important features).
  - **Average Pooling**: Computes the average value in a window.
- **Flattening**: Converting the 2D feature map into a 1D vector to pass it into a Fully Connected (Dense) layer.

---

## 4. Comparison: RNN vs. LSTM vs. Transformers

| Feature                | Basic RNN       | LSTM / GRU      | Transformers   |
| :--------------------- | :-------------- | :-------------- | :------------- |
| **Long-Term Memory**   | Poor            | Good            | Excellent      |
| **Vanishing Gradient** | High Risk       | Low Risk        | Not an Issue   |
| **Parallel Training**  | No (Sequential) | No (Sequential) | Yes (Parallel) |
| **Computation Cost**   | Low             | Medium          | High (Initial) |

---

## 5. Interview Focused Q&A

### ANN/FNN

1. **Q: What is the difference between a Perceptron and an MLP?**
   _A: A perceptron is a single-layer neural network, while an MLP has one or more hidden layers and can solve non-linearly separable problems._
2. **Q: Why is ReLU preferred over Sigmoid in hidden layers?**
   _A: ReLU avoids the vanishing gradient problem in positive regions and is computationally efficient._

### RNN

1. **Q: How does an RNN handle sequences of varying lengths?**
   _A: Through padding or by processing sequences one time step at a time until the end of the sequence._
2. **Q: How do GRU and LSTM solve the vanishing gradient problem?**
   _A: They use "gates" (forget, input, output gates) to regulate the flow of information and maintain long-term memory._

### CNN

1. **Q: What is the purpose of a 1x1 convolution?**
   _A: It is used for dimensionality reduction (reducing channels) and adding non-linearity._
2. **Q: What is the difference between Valid and Same padding?**
   _A: Valid padding means no padding (output size shrinks), whereas Same padding adds zeros to ensure output size equals input size._

---

```mermaid
graph TD
    A[Input Layer] --> B[Hidden Layer 1]
    B --> C[Hidden Layer 2]
    C --> D[Output Layer]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
```
