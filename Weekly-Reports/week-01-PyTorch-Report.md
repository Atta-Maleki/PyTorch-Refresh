Date: June 14th 2026

Hours: 2

I Learned:

# PyTorch Fundamentals

## Week 1 - Day 1

### Topics Covered

* Introduction to PyTorch
* PyTorch philosophy
* Tensors
* Tensor creation
* Tensor properties
* Tensor operations
* Tensor indexing and slicing
* Tensor reshaping
* Unsqueeze and squeeze
* Tensor dimension ordering (TensorFlow vs PyTorch)

---

## What I Learned

### What is PyTorch?

PyTorch is a deep learning framework built around tensors and automatic differentiation. It is widely used for machine learning and deep learning research and development.

One of PyTorch's biggest advantages is its dynamic computation graph, which allows operations to be executed immediately as Python code runs.

A useful mental model is:

PyTorch = NumPy + GPU Support + Automatic Differentiation

---

### Tensors

The fundamental data structure in PyTorch is the tensor.

A tensor is a multidimensional array.

Examples:

* Scalar → 0D tensor
* Vector → 1D tensor
* Matrix → 2D tensor
* Image batches and feature maps → 3D, 4D, or higher-dimensional tensors

Everything in PyTorch is represented as tensors:

* Inputs
* Labels
* Weights
* Activations
* Gradients
* Loss values

---

### Tensor Shapes and Dimensions

Examples:

Shape [4]

* Vector with 4 elements
* 1 dimension

Shape [2,3]

* Matrix
* 2 dimensions

Shape [32,3,224,224]

* Batch of 32 RGB images
* 3 channels
* Height 224
* Width 224

I learned that understanding tensor shapes is one of the most important PyTorch skills because many bugs come from shape mismatches.

---

### Creating Tensors

Common tensor creation methods:

```python
torch.tensor(...)
torch.zeros(...)
torch.ones(...)
torch.rand(...)
torch.randn(...)
```

These are frequently used for creating data, initializing weights, and testing code.

---

### Tensor Properties

Useful tensor attributes:

```python
x.shape
x.ndim
x.dtype
```

These help inspect tensor structure and data types.

---

### Tensor Operations

Basic operations:

```python
a + b
a * b
torch.matmul(a, b)
a @ b
```

Matrix multiplication is especially important because neural networks rely heavily on it.

---

### Indexing and Slicing

Examples:

```python
x[0]
x[0,0]
x[:,1]
x[0,:2]
```

I learned that indexing can reduce dimensions and that slicing is essential for selecting portions of tensors.

---

### Reshaping

Reshaping changes the structure of a tensor without changing the total number of elements.

Example:

```python
x.reshape(3,4)
```

Important rule:

Total number of elements must remain constant.

I also learned to use:

```python
x.reshape(3,-1)
```

where PyTorch automatically infers one dimension.

---

### Unsqueeze and Squeeze

Unsqueeze adds a dimension:

```python
x.unsqueeze(0)
```

Example:

```python
[5,10]
→
[1,5,10]
```

Squeeze removes dimensions of size 1:

```python
x.squeeze()
```

Example:

```python
[1,5,10]
→
[5,10]
```

---

### TensorFlow vs PyTorch Dimension Ordering

TensorFlow commonly uses:

```python
[batch, height, width, channels]
```

PyTorch commonly uses:

```python
[batch, channels, height, width]
```

Example:

```python
[16,224,224,3]
```

becomes

```python
[16,3,224,224]
```

using:

```python
x.permute(0,3,1,2)
```

---

## Key Takeaways

1. Tensors are the core object in PyTorch.
2. Shape management is one of the most important skills.
3. Neural networks ultimately operate on tensors.
4. Reshaping and dimension manipulation are used constantly.
5. PyTorch uses channel-first image format by default.
6. Understanding tensor shapes is critical for debugging models.

---
Date: June 14th 2026

Hours: less than 1

I Learned:
# PyTorch_Autograd
---
## week 1 day 2
## Autograd

### What is Autograd?

Autograd is PyTorch's automatic differentiation engine.

It automatically tracks operations performed on tensors and computes gradients using backpropagation. This eliminates the need to manually derive and implement derivatives for machine learning models.

Autograd is one of the core features that makes PyTorch suitable for deep learning.

---

### Why Autograd Exists

Machine learning models learn by minimizing error.

To reduce error, the model needs to know how changing a parameter affects the final output. This information is provided by gradients.

A gradient represents the rate of change of an output with respect to an input or parameter.

Autograd automatically computes these gradients.

---

### requires_grad

To enable gradient tracking:

```python
x = torch.tensor(3.0, requires_grad=True)
```

Setting `requires_grad=True` tells PyTorch to track all operations involving the tensor and build a computation graph.

Without this flag, PyTorch treats the tensor as regular data and does not calculate gradients.

---

### Computation Graph

When operations are performed on tensors with gradient tracking enabled, PyTorch automatically constructs a computation graph.

Example:

```python
x = torch.tensor(3.0, requires_grad=True)

y = x**2
```

Conceptually:

```text
x
↓
square
↓
y
```

The graph stores the sequence of operations required for gradient computation.

---

### Backward Pass

Gradients are calculated using:

```python
y.backward()
```

Calling `backward()` tells PyTorch to traverse the computation graph in reverse and compute gradients using the chain rule.

This process is known as backpropagation.

---

### Accessing Gradients

After calling:

```python
y.backward()
```

the gradient can be accessed through:

```python
x.grad
```

Example:

```python
x = torch.tensor(3.0, requires_grad=True)

y = x**2

y.backward()

print(x.grad)
```

Output:

```python
tensor(6.)
```

Because:

```text
d(x²)/dx = 2x
```

and when `x = 3`:

```text
2 × 3 = 6
```

---

### Gradient Accumulation

PyTorch accumulates gradients by default.

Multiple calls to:

```python
backward()
```

add gradients to existing values instead of replacing them.

Because of this, training loops typically contain:

```python
optimizer.zero_grad()
```

to clear previously stored gradients before computing new ones.

---

### Key Concepts Learned

* Autograd is PyTorch's automatic differentiation system.
* Gradients are required for model learning.
* `requires_grad=True` enables gradient tracking.
* PyTorch automatically builds computation graphs.
* `backward()` computes gradients through backpropagation.
* Gradients are stored in the `.grad` attribute.
* Gradients accumulate unless they are manually reset.

---

### Key Takeaway

Autograd is the mechanism that allows PyTorch models to learn. Instead of manually computing derivatives, PyTorch tracks tensor operations, builds computation graphs, and automatically calculates gradients needed for optimization.
****
