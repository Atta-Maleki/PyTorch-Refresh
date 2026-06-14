Date: June 14th 2026

Hours: 2

I Learned:

# PyTorch Learning Journey

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

## Next Topic

Autograd:

* Computational graphs
* requires_grad
* backward()
* gradients
* automatic differentiation
