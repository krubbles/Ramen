# Using TorchSharp
- Interop with pytorch is expensive. If you are accessing each item in a tensor array, use [Tensor].data<[]>().ToArray() instead of indexing each and using .item<>()