## RustTensor

A simple Tensor library in Rust similar to `torch.Tensor`

Example usage:

```rust
let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
input.set_requires_grad(true);

let squared = input.mul(&input);
let output = squared.add(&input);

let grads = output.backward();
let input_grad = grads.get(input.id).unwrap();
```
