use enum_dispatch::enum_dispatch;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1]
    }
    strides
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn stride_continuous(&self) -> Vec<usize> {
        compute_strides(&self.0)
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.0.clone()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Shape {
        Shape(dims)
    }
}

#[derive(Clone)]
struct Storage {
    data: Rc<RefCell<Vec<f32>>>,
}

impl Storage {
    fn new(data: Vec<f32>) -> Self {
        Self {
            data: Rc::new(RefCell::new(data)),
        }
    }

    fn get(&self, idx: usize) -> f32 {
        self.data.borrow()[idx]
    }

    fn len(&self) -> usize {
        self.data.borrow().len()
    }
}

type TensorId = usize;

#[enum_dispatch]
pub trait UnaryFn {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn backward(&self, grad: &Tensor, x: &Tensor) -> Tensor;
}

#[enum_dispatch]
pub trait BinaryFn {
    fn forward(&self, x: &Tensor, y: &Tensor) -> Tensor;
    fn backward(&self, grad: &Tensor, x: &Tensor, y: &Tensor) -> (Tensor, Tensor);
}

#[enum_dispatch(UnaryFn)]
#[derive(Clone)]
pub enum UnaryOp {
    Neg(Neg),
}

#[enum_dispatch(BinaryFn)]
#[derive(Clone)]
pub enum BinaryOp {
    Add(Add),
    Mul(Mul),
    Sub(Sub),
    Div(Div),
}

#[derive(Clone)]
pub enum Op {
    Unary(UnaryOp, Rc<Tensor>),
    Binary(BinaryOp, Rc<Tensor>, Rc<Tensor>),
    None,
}

#[derive(Default)]
pub struct GradientStore {
    grads: RefCell<HashMap<TensorId, Tensor>>,
}

impl GradientStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_grad(&self, id: TensorId, grad: Tensor) {
        self.grads.borrow_mut().insert(id, grad);
    }

    pub fn get_grad(&self, id: TensorId) -> Option<Tensor> {
        self.grads.borrow().get(&id).cloned()
    }
}

#[derive(Clone)]
pub struct Tensor {
    id: TensorId,
    storage: Storage,
    shape: Shape,
    stride: Vec<usize>,
    requires_grad: bool,
    grad: RefCell<Option<Rc<Tensor>>>,
    grad_fn: Option<Rc<Op>>,
    offset: usize,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let id = Self::new_id();
        let storage = Storage::new(data);
        let shape = Shape::from(shape);
        let stride = shape.stride_continuous();

        Self {
            id,
            storage,
            shape,
            stride,
            requires_grad: false,
            grad: RefCell::new(None),
            grad_fn: None,
            offset: 0,
        }
    }

    fn new_id() -> TensorId {
        static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let shape = Shape::from(shape.to_vec());
        let size = shape.elem_count();
        Self::new(vec![0.0; size], shape.to_vec())
    }

    pub fn ones(shape: &[usize]) -> Self {
        let shape = Shape::from(shape.to_vec());
        let size = shape.elem_count();
        Self::new(vec![1.0; size], shape.to_vec())
    }

    pub fn randn(shape: &[usize]) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();

        let shape = Shape::from(shape.to_vec());
        let size = shape.elem_count();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
        Self::new(data, shape.to_vec())
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    fn unary_op(&self, op: UnaryOp) -> Tensor {
        let result = op.forward(self);

        Tensor {
            id: result.id,
            storage: result.storage,
            shape: result.shape,
            stride: result.stride,
            requires_grad: self.requires_grad,
            grad: RefCell::new(None),
            grad_fn: if self.requires_grad {
                Some(Rc::new(Op::Unary(op, Rc::new(self.clone()))))
            } else {
                None
            },
            offset: result.offset,
        }
    }

    pub fn neg(&self) -> Tensor {
        self.unary_op(UnaryOp::Neg(Neg))
    }

    fn binary_op(&self, other: &Tensor, op: BinaryOp) -> Tensor {
        let result = op.forward(self, other);
        let requires_grad = self.requires_grad || other.requires_grad;

        Tensor {
            id: result.id,
            storage: result.storage,
            shape: result.shape,
            stride: result.stride,
            requires_grad,
            grad: RefCell::new(None),
            grad_fn: if requires_grad {
                Some(Rc::new(Op::Binary(
                    op,
                    Rc::new(self.clone()),
                    Rc::new(other.clone()),
                )))
            } else {
                None
            },
            offset: result.offset,
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, BinaryOp::Add(Add))
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, BinaryOp::Sub(Sub))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, BinaryOp::Mul(Mul))
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary_op(other, BinaryOp::Div(Div))
    }

    fn topological_sort(&self, nodes: &mut Vec<Rc<Tensor>>, visited: &mut HashSet<TensorId>) {
        if visited.insert(self.id) {
            if let Some(ref grad_fn) = self.grad_fn {
                match &**grad_fn {
                    Op::Unary(_, x) => x.topological_sort(nodes, visited),
                    Op::Binary(_, x, y) => {
                        x.topological_sort(nodes, visited);
                        y.topological_sort(nodes, visited);
                    }
                    _ => {}
                }
            }
            nodes.push(Rc::new(self.clone()));
        }
    }

    pub fn backward(&self) {
        let mut nodes = Vec::new();
        let mut visited = HashSet::new();
        self.topological_sort(&mut nodes, &mut visited);
        let mut grads = HashMap::new();
        grads.insert(self.id, Tensor::ones(self.shape.dims()));
        for node in nodes.into_iter().rev() {
            let grad = grads.get(&node.id).unwrap().clone();

            if let Some(ref grad_fn) = node.grad_fn {
                match &**grad_fn {
                    Op::Unary(ref op, ref x) => {
                        let grad_x = op.backward(&grad, x);
                        accumulate_grad(&mut grads, x.id, grad_x);
                    }
                    Op::Binary(ref op, ref x, ref y) => {
                        let (grad_x, grad_y) = op.backward(&grad, x, y);
                        accumulate_grad(&mut grads, x.id, grad_x);
                        accumulate_grad(&mut grads, y.id, grad_y);
                    }
                    _ => {}
                }
            }

            if node.requires_grad {
                *node.grad.borrow_mut() = Some(Rc::new(grad));
            }
        }
    }

    pub fn data(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.shape.elem_count());
        for idx in 0..self.shape.elem_count() {
            let index = self.offset + idx;
            data.push(self.storage.data.borrow()[index]);
        }
        data
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.grad.borrow().as_ref().map(|g| g.data())
    }
}

fn accumulate_grad(grads: &mut HashMap<TensorId, Tensor>, id: TensorId, grad: Tensor) {
    grads
        .entry(id)
        .and_modify(|g| {
            let data: Vec<f32> = g
                .data()
                .iter()
                .zip(grad.data().iter())
                .map(|(&a, &b)| a + b)
                .collect();
            *g = Tensor::new(data, g.shape().to_vec());
        })
        .or_insert(grad);
}

fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let mut result_shape = Vec::new();
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len2.max(len2);

    for i in 0..max_len {
        let dim1 = if i >= max_len - len1 {
            shape1[i - (max_len - len1)]
        } else {
            1
        };

        let dim2 = if i >= max_len - len2 {
            shape2[i - (max_len - len2)]
        } else {
            1
        };

        // dimensions must be the same, or one of them must be 1
        if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
            result_shape.push(dim1.max(dim2));
        } else {
            panic!("Cannot broadcast dimensions: {} and {}", dim1, dim2);
        }
    }
    result_shape
}

fn broadcast_strides(input_shape: &[usize], output_shape: &[usize]) -> Vec<usize> {
    let input_strides = compute_strides(input_shape);
    let input_rank = input_shape.len();
    let output_rank = output_shape.len();
    let mut broadcast_strides = vec![0; output_rank];
    for i in 0..output_rank {
        let input_dim = if i >= output_rank - input_rank {
            input_shape[i - (output_rank - input_rank)]
        } else {
            1
        };

        if input_dim == output_shape[i] {
            broadcast_strides[i] = input_strides[i - (output_rank - input_rank)];
        } else if input_dim == 1 {
            broadcast_strides[i] = 0;
        } else {
            panic!("Shapes cannot be broadcasted");
        }
    }
    broadcast_strides
}

fn flatten_index(idxs: &[usize], shape: &[usize]) -> usize {
    // Converts a multi-dimension index to a flat index
    idxs.iter()
        .zip(compute_strides(shape))
        .map(|(&idx, stride)| idx * stride)
        .sum()
}

fn unflatten_index(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    // Converts a flat index to a multi dimensional index
    let mut idxs = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        idxs[i] = idx % shape[i];
        idx /= shape[i];
    }
    idxs
}

fn ravel_index_broadcast(idxs: &[usize], strides: &[usize]) -> usize {
    idxs.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
}

fn elementwise_binary_op<F>(x: &Tensor, y: &Tensor, output_shape: &[usize], op: F) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32,
{
    let total_size: usize = output_shape.iter().product();
    let mut result = Vec::with_capacity(total_size);
    let x_broadcast_strides = broadcast_strides(x.shape.dims(), output_shape);
    let y_broadcast_strides = broadcast_strides(y.shape.dims(), output_shape);

    for idx in 0..total_size {
        let multi_idx = unflatten_index(idx, output_shape);
        let x_idx = ravel_index_broadcast(&multi_idx, &x_broadcast_strides) + x.offset;
        let y_idx = ravel_index_broadcast(&multi_idx, &y_broadcast_strides) + y.offset;
        let x_val = x.storage.get(x_idx);
        let y_val = y.storage.get(y_idx);
        result.push(op(x_val, y_val));
    }
    result
}

fn reduce_grad(grad_output: &Tensor, output_shape: &[usize], input_shape: &[usize]) -> Tensor {
    let axes_to_reduce: Vec<usize> = output_shape
        .iter()
        .zip(input_shape.iter())
        .enumerate()
        .filter_map(
            |(i, (&out_dim, &in_dim))| {
                if out_dim != in_dim {
                    Some(i)
                } else {
                    None
                }
            },
        )
        .collect();
    let data = sum_over_axes(&grad_output.data(), output_shape, &axes_to_reduce);
    Tensor::new(data, input_shape.to_vec())
}

fn sum_over_axes(data: &[f32], shape: &[usize], axes: &[usize]) -> Vec<f32> {
    // sums a tensor over a specified axes.
    let mut reduced_shape = shape.to_vec();
    for &axis in axes {
        reduced_shape[axis] = 1;
    }
    let total_size: usize = shape.iter().product();
    let reduced_size: usize = reduced_shape.iter().product();
    let mut result = vec![0.0; reduced_size];
    for idx in 0..total_size {
        let multi_idx = unflatten_index(idx, shape);
        let reduced_multi_idx: Vec<usize> = multi_idx
            .iter()
            .enumerate()
            .map(|(i, &v)| if axes.contains(&i) { 0 } else { v })
            .collect();
        let reduced_idx = flatten_index(&reduced_multi_idx, &reduced_shape);
        result[reduced_idx] += data[idx];
    }
    result
}

#[derive(Clone)]
pub struct Neg;

impl UnaryFn for Neg {
    fn forward(&self, x: &Tensor) -> Tensor {
        let data: Vec<f32> = x.data().iter().map(|&v| -v).collect();
        Tensor::new(data, x.shape().to_vec())
    }

    fn backward(&self, grad: &Tensor, _x: &Tensor) -> Tensor {
        grad.neg()
    }
}

#[derive(Clone)]
pub struct Add;

impl BinaryFn for Add {
    fn forward(&self, x: &Tensor, y: &Tensor) -> Tensor {
        let output_shape = broadcast_shapes(x.shape(), y.shape());
        let data = elementwise_binary_op(x, y, &output_shape, |a, b| a + b);
        Tensor::new(data, output_shape)
    }

    fn backward(&self, grad: &Tensor, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let grad_x = reduce_grad(grad, &grad.shape.dims(), x.shape());
        let grad_y = reduce_grad(grad, &grad.shape.dims(), y.shape());
        (grad_x, grad_y)
    }
}

#[derive(Clone)]
pub struct Sub;

impl BinaryFn for Sub {
    fn forward(&self, x: &Tensor, y: &Tensor) -> Tensor {
        let output_shape = broadcast_shapes(x.shape(), y.shape());
        let data = elementwise_binary_op(x, y, &output_shape, |a, b| a - b);
        Tensor::new(data, output_shape)
    }

    fn backward(&self, grad: &Tensor, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let grad_x = reduce_grad(grad, &grad.shape.dims(), x.shape());
        let grad_y = reduce_grad(&grad.neg(), &grad.shape.dims(), y.shape());
        (grad_x, grad_y)
    }
}

#[derive(Clone)]
pub struct Mul;

impl BinaryFn for Mul {
    fn forward(&self, x: &Tensor, y: &Tensor) -> Tensor {
        let output_shape = broadcast_shapes(x.shape(), y.shape());
        let data = elementwise_binary_op(x, y, &output_shape, |a, b| a * b);
        Tensor::new(data, output_shape)
    }

    fn backward(&self, grad: &Tensor, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        let grad_x = grad.mul(x);
        let grad_y = grad.mul(y);

        let grad_x = reduce_grad(&grad_x, &grad_x.shape.dims(), x.shape());
        let grad_y = reduce_grad(&grad_y, &grad_y.shape.dims(), y.shape());
        (grad_x, grad_y)
    }
}

#[derive(Clone)]
pub struct Div;

impl BinaryFn for Div {
    fn forward(&self, x: &Tensor, y: &Tensor) -> Tensor {
        let output_shape = broadcast_shapes(x.shape(), y.shape());
        let data = elementwise_binary_op(x, y, &output_shape, |a, b| a / b);
        Tensor::new(data, output_shape)
    }

    fn backward(&self, grad: &Tensor, x: &Tensor, y: &Tensor) -> (Tensor, Tensor) {
        // grad_x = grad / y
        let grad_x = grad.div(x);

        // grad_y = -grad * ( x / y ^ 2)
        let grad_y = grad.neg().mul(x).div(y).div(y);

        let grad_x = reduce_grad(&grad_x, &grad_x.shape.dims(), x.shape());
        let grad_y = reduce_grad(&grad_y, &grad_y.shape.dims(), y.shape());
        (grad_x, grad_y)
    }
}
