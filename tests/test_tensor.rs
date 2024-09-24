use rust_tensor::Tensor;
use std::error::Error;

#[test]
fn test_tensor_init() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.data(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_zeros() {
    let t = Tensor::zeros(&[2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.data(), vec![0.0; 6]);
}

#[test]
fn test_tensor_ones() {
    let t = Tensor::ones(&[2, 2]);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.data(), vec![1.0; 4]);
}

#[test]
fn test_tensor_randn() {
    let t = Tensor::randn(&[2, 2]);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.data().len(), 4);
}

#[test]
fn test_tensor_requires_grad() {
    let mut t = Tensor::new(vec![1.0, 2.0], vec![2]);
    t.set_requires_grad(true);
    assert!(t.requires_grad());
}

#[test]
fn test_tensor_neg() {
    let t = Tensor::new(vec![1.0, -2.0], vec![2]);
    let result = t.neg();
    assert_eq!(result.data(), vec![-1.0, 2.0]);
}

#[test]
fn test_tensor_add() {
    let t1 = Tensor::new(vec![1.0, 2.0], vec![2]);
    let t2 = Tensor::new(vec![3.0, 4.0], vec![2]);
    let result = t1.add(&t2);
    assert_eq!(result.data(), vec![4.0, 6.0]);
}

#[test]
fn test_tensor_sub() {
    let t1 = Tensor::new(vec![3.0, 4.0], vec![2]);
    let t2 = Tensor::new(vec![1.0, 2.0], vec![2]);
    let result = t1.sub(&t2);
    assert_eq!(result.data(), vec![2.0, 2.0]);
}

#[test]
fn test_tensor_mul() {
    let t1 = Tensor::new(vec![2.0, 3.0], vec![2]);
    let t2 = Tensor::new(vec![4.0, 5.0], vec![2]);
    let result = t1.mul(&t2);
    assert_eq!(result.data(), vec![8.0, 15.0]);
}

#[test]
fn test_tensor_div() {
    let t1 = Tensor::new(vec![6.0, 8.0], vec![2]);
    let t2 = Tensor::new(vec![2.0, 4.0], vec![2]);
    let result = t1.div(&t2);
    assert_eq!(result.data(), vec![3.0, 2.0]);
}
#[test]
fn test_simple_grad() -> Result<(), Box<dyn Error>> {
    let mut x = Tensor::new(vec![3f32, 1., 4.], vec![3]);
    x.set_requires_grad(true);
    let y = x.mul(&x).add(&x.mul(&Tensor::new(vec![5f32], vec![1])));
    let y = y.add(&Tensor::new(vec![4f32], vec![1]));

    let grads = y.backward()?;
    let grad_x = grads.get(x.id).unwrap();

    assert_eq!(x.to_vec1::<f32>()?, vec![3., 1., 4.]);
    assert_eq!(y.to_vec1::<f32>()?, vec![28., 10., 40.]);
    assert_eq!(grad_x.to_vec1::<f32>()?, vec![11., 7., 13.]);

    Ok(())
}
