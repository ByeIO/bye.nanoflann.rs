#![allow(unconditional_recursion)]
#![allow(unused_imports)]
#![allow(unused_variables)]
/*
定义各种数据静态特性以及数学特性。
*/

use rand::Rng;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::ops::{Index, IndexMut, Add};

use crate::sets::ResultSet;

/// 定义 PI 常量
pub fn pi_const<T>() -> T
where
    T: From<f32>,
{
    T::from(3.14159265358979323846)
}

/* start 静态特性 */

/// 判断类型是否具有 `resize` 方法
pub trait HasResize {
    fn resize(&mut self, n: usize);
}

impl<T> HasResize for Vec<T> {
    fn resize(&mut self, n: usize) {
        self.resize(n);
    }
}

/// 判断类型是否具有 `assign` 方法
pub trait HasAssign<T> {
    fn assign(&mut self, n: usize, value: T);
}

impl<T: Clone> HasAssign<T> for Vec<T> {
    fn assign(&mut self, n: usize, value: T) {
        self.clear();
        self.resize(n, value);
    }
}

/// 判断类型是否具有 `size` 方法
pub trait HasSize {
    fn size(&self) -> usize;
}

impl<T> HasSize for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }
}

// 为任意长度的数组 `[T; N]` 实现 `HasSize`
impl<T, const N: usize> HasSize for [T; N] {
    fn size(&self) -> usize {
        N
    }
}

/// 为可调整大小的容器实现 `resize` 函数
pub fn resize<Container>(c: &mut Container, n_elements: usize)
where
    Container: HasResize,
{
    c.resize(n_elements);
}

/// 为不可调整大小的容器实现 `resize` 函数
pub fn resize_fixed<Container>(c: &mut Container, n_elements: usize)
where
    Container: Debug + HasSize,
{
    if n_elements != c.size() {
        panic!("Try to change the size of a fixed-size container");
    }
}

// fn resize_fixed(arr: &mut [i32], new_size: usize) {
//     if new_size != arr.len() {
//         panic!("Try to change the size of a fixed-size container");
//     }
// }

/// 为可赋值的容器实现 `assign` 函数
pub fn assign<Container, T>(c: &mut Container, n_elements: usize, value: T)
where
    Container: HasAssign<T>,
{
    c.assign(n_elements, value);
}

/// 为固定大小的容器实现 `assign` 函数
pub fn assign_fixed<Container, T>(c: &mut Container, n_elements: usize, value: T)
where
    Container: Debug + IndexMut<usize, Output = T>,
    T: Copy,
{
    for i in 0..n_elements {
        c[i] = value;
    }
}

/// 用于排序的结构体
pub struct IndexDistSorter;

/// 距离排序的实现
impl IndexDistSorter {
    /// 比较函数，用于排序
    pub fn compare<PairType>(p1: &PairType, p2: &PairType) -> bool
    where
        PairType: HasDistance<IndexTypeAny = usize>,
    {
        // 默认比较第一个元素的距离
        p1.distance(0b0 as usize) < p2.distance(0b0 as usize)
    }
}

/// 可以通过数据和索引获得数据源当中的某点的特性
pub trait HasPointData {
    type ElementTypeAny; // 点的数据类型（如 f64 或 vec![f64,f64,f64,f64]) <- RGB-D数据)
    type IndexTypeAny; // 索引类型(如usize或vec![usize,usize,usize])
    /// 获取指定索引处的点的值
    fn get_point(&self, idx: Self::IndexTypeAny) -> Vec<Self::ElementTypeAny>;
    /// 获取指定索引处的点的指定维度的值
    fn get_point_dim(&self, idx: Self::IndexTypeAny, dim: usize) -> Self::ElementTypeAny;
    /// 获取所有的点的值
    fn get_point_vec(&self) -> Vec<Vec<Self::ElementTypeAny>>;
}

/// 拥有distance属性的特性
pub trait HasDistance {
    // 索引类型（如 usize）
    type IndexTypeAny; 
    // 距离类型（如 f64或vec![f64,f64,f64,f64])
    // 各个距离L1,L2,SO2,SO3
    type DistanceTypeAny: PartialOrd + Add; 
    // 获取某点的距离
    fn distance(&self, dim: Self::IndexTypeAny) -> Self::DistanceTypeAny;
    // 获取所有距离
    fn distance_vec(&self) -> Vec<Self::DistanceTypeAny>;
}

#[cfg(test)]
mod tests2 {
    use super::*;

    #[test]
    fn test_pi_const() {
        let pi: f32 = pi_const();
        assert!((pi - 3.14159265358979323846).abs() < 1e-10, "pi_const() returned an incorrect value");
    }

    // #[test]
    // fn test_resize_vector() {
    //     let mut v = vec![1, 2, 3];
    //     resize(&mut v, 5);
    //     assert_eq!(v.len(), 5, "Vector was not resized correctly");
    //     assert_eq!(v, vec![1, 2, 3, 0, 0], "Resized vector contains incorrect values");
    // }

    // #[test]
    // #[should_panic(expected = "Try to change the size of a fixed-size container")]
    // fn test_resize_fixed_panic() {
    //     let mut arr = [1, 2, 3];
    //     resize_fixed(&mut arr, 5);
    // }

    #[test]
    fn test_resize_fixed_no_panic() {
        let mut arr = [1, 2, 3];
        resize_fixed(&mut arr, 3); // Should not panic
        assert_eq!(arr, [1, 2, 3], "Fixed-size container should not change");
    }

    #[test]
    fn test_assign_vector() {
        let mut v = vec![1, 2, 3];
        assign(&mut v, 5, 0);
        assert_eq!(v, vec![0, 0, 0, 0, 0], "Vector was not assigned correctly");
    }

    #[test]
    fn test_assign_fixed() {
        let mut arr = [1, 2, 3];
        assign_fixed(&mut arr, 3, 0);
        assert_eq!(arr, [0, 0, 0], "Fixed-size container was not assigned correctly");
    }

    #[test]
    fn test_has_distance_trait() {
        let item = ResultSet::new(1, 2.5);
        assert_eq!(item.distance(), 2.5, "HasPointData trait implementation is incorrect");
    }

    #[test]
    fn test_has_size_trait() {
        let v = vec![1, 2, 3];
        assert_eq!(v.size(), 3, "HasSize trait implementation is incorrect");
    }

    #[test]
    fn test_has_resize_trait() {
        let mut v = vec![1, 2, 3];
        v.resize(5, 0); // 提供填充值 0
        assert_eq!(v.len(), 5, "HasResize trait implementation is incorrect");
    }

    #[test]
    fn test_has_assign_trait() {
        let mut v = vec![1, 2, 3];
        v.assign(5, 0);
        assert_eq!(v, vec![0, 0, 0, 0, 0], "HasAssign trait implementation is incorrect");
    }
}

/* end 静态特性 */

/* start 数学特性 */
pub trait Abs {
    fn abs(&self) -> Self;
}

impl Abs for f32 {
    fn abs(&self) -> Self {
        f32::abs(*self)
    }
}

impl Abs for f64 {
    fn abs(&self) -> Self {
        f64::abs(*self)
    }
}

// 最大值
pub trait MaxValue {
    fn max_value() -> Self;
}

impl MaxValue for f32 {
    fn max_value() -> Self {
        f32::MAX
    }
}

impl MaxValue for f64 {
    fn max_value() -> Self {
        f64::MAX
    }
}

#[cfg(test)]
mod tests5 {
    use super::*;

    #[test]
    fn test_abs_f32() {
        let x: f32 = -3.14;
        assert_eq!(x.abs(), 3.14);

        let y: f32 = 2.71;
        assert_eq!(y.abs(), 2.71);

        let z: f32 = 0.0;
        assert_eq!(z.abs(), 0.0);
    }

    #[test]
    fn test_abs_f64() {
        let x: f64 = -3.14;
        assert_eq!(x.abs(), 3.14);

        let y: f64 = 2.71;
        assert_eq!(y.abs(), 2.71);

        let z: f64 = 0.0;
        assert_eq!(z.abs(), 0.0);
    }
}
/* end 数学特性 */

/* start 调试工具 */
/// 打印内存使用情况（仅适用于Linux系统）
pub fn dump_mem_usage() {
    if let Ok(contents) = std::fs::read_to_string("/proc/self/statm") {
        println!("MEM: {}", contents);
    }
}
/* end 调试工具 */