#![allow(dead_code)]
#![allow(unconditional_recursion)]
#![allow(unused_imports)]

//! 本模块定义了多种距离度量函数，用于计算高维数据集中的距离。
//! 包括曼哈顿距离、欧几里得距离、SO2距离和SO3距离等。
//! 不需要建立kdtree也可以计算距离

use std::io::{
    self, Read, Write
};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{
    Add, Deref, Mul, Neg, Sub
};
use std::rc::Rc;
use rand::distributions::{
    Distribution, Standard
};

// 内部库
use crate::utils::{
    pi_const, HasResize, HasAssign, HasSize, IndexDistSorter, HasPointData, Abs
};
use crate::sets::{
    KNNResultSet, RKNNResultSet, RadiusResultSet
};

/// 距离度量基类
pub struct Metric;

// * 范型参数说明:
// * ElementTypeAny: 数据元素类型
// * DataSourceAny: 数据源
// * DistanceTypeAny: 距离类型
// * IndexTypeAny: 数据索引类型

/* start 距离度量适配器类 */

/// 曼哈顿距离适配器（L1距离）
/// 适用于高维数据集，优化了计算效率。
pub struct L1Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> {
    data_source: DataSourceAny,
    _phantom: std::marker::PhantomData<(ElementTypeAny, DistanceTypeAny, IndexTypeAny)>,
}

/// 曼哈顿距离适配器（L1距离）的实现, 计算给定点 a 和数据集中索引为 b_idx 的点之间的距离。
impl<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> L1Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>
where
    // ElementTypeAny需要实现的特性
    ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs + TryInto<DistanceTypeAny>,
    // DistanceTypeAny需要实现的特性
    DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
    // DataSourceAny需要实现的特性(可以访问数据源中的点)
    DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
    // IndexTypeAny需要实现的特性
    IndexTypeAny: TryInto<usize>,
{
    /// 构建L1距离适配器
    pub fn new(data_source: DataSourceAny) -> Self {
        Self {
            data_source,
            _phantom: std::marker::PhantomData,
        }
    }

    /// 计算曼哈顿距离
    pub fn eval_metric(
        &self, 
        a: &[ElementTypeAny], 
        b_idx: usize, 
        size: usize, 
        worst_dist: Option<DistanceTypeAny>
    ) -> DistanceTypeAny 
    where <DistanceTypeAny as Add<ElementTypeAny>>::Output: Add<ElementTypeAny>
    {
        let mut result = DistanceTypeAny::default();
        let last = a.len();
        let lastgroup = last - 3;
        let mut d = 0;

        // 每次循环处理4个元素以提高效率
        let mut a_ptr = a.as_ptr();
        while (a_ptr as usize) < (a.as_ptr().wrapping_add(lastgroup) as usize) {
            let diff0 = (unsafe { *a_ptr.offset(0) } - self.data_source.get_point_dim(b_idx, d)).abs();
            let diff1 = (unsafe { *a_ptr.offset(1) } - self.data_source.get_point_dim(b_idx, d + 1)).abs();
            let diff2 = (unsafe { *a_ptr.offset(2) } - self.data_source.get_point_dim(b_idx, d + 2)).abs();
            let diff3 = (unsafe { *a_ptr.offset(3) } - self.data_source.get_point_dim(b_idx, d + 3)).abs();
            result += (diff0 + diff1 + diff2 + diff3).into();
            a_ptr = unsafe { a_ptr.offset(4) };
            d += 4;

            if let Some(worst_dist) = worst_dist {
                if result > worst_dist {
                    return result;
                }
            }
        }

        // 处理剩余的0-3个元素
        while (a_ptr as usize) < (a.as_ptr().wrapping_add(last) as usize) {
            result += (unsafe { *a_ptr } - self.data_source.get_point_dim(b_idx.try_into().unwrap_or(0b0 as usize), d)).abs().into();
            a_ptr = unsafe { a_ptr.offset(1) };
            d += 1;
        }

        result
    }

    /// 计算两个标量值之间的曼哈顿距离
    pub fn accum_dist<U, V>(&self, a: U, b: V, _idx: usize) -> DistanceTypeAny
    where
        // U类型需要实现的特性
        U: Copy + std::ops::Sub<Output = U> + std::ops::Add<Output = DistanceTypeAny> + std::ops::Neg<Output = U> + Abs +
        Into<DistanceTypeAny>,
        // V类型需要实现的特性
        V: Copy + Into<U>,
    {
        // 先让a与b类型一致进行运算后再转为DistantTypeAny类型返回
        (a - b.into()).abs().into()
    }
}

/// 欧几里得距离适配器（L2距离）
/// 适用于高维数据集，优化了计算效率。
pub struct L2Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> {
    data_source: DataSourceAny,
    _phantom: std::marker::PhantomData<(ElementTypeAny, DistanceTypeAny, IndexTypeAny)>,
}

/// 欧几里得距离适配器（L2距离）行为的实现,  计算给定点 a 和数据集中索引为 b_idx 的点之间的距离。
impl<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> L2Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>
where
       // ElementTypeAny需要实现的特性
       ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs 
       + std::ops::Mul<Output = ElementTypeAny>,
       // DistanceTypeAny需要实现的特性
       DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
       // DataSourceAny需要实现的特性(可以访问数据源中的点)
       DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
       // IndexTypeAny需要实现的特性
       IndexTypeAny: TryInto<usize>,
{
    // 构造数据
    pub fn new(data_source: DataSourceAny) -> Self {
        Self {
            data_source,
            _phantom: std::marker::PhantomData,
        }
    }

    // 计算距离
    pub fn eval_metric(&self, a: &[ElementTypeAny], b_idx: usize, size: usize, worst_dist: Option<DistanceTypeAny>) -> DistanceTypeAny 
    {
        let mut result = DistanceTypeAny::default();
        let last = a.len();
        let lastgroup = last - 3;
        let mut d = 0;

        // 每次循环处理4个元素以提高效率
        let mut a_ptr = a.as_ptr();
        while (a_ptr as usize) < (a.as_ptr().wrapping_add(lastgroup) as usize) {
            let diff0 = unsafe { *a_ptr.offset(0) } - self.data_source.get_point_dim(b_idx.try_into().unwrap_or(0b0 as usize), d);
            let diff1 = unsafe { *a_ptr.offset(1) } - self.data_source.get_point_dim(b_idx.try_into().unwrap_or(0b0 as usize), d + 1);
            let diff2 = unsafe { *a_ptr.offset(2) } - self.data_source.get_point_dim(b_idx.try_into().unwrap_or(0b0 as usize), d + 2);
            let diff3 = unsafe { *a_ptr.offset(3) } - self.data_source.get_point_dim(b_idx.try_into().unwrap_or(0b0 as usize), d + 3);
            // 先类型一致进行运算后再转为DistantTypeAny类型返回
            result += (diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3).into();
            a_ptr = unsafe { a_ptr.offset(4) };
            d += 4;

            if let Some(worst_dist) = worst_dist {
                if result > worst_dist {
                    return result;
                }
            }
        }

        // 处理剩余的0-3个元素
        while (a_ptr as usize) < (a.as_ptr().wrapping_add(last) as usize) {
            let diff = unsafe { *a_ptr } - self.data_source.get_point_dim(b_idx.try_into().unwrap_or(0b0 as usize), d);
            result += (diff * diff).into();
            a_ptr = unsafe { a_ptr.offset(1) };
            d += 1;
        }

        result
    }

    // 计算两个标量值之间的欧几里得距离
    pub fn accum_dist<U, V>(&self, a: U, b: V, _idx: usize) -> DistanceTypeAny
    where
        // U类型需要实现的特性
        U: Copy + std::ops::Sub<Output = U> + std::ops::Add<Output = DistanceTypeAny> + std::ops::Neg<Output = DistanceTypeAny> + Abs +
        Into<DistanceTypeAny> + std::ops::Mul<Output = DistanceTypeAny>,
        // V类型需要实现的特性
        V: Copy + Into<U>,
    {
        // 先让a与b类型一致进行运算后再转为DistantTypeAny类型返回
        let diff = a - b.into();
        // 返回
        (diff * diff).into()
    }
}

// 简单计算两个点之间的欧几里得距离
pub fn l2_distance(p1: &Vec<f32>, p2: &Vec<f32>) -> f32 {
    p1.iter()
        .zip(p2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// 简单的欧几里得距离适配器（L2距离）
/// 适用于低维数据集，如2D或3D点云。
pub struct L2SimpleAdaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> {
    data_source: DataSourceAny,
    _phantom: std::marker::PhantomData<(ElementTypeAny, DistanceTypeAny, IndexTypeAny)>,
}

/// 简单的欧几里得距离适配器（L2距离）行为的实现, 计算给定点 a 和数据集中索引为 b_idx 的点之间的距离。
impl<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> L2SimpleAdaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>
where
       // ElementTypeAny需要实现的特性
       ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs
       + std::ops::Mul<Output = ElementTypeAny>,
       // DistanceTypeAny需要实现的特性
       DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
       // DataSourceAny需要实现的特性(可以访问数据源中的点)
       DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
       // IndexTypeAny需要实现的特性
       IndexTypeAny: TryInto<usize>,
{
    // 构造数据
    pub fn new(data_source: DataSourceAny) -> Self {
        Self {
            data_source,
            _phantom: std::marker::PhantomData,
        }
    }

    // 计算距离
    pub fn eval_metric(&self, a: &[ElementTypeAny], b_idx: usize, size: usize) -> DistanceTypeAny {
        let mut result = DistanceTypeAny::default();
        for i in 0..size as usize{
            let diff = a[i] - self.data_source.get_point_dim(b_idx.try_into().unwrap_or(0b0 as usize), i);
            result += (diff * diff).into();
        }
        result
    }

    // 计算两个标量值之间的欧几里得距离
    pub fn accum_dist<U, V>(&self, a: U, b: V, _idx: usize) -> DistanceTypeAny
    where
        // U类型需要实现的特性
        U: Copy + std::ops::Sub<Output = U> + std::ops::Add<Output = DistanceTypeAny> + std::ops::Neg<Output = DistanceTypeAny> + Abs +
        Into<DistanceTypeAny> + std::ops::Mul<Output = DistanceTypeAny>,
        // V类型需要实现的特性
        V: Copy + Into<U>,
    {
        // 先让a与b类型一致进行运算后再转为DistantTypeAny类型返回
        let diff = a - b.into();
        // 返回
        (diff * diff).into()
    }

}

/// SO2距离适配器
/// 用于计算二维旋转空间中的距离。
pub struct SO2Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> {
    data_source: DataSourceAny,
    _phantom: std::marker::PhantomData<(ElementTypeAny, DistanceTypeAny, IndexTypeAny)>,
}

/// SO2距离适配器行为的实现
impl<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> SO2Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>
where
       // ElementTypeAny需要实现的特性
       ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs
       + std::ops::Mul<Output = ElementTypeAny> 
       + std::ops::AddAssign
       + std::ops::SubAssign
       + Into<f32> + From<f32>,
       // DistanceTypeAny需要实现的特性
       DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny> + std::ops::Neg + From<f32> + Into<f32>,
       // DataSourceAny需要实现的特性(可以访问数据源中的点)
       DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
       // IndexTypeAny需要实现的特性
       IndexTypeAny: TryInto<usize>, 
{
    // 构造数据
    pub fn new(data_source: DataSourceAny) -> Self {
        Self {
            data_source,
            _phantom: std::marker::PhantomData,
        }
    }

    // 计算距离
    pub fn eval_metric(&self, a: &[ElementTypeAny], b_idx: usize, size: usize) -> DistanceTypeAny {
        // 返回值
        self.accum_dist(
            // a
            a[size - 1], 
            // b
            self.data_source.get_point_dim(b_idx.try_into().unwrap_or(0b0 as usize), size - 1), 
            // _idx
            size - 1)
    }

    // 计算两个标量值之间的SO2距离
    pub fn accum_dist<U, V>(&self, a: U, b: V, _idx: usize) -> DistanceTypeAny
    where
        // U类型需要实现的特性
        U: Copy + std::ops::Sub<Output = U> + std::ops::Add<Output = U> + std::ops::Neg<Output = U> + Abs +
        TryInto<DistanceTypeAny> + std::ops::Mul<Output = U> + std::ops::AddAssign<U> + std::ops::SubAssign<U> + Into<f32>,
        // V类型需要实现的特性
        V: Copy + Into<U>,
    {
        // 先让a与b类型一致进行运算后再转为DistantTypeAny类型返回
        let mut result : f32 = (b.into() - a).into();
        let pi = pi_const::<f32>();
        if result > pi {
            // result是f32类型
            result -= 2.0 * pi;
        } else if result < -pi {
            // result是f32类型
            result += 2.0 * pi;
        }
        // 返回, f32转成DistanceTypeAny类型
        result.into()
    }
}

/// SO3距离适配器（使用L2简单距离）
/// 用于计算三维旋转空间中的距离。
pub struct SO3Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> {
    l2_simple: L2SimpleAdaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>,
}

/// SO3距离适配器（使用L2简单距离）行为的实现
impl<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> SO3Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>
where
       // ElementTypeAny需要实现的特性
       ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs
       + std::ops::Mul<Output = ElementTypeAny>,
       // DistanceTypeAny需要实现的特性
       DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
       // DataSourceAny需要实现的特性(可以访问数据源中的点)
       DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
       // IndexTypeAny需要实现的特性
       IndexTypeAny: TryInto<usize>,
{
    // 构造数据
    pub fn new(data_source: DataSourceAny) -> Self {
        Self {
            l2_simple: L2SimpleAdaptor::new(data_source),
        }
    }

    // 计算距离
    pub fn eval_metric(&self, a: &[ElementTypeAny], b_idx: usize, size: usize) -> DistanceTypeAny {
        self.l2_simple.eval_metric(a, b_idx, size)
    }

    // 计算两个标量值之间的SO3距离
    pub fn accum_dist<U, V>(&self, a: U, b: V, idx: usize) -> DistanceTypeAny
    where
        // U类型需要实现的特性
        U: Copy + std::ops::Sub<Output = U> + std::ops::Add<Output = DistanceTypeAny> + std::ops::Neg<Output = DistanceTypeAny> + Abs +
        Into<DistanceTypeAny> + std::ops::Mul<Output = DistanceTypeAny>,
        // V类型需要实现的特性
        V: Copy + Into<U>,
    {
        // 返回
        self.l2_simple.accum_dist(a, b, idx).into()
    }

}

/* end 距离度量适配器类 */

/* start 距离度量类型的封装类 */

/// 曼哈顿距离
pub struct MetricL1;

impl MetricL1 {
    pub fn traits<
        // ElementTypeAny类型需要实现的特性
        ElementTypeAny : std::default::Default + std::ops::AddAssign 
        + std::ops::Add<Output = ElementTypeAny> + std::ops::Mul<Output = ElementTypeAny> 
        + std::ops::Sub<Output = ElementTypeAny> + std::marker::Copy
        + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default 
        + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
        // IndexTypeAny需要实现的特性
        IndexTypeAny: TryInto<usize>
        >
        (data_source: DataSourceAny) -> L1Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> 
        {
        L1Adaptor::new(data_source)
        }
}

/// 欧几里得距离
pub struct MetricL2;

impl MetricL2 {
    pub fn traits<
        // ElementTypeAny需要实现的特性
        ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs + std::ops::Mul<Output = ElementTypeAny>,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
        // IndexTypeAny需要实现的特性
        IndexTypeAny: TryInto<usize>,
        >
        (data_source: DataSourceAny) -> L2Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> 
        {
        L2Adaptor::new(data_source)
        }
}

/// 欧几里得距离（简单）
pub struct MetricL2Simple;

impl MetricL2Simple {
    pub fn traits<
        // ElementTypeAny需要实现的特性
        ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs 
        + std::ops::Mul<Output = ElementTypeAny>,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
        // IndexTypeAny需要实现的特性
        IndexTypeAny: TryInto<usize>,
        >
        (data_source: DataSourceAny) -> L2SimpleAdaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> 
        {
        L2SimpleAdaptor::new(data_source)
        }
}

/// SO2距离
pub struct MetricSO2;

impl MetricSO2 {
    pub fn traits<
        // ElementTypeAny需要实现的特性
        ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs
        + std::ops::Mul<Output = ElementTypeAny> 
        + std::ops::AddAssign
        + std::ops::SubAssign + Into<f32> + From<f32>,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny> + std::ops::Neg + From<f32> + Into<f32>,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
        // IndexTypeAny需要实现的特性
        IndexTypeAny: TryInto<usize>,
        >
        (data_source: DataSourceAny) -> SO2Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> 
        {
        SO2Adaptor::new(data_source)
        }
}

/// SO3距离
pub struct MetricSO3;

impl MetricSO3 {
    pub fn traits<
        // ElementTypeAny需要实现的特性
        ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs
        + std::ops::Mul<Output = ElementTypeAny>,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData<IndexTypeAny = usize, ElementTypeAny = ElementTypeAny>,
        // IndexTypeAny需要实现的特性
        IndexTypeAny: TryInto<usize>,
        >
        (data_source: DataSourceAny) -> SO3Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> 
        {
        SO3Adaptor::new(data_source)
        }
}

/* end 距离度量类型的封装类 */

// 单元测试
#[cfg(test)]
mod tests4 {
    use super::*;
    use crate::sets::SuperSet;
    use std::any::Any;

    #[test]
    fn test_l1_adaptor() {
        // 创建 SuperSet 存储数据
        let data_vec = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec: Vec<f32> = vec![0.0,0.0,0.0];
        let super_set = SuperSet::new(data_vec, distance_vec);

        // 创建 L1Adaptor
        let l1_adaptor : L1Adaptor<f32, SuperSet, f32, usize> = L1Adaptor::new(super_set);

        // 测试曼哈顿距离计算
        let a = vec![1.0, 2.0, 3.0];
        let dist = l1_adaptor.eval_metric(&a, 1, 3, None);
        assert_eq!(dist, 9.0); // |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9

        let dist = l1_adaptor.eval_metric(&a, 2, 3, None);
        assert_eq!(dist, 18.0); // |1-7| + |2-8| + |3-9| = 6 + 6 + 6 = 18
    }

    #[test]
    fn test_l2_adaptor() {
        // 创建 SuperSet 存储数据
        let data_vec = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec: Vec<f32> = vec![0.0,0.0,0.0];
        let super_set = SuperSet::new(data_vec, distance_vec);

        // 创建 L2Adaptor
        let l2_adaptor : L2Adaptor<f32, SuperSet, f32, usize> = L2Adaptor::new(super_set);

        // 测试欧几里得距离计算
        let a = vec![1.0, 2.0, 3.0];
        let dist = l2_adaptor.eval_metric(&a, 1, 3, None);
        assert_eq!(dist, 27.0); // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27

        let dist = l2_adaptor.eval_metric(&a, 2, 3, None);
        assert_eq!(dist, 108.0); // (1-7)^2 + (2-8)^2 + (3-9)^2 = 36 + 36 + 36 = 108
    }

    #[test]
    fn test_l2_simple_adaptor() {
        // 创建 SuperSet 存储数据
        let data_vec = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec: Vec<f32> = vec![0.0,0.0,0.0];
        let super_set = SuperSet::new(data_vec, distance_vec);

        // 创建 L2SimpleAdaptor
        let l2_simple_adaptor : L2SimpleAdaptor<f32, SuperSet, f32, usize> = L2SimpleAdaptor::new(super_set);

        // 测试简单欧几里得距离计算
        let a = vec![1.0, 2.0, 3.0];
        let dist = l2_simple_adaptor.eval_metric(&a, 1, 3);
        assert_eq!(dist, 27.0); // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27

        let dist = l2_simple_adaptor.eval_metric(&a, 2, 3);
        assert_eq!(dist, 108.0); // (1-7)^2 + (2-8)^2 + (3-9)^2 = 36 + 36 + 36 = 108
    }

    #[test]
    fn test_so2_adaptor() {
        // 创建 SuperSet 存储数据
        let data_vec = vec![
            vec![1.0],
            vec![4.0],
            vec![7.0],
        ];
        let distance_vec: Vec<f32> = vec![0.0,0.0,0.0];
        let super_set = SuperSet::new(data_vec, distance_vec);

        // 创建 SO2Adaptor
        let so2_adaptor : SO2Adaptor<f32, SuperSet, f32, usize> = SO2Adaptor::new(super_set);

        // 测试 SO2 距离计算
        let a = vec![1.0];
        let dist = so2_adaptor.eval_metric(&a, 1, 1);
        assert_eq!(dist, 3.0); // |1 - 4| = 3

        // pub fn eval_metric(&self, a: &[ElementTypeAny], b_idx: usize, size: usize) 
        let dist = so2_adaptor.eval_metric(&a, 2, 1);
        // FIXME: 这个测试用例有问题
        // assert_eq!(dist, 6.0); // |1 - 7| = 6
    }

    #[test]
    fn test_so3_adaptor() {
        // 创建 SuperSet 存储数据
        let data_vec = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec: Vec<f32> = vec![0.0,0.0,0.0];
        let super_set = SuperSet::new(data_vec, distance_vec);

        // 创建 SO3Adaptor
        let so3_adaptor : SO3Adaptor<f32, SuperSet, f32, usize> = SO3Adaptor::new(super_set);

        // 测试 SO3 距离计算
        let a = vec![1.0, 2.0, 3.0];
        let dist = so3_adaptor.eval_metric(&a, 1, 3);
        assert_eq!(dist, 27.0); // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27

        let dist = so3_adaptor.eval_metric(&a, 2, 3);
        assert_eq!(dist, 108.0); // (1-7)^2 + (2-8)^2 + (3-9)^2 = 36 + 36 + 36 = 108
    }
}