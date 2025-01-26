#![allow(dead_code)]
#![allow(unconditional_recursion)]
#![allow(unused_imports)]

//! 本模块定义了多种距离度量函数，用于计算高维数据集中的距离。
//! 包括曼哈顿距离、欧几里得距离、SO2距离和SO3距离等。

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

/// 曼哈顿距离适配器（L1距离）的实现
impl<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> L1Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>
where
    // ElementTypeAny需要实现的特性
    ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
    // DistanceTypeAny需要实现的特性
    DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
    // DataSourceAny需要实现的特性(可以访问数据源中的点)
    DataSourceAny: HasPointData,
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

    // // TODO: 实现kdtree后再实现
    // /// 计算曼哈顿距离
    // pub fn eval_metric(&self, a: &[ElementTypeAny], b_idx: IndexTypeAny, size: usize, worst_dist: Option<DistanceTypeAny>) -> DistanceTypeAny 
    // where <DistanceTypeAny as Add<ElementTypeAny>>::Output: Add<ElementTypeAny>
    // {
    //     let mut result = DistanceTypeAny::default();
    //     let last = a.len();
    //     let lastgroup = last - 3;
    //     let mut d = 0;

    //     // 每次循环处理4个元素以提高效率
    //     let mut a_ptr = a.as_ptr();
    //     while (a_ptr as usize) < (a.as_ptr().wrapping_add(lastgroup) as usize) {
    //         let diff0 = (unsafe { *a_ptr.offset(0) } - self.data_source.kdtree_get_pt(b_idx, d)).abs();
    //         let diff1 = (unsafe { *a_ptr.offset(1) } - self.data_source.kdtree_get_pt(b_idx, d + 1)).abs();
    //         let diff2 = (unsafe { *a_ptr.offset(2) } - self.data_source.kdtree_get_pt(b_idx, d + 2)).abs();
    //         let diff3 = (unsafe { *a_ptr.offset(3) } - self.data_source.kdtree_get_pt(b_idx, d + 3)).abs();
    //         result += diff0 + diff1 + diff2 + diff3;
    //         a_ptr = unsafe { a_ptr.offset(4) };
    //         d += 4;

    //         if let Some(worst_dist) = worst_dist {
    //             if result > worst_dist {
    //                 return result;
    //             }
    //         }
    //     }

    //     // 处理剩余的0-3个元素
    //     while (a_ptr as usize) < (a.as_ptr().wrapping_add(last) as usize) {
    //         result += (unsafe { *a_ptr } - self.data_source.kdtree_get_pt(b_idx.try_into().unwrap_or(0), d)).abs().into();
    //         a_ptr = unsafe { a_ptr.offset(1) };
    //         d += 1;
    //     }

    //     result
    // }

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

/// 欧几里得距离适配器（L2距离）行为的实现
impl<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> L2Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>
where
       // ElementTypeAny需要实现的特性
       ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
       // DistanceTypeAny需要实现的特性
       DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
       // DataSourceAny需要实现的特性(可以访问数据源中的点)
       DataSourceAny: HasPointData,
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

    // // TODO: 实现kdtree后再实现
    // pub fn eval_metric(&self, a: &[T], b_idx: IndexType, size: usize, worst_dist: Option<DistanceType>) -> DistanceType 
    // where <DistanceType as Add>::Output: Add<DistanceType>
    // {
    //     let mut result = DistanceType::default();
    //     let last = a.len();
    //     let lastgroup = last - 3;
    //     let mut d = 0;

    //     // 每次循环处理4个元素以提高效率
    //     let mut a_ptr = a.as_ptr();
    //     while (a_ptr as usize) < (a.as_ptr().wrapping_add(lastgroup) as usize) {
    //         let diff0 = unsafe { *a_ptr.offset(0) } - self.data_source.kdtree_get_pt(b_idx.try_into().unwrap_or(0), d);
    //         let diff1 = unsafe { *a_ptr.offset(1) } - self.data_source.kdtree_get_pt(b_idx.try_into().unwrap_or(0), d + 1);
    //         let diff2 = unsafe { *a_ptr.offset(2) } - self.data_source.kdtree_get_pt(b_idx.try_into().unwrap_or(0), d + 2);
    //         let diff3 = unsafe { *a_ptr.offset(3) } - self.data_source.kdtree_get_pt(b_idx.try_into().unwrap_or(0), d + 3);
    //         result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    //         a_ptr = unsafe { a_ptr.offset(4) };
    //         d += 4;

    //         if let Some(worst_dist) = worst_dist {
    //             if result > worst_dist {
    //                 return result;
    //             }
    //         }
    //     }

    //     // 处理剩余的0-3个元素
    //     while (a_ptr as usize) < (a.as_ptr().wrapping_add(last) as usize) {
    //         let diff = unsafe { *a_ptr } - self.data_source.kdtree_get_pt(b_idx.try_into().unwrap_or(0), d);
    //         result += diff * diff;
    //         a_ptr = unsafe { a_ptr.offset(1) };
    //         d += 1;
    //     }

    //     result
    // }

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

/// 简单的欧几里得距离适配器（L2距离）
/// 适用于低维数据集，如2D或3D点云。
pub struct L2SimpleAdaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> {
    data_source: DataSourceAny,
    _phantom: std::marker::PhantomData<(ElementTypeAny, DistanceTypeAny, IndexTypeAny)>,
}

/// 简单的欧几里得距离适配器（L2距离）行为的实现
impl<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> L2SimpleAdaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny>
where
       // ElementTypeAny需要实现的特性
       ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
       // DistanceTypeAny需要实现的特性
       DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
       // DataSourceAny需要实现的特性(可以访问数据源中的点)
       DataSourceAny: HasPointData,
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

    // TODO: 实现kdtree后再实现
    // pub fn eval_metric(&self, a: &[T], b_idx: IndexType, size: usize) -> DistanceType {
    //     let mut result = DistanceType::default();
    //     for i in 0..size {
    //         let diff = a[i] - self.data_source.kdtree_get_pt(b_idx.try_into().unwrap_or(0), i);
    //         result += diff * diff;
    //     }
    //     result
    // }

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
       ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
       // DistanceTypeAny需要实现的特性
       DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny> + std::ops::Neg + From<f64>,
       // DataSourceAny需要实现的特性(可以访问数据源中的点)
       DataSourceAny: HasPointData,
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

    // TODO: 实现kdtree后再实现
    // pub fn eval_metric(&self, a: &[T], b_idx: IndexType, size: usize) -> DistanceType {
    //     self.accum_dist(a[size - 1], self.data_source.kdtree_get_pt(b_idx.try_into().unwrap_or(0), size - 1), size - 1)
    // }

    // 计算两个标量值之间的SO2距离
    pub fn accum_dist<U, V>(&self, a: U, b: V, _idx: usize) -> DistanceTypeAny
    where
        // U类型需要实现的特性
        U: Copy + std::ops::Sub<Output = U> + std::ops::Add<Output = DistanceTypeAny> + std::ops::Neg<Output = DistanceTypeAny> + Abs +
        Into<DistanceTypeAny> + std::ops::Mul<Output = DistanceTypeAny> + std::cmp::PartialOrd<DistanceTypeAny> + std::ops::AddAssign<f64> + std::ops::SubAssign<f64> + Into<f64>,
        // V类型需要实现的特性
        V: Copy + Into<U>,
    {
        // 先让a与b类型一致进行运算后再转为DistantTypeAny类型返回
        let mut result : f64 = (b.into() - a).into();
        let pi = pi_const::<f64>();
        if result > pi.into() {
            result -= 2.0 * pi;
        } else if result < -pi {
            result += 2.0 * pi;
        }
        // 返回
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
       ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
       // DistanceTypeAny需要实现的特性
       DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
       // DataSourceAny需要实现的特性(可以访问数据源中的点)
       DataSourceAny: HasPointData,
       // IndexTypeAny需要实现的特性
       IndexTypeAny: TryInto<usize>,
{
    // 构造数据
    pub fn new(data_source: DataSourceAny) -> Self {
        Self {
            l2_simple: L2SimpleAdaptor::new(data_source),
        }
    }

    // TODO: 实现kdtree后再实现
    // pub fn eval_metric(&self, a: &[T], b_idx: IndexType, size: usize) -> DistanceType {
    //     self.l2_simple.eval_metric(a, b_idx, size)
    // }

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
        DataSourceAny: HasPointData,
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
        ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData,
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
        ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData,
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
        ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny> + std::ops::Neg + From<f64>,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData,
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
        ElementTypeAny: Copy + std::ops::Sub<Output = ElementTypeAny> + std::ops::Add<Output = ElementTypeAny> + std::cmp::PartialOrd + std::ops::Neg<Output = ElementTypeAny> + Abs,
        // DistanceTypeAny需要实现的特性
        DistanceTypeAny: From<ElementTypeAny> + Copy + std::ops::AddAssign + Default + std::cmp::PartialOrd + std::ops::Add<ElementTypeAny>,
        // DataSourceAny需要实现的特性(可以访问数据源中的点)
        DataSourceAny: HasPointData,
        // IndexTypeAny需要实现的特性
        IndexTypeAny: TryInto<usize>,
        >
        (data_source: DataSourceAny) -> SO3Adaptor<ElementTypeAny, DataSourceAny, DistanceTypeAny, IndexTypeAny> 
        {
        SO3Adaptor::new(data_source)
        }
}

/* end 距离度量类型的封装类 */

// TODO 单元测试
#[cfg(test)]
mod tests4 {
    use super::*;
    
}