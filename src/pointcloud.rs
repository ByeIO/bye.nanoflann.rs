#![allow(unconditional_recursion)]
#![allow(unused_imports)]
#![allow(unused_variables)]

/*
提供了几个模板类和相关函数，用于生成和管理不同类型的点云数据（三维点、四元数、方向）。这些类和方法可以用于构建和处理点云数据，适用于需要处理三维空间数据的应用场景。
*/

use rand::Rng;
use std::f64::consts::PI;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

// 工具类
use crate::utils::{
    pi_const, HasResize, HasAssign, HasSize, 
    IndexDistSorter, HasPointData, Abs
};

/* start 点云相关 */

/// 三维点结构体
#[derive(Clone)]
pub struct Point<ElementTypeAny> {
    pub x: ElementTypeAny,
    pub y: ElementTypeAny,
    pub z: ElementTypeAny,
}

/// 点云结构体，包含三维坐标
#[derive(Clone)]
pub struct PointCloud<ElementTypeAny> {
    pub pts: Vec<Point<ElementTypeAny>>,
}

/// PointCloud的实现
impl<T> PointCloud<T> {
    /// 返回点云数据集中点的数量
    pub fn get_point_count(&self) -> usize {
        self.pts.len()
    }

    /// 返回数据集中指定点的指定维度的值
    pub fn get_point(&self, idx: usize, dim: usize) -> &T {
        match dim {
            0 => &self.pts[idx].x,
            1 => &self.pts[idx].y,
            2 => &self.pts[idx].z,
            _ => panic!("Invalid dimension"),
        }
    }

    /// 可选：计算边界框，默认返回false
    pub fn get_bbox<BBOX>(&self, _bb: &mut BBOX) -> bool {
        false
    }
}

// 为PointCloud实现HasPointData特性
impl<ElementTypeAny: Copy + PartialOrd> HasPointData for PointCloud<ElementTypeAny> {
    type ElementTypeAny = ElementTypeAny;
    type IndexTypeAny = usize;
   
    // 获取某一点某一维度
    fn get_point_dim(&self, idx: Self::IndexTypeAny, dim: usize) -> Self::ElementTypeAny {
        match dim {
            0 => self.pts[idx].x,
            1 => self.pts[idx].y,
            2 => self.pts[idx].z,
            _ => panic!("Invalid dimension"),
        }
    }

    // 获取某一点
    fn get_point(&self, idx: Self::IndexTypeAny) -> Vec<Self::ElementTypeAny> {
        vec![self.pts[idx].x, self.pts[idx].y, self.pts[idx].z]
    }

    // 获取所有数据
    fn get_point_vec(&self) -> Vec<Vec<Self::ElementTypeAny>> {
        unimplemented!()
    }

}

/// 生成随机点云
pub fn generate_random_point_cloud_ranges<T: rand::distributions::uniform::SampleUniform + Copy + std::default::Default 
+ std::cmp::PartialOrd>(
    pc: &mut PointCloud<T>,
    n: usize,
    max_range_x: T,
    max_range_y: T,
    max_range_z: T,
) where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    let mut rng = rand::thread_rng();
    pc.pts.resize(n, Point { x: T::default(), y: T::default(), z: T::default() });
    for i in 0..n {
        pc.pts[i].x = rng.gen_range(T::default()..max_range_x);
        pc.pts[i].y = rng.gen_range(T::default()..max_range_y);
        pc.pts[i].z = rng.gen_range(T::default()..max_range_z);
    }
}

/// 生成随机点云，所有维度的范围相同
pub fn generate_random_point_cloud<
T: rand::distributions::uniform::SampleUniform + Copy + std::default::Default
+ std::cmp::PartialOrd
>(
    pc: &mut PointCloud<T>,
    n: usize,
    max_range: T,
) where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    generate_random_point_cloud_ranges(pc, n, max_range, max_range, max_range);
}

/// 四元数点云结构体
#[derive(Clone)]
pub struct PointCloudQuat<T> {
    pub pts: Vec<PointQuat<T>>,
}

/// 四元数点结构体
#[derive(Clone)]
pub struct PointQuat<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

/// PointCloudQuat的实现
impl<T> PointCloudQuat<T> {
    /// 返回点云中点的数量
    pub fn get_point_count(&self) -> usize {
        self.pts.len()
    }

    /// 返回指定点的指定维度的值
    pub fn get_point(&self, idx: usize, dim: usize) -> &T {
        match dim {
            0 => &self.pts[idx].w,
            1 => &self.pts[idx].x,
            2 => &self.pts[idx].y,
            3 => &self.pts[idx].z,
            _ => panic!("Invalid dimension"),
        }
    }

    /// 可选：计算边界框，默认返回false
    pub fn get_bbox<BBOX>(&self, _bb: &mut BBOX) -> bool {
        false
    }
}

/// PointCloudQuat的实现
impl<ElementTypeAny: Copy + PartialOrd> HasPointData for PointCloudQuat<ElementTypeAny> {
    type ElementTypeAny = ElementTypeAny;
    type IndexTypeAny = usize;

    // 获取某一点某一维度
    fn get_point_dim(&self, idx: Self::IndexTypeAny, dim: usize) -> Self::ElementTypeAny {
        match dim {
            0 => self.pts[idx].w,
            1 => self.pts[idx].x,
            2 => self.pts[idx].y,
            3 => self.pts[idx].z,
            _ => panic!("Invalid dimension"),
        }
    }

     // 获取某一点
     fn get_point(&self, idx: Self::IndexTypeAny) -> Vec<Self::ElementTypeAny> {
        vec![self.pts[idx].w, self.pts[idx].x, self.pts[idx].y, self.pts[idx].z]
    }

    // 获取所有数据
    fn get_point_vec(&self) -> Vec<Vec<Self::ElementTypeAny>> {
        unimplemented!()
    }

}

/// 生成随机四元数点云
pub fn generate_random_point_cloud_quat<T: rand::distributions::uniform::SampleUniform + Copy + std::default::Default>(
    point: &mut PointCloudQuat<T>,
    n: usize,
) where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
    T: std::ops::Mul<Output = T> + std::ops::Div<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Neg<Output = T> + std::fmt::Debug,
    f64: Into<T>,
{
    let mut rng = rand::thread_rng();
    point.pts.resize(n, PointQuat { w: T::default(), x: T::default(), y: T::default(), z: T::default() });
    for i in 0..n {
        let theta: f64 = rng.gen_range(0.0..PI);
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);
        let z: f64 = rng.gen_range(-1.0..1.0);
        let mag = (x * x + y * y + z * z).sqrt();
        let x = x / mag;
        let y = y / mag;
        let z = z / mag;
        let cos_ang = (theta / 2.0).cos();
        let sin_ang = (theta / 2.0).sin();
        point.pts[i].w = cos_ang.into();
        point.pts[i].x = (x * sin_ang).into();
        point.pts[i].y = (y * sin_ang).into();
        point.pts[i].z = (z * sin_ang).into();
    }
}

/// 方向点云结构体
#[derive(Clone)]
pub struct PointCloudOrient<T> {
    pub pts: Vec<PointOrient<T>>,
}

/// 方向点结构体
#[derive(Clone)]
pub struct PointOrient<T> {
    pub theta: T,
}

/// 方向点云行为的实现
impl<T> PointCloudOrient<T> {
    /// 返回点云中点的数量
    pub fn get_point_count(&self) -> usize {
        self.pts.len()
    }

    /// 返回指定点的指定维度的值
    pub fn get_point(&self, idx: usize, _dim: usize) -> &T {
        &self.pts[idx].theta
    }

    /// 可选：计算边界框，默认返回false
    pub fn get_bbox<BBOX>(&self, _bb: &mut BBOX) -> bool {
        false
    }
}

/// 方向点云行为的实现
impl<ElementTypeAny: Copy + PartialOrd> HasPointData for PointCloudOrient<ElementTypeAny> {
    type ElementTypeAny = ElementTypeAny;
    type IndexTypeAny = usize;

    // 获取某一点某一维度
    fn get_point_dim(&self, idx: Self::IndexTypeAny, _dim: usize) -> Self::ElementTypeAny {
        self.pts[idx].theta
    }

     // 获取某一点
     fn get_point(&self, idx: Self::IndexTypeAny) -> Vec<Self::ElementTypeAny> {
        vec![self.pts[idx].theta]
    }

    // 获取所有数据
    fn get_point_vec(&self) -> Vec<Vec<Self::ElementTypeAny>> {
        unimplemented!()
    }

}

/// 生成随机方向点云
pub fn generate_random_point_cloud_orient<T: rand::distributions::uniform::SampleUniform + Copy + std::default::Default>(
    point: &mut PointCloudOrient<T>,
    n: usize,
) where
    rand::distributions::Standard: rand::distributions::Distribution<T>,
    f64: Into<T>,
{
    let mut rng = rand::thread_rng();
    point.pts.resize(n, PointOrient { theta: T::default() });
    for i in 0..n {
        let theta: f64 = rng.gen_range(-PI..PI);
        point.pts[i].theta = theta.into();
    }
}

#[cfg(test)]
mod tests1 {
    use super::*;

    #[test]
    fn test_point_cloud() {
        let mut pc: PointCloud<f64> = PointCloud { pts: Vec::new() };
        generate_random_point_cloud(&mut pc, 100, 10.0);
        assert_eq!(pc.get_point_count(), 100);
    }

    #[test]
    fn test_point_cloud_quat() {
        let mut pc: PointCloudQuat<f64> = PointCloudQuat { pts: Vec::new() };
        generate_random_point_cloud_quat(&mut pc, 100);
        assert_eq!(pc.get_point_count(), 100);
    }

    #[test]
    fn test_point_cloud_orient() {
        let mut pc: PointCloudOrient<f64> = PointCloudOrient { pts: Vec::new() };
        generate_random_point_cloud_orient(&mut pc, 100);
        assert_eq!(pc.get_point_count(), 100);
    }
}

/* end 点云相关 */