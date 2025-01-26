#![allow(dead_code)]
#![allow(unconditional_recursion)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_variables)]

use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Deref, Mul, Neg, Sub};
use std::rc::Rc;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::vec::Vec;

/// nanoflann library version: 0xMmP (M=Major, m=minor, P=patch)
const NANOFLANN_VERSION: u32 = 0x163;

// kdtree模块: 实现静态KDTree和动态KDTree。
pub mod kdtree;

// file模块: 保存和加载数据。
pub mod file;

// set模块: 存储 K 近邻搜索和半径搜索的结果, 定义数据集和结果集的接口。
pub mod sets;

// metric模块: 距离度量。
pub mod metric;

// utils模块: 提供了一些辅助函数和公共常量。
pub mod utils;

// memalloc模块: 内存分配器。
pub mod memalloc;

// base模块: 元编程相关的struct,trait和impl。
pub mod base;

// params模块: 保存和加载参数。
pub mod params;

// pointcloud模块: pointcloud形态的kdtree
pub mod pointcloud;

/* start 内部库 */
// // 文件操作
// use crate::file::{
//     save_value, load_value
// };
// // 内存分配
// use crate::memalloc::PooledAllocator;
// // 参数配置
// use crate::params::{
//     KDTreeSingleIndexAdaptorParams, SearchParameters
// };
// // 结果集
// use crate::sets::{
//     KNNResultSet, RKNNResultSet, RadiusResultSet, ResultItem
// };
// // kdtree基类
// use crate::base::{
//     KDTreeBase, ArrayOrVector, NodeType,
//     Node, Interval, 
// };
// // 距离度量
// use crate::metric::{
//     L1Adaptor, L2Adaptor, L2SimpleAdaptor,
//     SO2Adaptor, SO3Adaptor, MetricL1,
//     MetricL2, MetricL2Simple, MetricSO2,
//     MetricSO3, 
// };
// // kdtree实现
// use crate::kdtree::{
//     StaticKDTree, DynamicKDTree, MatrixKDTree,
// };
// // 点云的kdtree
// use crate::pointcloud::{
//     PointCloud, Point, PointCloudQuat, PointQuat, PointCloudOrient, PointOrient, 
// };
// // 工具类
// use crate::utils::{
//     pi_const, HasResize, HasAssign, HasSize, IndexDistSorter, HasPointData, Abs
// };
/* end 内部库 */