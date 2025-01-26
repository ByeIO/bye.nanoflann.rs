#![allow(dead_code)]
#![allow(unconditional_recursion)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_variables)]

//! KD树的基本特性

use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};
use std::future::Future;
use std::pin::Pin;
use std::vec::Vec;
use std::marker::PhantomData;
use rand::distributions::{Distribution, Uniform};

use crate::file::{save_value, load_value};
use crate::memalloc::PooledAllocator;
use crate::params::{KDTreeSingleIndexAdaptorParams, SearchParameters};
use crate::sets::{KNNResultSet, RKNNResultSet, RadiusResultSet, ResultSet};

/* start 树叶 */

/// 节点类型
pub enum NodeType {
    Leaf { left: usize, right: usize },
    NonLeaf { divfeat: i32, divlow: f64, divhigh: f64 },
}

/// 树节点
pub struct Node {
    node_type: NodeType,
    child1: Option<Box<Node>>,
    child2: Option<Box<Node>>,
}

/// 区间
pub struct Interval {
    low: f64,
    high: f64,
}
/* end 树叶 */

/* start 数组或可变向量 */

/// 用于声明固定大小的数组或动态分配的向量
/// 当 `DIM > 0` 时，初始化一个固定大小的 `Vec<T>`；
/// 当 `DIM = None` 时，允许动态调整大小。
/// rust中Vec为动态大小, 不使用push和pop就变成固定大小了。
pub struct ArrayOrVector<const DIM: usize, T> {
    data: Vec<T>,
    _phantom: PhantomData<T>,
}

impl<const DIM: usize, T> ArrayOrVector<DIM, T> {
    /// 创建一个新的 `ArrayOrVector`
    pub fn new() -> Self {
        // DIM = None时就是动态大小
        let initial_capacity = if DIM > 0 { DIM } else { 0 };
        Self {
            data: Vec::with_capacity(initial_capacity),
            _phantom: PhantomData,
        }
    }

    /// 获取内部数据的引用
    pub fn as_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// 获取内部数据的可变引用
    pub fn as_mut_vec(&mut self) -> &mut Vec<T> {
        &mut self.data
    }

    /// 向 `ArrayOrVector` 中添加元素
    pub fn push(&mut self, value: T) {
        if DIM > 0 && self.data.len() >= DIM {
            panic!("Cannot push more elements: fixed size exceeded");
        }
        self.data.push(value);
    }

    /// 获取当前元素数量
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
/* end 数组或可变向量 */

/* start KDTreeBase公共特性 */

// TODO: 实现多线程
pub trait KDTreeBase {
    type ElementType;
    type DistanceType;
    type IndexType;
    // 初始化KD树
    fn init(&mut self, dimensionality: usize, leaf_max_size: usize);

    // 建立索引
    fn build_index(&mut self);

    // 更新索引, 确保辅助索引列表的大小与当前数据集一致，并在大小发生变化时重新生成。
    fn update_index(&mut self);

    // 获取数据点数量
    fn get_point_count(&self) -> usize;

    // 最近邻搜索
    fn find_neighbors(&self, query_point: &[Self::ElementType], num_closest: usize) -> Vec<(Self::IndexType, Self::DistanceType)>;

    // 半径搜索
    fn radius_search(&self, query_point: &[Self::ElementType], radius: Self::DistanceType) -> Vec<(Self::IndexType, Self::DistanceType)>;

    // K近邻搜索
    fn knn_search(&self, query_point: &[Self::ElementType], num_closest: usize) -> Vec<(Self::IndexType, Self::DistanceType)>;

    // 半径K近邻搜索
    fn rknn_search(&self, query_point: &[Self::ElementType], num_closest: usize, radius: Self::DistanceType) -> Vec<(Self::IndexType, Self::DistanceType)>;

    // 执行从节点开始的精确搜索
    fn search_level(&self, query_point: &[Self::ElementType], node: &Node, mindist: Self::DistanceType, eps_error: f32) -> Vec<(Self::IndexType, Self::DistanceType)>;

    // 计算边界框
    fn compute_bounding_box(&self) -> Option<Vec<Interval>>;

    // 节点分割, 用于实现节点的分割逻辑。它接受起始和结束索引，并返回一个 Node 对象。
    fn divide_tree(&mut self, start: usize, end: usize) -> Node;

    // 保存索引
    fn save_index(&self, path: &str) -> Result<(), std::io::Error>;

    // 加载索引
    fn load_index(&mut self, path: &str) -> Result<(), std::io::Error>;
}

/* end KDTreeBase公共特性 */