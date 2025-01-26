#![allow(unconditional_recursion)]
#![allow(unused_imports)]

/*
这段代码实现了三种不同的结果集结构，用于存储和管理近邻搜索的结果：`KNNResultSet`、`RKNNResultSet` 和 `RadiusResultSet`。`KNNResultSet` 用于存储K近邻搜索的结果，`RKNNResultSet` 在K近邻的基础上增加了最大搜索半径的限制，而 `RadiusResultSet` 则用于存储基于固定半径的搜索结果。每种结果集都提供了初始化、添加点、返回最差距离、排序等操作，并且通过泛型支持不同的距离类型和索引类型。
*/

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use rand::Rng;
use std::ops::Add;
use std::vec::Vec;
use std::any::Any;

use crate::utils::{
    HasPointData, IndexDistSorter, HasDistance, MaxValue
};

/* start 简单集合 */

/// 1. 超集, 数据 + 距离
#[derive(Debug, Clone, PartialEq)]
pub struct SuperSet<ElementTypeAny = f32, DistanceTypeAny = f32> {
    // N * N 维向量
    data_vec: Vec<Vec<ElementTypeAny>>,
    // N 维向量
    distance_vec: Vec<DistanceTypeAny>,
}

/// 实现SuperSet的构造方法
impl<ElementTypeAny, DistanceTypeAny> SuperSet<ElementTypeAny, DistanceTypeAny>
where
    ElementTypeAny: PartialOrd + Clone,
    DistanceTypeAny: Sized + Copy,
{
    /// 创建一个新的超集
    pub fn new(
        data_vec: Vec<Vec<ElementTypeAny>>, 
        distance_vec: Vec<DistanceTypeAny>
        ) -> Self {
        SuperSet { 
            data_vec, distance_vec
        }
    }// end fn new

}

/// 为 SuperSet 实现 HasPointData 特性
/// Box<dyn std::any::Any>确保编译时Sized是已知的
impl<ElementTypeAny> HasPointData for SuperSet<ElementTypeAny, f32>
where
    ElementTypeAny: PartialOrd + Clone,
{
    type ElementTypeAny = ElementTypeAny; // 与输入类型一致
    type IndexTypeAny = usize; // 固定为usize类型
    
    /// 获取指定索引处的点的指定维度的值
    fn get_point_dim(&self, idx: Self::IndexTypeAny, dim: usize)
    -> Self::ElementTypeAny {
        self.data_vec[idx][dim].clone()
    }

    /// 获取指定索引处的点的所有维度的值
    fn get_point(&self, idx: Self::IndexTypeAny) -> Vec<Self::ElementTypeAny> {
        self.data_vec[idx].clone()
    }

    /// 获取指定索引处的点的所有维度的值vec 
    fn get_point_vec(&self) -> Vec<Vec<Self::ElementTypeAny>> {
        self.data_vec.clone()
    }

}

/// 为 SuperSet 实现 HasDistance 特性
impl<ElementTypeAny, DistanceTypeAny> HasDistance for SuperSet<ElementTypeAny, DistanceTypeAny>
where
    DistanceTypeAny: PartialOrd + Clone + Add,
    ElementTypeAny: Copy,
{
    type DistanceTypeAny = DistanceTypeAny;
    type IndexTypeAny = usize; // 默认为usize

    /// 返回结果项的距离
    fn distance(&self, idx: Self::IndexTypeAny) -> Self::DistanceTypeAny {
        self.distance_vec[idx].clone()
    }

    /// 返回结果项的距离vec
    fn distance_vec(&self) -> Vec<Self::DistanceTypeAny> {
        self.distance_vec.clone()
    }

}

/// 2. 数据集, 用于存储原始数据
#[derive(Debug, Clone, PartialEq)]
pub struct DataSet<ElementTypeAny = f64> {
    // N * N 维向量
    data_vec: Vec<Vec<ElementTypeAny>>,
}

/// 为 DataSet 实现 HasPointData 特性
impl<ElementTypeAny> HasPointData for DataSet<ElementTypeAny>
where
    ElementTypeAny: PartialOrd + Clone,
{
    type ElementTypeAny = ElementTypeAny; // 与输入类型一致
    type IndexTypeAny = usize; // 默认为usize
    
    /// 获取指定索引处的点的指定维度的值
    fn get_point_dim(&self, idx: Self::IndexTypeAny, dim: usize)
    -> Self::ElementTypeAny {
        self.data_vec[idx][dim].clone()
    }

    /// 获取指定索引处的点的所有维度的值
    fn get_point(&self, idx: Self::IndexTypeAny) -> Vec<Self::ElementTypeAny> {
        self.data_vec[idx].clone()
    }

    /// 获取指定索引处的点的所有维度的值vec
    fn get_point_vec(&self) -> Vec<Vec<Self::ElementTypeAny>> {
        self.data_vec.clone()
    }

}

/// 3. 结果集，用于存储索引和距离
#[derive(Debug, Clone, PartialEq)]
pub struct ResultSet<IndexTypeAny = usize, DistanceTypeAny = f64> {
    index: IndexTypeAny,
    distance: DistanceTypeAny,
}

impl<IndexTypeAny, DistanceTypeAny> ResultSet<IndexTypeAny, DistanceTypeAny>
where
    DistanceTypeAny: Copy,
    IndexTypeAny: Copy + TryInto<usize>, // 假设索引类型是可复制的,
{
    /// 创建一个新的结果集
    pub fn new(index: IndexTypeAny, distance: DistanceTypeAny) -> Self {

        ResultSet { 
            index, distance
        }

    }

    /// 返回结果集中该点的距离
    pub fn distance(&self) -> DistanceTypeAny {
        self.distance
    }

    /// 返回结果项的该点索引
    pub fn index(&self) -> IndexTypeAny {
        self.index
    }

}

/// 为 ResultSet 实现 HasDistance 特性
impl<IndexTypeAny, DistanceTypeAny> HasDistance for ResultSet<IndexTypeAny, DistanceTypeAny>
where
    DistanceTypeAny: PartialOrd + Clone + Add,
    IndexTypeAny: Copy + TryInto<usize>, // 假设索引类型是可复制的
{
    type DistanceTypeAny = DistanceTypeAny; // 假设点的数据类型是 f64
    type IndexTypeAny = usize;
    
    /// 返回结果项的距离
    fn distance(&self, _:usize) -> DistanceTypeAny {
        self.distance.clone()
    }

    /// 返回结果项的距离vec(为了凑trait, 实际用不到)
    fn distance_vec(&self) -> Vec<Self::DistanceTypeAny> {
        unimplemented!()
    }

}
/* end 简单集合 */

/// KNN结果集，用于存储K近邻搜索的结果
pub struct KNNResultSet<DistanceTypeAny> {
    indices: Vec<usize>,
    dists: Vec<DistanceTypeAny>,
    // 只能使用usize作为索引
    capacity: usize,
    // 只能使用usize作为索引
    count: usize,
}

impl<DistanceTypeAny> KNNResultSet<DistanceTypeAny>
where
    DistanceTypeAny: PartialOrd + Copy + MaxValue,
{
    /// 创建一个新的KNN结果集
    pub fn new(capacity: usize) -> Self {
        KNNResultSet {
            indices: Vec::with_capacity(capacity.into()),
            dists: Vec::with_capacity(capacity.into()),
            capacity,
            count: 0,
        }
    }

    /// 初始化结果集
    pub fn init(&mut self, indices: Vec<usize>, dists: Vec<DistanceTypeAny>) {
        self.indices = indices;
        self.dists = dists;
        self.count = 0;
        if self.capacity > 0 {
            self.dists[self.capacity - 1] = DistanceTypeAny::max_value();
        }
    }

    /// 返回结果集中的元素数量
    pub fn size(&self) -> usize {
        self.count
    }

    /// 判断结果集是否为空
    pub fn empty(&self) -> bool {
        self.count == 0
    }

    /// 判断结果集是否已满
    pub fn full(&self) -> bool {
        self.count == self.capacity
    }

    /// 添加一个点到结果集中
    pub fn add_point(&mut self, dist: DistanceTypeAny, index: usize) -> bool {
        let mut i: usize = self.count.into();
        while i > 0 {
            if self.dists[i - 1] > dist {
                if i < self.capacity.into() {
                    self.dists[i] = self.dists[i - 1];
                    self.indices[i] = self.indices[i - 1];
                }
            } else {
                break;
            }
            i -= 1;
        }
        if i < self.capacity.into() {
            self.dists[i] = dist;
            self.indices[i] = index;
        }
        if self.count < self.capacity {
            self.count = self.count + 1;
        }
        true
    }

    /// 返回结果集中最差的距离
    pub fn worst_dist(&self) -> DistanceTypeAny {
        self.dists[self.capacity - 1]
    }

    /// 对结果集进行排序（已排序）
    pub fn sort(&self) {
        // 已经排序
    }
}

/// RKNN结果集，用于存储带最大半径的K近邻搜索的结果
pub struct RKNNResultSet<DistanceTypeAny> {
    indices: Vec<usize>,
    dists: Vec<DistanceTypeAny>,
    // 只能使用usize作为索引
    capacity: usize,
    // 只能使用usize作为索引
    count: usize,
    maximum_search_distance_squared: DistanceTypeAny,
}

impl<DistanceTypeAny> RKNNResultSet<DistanceTypeAny>
where
    DistanceTypeAny: PartialOrd + Copy + MaxValue,
{
    /// 创建一个新的RKNN结果集
    pub fn new(capacity: usize, maximum_search_distance_squared: DistanceTypeAny) -> Self {
        RKNNResultSet {
            indices: Vec::with_capacity(capacity.into()),
            dists: Vec::with_capacity(capacity.into()),
            capacity,
            count: 0,
            maximum_search_distance_squared,
        }
    }

    /// 初始化结果集
    pub fn init(&mut self, indices: Vec<usize>, dists: Vec<DistanceTypeAny>) {
        self.indices = indices;
        self.dists = dists;
        self.count = 0;
        if self.capacity > 0 {
            self.dists[self.capacity - 1] = self.maximum_search_distance_squared;
        }
    }

    /// 返回结果集中的元素数量
    pub fn size(&self) -> usize {
        self.count
    }

    /// 判断结果集是否为空
    pub fn empty(&self) -> bool {
        self.count == 0
    }

    /// 判断结果集是否已满
    pub fn full(&self) -> bool {
        self.count == self.capacity
    }

    /// 添加一个点到结果集中
    pub fn add_point(&mut self, dist: DistanceTypeAny, index: usize) -> bool {
        let mut i: usize = self.count;
        while i > 0 {
            if self.dists[i - 1] > dist {
                if i < self.capacity {
                    self.dists[i] = self.dists[i - 1];
                    self.indices[i] = self.indices[i - 1];
                }
            } else {
                break;
            }
            i -= 1;
        }
        if i < self.capacity {
            self.dists[i] = dist;
            self.indices[i] = index;
        }
        if self.count < self.capacity {
            self.count = self.count + 1;
        }
        true
    }

    /// 返回结果集中最差的距离
    pub fn worst_dist(&self) -> DistanceTypeAny {
        self.dists[self.capacity - 1]
    }

    /// 对结果集进行排序（已排序）
    pub fn sort(&self) {
        // 已经排序
    }
}

/// 半径结果集，用于存储基于半径的搜索结果
pub struct RadiusResultSet<DistanceTypeAny> {
    radius: DistanceTypeAny,
    indices_dists: Vec<ResultSet<usize, DistanceTypeAny>>,
}

impl<DistanceTypeAny> RadiusResultSet<DistanceTypeAny>
where
    DistanceTypeAny: PartialOrd + Copy,
{
    /// 创建一个新的半径结果集
    pub fn new(radius: DistanceTypeAny, indices_dists: Vec<ResultSet<usize, DistanceTypeAny>>) -> Self {
        RadiusResultSet {
            radius,
            indices_dists,
        }
    }

    /// 初始化结果集
    pub fn init(&mut self) {
        self.clear();
    }

    /// 清空结果集
    pub fn clear(&mut self) {
        self.indices_dists.clear();
    }

    /// 返回结果集中的元素数量
    pub fn size(&self) -> usize {
        self.indices_dists.len()
    }

    /// 判断结果集是否为空
    pub fn empty(&self) -> bool {
        self.indices_dists.is_empty()
    }

    /// 判断结果集是否已满
    pub fn full(&self) -> bool {
        true
    }

    /// 添加一个点到结果集中
    pub fn add_point(&mut self, dist: DistanceTypeAny, index: usize) -> bool {
        if dist < self.radius {
            self.indices_dists.push(ResultSet::new(index, dist));
        }
        true
    }

    /// 返回结果集中最差的距离
    pub fn worst_dist(&self) -> DistanceTypeAny {
        self.radius
    }

    /// 返回结果集中最差的项
    pub fn worst_item(&self) -> ResultSet<usize, DistanceTypeAny> {
        if self.indices_dists.is_empty() {
            panic!("Cannot invoke RadiusResultSet::worst_item() on an empty list of results.");
        }
        self.indices_dists.iter().max_by(|a, b| a.distance().partial_cmp(&b.distance()).unwrap()).unwrap().clone()
    }

    /// 对结果集进行排序
    pub fn sort(&mut self) {
        self.indices_dists.sort_by(|a, b| a.distance().partial_cmp(&b.distance()).unwrap());
    }
}

/// 测试模块
#[cfg(test)]
mod tests3 {
    use super::*;

    #[test]
    fn test_knn_result_set() {
        let mut knn_result_set = KNNResultSet::new(3);
        knn_result_set.init(vec![0, 0, 0], vec![0.0, 0.0, 0.0]);

        knn_result_set.add_point(1.0, 1);
        knn_result_set.add_point(0.5, 2);
        knn_result_set.add_point(2.0, 3);

        assert_eq!(knn_result_set.size(), 3);
        assert_eq!(knn_result_set.worst_dist(), 2.0);
    }

    #[test]
    fn test_rknn_result_set() {
        let mut rknn_result_set = RKNNResultSet::new(3, 2.0);
        rknn_result_set.init(vec![0, 0, 0], vec![0.0, 0.0, 0.0]);

        rknn_result_set.add_point(1.0, 1);
        rknn_result_set.add_point(0.5, 2);
        rknn_result_set.add_point(2.0, 3);

        assert_eq!(rknn_result_set.size(), 3);
        assert_eq!(rknn_result_set.worst_dist(), 2.0);
    }

    #[test]
    fn test_radius_result_set() {
        let mut radius_result_set = RadiusResultSet::new(2.0, vec![]);
        radius_result_set.init();

        radius_result_set.add_point(1.0, 1);
        radius_result_set.add_point(0.5, 2);
        radius_result_set.add_point(2.0, 3);

        assert_eq!(radius_result_set.size(), 2);
        assert_eq!(radius_result_set.worst_dist(), 2.0);
    }

    #[test]
    fn test_result_item() {
        let item = ResultSet::new(1, 2.5);
        assert_eq!(item.index(), 1, "ResultSet index is incorrect");
        assert_eq!(item.distance(), 2.5, "ResultSet distance is incorrect");
    }

    #[test]
    fn test_index_dist_sorter() {
        let item1 = ResultSet::new(1, 2.5);
        let item2 = ResultSet::new(2, 1.5);
        // item2 的距离大于 item1 的距离
        assert!(IndexDistSorter::compare(&item2, &item1), "IndexDistSorter comparison failed");

        let item3 = ResultSet::new(3, 3.0);
        // item3 的距离小于 item1 的距离
        assert!(!IndexDistSorter::compare(&item3, &item1), "IndexDistSorter comparison failed");
    }
}