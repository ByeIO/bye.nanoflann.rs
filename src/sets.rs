#![allow(unconditional_recursion)]
#![allow(unused_imports)]

/*
这段代码实现了三种不同的结果集结构，用于存储和管理近邻搜索的结果：`KNNResultSet`、`RKNNResultSet` 和 `RadiusResultSet`。`KNNResultSet` 用于存储K近邻搜索的结果，`RKNNResultSet` 在K近邻的基础上增加了最大搜索半径的限制，而 `RadiusResultSet` 则用于存储基于固定半径的搜索结果。每种结果集都提供了初始化、添加点、返回最差距离、排序等操作，并且通过泛型支持不同的距离类型和索引类型。
*/

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use rand::Rng;

use crate::utils::HasPointData;

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

/// KNN结果集，用于存储K近邻搜索的结果
pub struct KNNResultSet<DistanceTypeAny, IndexTypeAny = usize, CountTypeAny = usize> {
    indices: Vec<IndexTypeAny>,
    dists: Vec<DistanceTypeAny>,
    capacity: CountTypeAny,
    count: CountTypeAny,
}

impl<DistanceTypeAny, IndexTypeAny, CountTypeAny> KNNResultSet<DistanceTypeAny, IndexTypeAny, CountTypeAny>
where
    DistanceTypeAny: PartialOrd + Copy + MaxValue,
    IndexTypeAny: Copy,
    CountTypeAny: PartialOrd + Copy + From<usize> + Into<usize>,
{
    /// 创建一个新的KNN结果集
    pub fn new(capacity: CountTypeAny) -> Self {
        KNNResultSet {
            indices: Vec::with_capacity(capacity.into()),
            dists: Vec::with_capacity(capacity.into()),
            capacity,
            count: CountTypeAny::from(0),
        }
    }

    /// 初始化结果集
    pub fn init(&mut self, indices: Vec<IndexTypeAny>, dists: Vec<DistanceTypeAny>) {
        self.indices = indices;
        self.dists = dists;
        self.count = CountTypeAny::from(0);
        if self.capacity > CountTypeAny::from(0) {
            self.dists[self.capacity.into() - 1] = DistanceTypeAny::max_value();
        }
    }

    /// 返回结果集中的元素数量
    pub fn size(&self) -> CountTypeAny {
        self.count
    }

    /// 判断结果集是否为空
    pub fn empty(&self) -> bool {
        self.count == CountTypeAny::from(0)
    }

    /// 判断结果集是否已满
    pub fn full(&self) -> bool {
        self.count == self.capacity
    }

    /// 添加一个点到结果集中
    pub fn add_point(&mut self, dist: DistanceTypeAny, index: IndexTypeAny) -> bool {
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
            self.count = CountTypeAny::from(self.count.into() + 1);
        }
        true
    }

    /// 返回结果集中最差的距离
    pub fn worst_dist(&self) -> DistanceTypeAny {
        self.dists[self.capacity.into() - 1]
    }

    /// 对结果集进行排序（已排序）
    pub fn sort(&self) {
        // 已经排序
    }
}

/// RKNN结果集，用于存储带最大半径的K近邻搜索的结果
pub struct RKNNResultSet<DistanceTypeAny, IndexTypeAny = usize, CountTypeAny = usize> {
    indices: Vec<IndexTypeAny>,
    dists: Vec<DistanceTypeAny>,
    capacity: CountTypeAny,
    count: CountTypeAny,
    maximum_search_distance_squared: DistanceTypeAny,
}

impl<DistanceTypeAny, IndexTypeAny, CountTypeAny> RKNNResultSet<DistanceTypeAny, IndexTypeAny, CountTypeAny>
where
    DistanceTypeAny: PartialOrd + Copy + MaxValue,
    IndexTypeAny: Copy,
    CountTypeAny: PartialOrd + Copy + From<usize> + Into<usize>,
{
    /// 创建一个新的RKNN结果集
    pub fn new(capacity: CountTypeAny, maximum_search_distance_squared: DistanceTypeAny) -> Self {
        RKNNResultSet {
            indices: Vec::with_capacity(capacity.into()),
            dists: Vec::with_capacity(capacity.into()),
            capacity,
            count: CountTypeAny::from(0),
            maximum_search_distance_squared,
        }
    }

    /// 初始化结果集
    pub fn init(&mut self, indices: Vec<IndexTypeAny>, dists: Vec<DistanceTypeAny>) {
        self.indices = indices;
        self.dists = dists;
        self.count = CountTypeAny::from(0);
        if self.capacity > CountTypeAny::from(0) {
            self.dists[self.capacity.into() - 1] = self.maximum_search_distance_squared;
        }
    }

    /// 返回结果集中的元素数量
    pub fn size(&self) -> CountTypeAny {
        self.count
    }

    /// 判断结果集是否为空
    pub fn empty(&self) -> bool {
        self.count == CountTypeAny::from(0)
    }

    /// 判断结果集是否已满
    pub fn full(&self) -> bool {
        self.count == self.capacity
    }

    /// 添加一个点到结果集中
    pub fn add_point(&mut self, dist: DistanceTypeAny, index: IndexTypeAny) -> bool {
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
            self.count = CountTypeAny::from(self.count.into() + 1);
        }
        true
    }

    /// 返回结果集中最差的距离
    pub fn worst_dist(&self) -> DistanceTypeAny {
        self.dists[self.capacity.into() - 1]
    }

    /// 对结果集进行排序（已排序）
    pub fn sort(&self) {
        // 已经排序
    }
}

/// 半径结果集，用于存储基于半径的搜索结果
pub struct RadiusResultSet<DistanceTypeAny, IndexTypeAny = usize> {
    radius: DistanceTypeAny,
    indices_dists: Vec<ResultItem<IndexTypeAny, DistanceTypeAny>>,
}

impl<DistanceTypeAny, IndexTypeAny> RadiusResultSet<DistanceTypeAny, IndexTypeAny>
where
    DistanceTypeAny: PartialOrd + Copy,
    IndexTypeAny: Copy,
{
    /// 创建一个新的半径结果集
    pub fn new(radius: DistanceTypeAny, indices_dists: Vec<ResultItem<IndexTypeAny, DistanceTypeAny>>) -> Self {
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
    pub fn add_point(&mut self, dist: DistanceTypeAny, index: IndexTypeAny) -> bool {
        if dist < self.radius {
            self.indices_dists.push(ResultItem::new(index, dist));
        }
        true
    }

    /// 返回结果集中最差的距离
    pub fn worst_dist(&self) -> DistanceTypeAny {
        self.radius
    }

    /// 返回结果集中最差的项
    pub fn worst_item(&self) -> ResultItem<IndexTypeAny, DistanceTypeAny> {
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

/// 结果项，用于存储索引和距离
#[derive(Debug, Clone, PartialEq)]
pub struct ResultItem<IndexTypeAny = usize, DistanceTypeAny = f64> {
    index: IndexTypeAny,
    distance: DistanceTypeAny,
}

impl<IndexTypeAny, DistanceTypeAny> ResultItem<IndexTypeAny, DistanceTypeAny>
where
    DistanceTypeAny: Copy,
    IndexTypeAny: Copy,
{
    /// 创建一个新的结果项
    pub fn new(index: IndexTypeAny, distance: DistanceTypeAny) -> Self {
        ResultItem { index, distance }
    }

    /// 返回结果项的距离
    pub fn distance(&self) -> DistanceTypeAny {
        self.distance
    }

    /// 返回结果项的索引
    pub fn index(&self) -> IndexTypeAny {
        self.index
    }
}

/// 为 ResultItem 实现 HasPointData 特性
impl<IndexTypeAny, DistanceTypeAny> HasPointData for ResultItem<IndexTypeAny, DistanceTypeAny>
where
    DistanceTypeAny: PartialOrd + Clone,
    IndexTypeAny: Copy, // 假设索引类型是可复制的
{
    type Item = f64; // 假设点的数据类型是 f64
    type IndexTypeAny = IndexTypeAny;
    type DistanceTypeAny = DistanceTypeAny;

    /// 获取距离
    fn distance(&self) -> Self::DistanceTypeAny {
        self.distance().clone()
    }

    /// 获取指定索引处的点的指定维度的值
    fn kdtree_get_pt(&self, idx: Self::IndexTypeAny, dim: usize) -> Self::Item {
        unimplemented!()
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
        let item = ResultItem::new(1, 2.5);
        assert_eq!(item.index(), 1, "ResultItem index is incorrect");
        assert_eq!(item.distance(), 2.5, "ResultItem distance is incorrect");
    }

    #[test]
    fn test_index_dist_sorter() {
        let item1 = ResultItem::new(1, 2.5);
        let item2 = ResultItem::new(2, 1.5);
        assert!(IndexDistSorter::compare(&item2, &item1), "IndexDistSorter comparison failed");

        let item3 = ResultItem::new(3, 3.0);
        assert!(IndexDistSorter::compare(&item3, &item1), "IndexDistSorter comparison failed");
    }
}