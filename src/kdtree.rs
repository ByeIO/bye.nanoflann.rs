#![allow(dead_code)]
#![allow(unconditional_recursion)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_assignments)]

use std::io::{self, Read, Write};
use std::ops::{Add, Deref, Mul, Neg, Sub};
use std::sync::{Arc, Mutex};
use std::borrow::Borrow;
use std::cell::RefCell;
use std::rc::Rc;
use std::fmt::Debug;
use std::ptr;
use std::alloc::{alloc, dealloc, Layout};
use std::future::Future;
use std::pin::Pin;
use std::vec::Vec;
use std::fs;
use std::marker::PhantomData;
use rand::distributions::{Distribution, Uniform};

// 引入内部库
use crate::file::{
    save_value, load_value
};
use crate::memalloc::PooledAllocator;
use crate::params::{
    KDTreeSingleIndexAdaptorParams, SearchParameters, KDTreeSingleIndexAdaptorFlags
};
use crate::sets::{
    KNNResultSet, RKNNResultSet, RadiusResultSet, ResultSet, SuperSet
};
use crate::base::{
    KDTreeBase, ArrayOrVector, NodeType,
    Node, Interval, 
};
use crate::utils::{
    HasPointData, HasDistance
};
use crate::metric::{
    L1Adaptor, L2Adaptor, L2SimpleAdaptor,
    SO2Adaptor, SO3Adaptor, MetricL1,
    MetricL2, MetricL2Simple, MetricSO2,
    MetricSO3, l2_distance, 
};

/* start 静态KD树 */
// SingleIndexKDTree, 使用index表示整棵树
pub struct StaticKDTree<'a>{
    /// 引用数据集的数据, 数据集需要符合SuperSet的格式要求
    dataset: &'a mut SuperSet<f32, f32>,

    /// 树的维度
    dim: usize,

    /// 索引参数
    index_params: KDTreeSingleIndexAdaptorParams,

    /// 搜索参数
    search_params: SearchParameters,

    /// 树的根节点, 
    /// Node 用于表示树的节点。每个节点包含指向其子节点的指针（ child1 和 child2 ），以及用于划分数据的特征（ divfeat ）和划分值（ divlow 和 divhigh ）,
    /// 根节点是树的入口点，它包含整个数据集的信息。
    root_node: Option<Box<Node>>,

    /// 边界框
    root_bbox: Vec<Interval>,

    /// 辅助索引列表
    index_acc: Vec<usize>,
    
    /// 叶子节点的最大大小
    leaf_max_size: usize,
}

// 'a: 接受任意生命周期
impl <'a> StaticKDTree<'a> 
{
    /// 创建KD树(返回StaticKDTree类型,无法定义到KDTreeBase中)
    fn new(
        data_source: &'a mut SuperSet, dimension: usize, leaf_max_size: usize
    ) -> Self {
        let mut tree = StaticKDTree {
            dataset: data_source,
            dim: dimension,
            // index_params,
            index_params: KDTreeSingleIndexAdaptorParams::new(2, KDTreeSingleIndexAdaptorFlags::None, 1),
            // search_params,
            search_params: SearchParameters::new(0.0, true),
            root_node: None,
            root_bbox: Vec::new(),
            index_acc: Vec::new(),
            leaf_max_size: leaf_max_size,
        };
        tree.index_acc = (0..tree.dataset.get_point_vec().len()).collect();
        tree.root_bbox = tree.compute_bounding_box().expect("REASON");
        tree.root_node = Some(Box::new(tree.divide_tree(0, tree.dataset.get_point_vec().len())));
        // 返回实例
        tree
    }
}

impl KDTreeBase for StaticKDTree<'_> {
    // 默认为f32
    type ElementTypeAny = f32;
    // 默认为f32
    type DistanceTypeAny = f32;
    // 默认为usize
    type IndexTypeAny = usize;
    // 默认为SuperSet
    type DataSourceAny = SuperSet; 

    /// 初始化KD树
    fn init(&mut self){
        unimplemented!()
    }

    /// 构建索引
    fn _build_index(&mut self) {
        self.root_node = Some(Box::new(self.divide_tree(0, self.dataset.get_point_vec().len())));
    }

    /// 计算边界框
    fn compute_bounding_box(&self) -> Option<Vec<Interval>> {
        let mut bbox = vec![
            Interval {
                low: f32::INFINITY,
                high: f32::NEG_INFINITY,
            };
            self.dim
        ];
        for point in &self.dataset.get_point_vec() {
            for (i, &value) in point.iter().enumerate() {
                if value < bbox[i].low {
                    bbox[i].low = value as f32;
                }
                if value > bbox[i].high {
                    bbox[i].high = value as f32;
                }
            }
        }
        // 返回边界框的vec
        Some(bbox)
    }

    /// 节点分割
    fn divide_tree(&mut self, start: usize, end: usize) -> Node {
        if end - start <= self.leaf_max_size {
            return Node {
                node_type: NodeType::Leaf { left: start, right: end },
                child1: None,
                child2: None,
            };
        }

        // 确保 dim 不超过数据点的维度
        let dim = self.dim.min(self.dataset.get_point_vec()[0].len() - 1);
        let mid = (start + end) / 2;
        self.index_acc[start..end].sort_by(|&a, &b| {
            self.dataset.get_point_vec()[a][dim]
                .partial_cmp(&self.dataset.get_point_vec()[b][dim])
                .unwrap()
        });

        let divfeat = dim as i32;
        let divlow = self.dataset.get_point_vec()[self.index_acc[start]][dim] as f32;
        let divhigh = self.dataset.get_point_vec()[self.index_acc[end - 1]][dim] as f32;

        Node {
            node_type: NodeType::NonLeaf {
                divfeat,
                divlow,
                divhigh,
            },
            child1: Some(Box::new(self.divide_tree(start, mid))),
            child2: Some(Box::new(self.divide_tree(mid, end))),
        }

    }

    /// 更新索引
    fn update_index(&mut self) {
        self._build_index();
    }

    /// 获取数据点数量
    fn get_point_count(&self) -> usize {
        self.dataset.get_point_vec().len()
    }

    /* start 搜索算法 */

    /// 1. 查找最近邻居
    fn find_neighbors(&self, query_point: &Vec<f32>, num_closest: usize, result_set: &mut SuperSet<f32, f32>) -> bool {
        // 检查输入点 vec 是否为空，若为空则返回 false
        if query_point.is_empty() {
            return false;
        }

        // 检查 KDTree 是否为空，若为空则返回 false
        if self.root_node.is_none() {
            return false;
        }

        // 初始化结果集
        let mut result_distance_vec = vec![0.0 as f32];
        let mut result_point_vec = vec![vec![0.0 as f32]];

        // 计算初始距离 dist 和距离向量 dists
        let mut min_dist = f32::MAX;
        let mut eps_error = 0.0;

        // 调用 search_level 函数，从根节点开始递归搜索最近邻
        if let Some(ref root) = self.root_node {
            self.search_level(query_point, root, min_dist, eps_error, result_set);
        }

        // 如果全部节点都搜索完成，则对结果集距离进行升序排序
        let sorted_distances = result_set.into_sorted_vec();
        result_distance_vec = sorted_distances;

        // 返回结果集是否已满（即是否找到足够的最近邻）
        result_distance_vec.len() == num_closest
    }

    // /// 2. 半径搜索
    fn radius_search(&self, query_point: &Vec<f32>, radius: f32, result_set: &mut SuperSet<f32, f32>) -> usize{
        unimplemented!()
        // let mut result_set = RadiusResultSet::new(radius);
        // self.search_level(query_point, self.root_node.as_ref().unwrap(), 0.0, self.search_params.eps, &mut result_set);
        // result_set.into_sorted_vec()
    }

    /// 3. K近邻搜索
    fn knn_search(&self, query_point: &Vec<f32>, num_closest: usize, result_set: &mut SuperSet<f32, f32>) -> usize{
        unimplemented!()
        // self.find_neighbors(query_point, num_closest)
    }

    // /// 4. 半径K近邻搜索
    fn rknn_search(&self, query_point: &Vec<f32>, num_closest: usize, radius: f32, result_set: &mut SuperSet<f32, f32>) -> usize {
        unimplemented!()
    //     let mut result_set = RKNNResultSet::new(num_closest, radius);
    //     self.search_level(query_point, self.root_node.as_ref().unwrap(), 0.0, self.search_params.eps, &mut result_set);
    //     result_set.into_sorted_vec()
    }

    /// 5. 执行从节点开始的精确搜索
    fn search_level(
        &self,
        query_point: &Vec<f32>,
        node: &Node,
        min_dist: f32,
        eps_error: f32,
        result_set: &mut SuperSet<f32, f32>,
    ) -> bool {
        match &node.node_type {
            NodeType::Leaf { left, right } => {
                // 遍历叶子节点中的所有点
                for i in *left..*right {
                    let point = &self.dataset.get_point_vec()[self.index_acc[i]];
                    let distance = l2_distance(query_point, point);
                    
                    // 如果距离小于结果集中的最坏距离，则将点加入结果集
                    if distance < result_set.worst_dist() {
                        result_set.add_point(i, distance);
                    }
                    
                    // 如果结果集已满，则停止搜索并返回 false
                    if result_set.distance_vec().len() > self.leaf_max_size {
                        return false;
                    }
                }
                return true;
            }
            NodeType::NonLeaf { divfeat, divlow, divhigh } => {
                // 计算当前节点的分割特征值
                let split_value = query_point[*divfeat as usize];
    
                // 确定先搜索哪个子节点
                let (best_child, other_child) = if split_value < *divlow {
                    (node.child1.as_ref(), node.child2.as_ref())
                } else {
                    (node.child2.as_ref(), node.child1.as_ref())
                };
    
                // 递归搜索最佳子节点
                if let Some(ref best) = best_child {
                    if !self.search_level(query_point, best, min_dist, eps_error, result_set) {
                        return false;
                    }
                }
    
                // 计算其他子节点的搜索条件
                let distance_to_split = (split_value - *divlow).abs();
                if distance_to_split < min_dist + eps_error {
                    if let Some(ref other) = other_child {
                        self.search_level(query_point, other, min_dist, eps_error, result_set);
                    }
                }
            }
        }
        true
    }

    /* end 搜索算法 */


    /* start 数据保存 */

    /// 保存索引(根节点)
    fn save_index(&self, path: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;
        if let Some(root) = &self.root_node {
            save_value(&mut file, root)?;
        }
        Ok(())
    }

    /// 加载索引(根节点)
    fn load_index(&mut self, path: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::open(path)?;
        self.root_node = Some(Box::new(load_value(&mut file)?));
        Ok(())
    }

    /// 保存整棵树(tree)
    fn save_tree(&self, path: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(path)?;
        save_value(&mut file, &self.root_bbox)?;
        save_value(&mut file, &self.index_acc)?;
        if let Some(root) = &self.root_node {
            save_value(&mut file, root)?;
        }
        Ok(())
    }
    
    // 加载整棵树(tree)
    fn load_tree(&mut self, path: &str) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::open(path)?;
        self.root_bbox = load_value(&mut file)?;
        self.index_acc = load_value(&mut file)?;
        self.root_node = Some(Box::new(load_value(&mut file)?));
        Ok(())
    }
    /* end 数据保存 */

}

#[cfg(test)]
mod tests9 {
    use super::*;
    use crate::sets::SuperSet;
    use crate::params::{KDTreeSingleIndexAdaptorParams, KDTreeSingleIndexAdaptorFlags, SearchParameters};

    #[test]
    fn test_static_kdtree_initialization() {
        // 构造SuperSet数据源
        let data_vec = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec = vec![0.0, 0.0, 0.0];
        let mut dataset = SuperSet::new(data_vec, distance_vec);

        // 建立kdtree
        let kdtree = StaticKDTree::new(&mut dataset, 3, 2);

        assert!(kdtree.root_node.is_some());
        assert_eq!(kdtree.dim, 3);
        assert_eq!(kdtree.leaf_max_size, 2);
    }

    #[test]
    fn test_save_and_load_index() {
        let data_vec = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec = vec![0.0, 0.0, 0.0];
        let mut dataset = SuperSet::new(data_vec, distance_vec);

        let mut kdtree = StaticKDTree::new(&mut dataset, 3, 2);

        // 保存索引
        let index_path = "test_index.bin";
        kdtree.save_index(index_path).expect("Failed to save index");

        // 加载索引
        let data_vec2 = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec2 = vec![0.0, 1.0, 0.0];
        let mut dataset2 = SuperSet::new(data_vec2, distance_vec2);
        let mut loaded_kdtree = StaticKDTree::new(&mut dataset2, 3, 2);
        loaded_kdtree.load_index(index_path).expect("Failed to load index");

        // 验证加载的索引
        assert!(loaded_kdtree.root_node.is_some());

        // 清理测试文件
        std::fs::remove_file(index_path).expect("Failed to remove test index file");
    }

    #[test]
    fn test_save_and_load_tree() {
        let data_vec = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec = vec![0.0, 0.0, 0.0];
        let mut dataset = SuperSet::new(data_vec, distance_vec);

        let mut kdtree = StaticKDTree::new(&mut dataset, 3, 2);

        // 保存整棵树
        let tree_path = "test_tree.bin";
        kdtree.save_tree(tree_path).expect("Failed to save tree");

        // 加载整棵树
        let data_vec2 = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let distance_vec2 = vec![0.0, 1.0, 0.0];
        let mut dataset2 = SuperSet::new(data_vec2, distance_vec2);
        let mut loaded_kdtree = StaticKDTree::new(&mut dataset2, 3, 2);
        loaded_kdtree.load_tree(tree_path).expect("Failed to load tree");

        // 验证加载的树
        assert!(loaded_kdtree.root_node.is_some());
        assert_eq!(loaded_kdtree.root_bbox.len(), kdtree.root_bbox.len());
        assert_eq!(loaded_kdtree.index_acc.len(), kdtree.index_acc.len());

        // 清理测试文件
        std::fs::remove_file(tree_path).expect("Failed to remove test tree file");
    }

}

#[cfg(test)]
mod tests12 {
    use super::*;
    use crate::sets::SuperSet;
    use crate::params::{KDTreeSingleIndexAdaptorParams, KDTreeSingleIndexAdaptorFlags, SearchParameters};

    #[test]
    fn test_search_level() {
        // 构造数据源
        let data_vec = vec![
            vec![1.0, 2.0, 3.0], // 0
            vec![4.0, 5.0, 6.0], // 1
            vec![7.0, 8.0, 9.0], // 2
            vec![10.0, 11.0, 12.0], // 3
            vec![13.0, 14.0, 15.0], // 4
        ];
        let distance_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let mut dataset = SuperSet::new(data_vec, distance_vec);

        // 建立kdtree, leaf_max_size=2
        let kdtree = StaticKDTree::new(&mut dataset, 3, 2);

        // 构造查询点
        let query_point = vec![5.0, 6.0, 7.0];
        let min_dist = 10.0;
        let eps_error = 0.1;
        let mut result_set = SuperSet::new(vec![vec![0.0 as f32]], vec![0.0 as f32]);

        // 获取根节点
        if let Some(ref root) = kdtree.root_node {
            // 调用search_level函数
            let result = kdtree.search_level(&query_point, root, min_dist, eps_error, &mut result_set);

            // 验证结果
            assert!(result);
            assert!(!result_set.get_point_vec().is_empty());
            // 验证返回的点是否在预期范围内，这里假设距离计算正确，只验证返回的点
            // 预期的点(与叶子数对应)
            let expected_indices = vec![vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]]; 
            // let actual_indices: Vec<Vec<f32>> = result_set.get_point_vec().iter().map(|p| p.clone()).collect();
            let actual_indices: Vec<Vec<f32>> = result_set.get_point_vec();
            println!("points: {:?}", actual_indices);
            for point in expected_indices {
                println!("point: {:?}", point);
                assert!(actual_indices.contains(&point));
            }
        } else {
            assert!(false); // 根节点为空，测试失败
        }
    }

    // #[test]
    // fn test_find_neighbors() {
    //     // 构造数据源
    //     let data_vec = vec![
    //         vec![1.0, 2.0, 3.0],
    //         vec![4.0, 5.0, 6.0],
    //         vec![7.0, 8.0, 9.0],
    //         vec![10.0, 11.0, 12.0],
    //         vec![13.0, 14.0, 15.0],
    //     ];
    //     let distance_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    //     let mut dataset = SuperSet::new(data_vec, distance_vec);

    //     // 建立kdtree
    //     let kdtree = StaticKDTree::new(&mut dataset, 3, 2);

    //     // 构造查询点
    //     let query_point = vec![5.0, 6.0, 7.0];
    //     let num_closest = 3;
    //     let mut result_set = SuperSet::new(vec![vec![0.0 as f32]], vec![0.0 as f32]);

    //     // 调用find_neighbors函数
    //     let result = kdtree.find_neighbors(&query_point, num_closest, &mut result_set);

    //     // 验证结果
    //     assert!(result);
    //     assert_eq!(result_set.get_point_vec().len(), num_closest);
    //     // 验证返回的最近邻点是否正确，这里假设距离计算正确，只验证返回的点的索引
    //     let expected_indices = vec![1, 2, 3]; // 预期的最近邻点索引
    //     let actual_indices: Vec<usize> = result_set.get_point_vec().iter().map(|p| p[0] as usize).collect();
    //     assert_eq!(actual_indices, expected_indices);
    // }

}

/* end 静态KD树 */


/* start 动态KD树 */
// SingleIndexKDTree
pub struct DynamicKDTree;

// 定义一个特性来添加和移除点
pub trait KDTreeOperations {
    fn add_points(&mut self, start: usize, end: usize);
    fn remove_point(&mut self, idx: usize);
}

// 为DynamicKDTree实现KDTreeOperations特性
impl KDTreeOperations for DynamicKDTree {
    fn add_points(&mut self, start: usize, end: usize) {
        // 实现添加点的逻辑
        unimplemented!()
    }

    fn remove_point(&mut self, idx: usize) {
        // 实现移除点的逻辑
        unimplemented!()
    }
}

// impl KDTree for DynamicKDTree;

/* end 动态KD树 */

