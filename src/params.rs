#![allow(dead_code)]
#![allow(unconditional_recursion)]
#![allow(unused_imports)]

//! 这段代码实现了一个用于构建和搜索KD树（K-dimensional tree）的基本结构和参数配置。

use std::borrow::Borrow;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Deref, Mul, Neg, Sub};
use std::rc::Rc;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::vec::Vec;

/// KDTreeSingleIndexAdaptorFlags 枚举类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KDTreeSingleIndexAdaptorFlags {
    None = 0,
    SkipInitialBuildIndex = 1,
}

/// 重载 `&` 操作符
impl std::ops::BitAnd for KDTreeSingleIndexAdaptorFlags {
    type Output = u32;

    fn bitand(self, rhs: Self) -> Self::Output {
        self as u32 & rhs as u32
    }
}

/// KDTreeSingleIndexAdaptorParams 结构体
#[derive(Debug, Clone)]
pub struct KDTreeSingleIndexAdaptorParams {
    /// 叶子节点的最大大小
    pub leaf_max_size: usize,
    /// 标志位
    pub flags: KDTreeSingleIndexAdaptorFlags,
    /// 构建时使用的线程数
    pub n_thread_build: usize,
}

impl KDTreeSingleIndexAdaptorParams {
    /// 构造函数
    pub fn new(
        leaf_max_size: usize,
        flags: KDTreeSingleIndexAdaptorFlags,
        n_thread_build: usize,
    ) -> Self {
        KDTreeSingleIndexAdaptorParams {
            leaf_max_size,
            flags,
            n_thread_build,
        }
    }
}

/// SearchParameters 结构体
#[derive(Debug, Clone)]
pub struct SearchParameters {
    /// 搜索 eps-近似邻居 (默认: 0)
    pub eps: f32,
    /// 仅用于半径搜索，要求邻居按距离排序 (默认: true)
    pub sorted: bool,
}

impl SearchParameters {
    /// 构造函数
    pub fn new(eps: f32, sorted: bool) -> Self {
        SearchParameters { eps, sorted }
    }
}

#[cfg(test)]
mod tests8 {
    use super::*;

    #[test]
    fn test_kd_tree_single_index_adaptor_flags() {
        let flag1 = KDTreeSingleIndexAdaptorFlags::None;
        let flag2 = KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex;

        assert_eq!(flag1 & flag2, 0);
        assert_eq!(flag2 & flag2, 1);
    }

    #[test]
    fn test_kd_tree_single_index_adaptor_params() {
        let params = KDTreeSingleIndexAdaptorParams::new(
            10,
            KDTreeSingleIndexAdaptorFlags::None,
            1,
        );

        assert_eq!(params.leaf_max_size, 10);
        assert_eq!(params.flags, KDTreeSingleIndexAdaptorFlags::None);
        assert_eq!(params.n_thread_build, 1);
    }

    #[test]
    fn test_search_parameters() {
        let params = SearchParameters::new(0.0, true);

        assert_eq!(params.eps, 0.0);
        assert_eq!(params.sorted, true);
    }
}