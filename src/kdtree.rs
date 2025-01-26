#![allow(dead_code)]
#![allow(unconditional_recursion)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_variables)]

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
use std::marker::PhantomData;
use rand::distributions::{Distribution, Uniform};

// 引入内部库
use crate::file::{save_value, load_value};
use crate::memalloc::PooledAllocator;
use crate::params::{KDTreeSingleIndexAdaptorParams, SearchParameters};
use crate::sets::{KNNResultSet, RKNNResultSet, RadiusResultSet, ResultItem};
use crate::base::{
    KDTreeBase, ArrayOrVector, NodeType,
    Node, Interval, 
};

/* start 静态KD树 */
pub struct StaticKDTree;

// impl KDTree for StaticKDTree;

/* end 静态KD树 */


/* start 动态KD树 */
pub struct DynamicKDTree;

// impl KDTree for DynamicKDTree;

/* end 动态KD树 */

