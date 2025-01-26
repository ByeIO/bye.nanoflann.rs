#![allow(dead_code)]
#![allow(unconditional_recursion)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_variables)]

/// 内存分配模块
/// 使用 Vec 来自动管理内存块, 避免手动分配内存

use std::rc::Rc;
use std::cell::RefCell;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::vec::Vec;

/// 池化分配器
///
/// 该结构体维护了一个内存池，用于高效地分配小块内存。
pub struct PooledAllocator {
    wordsize: usize,  // 字大小，必须大于等于8
    blocksize: usize, // 每次从系统请求的最小字节数，必须是字大小的倍数
    blocks: RefCell<Vec<Vec<u8>>>, // 存储所有分配的内存块
    used_memory: usize,   // 已使用的内存量
    wasted_memory: usize, // 浪费的内存量
}

impl PooledAllocator {
    /// 创建一个新的池化分配器
    pub fn new() -> Self {
        PooledAllocator {
            wordsize: 16,  // 字大小，必须大于等于8
            blocksize: 8192, // 每次从系统请求的最小字节数
            blocks: RefCell::new(Vec::new()),
            used_memory: 0,
            wasted_memory: 0,
        }
    }

    /// 释放所有分配的内存块
    pub fn free_all(&mut self) {
        self.blocks.borrow_mut().clear();
        self.used_memory = 0;
        self.wasted_memory = 0;
    }

    /// 分配指定大小的内存
    ///
    /// # 参数
    /// - `req_size`: 请求的内存大小（字节）
    ///
    /// # 返回
    /// - 返回一个分配的内存块
    pub fn malloc(&mut self, req_size: usize) -> Vec<u8> {
        let size = (req_size + (self.wordsize - 1)) & !(self.wordsize - 1);

        let mut blocks = self.blocks.borrow_mut();

        // 检查最后一个块是否有足够的空间
        if let Some(block) = blocks.last_mut() {
            if block.len() + size <= self.blocksize {
                let start = block.len();
                block.extend(vec![0; size]);
                self.used_memory += size;
                return block[start..].to_vec();
            } else {
                self.wasted_memory += self.blocksize - block.len();
            }
        }

        // 如果没有足够的空间，创建一个新块
        let mut new_block = Vec::with_capacity(self.blocksize);
        new_block.extend(vec![0; size]);
        blocks.push(new_block);
        self.used_memory += size;

        // 返回新块的引用
        blocks.last().unwrap().clone()
    }

    /// 分配指定数量的类型 `T` 的内存
    ///
    /// # 参数
    /// - `count`: 要分配的实例数量
    ///
    /// # 返回
    /// - 返回一个指向分配内存的指针
    pub fn allocate<T: Default + Clone>(&mut self, count: usize) -> Vec<T> {
        let size = std::mem::size_of::<T>() * count;
        let mem = self.malloc(size);
        let mut vec = Vec::with_capacity(count);
        vec.extend(std::iter::repeat(T::default()).take(count));
        vec
    }
}

#[cfg(test)]
mod tests7 {
    use super::*;

    #[test]
    fn test_pooled_allocator() {
        let mut allocator = PooledAllocator::new();

        // 分配一个u32数组
        let vec_u32: Vec<u32> = allocator.allocate(10);
        assert_eq!(vec_u32.len(), 10);

        // 分配一个u8数组
        let req_size = 100;
        let expected_size = (req_size + 15) & !15; // 对齐后的预期大小
        let vec_u8: Vec<u8> = allocator.malloc(req_size);
        assert_eq!(vec_u8.len(), expected_size); // 使用对齐后的预期值

        // 释放所有内存
        allocator.free_all();
        assert_eq!(allocator.used_memory, 0);
        assert_eq!(allocator.wasted_memory, 0);
    }

    #[test]
    fn test_large_allocation() {
        let mut allocator = PooledAllocator::new();

        // 分配一个大于BLOCKSIZE的内存块
        let req_size = 9000;
        let expected_size = (req_size + 15) & !15; // 对齐后的预期大小
        let large_vec: Vec<u8> = allocator.malloc(req_size);
        assert_eq!(large_vec.len(), expected_size); // 使用对齐后的预期值

        // 释放所有内存
        allocator.free_all();
        assert_eq!(allocator.used_memory, 0);
        assert_eq!(allocator.wasted_memory, 0);
    }
}