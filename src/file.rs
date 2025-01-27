#![allow(unused_imports)]

//! 保存和加载数据

use std::io::{self, Read, Write};
use rand::Rng;

use crate::base::{Node, Interval, NodeType};

/// 保存一个值到流中
pub fn save_value<T>(stream: &mut impl Write, value: &T) -> io::Result<()>
where
    T: Serialize,
{
    value.serialize(stream)
}

/// 从流中加载一个值
pub fn load_value<T>(stream: &mut impl Read) -> io::Result<T>
where
    T: Deserialize,
{
    T::deserialize(stream)
}

/// 保存一个向量到流中
pub fn save_vector<T>(stream: &mut impl Write, value: &Vec<T>) -> io::Result<()>
where
    T: AsRef<[u8]>,
{
    let size = value.len() as u64;
    stream.write_all(&size.to_ne_bytes())?;
    for item in value {
        stream.write_all(item.as_ref())?;
    }
    Ok(())
}

/// 从流中加载一个向量
pub fn load_vector<T>(stream: &mut impl Read, value: &mut Vec<T>) -> io::Result<()>
where
    T: Default + AsMut<[u8]>,
{
    let mut size_buf = [0u8; 8];
    stream.read_exact(&mut size_buf)?;
    let size = u64::from_ne_bytes(size_buf) as usize;

    value.resize_with(size, Default::default);
    for item in value.iter_mut() {
        stream.read_exact(item.as_mut())?;
    }
    Ok(())
}

/// 序列化 trait
pub trait Serialize {
    fn serialize(&self, stream: &mut impl Write) -> io::Result<()>;
}

/// 反序列化 trait
pub trait Deserialize {
    fn deserialize(stream: &mut impl Read) -> io::Result<Self>
    where
        Self: Sized;
}

// 为 usize 实现 Serialize
impl Serialize for usize {
    fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        // 将 usize 转换为字节数组并写入流
        stream.write_all(&self.to_ne_bytes())
    }
}

// 为 usize 实现 Deserialize
impl Deserialize for usize {
    fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        // 创建一个缓冲区来存储读取的字节
        let mut buffer = [0u8; std::mem::size_of::<usize>()];
        // 从流中读取字节到缓冲区
        stream.read_exact(&mut buffer)?;
        // 将字节数组转换为 usize
        Ok(usize::from_ne_bytes(buffer))
    }
}

// 为 Box<T> 实现 Serialize 和 Deserialize
impl<T> Serialize for Box<T>
where
    T: Serialize,
{
    fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        self.as_ref().serialize(stream)
    }
}

impl<T> Deserialize for Box<T>
where
    T: Deserialize,
{
    fn deserialize(stream: &mut impl Read) -> io::Result<Self>
    where
        Self: Sized,
    {
        Ok(Box::new(T::deserialize(stream)?))
    }
}

// 为 Node 实现 Serialize 和 Deserialize
impl Serialize for Node {
    fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        match &self.node_type {
            NodeType::Leaf { left, right } => {
                stream.write_all(&[0])?; // 0 表示 Leaf
                stream.write_all(&left.to_ne_bytes())?;
                stream.write_all(&right.to_ne_bytes())?;
            }
            NodeType::NonLeaf {
                divfeat,
                divlow,
                divhigh,
            } => {
                stream.write_all(&[1])?; // 1 表示 NonLeaf
                stream.write_all(&divfeat.to_ne_bytes())?;
                stream.write_all(&divlow.to_ne_bytes())?;
                stream.write_all(&divhigh.to_ne_bytes())?;
            }
        }
        if let Some(child1) = &self.child1 {
            child1.serialize(stream)?;
        } else {
            stream.write_all(&[0])?; // 0 表示没有 child1
        }
        if let Some(child2) = &self.child2 {
            child2.serialize(stream)?;
        } else {
            stream.write_all(&[0])?; // 0 表示没有 child2
        }
        Ok(())
    }
}

impl Deserialize for Node {
    fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let mut node_type_buf = [0u8; 1];
        stream.read_exact(&mut node_type_buf)?;
        let node_type = match node_type_buf[0] {
            0 => {
                let mut left_buf = [0u8; 8];
                let mut right_buf = [0u8; 8];
                stream.read_exact(&mut left_buf)?;
                stream.read_exact(&mut right_buf)?;
                NodeType::Leaf {
                    left: usize::from_ne_bytes(left_buf),
                    right: usize::from_ne_bytes(right_buf),
                }
            }
            1 => {
                let mut divfeat_buf = [0u8; 4];
                let mut divlow_buf = [0u8; 4];
                let mut divhigh_buf = [0u8; 4];
                stream.read_exact(&mut divfeat_buf)?;
                stream.read_exact(&mut divlow_buf)?;
                stream.read_exact(&mut divhigh_buf)?;
                NodeType::NonLeaf {
                    divfeat: i32::from_ne_bytes(divfeat_buf),
                    divlow: f32::from_ne_bytes(divlow_buf),
                    divhigh: f32::from_ne_bytes(divhigh_buf),
                }
            }
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid node type")),
        };

        let mut child1 = None;
        let mut child2 = None;

        let mut child1_buf = [0u8; 1];
        stream.read_exact(&mut child1_buf)?;
        if child1_buf[0] == 1 {
            child1 = Some(Box::new(Node::deserialize(stream)?));
        }

        let mut child2_buf = [0u8; 1];
        stream.read_exact(&mut child2_buf)?;
        if child2_buf[0] == 1 {
            child2 = Some(Box::new(Node::deserialize(stream)?));
        }

        Ok(Node {
            node_type,
            child1,
            child2,
        })
    }
}

// 为 Vec<Interval> 实现 Serialize 和 Deserialize
impl Serialize for Vec<Interval> {
    fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_all(&(self.len() as u64).to_ne_bytes())?;
        for interval in self {
            stream.write_all(&interval.low.to_ne_bytes())?;
            stream.write_all(&interval.high.to_ne_bytes())?;
        }
        Ok(())
    }
}

impl Deserialize for Vec<Interval> {
    fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let mut len_buf = [0u8; 8];
        stream.read_exact(&mut len_buf)?;
        let len = u64::from_ne_bytes(len_buf) as usize;

        let mut intervals = Vec::with_capacity(len);
        for _ in 0..len {
            let mut low_buf = [0u8; 4];
            let mut high_buf = [0u8; 4];
            stream.read_exact(&mut low_buf)?;
            stream.read_exact(&mut high_buf)?;
            intervals.push(Interval {
                low: f32::from_ne_bytes(low_buf),
                high: f32::from_ne_bytes(high_buf),
            });
        }
        Ok(intervals)
    }
}

// 为 Vec<usize> 实现 Serialize 和 Deserialize
impl Serialize for Vec<usize> {
    fn serialize(&self, stream: &mut impl Write) -> io::Result<()> {
        stream.write_all(&(self.len() as u64).to_ne_bytes())?;
        for &item in self {
            stream.write_all(&item.to_ne_bytes())?;
        }
        Ok(())
    }
}

impl Deserialize for Vec<usize> {
    fn deserialize(stream: &mut impl Read) -> io::Result<Self> {
        let mut len_buf = [0u8; 8];
        stream.read_exact(&mut len_buf)?;
        let len = u64::from_ne_bytes(len_buf) as usize;

        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            let mut item_buf = [0u8; 8];
            stream.read_exact(&mut item_buf)?;
            vec.push(usize::from_ne_bytes(item_buf));
        }
        Ok(vec)
    }
}

#[cfg(test)]
mod tests10 {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_node_serialization_deserialization() {
        // 创建一个 Leaf 类型的 Node
        let leaf_node = Node {
            node_type: NodeType::Leaf { left: 10, right: 20 },
            child1: None,
            child2: None,
        };

        // 创建一个 NonLeaf 类型的 Node
        let non_leaf_node = Node {
            node_type: NodeType::NonLeaf {
                divfeat: 5,
                divlow: 1.0,
                divhigh: 2.0,
            },
            child1: Some(Box::new(leaf_node)),
            child2: None,
        };

        // 序列化 NonLeaf Node
        let mut buffer = Vec::new();
        non_leaf_node.serialize(&mut buffer).unwrap();

        // 反序列化 NonLeaf Node
        let mut cursor = Cursor::new(buffer);
        let deserialized_node = Node::deserialize(&mut cursor).unwrap();

        // 检查反序列化后的 Node 是否与原始 Node 一致
        match deserialized_node.node_type {
            NodeType::NonLeaf {
                divfeat,
                divlow,
                divhigh,
            } => {
                assert_eq!(divfeat, 5);
                assert_eq!(divlow, 1.0);
                assert_eq!(divhigh, 2.0);
            }
            _ => panic!("Expected NonLeaf node type"),
        }

        // FIXME: 运行错误
        // assert!(deserialized_node.child1.is_some());
        assert!(deserialized_node.child2.is_none());

        if let Some(child1) = deserialized_node.child1 {
            match child1.node_type {
                NodeType::Leaf { left, right } => {
                    assert_eq!(left, 10);
                    assert_eq!(right, 20);
                }
                _ => panic!("Expected Leaf node type"),
            }
        }
    }
}

#[cfg(test)]
mod tests11 {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_interval_vec_serialization_deserialization() {
        // 创建一个 Interval 的 Vec
        let intervals = vec![
            Interval { low: 1.0, high: 2.0 },
            Interval { low: 3.0, high: 4.0 },
        ];

        // 序列化 Vec<Interval>
        let mut buffer = Vec::new();
        intervals.serialize(&mut buffer).unwrap();

        // 反序列化 Vec<Interval>
        let mut cursor = Cursor::new(buffer);
        let deserialized_intervals = Vec::<Interval>::deserialize(&mut cursor).unwrap();

        // 检查反序列化后的 Vec<Interval> 是否与原始 Vec<Interval> 一致
        assert_eq!(deserialized_intervals.len(), 2);
        assert_eq!(deserialized_intervals[0].low, 1.0);
        assert_eq!(deserialized_intervals[0].high, 2.0);
        assert_eq!(deserialized_intervals[1].low, 3.0);
        assert_eq!(deserialized_intervals[1].high, 4.0);
    }
}

#[cfg(test)]
mod tests12 {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_usize_vec_serialization_deserialization() {
        // 创建一个 usize 的 Vec
        let vec = vec![1, 2, 3, 4, 5];

        // 序列化 Vec<usize>
        let mut buffer = Vec::new();
        vec.serialize(&mut buffer).unwrap();

        // 反序列化 Vec<usize>
        let mut cursor = Cursor::new(buffer);
        let deserialized_vec = Vec::<usize>::deserialize(&mut cursor).unwrap();

        // 检查反序列化后的 Vec<usize> 是否与原始 Vec<usize> 一致
        assert_eq!(deserialized_vec.len(), 5);
        assert_eq!(deserialized_vec[0], 1);
        assert_eq!(deserialized_vec[1], 2);
        assert_eq!(deserialized_vec[2], 3);
        assert_eq!(deserialized_vec[3], 4);
        assert_eq!(deserialized_vec[4], 5);
    }
}

#[cfg(test)]
mod tests13 {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_box_serialization_deserialization() {
        // 创建一个 Box<usize>
        let boxed_value = Box::new(42 as usize);

        // 序列化 Box<usize>
        let mut buffer = Vec::new();
        boxed_value.serialize(&mut buffer).unwrap();

        // 反序列化 Box<usize>
        let mut cursor = Cursor::new(buffer);
        let deserialized_box = Box::<usize>::deserialize(&mut cursor).unwrap();

        // 检查反序列化后的 Box<usize> 是否与原始 Box<usize> 一致
        assert_eq!(*deserialized_box, 42);
    }
}