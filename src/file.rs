#![allow(unused_imports)]
use std::io::{self, Read, Write};
use rand::Rng;

/// 保存一个值到流中
pub fn save_value<T>(stream: &mut impl Write, value: &T) -> io::Result<()>
where
    T: ?Sized + AsRef<[u8]>,
{
    stream.write_all(value.as_ref())
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

/// 从流中加载一个值
pub fn load_value<T>(stream: &mut impl Read, value: &mut T) -> io::Result<()>
where
    T: ?Sized + AsMut<[u8]>,
{
    stream.read_exact(value.as_mut())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_save_load_value() {
        let mut buffer = Vec::new();
        let original_value: u32 = 123456;

        // 保存值
        save_value(&mut buffer, &original_value.to_ne_bytes()).unwrap();

        // 加载值
        let mut loaded_value = [0u8; 4];
        load_value(&mut Cursor::new(&buffer), &mut loaded_value).unwrap();
        let loaded_value = u32::from_ne_bytes(loaded_value);

        assert_eq!(original_value, loaded_value);
    }

    #[test]
    fn test_save_load_vector() {
        let mut buffer = Vec::new();
        let original_vector: Vec<u32> = vec![1, 2, 3, 4, 5];

        // 保存向量
        save_vector(&mut buffer, &original_vector.iter().map(|&x| x.to_ne_bytes()).collect()).unwrap();

        // 加载向量
        let mut loaded_vector: Vec<[u8; 4]> = Vec::new();
        load_vector(&mut Cursor::new(&buffer), &mut loaded_vector).unwrap();
        let loaded_vector: Vec<u32> = loaded_vector.iter().map(|&x| u32::from_ne_bytes(x)).collect();

        assert_eq!(original_vector, loaded_vector);
    }

    #[test]
    fn test_save_load_random_vector() {
        let mut rng = rand::thread_rng();
        let original_vector: Vec<u32> = (0..100).map(|_| rng.gen()).collect();

        let mut buffer = Vec::new();

        // 保存向量
        save_vector(&mut buffer, &original_vector.iter().map(|&x| x.to_ne_bytes()).collect()).unwrap();

        // 加载向量
        let mut loaded_vector: Vec<[u8; 4]> = Vec::new();
        load_vector(&mut Cursor::new(&buffer), &mut loaded_vector).unwrap();
        let loaded_vector: Vec<u32> = loaded_vector.iter().map(|&x| u32::from_ne_bytes(x)).collect();

        assert_eq!(original_vector, loaded_vector);
    }
}