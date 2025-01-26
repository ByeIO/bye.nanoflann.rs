fn main() {
    println!("Hello, world!");
}

// use nalgebra::{
//     Quaternion, Vector3, Matrix3, UnitQuaternion, 
//     Isometry3, Translation3, Const,
//     Matrix, Point3, ViewStorage, Rotation3,
//     Matrix3x1, VectorView3, DMatrix, DVector, SVector,
//     Vector6, Matrix6
// };

// 导入nalgebra的所有公共项, 然后将nalgebra作为子模块重新导出
// pub use nalgebra::*;
// pub mod nalgebra {
//     pub use nalgebra::*;
// }

/* start 适配至nalgebra矩阵的KD树 */
pub struct MatrixKDTree;

// impl KDTree for MatrixKDTree;

/* end 适配至nalgebra矩阵的KD树 */