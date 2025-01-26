# bye_nanoflann_rs

## 简介
*bye_nanoflann_rs* 是一个 **纯RUST库, 依赖rand库和num-traits库**，其用于构建不同拓扑结构的KD-树数据集：R<sup>2</sup>、R<sup>3</sup>（点云）、SO(2) 和 SO(3)（2D和3D旋转群）。不提供近似>NN<的支持。*bye_nanoflann* 只需在Cargo.toml代码中包含此库即可使用。示例在github仓库的examples文件夹。

### 代码说明：
1. **KNNResultSet** 和 **RadiusResultSet**：分别用于存储 K 近邻搜索和半径搜索的结果。
2. **IndexDistSorter**：用于对结果集进行排序。
3. **loadsave** 模块：提供了保存和加载数据的辅助函数。
4. **metric** 模块：实现了 L1、L2 和 L2 简单距离度量。
5. **KDTreeSingleIndexAdaptor**：实现了 KDTree 的主要功能，包括构建索引、搜索最近邻居等。
6. **Dataset** 和 **ResultSet**：定义了数据集和结果集的接口。
7. **Metric**：定义了距离度量的接口。

struct及fun的签名:
```rs
// crate::file
pub fn save_value<T>(stream: &mut impl Write, value: &T) -> io::Result<()>
where T: ?Sized + AsRef<[u8]>{}
pub fn save_vector<T>(stream: &mut impl Write, value: &Vec<T>) -> io::Result<()>
where T: AsRef<[u8]>{}
pub fn load_value<T>(stream: &mut impl Read, value: &mut T) -> io::Result<()>
where T: ?Sized + AsMut<[u8]>{}
pub fn load_vector<T>(stream: &mut impl Read, value: &mut Vec<T>) -> io::Result<()>
where T: Default + AsMut<[u8]>{}
// crate::memalloc
pub struct PooledAllocator {
    wordsize: usize,  // 字大小，必须大于等于8
    blocksize: usize, // 每次从系统请求的最小字节数，必须是字大小的倍数
    blocks: RefCell<Vec<Vec<u8>>>, // 存储所有分配的内存块
    used_memory: usize,   // 已使用的内存量
    wasted_memory: usize, // 浪费的内存量
}
impl PooledAllocator {
  pub fn new() -> Self {}
  pub fn free_all(&mut self) {}
  pub fn malloc(&mut self, req_size: usize) -> Vec<u8> {}
  pub fn allocate<T: Default + Clone>(&mut self, count: usize) -> Vec<T> {}
}
// crate::params
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KDTreeSingleIndexAdaptorFlags {
    None = 0,
    SkipInitialBuildIndex = 1,
}
impl std::ops::BitAnd for KDTreeSingleIndexAdaptorFlags {}
#[derive(Debug, Clone)]
pub struct KDTreeSingleIndexAdaptorParams {
    pub leaf_max_size: usize,
    pub flags: KDTreeSingleIndexAdaptorFlags,
    pub n_thread_build: usize,
}
impl KDTreeSingleIndexAdaptorParams {
    pub fn new(
        leaf_max_size: usize,
        flags: KDTreeSingleIndexAdaptorFlags,
        n_thread_build: usize,
    ) -> Self {}
}
#[derive(Debug, Clone)]
pub struct SearchParameters {
    pub eps: f32,
    pub sorted: bool,
}
impl SearchParameters {
    pub fn new(eps: f32, sorted: bool) -> Self {}
}
// crate::sets
pub struct KNNResultSet<DistanceType, IndexType = usize, CountType = usize> {
    indices: Vec<IndexType>,
    dists: Vec<DistanceType>,
    capacity: CountType,
    count: CountType,
}
impl<DistanceType, IndexType, CountType> KNNResultSet<DistanceType, IndexType, CountType>
where
    DistanceType: PartialOrd + Copy + MaxValue,
    IndexType: Copy,
    CountType: PartialOrd + Copy + From<usize> + Into<usize>,
{
    pub fn new(capacity: CountType) -> Self;
    pub fn init(&mut self, indices: Vec<IndexType>, dists: Vec<DistanceType>);
    pub fn size(&self) -> CountType;
    pub fn empty(&self) -> bool;
    pub fn full(&self) -> bool;
    pub fn add_point(&mut self, dist: DistanceType, index: IndexType) -> bool;
    pub fn worst_dist(&self) -> DistanceType;
    pub fn sort(&self);
}
pub struct RKNNResultSet<DistanceType, IndexType = usize, CountType = usize> {
    indices: Vec<IndexType>,
    dists: Vec<DistanceType>,
    capacity: CountType,
    count: CountType,
    maximum_search_distance_squared: DistanceType,
}
impl<DistanceType, IndexType, CountType> RKNNResultSet<DistanceType, IndexType, CountType>
where
    DistanceType: PartialOrd + Copy + MaxValue,
    IndexType: Copy,
    CountType: PartialOrd + Copy + From<usize> + Into<usize>,
{
    pub fn new(capacity: CountType, maximum_search_distance_squared: DistanceType) -> Self;
    pub fn init(&mut self, indices: Vec<IndexType>, dists: Vec<DistanceType>);
    pub fn size(&self) -> CountType;
    pub fn empty(&self) -> bool;
    pub fn full(&self) -> bool;
    pub fn add_point(&mut self, dist: DistanceType, index: IndexType) -> bool;
    pub fn worst_dist(&self) -> DistanceType;
    pub fn sort(&self);
}
pub struct RadiusResultSet<DistanceType, IndexType = usize> {
    radius: DistanceType,
    indices_dists: Vec<ResultItem<IndexType, DistanceType>>,
}
impl<DistanceType, IndexType> RadiusResultSet<DistanceType, IndexType>
where
    DistanceType: PartialOrd + Copy,
    IndexType: Copy,
{
    pub fn new(radius: DistanceType, indices_dists: Vec<ResultItem<IndexType, DistanceType>>) -> Self;
    pub fn init(&mut self);
    pub fn clear(&mut self);
    pub fn size(&self) -> usize;
    pub fn empty(&self) -> bool;
    pub fn full(&self) -> bool;
    pub fn add_point(&mut self, dist: DistanceType, index: IndexType) -> bool;
    pub fn worst_dist(&self) -> DistanceType;
    pub fn worst_item(&self) -> ResultItem<IndexType, DistanceType>;
    pub fn sort(&mut self);
}
pub struct ResultItem<IndexType = usize, DistanceType = f64> {
    index: IndexType,
    distance: DistanceType,
}
impl<IndexType, DistanceType> ResultItem<IndexType, DistanceType>
where
    DistanceType: Copy,
{
    pub fn new(index: IndexType, distance: DistanceType) -> Self;
    pub fn distance(&self) -> DistanceType;
}
// crate::utils
#[derive(Clone)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
#[derive(Clone)]
pub struct PointCloud<T> {
    pub pts: Vec<Point<T>>,
}
impl<T> PointCloud<T> {
    pub fn kdtree_get_point_count(&self) -> usize;
    pub fn kdtree_get_pt(&self, idx: usize, dim: usize) -> &T;
    pub fn kdtree_get_bbox<BBOX>(&self, _bb: &mut BBOX) -> bool;
}
pub fn generate_random_point_cloud_ranges<T>(pc: &mut PointCloud<T>, n: usize, max_range_x: T, max_range_y: T, max_range_z: T)
where
    T: rand::distributions::uniform::SampleUniform + Copy + Default + PartialOrd,
    rand::distributions::Standard: rand::distributions::Distribution<T>;
pub fn generate_random_point_cloud<T>(pc: &mut PointCloud<T>, n: usize, max_range: T)
where
    T: rand::distributions::uniform::SampleUniform + Copy + Default + PartialOrd,
    rand::distributions::Standard: rand::distributions::Distribution<T>;
#[derive(Clone)]
pub struct PointCloudQuat<T> {
    pub pts: Vec<PointQuat<T>>,
}
#[derive(Clone)]
pub struct PointQuat<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}
impl<T> PointCloudQuat<T> {
    pub fn kdtree_get_point_count(&self) -> usize;
    pub fn kdtree_get_pt(&self, idx: usize, dim: usize) -> &T;
    pub fn kdtree_get_bbox<BBOX>(&self, _bb: &mut BBOX) -> bool;
}
pub fn generate_random_point_cloud_quat<T>(point: &mut PointCloudQuat<T>, n: usize)
where
    T: rand::distributions::uniform::SampleUniform + Copy + Default + Mul<Output = T> + Div<Output = T> + Add<Output = T> + Sub<Output = T> + Neg<Output = T> + Debug,
    f64: Into<T>,
    rand::distributions::Standard: rand::distributions::Distribution<T>;
#[derive(Clone)]
pub struct PointCloudOrient<T> {
    pub pts: Vec<PointOrient<T>>,
}
#[derive(Clone)]
pub struct PointOrient<T> {
    pub theta: T,
}
impl<T> PointCloudOrient<T> {
    pub fn kdtree_get_point_count(&self) -> usize;
    pub fn kdtree_get_pt(&self, idx: usize, _dim: usize) -> &T;
    pub fn kdtree_get_bbox<BBOX>(&self, _bb: &mut BBOX) -> bool;
}
pub fn generate_random_point_cloud_orient<T>(point: &mut PointCloudOrient<T>, n: usize)
where
    T: rand::distributions::uniform::SampleUniform + Copy + Default,
    f64: Into<T>,
    rand::distributions::Standard: rand::distributions::Distribution<T>;
pub fn dump_mem_usage();
pub fn pi_const<T>() -> T
where
    T: From<f64>;
pub trait HasResize {
    fn resize(&mut self, n: usize);
}
impl<T> HasResize for Vec<T> {
    fn resize(&mut self, n: usize);
}
pub trait HasAssign<T> {
    fn assign(&mut self, n: usize, value: T);
}
impl<T: Clone> HasAssign<T> for Vec<T> {
    fn assign(&mut self, n: usize, value: T);
}
pub trait HasSize {
    fn size(&self) -> usize;
}
impl<T> HasSize for Vec<T> {
    fn size(&self) -> usize;
}
impl<T, const N: usize> HasSize for [T; N] {
    fn size(&self) -> usize;
}
pub fn resize<Container>(c: &mut Container, n_elements: usize)
where
    Container: HasResize;
pub fn resize_fixed<Container>(c: &mut Container, n_elements: usize)
where
    Container: Debug + HasSize;
pub fn assign<Container, T>(c: &mut Container, n_elements: usize, value: T)
where
    Container: HasAssign<T>;
pub fn assign_fixed<Container, T>(c: &mut Container, n_elements: usize, value: T)
where
    Container: Debug + IndexMut<usize, Output = T>,
    T: Copy;
pub struct IndexDistSorter;
impl IndexDistSorter {
    pub fn compare<PairType>(p1: &PairType, p2: &PairType) -> bool
    where
        PairType: HasDistance;
}
#[derive(Debug, Clone, PartialEq)]
pub struct ResultItem<IndexType = usize, DistanceType = f64> {
    pub index: IndexType,
    pub distance: DistanceType,
}
impl<IndexType, DistanceType> ResultItem<IndexType, DistanceType> {
    pub fn new(index: IndexType, distance: DistanceType) -> Self;
}
pub trait HasDistance {
    type DistanceType: PartialOrd + Clone;
    fn distance(&self) -> Self::DistanceType;
}
impl<IndexType, DistanceType: PartialOrd + Clone> HasDistance for ResultItem<IndexType, DistanceType> {
    type DistanceType = DistanceType;
    fn distance(&self) -> Self::DistanceType;
}
```

### 注意事项：
- 该代码是对 C++ 版本的翻译，保留了主要功能，但由于 Rust 和 C++ 的差异，部分实现细节可能有所不同。
- 该代码仅使用了 Rust 的标准库，没有依赖其他第三方库。
- 由于 Rust 的所有权机制，部分代码可能需要进一步优化以确保内存安全。

## 1. 关于
原库nanoflann是 [flann库](https://github.com/flann-lib/flann) 的一个 *分支*，由 Marius Muja 和 David G. Lowe 开发，作为 [MRPT](https://www.mrpt.org/) 的子项目诞生。遵循原始许可条款，*nanoflann* 在 BSD 许可下分发。请使用问题按钮报告错误或分叉并打开一个拉取请求。

引用如下：
```
@misc{blanco2014nanoflann,
  title        = {nanoflann: a {C}++ header-only fork of {FLANN}, a library for Nearest Neighbor ({NN}) with KD-trees},
  author       = {Blanco, Jose Luis and Rai, Pranjal Kumar},
  howpublished = {\url{https://github.com/jlblancoc/nanoflann}},
  year         = {2014}
}
```
查看发布 [CHANGELOG](CHANGELOG.md) 了解项目变更列表。
### 1.1. 获取代码
* 最简单的方法：克隆此 GIT 仓库并使用 `include/nanoflann.hpp` 文件。
* Debian 或 Ubuntu ([21.04 或更新版本](https://packages.ubuntu.com/source/hirsute/nanoflann)) 用户可以简单安装：
  ```bash
  $ sudo apt install libnanoflann-dev
  ```
* macOS 用户可以使用 [Homebrew](https://brew.sh) 安装 `nanoflann`：
  ```shell
  $ brew install brewsci/science/nanoflann
  ```
  或者
  ```shell
  $ brew tap brewsci/science
  $ brew install nanoflann
  ```
  MacPorts 用户可以使用：
  ```shell
  $ sudo port install nanoflann
  ```
* Linux 用户也可以使用 [Linuxbrew](https://docs.brew.sh/Homebrew-on-Linux) 安装：`brew install homebrew/science/nanoflann`
* [**稳定版本列表**](https://github.com/jlblancoc/nanoflann/releases)。查看 [CHANGELOG](https://github.com/jlblancoc/nanoflann/blob/master/CHANGELOG.md)
尽管 nanoflann 本身无需编译，但你也可以构建一些示例和测试：
```shell
$ sudo apt-get install build-essential cmake libgtest-dev libeigen3-dev
$ mkdir build && cd build && cmake ..
$ make && make test
```
### 1.2. C++ API 参考
* 浏览 [Doxygen 文档](https://jlblancoc.github.io/nanoflann/)。
* **重要提示**：如果使用 L2 范数，请注意搜索半径和所有传递和返回的距离实际上是 *平方距离*。
### 1.3. 代码示例
* 使用 `knnSearch()` 和 `radiusSearch()` 进行 KD-tree 查找：[pointcloud_kdd_radius.cpp](https://github.com/jlblancoc/nanoflann/blob/master/examples/pointcloud_kdd_radius.cpp)
* 在点云数据集上使用 KD-tree 查找：[pointcloud_example.cpp](https://github.com/jlblancoc/nanoflann/blob/master/examples
