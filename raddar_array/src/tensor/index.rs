#[derive(Clone)]
pub struct IndexInfo {
    pub infos: Vec<IndexInfoItem>,
}

#[derive(Clone)]
pub enum IndexInfoItem {
    Single(isize),
    Range(isize, isize, isize),
    NewAxis,
}
