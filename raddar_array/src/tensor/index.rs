pub struct IndexInfo {
    pub infos: Vec<IndexInfoItem>,
}

pub enum IndexInfoItem {
    Single(usize),
    Range(usize, usize, usize),
    Slice(Vec<usize>),
}
