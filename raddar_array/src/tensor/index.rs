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

impl IndexInfo {
    pub fn rest_full_for(self, shape: &[usize]) -> Self {
        let mut infos = self.infos;
        let mut rest = shape.len() - infos.len();
        while rest > 0 {
            infos.push(IndexInfoItem::Range(0, -1, 1));
            rest -= 1;
        }
        IndexInfo { infos }
    }
}