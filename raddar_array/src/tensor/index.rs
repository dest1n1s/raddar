#[derive(Clone, Debug)]
pub struct IndexInfo {
    pub infos: Vec<IndexInfoItem>,
}

#[derive(Clone, Debug)]
pub enum IndexInfoItem {
    Single(isize),
    Range(isize, Option<isize>, isize),
    NewAxis,
}

pub const ALL: IndexInfoItem = IndexInfoItem::Range(0, None, 1);

impl IndexInfo {
    pub fn rest_full_for(self, shape: &[usize]) -> Self {
        let mut infos = self.infos;
        let mut rest = shape.len() - infos.len();
        while rest > 0 {
            infos.push(ALL);
            rest -= 1;
        }
        IndexInfo { infos }
    }
}

impl From<Vec<IndexInfoItem>> for IndexInfo {
    fn from(infos: Vec<IndexInfoItem>) -> Self {
        IndexInfo { infos }
    }
}