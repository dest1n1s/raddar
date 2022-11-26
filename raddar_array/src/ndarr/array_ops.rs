use ndarray::SliceInfoElem;

use crate::tensor::{
    index::{IndexInfo, IndexInfoItem},
    ops::Operation,
};

use super::{AsView, KindedArrayViewD, KindedArrayViewMutD, ViewMethods, ViewMutMethods};

impl From<IndexInfoItem> for SliceInfoElem {
    fn from(item: IndexInfoItem) -> Self {
        match item {
            IndexInfoItem::Single(i) => SliceInfoElem::Index(i),
            IndexInfoItem::Range(start, end, step) => SliceInfoElem::Slice {
                start: start,
                end: Some(end),
                step: step,
            },
            IndexInfoItem::NewAxis => SliceInfoElem::NewAxis,
        }
    }
}

impl From<IndexInfo> for Vec<SliceInfoElem> {
    fn from(info: IndexInfo) -> Self {
        info.infos.into_iter().map(|item| item.into()).collect()
    }
}

pub(crate) struct SliceView {
    slice: IndexInfo,
}

impl SliceView {
    pub fn new(slice: IndexInfo) -> Self {
        Self { slice }
    }
}

impl AsView for SliceView {
    fn op(&self) -> std::sync::Arc<dyn Operation> {
        todo!()
    }

    fn view<'a>(&self, tensor: KindedArrayViewD<'a>) -> KindedArrayViewD<'a> {
        tensor.into_slice(self.slice.clone())
    }

    fn view_mut<'a>(&self, tensor: KindedArrayViewMutD<'a>) -> KindedArrayViewMutD<'a> {
        tensor.into_slice_mut(self.slice.clone())
    }
}
