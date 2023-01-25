use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard},
    vec,
};

use owning_ref::OwningHandle;

use super::{
    KindedArrayD, KindedArrayViewD, KindedArrayViewMutD, NdArrayTensor, NdArrayTensorInternal,
};

pub(crate) trait LookThrough<Input, Output, OutputFunction>
where
    OutputFunction: FnOnce(Input) -> Output,
{
    fn look_through(self, f: OutputFunction) -> Output;
}

pub(crate) trait CompositeLen<InputTarget, OutputLen> {
    fn and(self, next_target: InputTarget) -> OutputLen;
}

pub(crate) trait CompositeMutLen<InputTarget, OutputLen> {
    fn and_mut(self, next_target: InputTarget) -> OutputLen;
}

pub(crate) trait SimpleLock<T> {
    type Output<'a>: DerefMut<Target = T>
    where
        T: 'a,
        Self: 'a;
    fn lock<'a>(&'a self) -> Self::Output<'a>;
    fn new(t: T) -> Self;
}

trait SimpleReadOnlyLock<T>: SimpleLock<T> {
    type Output<'a>: Deref<Target = T>
    where
        T: 'a,
        Self: 'a;
    fn ro_lock<'a>(&'a self) -> <Self as SimpleReadOnlyLock<T>>::Output<'a>;
}

impl<T> SimpleLock<T> for Mutex<T> {
    type Output<'a> = MutexGuard<'a, T>  where T: 'a, Self: 'a;
    fn lock<'a>(&'a self) -> Self::Output<'a> {
        Mutex::lock(self).expect("The mutex is poisoned")
    }
    fn new(t: T) -> Self {
        Mutex::new(t)
    }
}

impl<T> SimpleLock<T> for RwLock<T> {
    type Output<'a> = RwLockWriteGuard<'a, T>  where T: 'a, Self: 'a;
    fn lock<'a>(&'a self) -> Self::Output<'a> {
        RwLock::write(self).expect("The rwlock is poisoned")
    }
    fn new(t: T) -> Self {
        RwLock::new(t)
    }
}

impl<T> SimpleReadOnlyLock<T> for RwLock<T> {
    type Output<'a> = RwLockReadGuard<'a, T>  where T: 'a, Self: 'a;
    fn ro_lock<'a>(&'a self) -> <Self as SimpleReadOnlyLock<T>>::Output<'a> {
        RwLock::read(self).expect("The rwlock is poisoned")
    }
}

pub(crate) struct ViewLens<ToBeMutView, ToView> {
    mut_views: ToBeMutView,
    views: Vec<ToView>,
}

impl<ToBeMutView, ToView> CompositeLen<ToView, ViewLens<ToBeMutView, ToView>>
    for ViewLens<ToBeMutView, ToView>
{
    fn and(mut self, next_target: ToView) -> ViewLens<ToBeMutView, ToView> {
        self.views.push(next_target);
        self
    }
}

impl<ToBeMutView, ToView> CompositeMutLen<ToBeMutView, ViewLens<ToBeMutView, ToView>>
    for ViewLens<(), ToView>
{
    fn and_mut(self, next_target: ToBeMutView) -> ViewLens<ToBeMutView, ToView> {
        ViewLens {
            mut_views: next_target,
            views: self.views,
        }
    }
}

fn borrow_arc_locks<ReturnValue, Inner, Lock: SimpleLock<Inner>>(
    inputs: Vec<Arc<Lock>>,
    f: impl FnOnce(Vec<&Lock::Output<'_>>) -> ReturnValue,
) -> ReturnValue {
    let mut guards = HashMap::with_capacity(inputs.len());
    let mut guard_refs = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        let duplicated = inputs
            .iter()
            .enumerate()
            .take(i)
            .find(|(_, x)| Arc::ptr_eq(x, input));
        if let Some((j, _)) = duplicated {
            guard_refs.push(j);
        } else {
            guards.insert(i, input.lock());
            guard_refs.push(i);
        }
    }
    f(guard_refs
        .into_iter()
        .map(|i| &guards[&i])
        .collect::<Vec<_>>())
}

fn borrow_arc_rwlocks<T, U: Clone, Lock: SimpleReadOnlyLock<U>>(
    input_mut: &Arc<Lock>,
    inputs: &[&Arc<Lock>],
    f: impl FnOnce(
        <Lock as SimpleLock<U>>::Output<'_>,
        Vec<<Lock as SimpleReadOnlyLock<U>>::Output<'_>>,
    ) -> T,
) -> T {
    let mut cloned_input_mut: Option<Lock> = None;
    let mut guards = Vec::with_capacity(inputs.len());
    let mut_guard = input_mut.lock();
    for input in inputs.iter() {
        if Arc::ptr_eq(input, input_mut) {
            if cloned_input_mut.is_none() {
                cloned_input_mut = Some(Lock::new(U::clone(&*mut_guard)));
            }
            guards.push(None);
        } else {
            guards.push(Some(input.ro_lock()));
        }
    }
    f(
        mut_guard,
        guards
            .into_iter()
            .map(|guard| match guard {
                Some(guard) => guard,
                None => cloned_input_mut.as_ref().unwrap().ro_lock(),
            })
            .collect::<Vec<_>>(),
    )
}

type ReadViewHandle<'a, 'b> =
    OwningHandle<RwLockReadGuard<'a, KindedArrayD>, Box<KindedArrayViewD<'b>>>;

type WriteViewHandle<'a, 'b> =
    OwningHandle<RwLockWriteGuard<'a, KindedArrayD>, Box<KindedArrayViewMutD<'b>>>;

impl ViewLens<(), ()> {
    pub(crate) fn with(
        view: Arc<Mutex<NdArrayTensorInternal>>,
    ) -> ViewLens<(), Arc<Mutex<NdArrayTensorInternal>>> {
        ViewLens {
            views: vec![view],
            mut_views: (),
        }
    }

    pub(crate) fn with_mut(
        view: Arc<Mutex<NdArrayTensorInternal>>,
    ) -> ViewLens<Arc<Mutex<NdArrayTensorInternal>>, Arc<Mutex<NdArrayTensorInternal>>> {
        ViewLens {
            views: vec![],
            mut_views: view,
        }
    }

    pub(crate) fn with_tensor(
        view: &NdArrayTensor,
    ) -> ViewLens<(), Arc<Mutex<NdArrayTensorInternal>>> {
        Self::with(view.i_copy())
    }

    pub(crate) fn with_mut_tensor(
        view: &NdArrayTensor,
    ) -> ViewLens<Arc<Mutex<NdArrayTensorInternal>>, Arc<Mutex<NdArrayTensorInternal>>> {
        Self::with_mut(view.i_copy())
    }
}

impl<ToViewMut> CompositeLen<&NdArrayTensor, ViewLens<ToViewMut, Arc<Mutex<NdArrayTensorInternal>>>>
    for ViewLens<ToViewMut, Arc<Mutex<NdArrayTensorInternal>>>
{
    fn and(self, next_target: &NdArrayTensor) -> Self {
        self.and(next_target.i_copy())
    }
}

impl
    CompositeMutLen<
        &NdArrayTensor,
        ViewLens<Arc<Mutex<NdArrayTensorInternal>>, Arc<Mutex<NdArrayTensorInternal>>>,
    > for ViewLens<(), Arc<Mutex<NdArrayTensorInternal>>>
{
    fn and_mut(
        self,
        next_target: &NdArrayTensor,
    ) -> ViewLens<Arc<Mutex<NdArrayTensorInternal>>, Arc<Mutex<NdArrayTensorInternal>>> {
        self.and_mut(next_target.i_copy())
    }
}

impl<Output, OutputFunction> LookThrough<Vec<ReadViewHandle<'_, '_>>, Output, OutputFunction>
    for ViewLens<(), Arc<Mutex<NdArrayTensorInternal>>>
where
    OutputFunction: FnOnce(Vec<ReadViewHandle<'_, '_>>) -> Output,
{
    fn look_through(self, f: OutputFunction) -> Output {
        borrow_arc_locks(self.views, |guards| {
            f(guards
                .iter()
                .map(|guard| guard.as_view())
                .collect::<Vec<_>>())
        })
    }
}

impl<Output, OutputFunction>
    LookThrough<(WriteViewHandle<'_, '_>, Vec<ReadViewHandle<'_, '_>>), Output, OutputFunction>
    for ViewLens<Arc<Mutex<NdArrayTensorInternal>>, Arc<Mutex<NdArrayTensorInternal>>>
where
    OutputFunction: FnOnce((WriteViewHandle<'_, '_>, Vec<ReadViewHandle<'_, '_>>)) -> Output,
{
    fn look_through(mut self, f: OutputFunction) -> Output {
        self.views.push(self.mut_views.clone());
        let ret = borrow_arc_locks(self.views, |mut guards| {
            let mut_guard = guards.pop().unwrap();
            let view_guards = guards.iter().map(|guard| &guard.data).collect::<Vec<_>>();
            borrow_arc_rwlocks(
                &mut_guard.data,
                view_guards.as_slice(),
                |mut_data_guard, data_guards| {
                    f((
                        mut_guard.array_as_view_mut(mut_data_guard),
                        data_guards
                            .into_iter()
                            .zip(guards.iter())
                            .map(|(data, internal)| internal.array_as_view(data))
                            .collect::<Vec<_>>(),
                    ))
                },
            )
        });
        ret
    }
}
