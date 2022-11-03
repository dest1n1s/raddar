use std::{mem::ManuallyDrop, ops::{Deref, DerefMut}};

/// A guard to execute some function before the inner object drops.
pub struct DropGuard<T>{
    // Both the inner object and the function should be wrapped in a [ManuallyDrop], to avoid dropping before the guard drops.
    inner: ManuallyDrop<T>,
    drop_cb: ManuallyDrop<Box<dyn FnMut(&mut T)>>,
}

impl<T> Deref for DropGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<T> DerefMut for DropGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}

impl<T> DropGuard<T> {
    pub fn new(inner: T, drop_cb: Box<dyn FnMut(&mut T)>) -> Self {
        Self {
            inner: ManuallyDrop::new(inner),
            drop_cb: ManuallyDrop::new(drop_cb),
        }
    }
}

impl<T> Drop for DropGuard<T> {
    fn drop(&mut self) {
        // Execute the drop callback.
        (self.drop_cb)(&mut *self.inner);

        // Drop the inner object and the callback.
        unsafe {
            ManuallyDrop::drop(&mut self.drop_cb);
            ManuallyDrop::drop(&mut self.inner);
        }
    }
}