use std::sync::Arc;

use image::{DynamicImage, ImageBuffer, Pixel};
use walkdir::WalkDir;

use super::{LoadFromImageFolder, UnsupervisedDataset};

pub type DynImageDataset = UnsupervisedDataset<DynamicImage>;
pub type ImageDataset<P: Pixel, Container> = UnsupervisedDataset<ImageBuffer<P, Container>>;

impl LoadFromImageFolder for DynImageDataset {
    type ConfigType = ();

    fn from_image_folder(path: &str, _config: Self::ConfigType) -> Self {
        let mut inputs = Vec::new();
        WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .into_iter()
            .for_each(|entry| {
                let path = entry.path();
                let image = image::open(path).expect("Failed to open image");
                inputs.push(Arc::new(image));
            });
        Self::from_vectors(inputs)
    }
}

pub mod image_mappings {
    use std::{ops::Deref, sync::Arc};

    use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};
    use tch::{kind::Element, Tensor};

    use crate::dataset::{Dataset, UnsupervisedTensorDataset};

    use super::DynImageDataset;

    pub fn crop_center(
        width: u32,
        height: u32,
    ) -> impl FnMut(
        <DynImageDataset as Dataset>::SampleType,
    ) -> <DynImageDataset as Dataset>::SampleType
           + Send
           + Clone
           + 'static {
        move |input: Arc<DynamicImage>| {
            let mut input = (*input).clone();
            let (w, h) = input.dimensions();
            let (x, y) = ((w - width) / 2, (h - height) / 2);
            let cropped = input.crop(x, y, width, height);
            Arc::new(cropped)
        }
    }

    pub fn resize(
        width: u32,
        height: u32,
    ) -> impl FnMut(
        <DynImageDataset as Dataset>::SampleType,
    ) -> <DynImageDataset as Dataset>::SampleType
           + Send
           + Clone
           + 'static {
        move |input: Arc<DynamicImage>| {
            let input = (*input).clone();
            let resized = input.resize(width, height, image::imageops::FilterType::Lanczos3);
            Arc::new(resized)
        }
    }

    pub fn to_tensor<
        P: Pixel,
        Container: Deref<Target = [P::Subpixel]> + Clone,
        F: Fn(DynamicImage) -> ImageBuffer<P, Container> + Send + Clone + 'static,
    >(
        into_image_buffer: F,
    ) -> impl FnMut(
        <DynImageDataset as Dataset>::SampleType,
    ) -> <UnsupervisedTensorDataset as Dataset>::SampleType
           + Send
           + Clone
           + 'static
    where
        <P as Pixel>::Subpixel: Element,
    {
        move |input: Arc<DynamicImage>| {
            let input = (*input).clone();
            let (w, h) = input.dimensions();
            let input = into_image_buffer(input);
            let channels = P::CHANNEL_COUNT as i64;
            let tensor =
                Tensor::of_slice(&input.into_raw()).reshape(&[h as i64, w as i64, channels]);
            Arc::new(tensor)
        }
    }
}
