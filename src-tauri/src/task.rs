use image::DynamicImage;
use ort::SessionOutputs;

pub trait IdentifyTask<'a> {
    fn identify(&self, image: DynamicImage) -> anyhow::Result<SessionOutputs<'a>>;
}
