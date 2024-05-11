use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use image::DynamicImage;

use crate::br_classes;
use crate::model::yolo_v8::{Multiples, YoloV8};

pub type BBoxes = Vec<Vec<Bbox<Vec<KeyPoint>>>>;

pub trait IdentifyTask {
    fn identify(&self, image: DynamicImage) -> anyhow::Result<BBoxes>;
}

pub struct IdentifyModel<T: Module> {
    model: T,
    device: Device,
    d_type: DType,
}

impl IdentifyModel<YoloV8> {
    pub fn load(path: &Path) -> anyhow::Result<IdentifyModel<YoloV8>> {
        let device = Device::Cpu;
        let d_type = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], d_type, &device)? };
        let model = YoloV8::load(vb, Multiples::n(), br_classes::NAMES.len())?;
        Ok(IdentifyModel {
            model,
            device,
            d_type,
        })
    }
}

impl IdentifyTask for IdentifyModel<YoloV8> {
    fn identify(&self, image: DynamicImage) -> anyhow::Result<BBoxes> {
        let pred = predictions(self, &image)?;
        yolo_v8_report(&pred, 0.25, 0.45)
    }
}

fn predictions<T: Module>(
    i_model: &IdentifyModel<T>,
    image: &DynamicImage,
) -> anyhow::Result<Tensor> {
    let (width, height) = {
        let w = image.width() as usize;
        let h = image.height() as usize;
        if w < h {
            let w = w * 640 / h;
            // Sizes have to be divisible by 32.
            (w / 32 * 32, 640)
        } else {
            let h = h * 640 / w;
            (640, h / 32 * 32)
        }
    };
    let image_t = {
        let image = image.resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let data = image.to_rgb8().into_raw();
        Tensor::from_vec(
            data,
            (image.height() as usize, image.width() as usize, 3),
            &i_model.device,
        )?
        .permute((2, 0, 1))?
    };
    let image_t = (image_t.unsqueeze(0)?.to_dtype(i_model.d_type)? * (1. / 255.))?;
    Ok(i_model.model.forward(&image_t)?.squeeze(0)?)
}

fn yolo_v8_report(
    pred: &Tensor,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> anyhow::Result<BBoxes> {
    // 将张量结果提取到内存
    let pred = pred.to_device(&Device::Cpu)?;
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);

    Ok(bboxes)
}
