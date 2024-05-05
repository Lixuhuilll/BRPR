use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use clap::Parser;
use image::DynamicImage;

use buckshot_roulette_projectile_recorder::br_classes;
use buckshot_roulette_projectile_recorder::model::yolo_v8::{Multiples, YoloV8};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    // Create the model and load the weights from the file.
    let multiples = Multiples::n();
    let model = Path::new("model/yolov8n_imgsz640.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let model = YoloV8::load(vb, multiples, br_classes::NAMES.len())?;
    println!("model loaded");
    for image_name in args.images.iter() {
        println!("processing {image_name}");
        let mut image_name = std::path::PathBuf::from(image_name);
        let original_image = image::io::Reader::open(&image_name)?
            .decode()
            .map_err(candle_core::Error::wrap)?;
        let (width, height) = {
            let w = original_image.width() as usize;
            let h = original_image.height() as usize;
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
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &device,
            )?
            .permute((2, 0, 1))?
        };
        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = model.forward(&image_t)?.squeeze(0)?;
        println!("generated predictions {predictions:?}");
        let image_t = report(
            &predictions,
            original_image,
            width,
            height,
            args.confidence_threshold,
            args.nms_threshold,
            args.legend_size,
        )?;
        image_name.set_extension("pp.jpg");
        println!("writing {image_name:?}");
        image_t.save(image_name)?
    }
    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    images: Vec<String>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.45)]
    nms_threshold: f32,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 14)]
    legend_size: u32,
}

fn report(
    pred: &Tensor,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
) -> Result<DynamicImage> {
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

    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("../roboto-mono-stripped.ttf") as &[u8]);
    let font = ab_glyph::FontRef::try_from_slice(&font).map_err(candle_core::Error::wrap)?;
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", br_classes::NAMES[class_index], b);
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let dx = (b.xmax - b.xmin) * w_ratio;
            let dy = (b.ymax - b.ymin) * h_ratio;
            if dx >= 0. && dy >= 0. {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                    image::Rgb([255, 0, 0]),
                );
            }
            if legend_size > 0 {
                imageproc::drawing::draw_filled_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, legend_size),
                    image::Rgb([170, 0, 0]),
                );
                let legend = format!(
                    "{}   {:.0}%",
                    br_classes::NAMES[class_index],
                    100. * b.confidence
                );
                imageproc::drawing::draw_text_mut(
                    &mut img,
                    image::Rgb([255, 255, 255]),
                    xmin,
                    ymin,
                    ab_glyph::PxScale {
                        x: legend_size as f32 - 1.,
                        y: legend_size as f32 - 1.,
                    },
                    &font,
                    &legend,
                )
            }
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}
