use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use clap::{Parser, ValueEnum};
use image::DynamicImage;

use buckshot_roulette_projectile_recorder::model::yolo_v8::{Multiples, YoloV8, YoloV8Pose};

mod coco_classes;

fn main() -> anyhow::Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    match args.task {
        YoloTask::Detect => run::<YoloV8>(args)?,
        YoloTask::Pose => run::<YoloV8Pose>(args)?,
    }
    Ok(())
}

const KP_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];

trait Task: Module + Sized {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self>;
    fn report(
        pred: &Tensor,
        img: DynamicImage,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> Result<DynamicImage>;
}

impl Task for YoloV8 {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self> {
        YoloV8::load(vb, multiples, /* num_classes=*/ 80)
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
        let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
        let font = ab_glyph::FontRef::try_from_slice(&font).map_err(candle_core::Error::wrap)?;
        for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
            for b in bboxes_for_class.iter() {
                println!("{}: {:?}", coco_classes::NAMES[class_index], b);
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
                        coco_classes::NAMES[class_index],
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
}

impl Task for YoloV8Pose {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self> {
        YoloV8Pose::load(vb, multiples, /* num_classes=*/ 1, (17, 3))
    }

    fn report(
        pred: &Tensor,
        img: DynamicImage,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        _legend_size: u32,
    ) -> Result<DynamicImage> {
        let pred = pred.to_device(&Device::Cpu)?;
        let (pred_size, npreds) = pred.dims2()?;
        if pred_size != 17 * 3 + 4 + 1 {
            candle_core::bail!("unexpected pred-size {pred_size}");
        }
        let mut bboxes = vec![];
        // Extract the bounding boxes for which confidence is above the threshold.
        for index in 0..npreds {
            let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
            let confidence = pred[4];
            if confidence > confidence_threshold {
                let keypoints = (0..17)
                    .map(|i| KeyPoint {
                        x: pred[3 * i + 5],
                        y: pred[3 * i + 6],
                        mask: pred[3 * i + 7],
                    })
                    .collect::<Vec<_>>();
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: keypoints,
                };
                bboxes.push(bbox)
            }
        }

        let mut bboxes = vec![bboxes];
        non_maximum_suppression(&mut bboxes, nms_threshold);
        let bboxes = &bboxes[0];

        // Annotate the original image and print boxes information.
        let (initial_h, initial_w) = (img.height(), img.width());
        let w_ratio = initial_w as f32 / w as f32;
        let h_ratio = initial_h as f32 / h as f32;
        let mut img = img.to_rgb8();
        for b in bboxes.iter() {
            println!("{b:?}");
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
            for kp in b.data.iter() {
                if kp.mask < 0.6 {
                    continue;
                }
                let x = (kp.x * w_ratio) as i32;
                let y = (kp.y * h_ratio) as i32;
                imageproc::drawing::draw_filled_circle_mut(
                    &mut img,
                    (x, y),
                    2,
                    image::Rgb([0, 255, 0]),
                );
            }

            for &(idx1, idx2) in KP_CONNECTIONS.iter() {
                let kp1 = &b.data[idx1];
                let kp2 = &b.data[idx2];
                if kp1.mask < 0.6 || kp2.mask < 0.6 {
                    continue;
                }
                imageproc::drawing::draw_line_segment_mut(
                    &mut img,
                    (kp1.x * w_ratio, kp1.y * h_ratio),
                    (kp2.x * w_ratio, kp2.y * h_ratio),
                    image::Rgb([255, 255, 0]),
                );
            }
        }
        Ok(DynamicImage::ImageRgb8(img))
    }
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum YoloTask {
    Detect,
    Pose,
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Which {
    N,
    S,
    M,
    L,
    X,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::S)]
    which: Which,

    images: Vec<String>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.45)]
    nms_threshold: f32,

    /// The task to be run.
    #[arg(long, default_value = "detect")]
    task: YoloTask,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 14)]
    legend_size: u32,
}

impl Args {
    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/candle-yolo-v8".to_string());
                let size = match self.which {
                    Which::N => "n",
                    Which::S => "s",
                    Which::M => "m",
                    Which::L => "l",
                    Which::X => "x",
                };
                let task = match self.task {
                    YoloTask::Pose => "-pose",
                    YoloTask::Detect => "",
                };
                api.get(&format!("yolov8{size}{task}.safetensors"))?
            }
        };
        Ok(path)
    }
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

fn run<T: Task>(args: Args) -> anyhow::Result<()> {
    let device = device(args.cpu)?;
    // Create the model and load the weights from the file.
    let multiples = match args.which {
        Which::N => Multiples::n(),
        Which::S => Multiples::s(),
        Which::M => Multiples::m(),
        Which::L => Multiples::l(),
        Which::X => Multiples::x(),
    };
    let model = args.model()?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let model = T::load(vb, multiples)?;
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
        let image_t = T::report(
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
