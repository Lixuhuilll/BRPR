use std::path::Path;

use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, Axis, Dim, Ix};
use ort::{inputs, GraphOptimizationLevel, Session, SessionOutputs};

pub const YOLO_V8_CLASS_LABELS: [&str; 4] =
    ["real bullet", "empty bullet", "inverter", "bullet display"];

#[derive(Debug)]
pub struct YoloV8 {
    model: Session,
    resizer: Resizer,
    resize_options: ResizeOptions,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl YoloV8 {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        // 如果不配置内部线程池且无全局线程池，默认创建一个使用全部 CPU 核心的线程池
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;

        Ok(Self {
            model,
            resizer: Resizer::new(),
            resize_options: ResizeOptions {
                algorithm: ResizeAlg::Convolution(FilterType::CatmullRom),
                mul_div_alpha: false,
                ..Default::default()
            },
        })
    }

    /// 需要存在 ONNX 线程池且并行度至少为 2 ，否则 AI 推理必定执行错误
    pub async fn run_async(
        &mut self,
        image: DynamicImage,
    ) -> anyhow::Result<Vec<Vec<(BoundingBox, f32)>>> {
        let (img_w, img_h) = (image.width(), image.height());
        let input = self.pre_processing(image)?;
        // Run YOLOv8 inference
        let outputs = self
            .model
            .run_async(inputs!["images" => input.view()]?)?
            .await?;
        Ok(Self::post_processing(outputs, img_w, img_h)?)
    }

    fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        (box1.x2.min(box2.x2) - box1.x1.max(box2.x1))
            * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
    }

    fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
            - Self::intersection(box1, box2)
    }

    fn pre_processing(&mut self, image: DynamicImage) -> anyhow::Result<Array<f32, Dim<[Ix; 4]>>> {
        let mut img = DynamicImage::new(640, 640, image.color());
        self.resizer
            .resize(&image, &mut img, &self.resize_options)?;

        let mut input = Array::zeros((1, 3, 640, 640));
        for pixel in img.pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2 .0;
            input[[0, 0, y, x]] = (r as f32) / 255.;
            input[[0, 1, y, x]] = (g as f32) / 255.;
            input[[0, 2, y, x]] = (b as f32) / 255.;
        }
        Ok(input)
    }

    fn post_processing(
        outputs: SessionOutputs,
        img_width: u32,
        img_height: u32,
    ) -> anyhow::Result<Vec<Vec<(BoundingBox, f32)>>> {
        let output = outputs["output0"]
            .try_extract_tensor::<f32>()?
            .t()
            .into_owned();

        let mut boxes = Vec::new();
        let output = output.slice(s![.., .., 0]);
        for row in output.axis_iter(Axis(0)) {
            let row: Vec<_> = row.iter().copied().collect();
            let (class_id, prob) = row
                .iter()
                // skip bounding box coordinates
                .skip(4)
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .unwrap();

            if class_id >= YOLO_V8_CLASS_LABELS.len() || prob < 0.5 {
                continue;
            }

            let xc = row[0] / 640. * (img_width as f32);
            let yc = row[1] / 640. * (img_height as f32);
            let w = row[2] / 640. * (img_width as f32);
            let h = row[3] / 640. * (img_height as f32);
            boxes.push((
                BoundingBox {
                    x1: xc - w / 2.,
                    y1: yc - h / 2.,
                    x2: xc + w / 2.,
                    y2: yc + h / 2.,
                },
                class_id,
                prob,
            ));
        }

        boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
        let mut result = vec![vec![]; YOLO_V8_CLASS_LABELS.len()];

        while !boxes.is_empty() {
            result[boxes[0].1].push((boxes[0].0, boxes[0].2));
            boxes = boxes
                .iter()
                .filter(|box1| {
                    Self::intersection(&boxes[0].0, &box1.0) / Self::union(&boxes[0].0, &box1.0)
                        < 0.7
                })
                .copied()
                .collect();
        }

        Ok(result)
    }
}
