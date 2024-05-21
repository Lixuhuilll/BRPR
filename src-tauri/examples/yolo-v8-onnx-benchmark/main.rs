use std::time::{Duration, Instant};

use buckshot_roulette_projectile_recorder::utils::{ort_init, tracing_subscriber_init};
use ndarray::Array;
use ort::{inputs, GraphOptimizationLevel, Session};
use tracing::info;

const COUNT: u32 = 30;

async fn run() -> anyhow::Result<()> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("models/yolov8n_imgsz640.onnx")?;

    info!("{model:?}");

    let input: Array<f32, _> = Array::zeros((1, 3, 640, 640));

    let mut avg = Duration::new(0, 0);
    for _ in 0..COUNT {
        let start = Instant::now();
        model.run_async(inputs!["images" => input.view()]?)?.await?;
        let dura = start.elapsed();
        avg += dura / COUNT;
        info!("Time: {:?}", dura);
    }
    info!("Avg Time: {:?}", avg);

    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber_init();
    ort_init()?;

    tauri::async_runtime::block_on(run())?;

    Ok(())
}
