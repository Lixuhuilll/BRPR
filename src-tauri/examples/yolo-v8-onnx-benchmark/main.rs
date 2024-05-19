use std::time::{Duration, Instant};

use ndarray::Array;
use ort::{inputs, DirectMLExecutionProvider, GraphOptimizationLevel, Session};

const COUNT: u32 = 30;

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    use windows_sys::s;
    use windows_sys::Win32::System::LibraryLoader::SetDllDirectoryA;

    unsafe {
        SetDllDirectoryA(s!("runtimes"));
    }

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([DirectMLExecutionProvider::default().build()])?
        .commit_from_file("models/yolov8n_imgsz640_fp16.onnx")?;

    let input: Array<half::f16, _> = Array::zeros((1, 3, 640, 640));

    let mut avg = Duration::new(0, 0);
    for _ in 0..COUNT {
        let start = Instant::now();
        model.run(inputs!["images" => input.view()]?)?;
        let dura = start.elapsed();
        avg += dura / COUNT;
        println!("DirectML Time: {:?}", dura);
    }
    println!("DirectML Avg Time: {:?}", avg);

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("models/yolov8n_imgsz640_fp16.onnx")?;

    let input: Array<half::f16, _> = Array::zeros((1, 3, 640, 640));

    let mut avg = Duration::new(0, 0);
    for _ in 0..COUNT {
        let start = Instant::now();
        model.run(inputs!["images" => input.view()]?)?;
        let dura = start.elapsed();
        avg += dura / COUNT;
        println!("CPU Time: {:?}", dura);
    }
    println!("CPU Avg Time: {:?}", avg);

    Ok(())
}
