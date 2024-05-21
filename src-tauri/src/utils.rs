use anyhow::anyhow;
use image::DynamicImage;
use ort::EnvironmentGlobalThreadPoolOptions;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;
use xcap::Window;

pub fn find_br_window() -> anyhow::Result<Window> {
    let windows = Window::all()?;
    for window in windows {
        if window.title() == "Buckshot Roulette" {
            return Ok(window);
        }
    }
    Err(anyhow!("Unable to find Buckshot Roulette's window."))
}

pub fn screenshot(window: &Window) -> anyhow::Result<DynamicImage> {
    if window.is_minimized() {
        Err(anyhow!("The window is minimized."))
    } else {
        let image = DynamicImage::ImageRgba8(window.capture_image()?);
        Ok(image)
    }
}

pub fn tracing_subscriber_init() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .init();
}

pub fn ort_init() -> ort::Result<()> {
    // 配置全局线程池
    ort::init()
        .with_global_thread_pool(EnvironmentGlobalThreadPoolOptions {
            // intra 有两个特殊值：
            // 0 = 使用默认线程数（CPU 的全部物理核心）
            // 1 = 使用调用 run 函数的线程，不会在线程池中创建任何线程
            // 因此想配合 run_async 使用至少应该配置为 2
            intra_op_parallelism: Some(2),
            // 禁止在任务队列为空时空转（禁用后推理速度可能变慢，但 CPU 占用会降低）
            spin_control: Some(false),
            ..Default::default()
        })
        .commit()
}
