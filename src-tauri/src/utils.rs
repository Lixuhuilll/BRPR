use anyhow::anyhow;
use image::DynamicImage;
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
