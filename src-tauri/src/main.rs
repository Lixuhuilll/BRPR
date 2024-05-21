// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use tauri::Manager;
use tracing::error;
use xcap::Window;

use buckshot_roulette_projectile_recorder::utils::{
    find_br_window, ort_init, screenshot, tracing_subscriber_init,
};
use buckshot_roulette_projectile_recorder::yolo_v8::{BoundingBox, YoloV8};

#[derive(Debug, Default)]
struct AIState {
    enabled: bool,
    first_enabled: bool,
    window: Option<Window>,
}

impl AIState {
    fn find_and_set_window(&mut self) -> Option<&Window> {
        if let Ok(window) = find_br_window() {
            self.window = Some(window);
        }
        self.window.as_ref()
    }
}

// 状态码不为 0 则自动通知前端
struct AutoEmit<'a>(u8, &'a tauri::AppHandle);
impl Drop for AutoEmit<'_> {
    fn drop(&mut self) {
        let event = match self.0 {
            1 => "screenshot-failed",
            2 => "identify-failed",
            3 => "model-load-failed",
            _ => {
                return;
            }
        };
        if self.1.emit_all(event, ()).is_err() {
            error!("Emit failed.");
        }
    }
}

async fn ai_background_task(app: tauri::AppHandle) {
    let ai_state = app.state::<Mutex<AIState>>();

    let mut intv = tokio::time::interval(tokio::time::Duration::from_millis(300));
    intv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut max_bullet = [0; 2];
    let mut model = None;

    loop {
        // 保持间隔执行
        intv.tick().await;

        // 初始认为模型载入异常
        let mut auto_emit = AutoEmit(3, &app);

        let image = {
            let mut ai_state = ai_state.lock().unwrap();

            // AI 功能未启用不继续执行
            if !ai_state.enabled {
                continue;
            }

            // 载入模型
            if model.is_none() {
                match app
                    .path_resolver()
                    .resolve_resource("models/yolov8n_imgsz640.onnx")
                {
                    Some(model_path) => match YoloV8::load(&model_path) {
                        Ok(m) => model = Some(m),
                        Err(msg) => {
                            // 模型载入失败
                            error!("{msg}");
                            ai_state.enabled = false;
                            continue;
                        }
                    },
                    None => {
                        error!("Failed to resolve model resource.");
                        ai_state.enabled = false;
                        continue;
                    }
                }
            }

            // 模型载入完毕，认为截图异常
            auto_emit.0 = 1;

            // 首次执行时获取窗口句柄
            let window = match ai_state.window.as_ref() {
                Some(window) => window,
                None => match ai_state.find_and_set_window() {
                    Some(window) => window,
                    None => continue,
                },
            };
            match screenshot(window) {
                Ok(image) => image,
                Err(_) => {
                    // 截图失败可能是窗口句柄失效，重新获取窗口并截图
                    let window = match ai_state.find_and_set_window() {
                        Some(window) => window,
                        None => continue,
                    };
                    match screenshot(window) {
                        Ok(image) => image,
                        Err(_) => continue,
                    }
                }
            }
        };

        // 截图完毕，认为识别异常
        auto_emit.0 = 2;

        let model = model.as_ref().unwrap();
        let result = model.run_async(image).await;

        if result.is_err() {
            error!("{}", result.unwrap_err());
            continue;
        }

        if let [reals, empties, .., display] = result.unwrap().as_mut_slice() {
            // 识别成功，认为无错误
            auto_emit.0 = 0;
            // 有且只有一个弹药展示区域
            if display.len() != 1 {
                continue;
            }
            let display = &display[0].0;
            let check_bullet = |arg: &(BoundingBox, f32)| {
                let bbox = &arg.0;
                let mid_x = bbox.x1 / 2. + bbox.x2 / 2.;
                let mid_y = bbox.y1 / 2. + bbox.y2 / 2.;
                if mid_x >= display.x1
                    && mid_x <= display.x2
                    && mid_y >= display.y1
                    && mid_y <= display.y2
                {
                    true
                } else {
                    false
                }
            };
            reals.retain(check_bullet);
            empties.retain(check_bullet);
            // Bullet must greater than or equal to 2.
            if reals.len() + empties.len() < 2 {
                max_bullet.fill(0);
                continue;
            }
            // 减少误判（子弹展示到退出展示这个过程中，子弹数量呈现 少 -> 全部 -> 少 的变化）
            if max_bullet[0] >= reals.len() && max_bullet[1] >= empties.len() {
                continue;
            }
            max_bullet[0] = std::cmp::max(max_bullet[0], reals.len());
            max_bullet[1] = std::cmp::max(max_bullet[1], empties.len());
            if app.emit_all("bullet-filling", max_bullet).is_err() {
                error!("Emit failed.");
            }
        }
    }
}

#[tauri::command]
async fn set_ai_enabled(enabled: bool, app: tauri::AppHandle) -> Result<(), String> {
    let ai_state = app.state::<Mutex<AIState>>();
    let mut ai_state = ai_state.lock().unwrap();

    ai_state.enabled = enabled;
    if enabled && ai_state.first_enabled {
        ai_state.first_enabled = false;
        // 提前销毁 ai_state 以便于转移 app 的所有权
        drop(ai_state);
        // 首次启用需要启动后台任务
        tauri::async_runtime::spawn(ai_background_task(app));
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber_init();
    ort_init()?;

    let ai_state = Mutex::new(AIState {
        first_enabled: true,
        ..Default::default()
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .manage(ai_state)
        .invoke_handler(tauri::generate_handler![set_ai_enabled])
        .run(tauri::generate_context!())?;

    Ok(())
}
