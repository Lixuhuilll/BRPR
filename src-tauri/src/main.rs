// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use candle_transformers::object_detection::{Bbox, KeyPoint};
use tauri::Manager;
use tracing::error;
use xcap::Window;

use buckshot_roulette_projectile_recorder::model::yolo_v8::YoloV8;
use buckshot_roulette_projectile_recorder::screenshot::{find_br_window, screenshot};
use buckshot_roulette_projectile_recorder::task::{IdentifyModel, IdentifyTask};

struct AIState {
    enabled: bool,
    identify_task: Option<Box<dyn IdentifyTask + Send>>,
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

async fn ai_background_task(app: tauri::AppHandle) {
    let ai_state = app.state::<Mutex<AIState>>();
    let mut intv = tokio::time::interval(tokio::time::Duration::from_millis(1000));
    let mut max_bullet = [0; 2];

    loop {
        // 保持间隔执行
        intv.tick().await;
        let mut ai_state = ai_state.lock().unwrap();
        // AI 功能未启用或者识别任务未加载时，不继续执行
        if !ai_state.enabled || ai_state.identify_task.is_none() {
            continue;
        }

        // 状态码不为 0 则自动通知前端
        struct AutoEmit<'a>(u8, &'a tauri::AppHandle);
        impl Drop for AutoEmit<'_> {
            fn drop(&mut self) {
                let event = match self.0 {
                    1 => "screenshot-failed",
                    2 => "identify-failed",
                    _ => {
                        return;
                    }
                };
                if self.1.emit_all(event, ()).is_err() {
                    error!("Emit failed.");
                }
            }
        }
        // 初始认为截图异常
        let mut auto_emit = AutoEmit(1, &app);

        // 首次执行时获取窗口句柄
        let window = match ai_state.window.as_ref() {
            Some(window) => window,
            None => match ai_state.find_and_set_window() {
                Some(window) => window,
                None => continue,
            },
        };
        let image = match screenshot(window) {
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
        };
        // 截图完毕，认为识别异常
        auto_emit.0 = 2;

        let identify_task = ai_state.identify_task.as_ref().unwrap();
        let result = identify_task.identify(image);

        if result.is_err() {
            continue;
        }

        if let [reals, empties, .., display] = result.unwrap().as_mut_slice() {
            // 有且只有一个弹药展示区域
            if display.len() != 1 {
                continue;
            }
            let display = &display[0];
            let check_bullet = |bbox: &Bbox<Vec<KeyPoint>>| {
                let mid_x = bbox.xmin / 2. + bbox.xmax / 2.;
                let mid_y = bbox.ymin / 2. + bbox.ymax / 2.;
                if mid_x >= display.xmin
                    && mid_x <= display.xmax
                    && mid_y >= display.ymin
                    && mid_y <= display.ymax
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
                max_bullet.fill_with(|| 0);
                continue;
            }
            // 识别成功，认为无错误
            auto_emit.0 = 0;
            // 减少误判（子弹展示到退出展示这个过程中，子弹数量呈现 少 -> 全部 -> 少 的变化）
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
    if enabled && ai_state.identify_task.is_none() {
        // 首次启用需要载入模型
        let model_path = app
            .path_resolver()
            .resolve_resource("model/yolov8n_imgsz640.safetensors")
            .ok_or("Failed to resolve model resource.")?;
        match IdentifyModel::<YoloV8>::load(&model_path).map_err(|err| err.to_string()) {
            Ok(model) => ai_state.identify_task = Some(Box::new(model)),
            Err(msg) => {
                // 模型载入失败
                ai_state.enabled = false;
                return Err(msg);
            }
        }
        // 提前销毁 ai_state 以便于转移 app 的所有权
        drop(ai_state);
        // 首次启用需要启动后台任务
        tauri::async_runtime::spawn(ai_background_task(app));
    }
    Ok(())
}

fn main() {
    let ai_state = Mutex::new(AIState {
        enabled: false,
        identify_task: None,
        window: None,
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .manage(ai_state)
        .invoke_handler(tauri::generate_handler![set_ai_enabled])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
