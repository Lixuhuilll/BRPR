// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use std::time::Instant;

use tauri::Manager;
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
    let mut intv = tokio::time::interval(tokio::time::Duration::from_millis(500));

    loop {
        // 保持间隔执行
        intv.tick().await;
        println!("等待周期结束");
        let mut ai_state = ai_state.lock().unwrap();
        // AI 功能未启用或者识别任务未加载时，不继续执行
        if !ai_state.enabled || ai_state.identify_task.is_none() {
            continue;
        }
        let start = Instant::now();
        // 首次执行时获取窗口句柄
        let window = match ai_state.window.as_ref() {
            Some(window) => window,
            None => match ai_state.find_and_set_window() {
                Some(window) => window,
                None => {
                    todo!("当获取窗口失败时应该向前端发出事件以提示前端");
                    continue;
                }
            },
        };
        let image = match screenshot(window) {
            Ok(image) => image,
            Err(_) => {
                // 截图失败可能是窗口句柄失效，重新获取窗口并截图
                let window = match ai_state.find_and_set_window() {
                    Some(window) => window,
                    None => {
                        todo!("当获取窗口失败时应该向前端发出事件以提示前端");
                        continue;
                    }
                };
                match screenshot(window) {
                    Ok(image) => image,
                    Err(_) => {
                        todo!("当获取截图失败时应该向前端发出事件以提示前端");
                        continue;
                    }
                }
            }
        };
        println!("截图用时：{:?}", start.elapsed());
        let start = Instant::now();
        let identify_task = ai_state.identify_task.as_ref().unwrap();
        match identify_task.identify(image) {
            Ok(result) => {
                println!("AI 识别结果：{:?}", result);
            }
            Err(_) => todo!("当获取 AI 识别失败时应该向前端发出事件以提示前端"),
        }
        println!("AI 识别用时：{:?}", start.elapsed());
    }
}

#[tauri::command]
async fn set_ai_enabled(enabled: bool, app: tauri::AppHandle) -> Result<(), String> {
    let ai_state = app.state::<Mutex<AIState>>();
    let mut ai_state = ai_state.lock().unwrap();

    ai_state.enabled = enabled;
    println!("{}", ai_state.enabled);
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
