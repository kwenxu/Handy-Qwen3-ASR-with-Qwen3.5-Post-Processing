use crate::managers::post_process_model::{PostProcessModelInfo, PostProcessModelManager};
use std::sync::Arc;
use tauri::{AppHandle, Manager};

#[tauri::command]
#[specta::specta]
pub fn get_local_post_process_models(app: AppHandle) -> Result<Vec<PostProcessModelInfo>, String> {
    let manager = app.state::<Arc<PostProcessModelManager>>();
    Ok(manager.get_available_models())
}

#[tauri::command]
#[specta::specta]
pub async fn download_local_post_process_model(
    app: AppHandle,
    model_id: String,
) -> Result<(), String> {
    let manager = app.state::<Arc<PostProcessModelManager>>();
    manager.download_model(&model_id).await.map_err(|e| {
        format!(
            "Failed to download local post-process model '{}': {}",
            model_id, e
        )
    })
}

#[tauri::command]
#[specta::specta]
pub fn delete_local_post_process_model(app: AppHandle, model_id: String) -> Result<(), String> {
    let manager = app.state::<Arc<PostProcessModelManager>>();
    manager.delete_model(&model_id).map_err(|e| {
        format!(
            "Failed to delete local post-process model '{}': {}",
            model_id, e
        )
    })
}

#[tauri::command]
#[specta::specta]
pub fn cancel_local_post_process_model_download(
    app: AppHandle,
    model_id: String,
) -> Result<(), String> {
    let manager = app.state::<Arc<PostProcessModelManager>>();
    manager.cancel_download(&model_id).map_err(|e| {
        format!(
            "Failed to cancel local post-process model download '{}': {}",
            model_id, e
        )
    })
}
