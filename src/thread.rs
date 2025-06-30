use std::thread::{JoinHandle};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ThreadManager {
    pub handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl ThreadManager {
    pub fn new() -> Self {
        Self {
            handles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn wait_all(&self) {
        let mut handles = self.handles.lock().unwrap();
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }
}

