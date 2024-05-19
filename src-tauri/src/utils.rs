use ort::DirectMLExecutionProvider;

pub fn onnx_init() -> Result<(), ort::Error> {
    #[cfg(windows)]
    {
        use windows_sys::s;
        use windows_sys::Win32::System::LibraryLoader::SetDllDirectoryA;

        unsafe {
            SetDllDirectoryA(s!("runtimes"));
        }
    }

    ort::init()
        .with_execution_providers([
            #[cfg(windows)]
            {
                DirectMLExecutionProvider::default().build()
            },
        ])
        .commit()
}
