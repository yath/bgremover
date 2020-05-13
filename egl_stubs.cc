#ifdef __cplusplus
extern "C" {
#endif

// EGL_KHR_fence_sync calls my libEGL doesn’t have. Doesn’t seem to affect the OpenGL delegate.
void __attribute__((weak)) eglCreateSyncKHR() { }
void __attribute__((weak)) eglClientWaitSyncKHR() { }
void __attribute__((weak)) eglWaitSyncKHR() { }
void __attribute__((weak)) eglDestroySyncKHR() { }

#ifdef __cplusplus
} // extern "C"
#endif
