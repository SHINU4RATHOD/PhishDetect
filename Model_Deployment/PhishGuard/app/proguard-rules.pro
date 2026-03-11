# PhishGuard ProGuard Rules
# ============================================================================

# ONNX Runtime — keep JNI and API classes
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }

# Gson — keep serialized model classes
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.google.gson.** { *; }
-keep class com.phishguard.app.domain.** { *; }
-keep class com.phishguard.app.core.PhishGuardConfig { *; }

# Kotlin coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}

# Keep data classes used in JSON serialization/deserialization
-keepclassmembers class * {
    @com.google.gson.annotations.SerializedName <fields>;
}
