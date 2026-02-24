# Mobile Build Setup Guide

## Prerequisites

### Android
- Android Studio with SDK (API 24+)
- NDK r25+
- Set `ANDROID_HOME` and `NDK_HOME` environment variables

### iOS
- macOS with Xcode 15+
- CocoaPods (`gem install cocoapods`)

## Initialize Mobile Targets

```bash
# Android
npx tauri android init

# iOS (macOS only)
npx tauri ios init
```

## Android Permissions

After `npx tauri android init`, add to `src-tauri/gen/android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_MEDIA_IMAGES" />
```

## Build & Run

```bash
# Android development
npx tauri android dev

# Android release
npx tauri android build

# iOS development
npx tauri ios dev

# iOS release
npx tauri ios build
```

## ONNX Runtime Mobile Optimization

For better performance on mobile:
- Android: ONNX Runtime can use NNAPI execution provider
- iOS: ONNX Runtime can use CoreML execution provider

These are configured in the Rust `ort` SessionBuilder when available on the target platform.
