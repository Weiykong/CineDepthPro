# Android Prototype (Jetpack Compose)

This folder is now a **minimal Android Studio project scaffold** for the `Retouch Depth` screen.

Included:

- `Manual Paint` mode (brush controls)
- `Auto Segment` mode (mask selection + depth slider)
- Shared controls (`target depth`, `feather`, `edge snap`, `undo/redo`)
- `ViewModel` + `StateFlow` action/event pipeline (`RetouchDepthViewModel`)

## Files

- `settings.gradle.kts`
- `build.gradle.kts`
- `app/src/main/java/com/cinedepth/pro/MainActivity.kt`
- `app/src/main/java/com/cinedepth/pro/ui/retouch/RetouchDepthScreen.kt`
- `app/src/main/java/com/cinedepth/pro/ui/retouch/RetouchDepthUiState.kt`
- `app/src/main/java/com/cinedepth/pro/ui/retouch/RetouchDepthViewModel.kt`
- `app/src/main/java/com/cinedepth/pro/ui/theme/Theme.kt`
- `app/src/main/AndroidManifest.xml`
- `app/build.gradle.kts`

## Open In Android Studio

1. Open `/Users/weiyuankong/Projects/real_depth_photo/android_ui_prototype`
2. Let Gradle sync
3. Run on an emulator/device

## Next wiring

- Replace the placeholder canvas in `RetouchDepthScreen.kt` with the real photo/depth preview surface.
- Hook `RetouchDepthViewModel` actions to the depth retouch engine (brush strokes, masks, undo stack).
- Keep depth edits numeric (`0..255`) and colorize only for preview.
