# aar build 명령어
```bash
bazel build -c opt --strip=ALWAYS \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --fat_apk_cpu=arm64-v8a,armeabi-v7a \
    --legacy_whole_archive=0 \
    --features=-legacy_whole_archive \
    --copt=-fvisibility=hidden \
    --copt=-ffunction-sections \
    --copt=-fdata-sections \
    --copt=-fstack-protector \
    --copt=-Oz \
    --copt=-fomit-frame-pointer \
    --copt=-DABSL_MIN_LOG_LEVEL=2 \
    --linkopt=-Wl,--gc-sections,--strip-all \
    //mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_example:mediapipe_pose_tracking.aar
```
# sample apk 빌드 명령어
```bash
bazelisk build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/posetrackinggpu:posetrackinggpu
```
# sample apk 설치 명령어
```bash
adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/posetrackinggpu/posetrackinggpu.apk
```
# custom sample app
https://github.com/seungy0/PoseRecognitionSample
