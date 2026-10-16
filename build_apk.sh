#!/bin/bash
# APK 构建脚本 (Linux/macOS 或 WSL/Docker)
set -e

echo "===== Flet APK 构建 ====="

cd "$(dirname "$0")"

flet build apk \
  --output build/app-release.apk \
  --android-permission android.permission.INTERNET \
  --company "MaterialAI" \
  --org com.materialai.diagnosis \
  --product "材料AI诊断" \
  --description "新材料工艺偏差排查AI系统" \
  --build-number 1

echo "APK 构建完成: build/app-release.apk"
