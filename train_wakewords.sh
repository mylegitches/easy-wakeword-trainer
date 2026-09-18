#!/bin/bash
set -e
cd "$(dirname "$0")"

# Warm page cache for features_neg.npy (run once per session)
echo "=== Warming page cache ==="
dd if=../features_neg.npy of=/dev/null bs=4M

export_model() {
    local name=$1
    local onnx_path="my_custom_model/${name}.onnx"
    echo "=== Exporting ${name} to TFLite ==="
    onnx2tf -i "${onnx_path}" -o my_custom_model/ -kat onnx____Flatten_0 2>&1
    local tflite_src
    tflite_src=$(ls my_custom_model/${name}*float32*.tflite 2>/dev/null | head -1 \
        || ls my_custom_model/${name}*.tflite 2>/dev/null | grep -v "int8\|float16" | head -1 \
        || true)
    if [ -n "${tflite_src}" ] && [ "${tflite_src}" != "my_custom_model/${name}.tflite" ]; then
        mv "${tflite_src}" "my_custom_model/${name}.tflite"
    fi
    echo "Files: $(ls -lh my_custom_model/${name}.onnx my_custom_model/${name}.tflite 2>/dev/null)"
}

train_wakeword() {
    local name=$1
    local yaml=$2
    echo ""
    echo "### STARTING: ${name} ###"
    python3 openwakeword/openwakeword/train.py --training_config "${yaml}" --generate_clips 2>&1
    python3 openwakeword/openwakeword/train.py --training_config "${yaml}" --augment_clips 2>&1
    # || true: train.py exits non-zero after saving ONNX due to missing onnx_tf; harmless
    python3 openwakeword/openwakeword/train.py --training_config "${yaml}" --train_model 2>&1 || true
    export_model "${name}"
    echo "### DONE: ${name} ###"
}

# --- Add wakewords here ---
# train_wakeword "yo_hal"     "configs/yo_hal.yaml"
# train_wakeword "yo_gladdis" "configs/yo_gladdis.yaml"
# train_wakeword "hey_fucker" "configs/hey_fucker.yaml"
# --------------------------

echo ""
echo "=== ALL DONE ==="
ls -lh my_custom_model/*.onnx my_custom_model/*.tflite 2>/dev/null
