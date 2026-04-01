#!/bin/bash
# run_pipeline.sh

# 預設路徑 (可透過參數覆蓋)
DATA_DIR="./data"
OUTPUT_DIR="./output"

# 1. 解析命令列參數
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "未知的參數: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================"
echo " 專案啟動"
echo " 資料集路徑: $DATA_DIR"
echo " 輸出路徑: $OUTPUT_DIR"
echo "========================================"

# 2. 安裝專案依賴
echo " 正在安裝 requirements.txt 中的套件..."
# # 在無 root 權限的容器中，加上 --user 可避免權限報錯
# pip install --user -q -r requirements.txt

# 確保 --user 安裝的執行檔路徑在 PATH 中 (例如 accelerate)
export PATH="$HOME/.local/bin:$PATH"

# 3. 確保輸出目錄存在
mkdir -p "$OUTPUT_DIR"

# 4. 啟動 Python 訓練程式
# 將收到的路徑參數直接往下傳遞給 Python 腳本
echo " 開始執行 Diffusion 訓練..."
python train_diffusion.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

echo " 執行完畢！"