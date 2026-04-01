# H200 Diffusion Project

這是一個標準的 PyTorch 專案，設計用於在 AI Stack 等提供預建置容器的平台上運行。

## 專案結構
* `requirements.txt`: 專案需要的第三方套件。
* `run_pipeline.sh`: 一鍵安裝依賴並啟動訓練的進入點。
* `train_diffusion.py`: 核心訓練腳本。

## 如何在平台上執行

在 AI Stack 提交任務時，將執行指令設定為：

```bash
bash run_pipeline.sh --data_dir /平台的/資料/掛載路徑 --output_dir ./output