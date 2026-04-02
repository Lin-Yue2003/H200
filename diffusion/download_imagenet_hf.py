import os
import argparse
from huggingface_hub import login
from datasets import load_dataset

def main(hf_token, cache_dir):
    print("登入 Hugging Face...")
    login(token=hf_token)
    
    print(f"開始下載 ImageNet-1k 資料集至 {cache_dir} ...")
    print("這會花費相當長的時間，請耐心等候。")
    
    # 下載並快取資料集
    dataset = load_dataset(
        "imagenet-1k", 
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    print("下載與處理完成！資料已安全存放在外部儲存空間。")

if __name__ == "__main__":
    # 設定命令列參數
    parser = argparse.ArgumentParser(description="下載 ImageNet 資料集並存入指定路徑")
    parser.add_argument("--token", type=str, required=True, help="請填入 Hugging Face Access Token")
    parser.add_argument("--dir", type=str, default="/mnt/data/imagenet", help="資料集儲存路徑 (預設為 /mnt/data/imagenet)")
    
    args = parser.parse_args()
    
    # 確保儲存目錄存在
    os.makedirs(args.dir, exist_ok=True)
    
    # 執行主程式
    main(args.token, args.dir)
