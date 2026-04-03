import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 引入 Hugging Face Diffusers 與 Accelerate 套件
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="H200 優化的完整 Diffusion 模型訓練腳本")
    parser.add_argument("--data_dir", type=str, required=True, help="資料集路徑 (包含圖片子目錄的根目錄)")
    parser.add_argument("--output_dir", type=str, required=True, help="模型權重與結果輸出路徑")
    
    # 訓練超參數 (可靈活調整)
    parser.add_argument("--resolution", type=int, default=512, help="圖片訓練解析度 (H200 VRAM 充足，可設為 256 或 512)")
    parser.add_argument("--batch_size", type=int, default=32, help="單卡 Batch Size")
    parser.add_argument("--epochs", type=int, default=50, help="總訓練 Epoch 數")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始學習率")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="學習率預熱步數")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="混合精度 (H200 強烈建議使用 bf16)")
    parser.add_argument("--save_model_epochs", type=int, default=10, help="每隔幾個 Epoch 儲存一次模型")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 的 CPU 讀取執行緒數")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 初始化 Accelerator (負責處理多卡、混合精度與設備擺放)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
        log_with="wandb" # 新增這行
    )
    
    # H200 架構優化：開啟 TF32 加速矩陣運算
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(" H200 優化：已啟用 TensorFloat-32 (TF32)")

    # 2. 資料集與 DataLoader 準備
    # 使用標準的影像前處理：縮放、中心裁切、轉為 Tensor、正規化至 [-1, 1] (符合 DDPM 標準)
    data_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # --- 原本的 ImageFolder 刪掉，換成這個 ---
    if accelerator.is_main_process:
        print(f"📂 正在從 Arrow 格式載入資料集...")
    
    # 載入 Arrow 資料集 (指向你那個包含大量 .arrow 的資料夾)
    raw_dataset = load_from_disk(args.data_dir)
    
    # 定義轉換函數 (因為 Arrow 裡面的圖片是 PIL 物件)
    def transform_fn(examples):
        images = [data_transforms(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    # 設定轉換邏輯 (這不會立刻執行，而是在 DataLoader 讀取時才動態轉換)
    dataset = raw_dataset["train"].with_transform(transform_fn)

    # 修改 DataLoader 的取樣方式 (因為 datasets 格式結構稍微不同)
    def collate_fn(examples):
        pixel_values = torch.stack([example["input"] for example in examples])
        return (pixel_values,) # 回傳 Tuple 保持跟原本程式碼相容

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    # ---------------------------------------

    # 3. 初始化模型與排程器 (Scheduler)
    # 這裡建立一個標準的 UNet 結構
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    )
    model.to(accelerator.device) # 先放到設備上
    
    
    # DDPM 雜訊排程器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 4. 初始化優化器與學習率排程器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    

    # 5. 讓 Accelerate 接管所有 PyTorch 物件
    # 這是最關鍵的一步：它會自動把 model 丟到對應的 H200 上，並配置分散式通訊
    # 1. 先 Prepare Model, Optimizer, DataLoader
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    model = torch.compile(model, mode="reduce-overhead")
    
    # 2. 這時候的 len(dataloader) 才是真正分發到單卡後的步數
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs)
    )
    
    # 3. 再單獨 Prepare Scheduler
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # 6. 開始訓練迴圈
    global_step = 0
    # 新增：用來記錄歷史最低 Loss
    best_loss = float("inf") 

    if accelerator.is_main_process:
        print(f"🔥 開始訓練，總 Epoch 數: {args.epochs}，單卡 Batch Size: {args.batch_size}")
        accelerator.init_trackers(
                    project_name="h200-diffusion-training", # 你的 W&B 專案名稱 (可自訂)
                    config=vars(args) # 自動記錄所有 argparse 參數！
                )
    
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)   
        progress_bar.set_description(f"Epoch {epoch+1}")
        # 新增：用來計算這個 Epoch 的總 Loss
        epoch_total_loss = 0.0 
        
        for step, batch in enumerate(dataloader):
            clean_images = batch[0]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 累加 Loss
            epoch_total_loss += loss.detach().item()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0],
                "best": best_loss if best_loss != float("inf") else "N/A"
            }
            progress_bar.set_postfix(**logs)
            # 🟢 新增這行：把數據推送到 W&B 雲端
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        progress_bar.close()

        # 新增：計算該 Epoch 的平均 Loss
        avg_epoch_loss = epoch_total_loss / len(dataloader)

        # 7. 儲存模型 (Last 與 Best 策略)
        accelerator.wait_for_everyone() # 確保所有 GPU 都跑到這裡才存檔
        
        if accelerator.is_main_process:
            unwrap_model = accelerator.unwrap_model(model)
            pipeline = DDPMPipeline(unet=unwrap_model, scheduler=noise_scheduler)
            
            # (A) 永遠覆寫儲存「最新」的模型 (last)
            last_save_path = os.path.join(args.output_dir, "last_model")
            pipeline.save_pretrained(last_save_path)
            print(f"✅ Epoch {epoch+1} 結束，已覆寫 last_model (Avg Loss: {avg_epoch_loss:.5f})")

            # (B) 判斷是否為「最佳」模型 (best)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_save_path = os.path.join(args.output_dir, "best_model")
                pipeline.save_pretrained(best_save_path)
                print(f"🏆 發現更低的 Loss ({best_loss:.5f})！已更新 best_model")

    if accelerator.is_main_process:
        print(" 訓練管線執行完畢！")
        accelerator.end_training()

if __name__ == "__main__":
    main()
