import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
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
        gradient_accumulation_steps=1
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
    
    dataset = ImageFolder(root=args.data_dir, transform=data_transforms)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True # 加速 CPU 到 GPU 的記憶體傳輸
    )
    
    if accelerator.is_main_process:
        print(f"📂 成功載入資料集：共 {len(dataset)} 張圖片")

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
    
    # DDPM 雜訊排程器
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 4. 初始化優化器與學習率排程器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs)
    )

    # 5. 讓 Accelerate 接管所有 PyTorch 物件
    # 這是最關鍵的一步：它會自動把 model 丟到對應的 H200 上，並配置分散式通訊
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # 6. 開始訓練迴圈
    global_step = 0
    if accelerator.is_main_process:
        print(f" 開始訓練，總 Epoch 數: {args.epochs}，單卡 Batch Size: {args.batch_size}")

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        for step, batch in enumerate(dataloader):
            # batch 包含 (images, labels)，我們只取 images
            clean_images = batch[0]
            
            # (a) 為圖片隨機採樣雜訊
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bsz = clean_images.shape[0]
            
            # (b) 為每個 batch 隨機採樣一個時間步 (timestep)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()
            
            # (c) 根據時間步將雜訊加入乾淨的圖片 (Forward process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # (d) 預測雜訊
            with accelerator.accumulate(model):
                # 預測圖片中加入的雜訊殘差
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                
                # 計算損失 (MSE)
                loss = F.mse_loss(noise_pred, noise)
                
                # 反向傳播
                accelerator.backward(loss)
                
                # 梯度裁切防爆與參數更新
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # 更新進度條
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step += 1
            
        progress_bar.close()

        # 7. 定期儲存模型 (僅由主進程負責，避免多卡覆寫)
        if accelerator.is_main_process:
            if (epoch + 1) % args.save_model_epochs == 0 or (epoch + 1) == args.epochs:
                save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
                # 將包裹在 accelerator 中的模型還原，並打包成 Pipeline 儲存
                unwrap_model = accelerator.unwrap_model(model)
                pipeline = DDPMPipeline(unet=unwrap_model, scheduler=noise_scheduler)
                pipeline.save_pretrained(save_path)
                print(f" 模型已儲存至: {save_path}")

    if accelerator.is_main_process:
        print(" 訓練管線執行完畢！")

if __name__ == "__main__":
    main()