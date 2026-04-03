import os
import argparse
import torch
import torch.nn.functional as F
import glob
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pyarrow as pa  # 確保 pyarrow 有安裝
import pyarrow.dataset as ds

# 引入 Hugging Face 核心套件
from datasets import Dataset, load_dataset
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="H200 優化的 ImageNet Diffusion 訓練")
    parser.add_argument("--data_dir", type=str, required=True, help="Arrow 檔案所在目錄")
    parser.add_argument("--output_dir", type=str, required=True, help="模型輸出路徑")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision, log_with="wandb")
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            print("🚀 H200 優化：已啟用 TF32")

    # 1. 資料處理
    data_transforms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 核心修正：正確載入碎片化的 Arrow 檔案
    if accelerator.is_main_process:
        print(f"📂 使用官方 load_dataset 載入 Arrow 碎片 (安全省 RAM 模式)...")

    # 1. 精確抓出所有的 .arrow 檔案，徹底避開 .json 和 .lock 檔
    arrow_files = sorted(glob.glob(os.path.join(args.data_dir, "imagenet-1k-train-*.arrow")))
    
    if not arrow_files:
        raise FileNotFoundError(f"在 {args.data_dir} 找不到任何 imagenet-1k-train-*.arrow 檔案！")

    # 2. 這是 HF 官方的標準讀取法：自動處理 Schema，自動使用 Zero-copy 記憶體映射
    dataset = load_dataset("arrow", data_files=arrow_files, split="train")
    
    # 3. 定義影像前處理轉換
    def transform_fn(examples):
        images = [data_transforms(img.convert("RGB")) for img in examples["image"]]
        return {"input": images}

    # 4. 懶加載轉換 (用到時才算)
    dataset = dataset.with_transform(transform_fn)

    # 5. 設定 DataLoader
    def collate_fn(examples):
        pixel_values = torch.stack([example["input"] for example in examples])
        return (pixel_values,)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 2. 模型初始化 (UNet)
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3, out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 3. Accelerate 準備
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # 這裡才做 torch.compile (針對 H200 再次提速)
    model = torch.compile(model)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs)
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # 4. 訓練
    if accelerator.is_main_process:
        accelerator.init_trackers("h200-diffusion-training", config=vars(args))
        print(f"🔥 開始訓練，樣本數: {len(dataset)}")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            clean_images = batch[0]
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, 1000, (clean_images.shape[0],), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.detach().item()
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # 5. 存檔邏輯
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            avg_loss = epoch_loss / len(dataloader)
            unwrap_model = accelerator.unwrap_model(model)
            pipeline = DDPMPipeline(unet=unwrap_model, scheduler=noise_scheduler)
            
            pipeline.save_pretrained(os.path.join(args.output_dir, "last_model"))
            if avg_loss < best_loss:
                best_loss = avg_loss
                pipeline.save_pretrained(os.path.join(args.output_dir, "best_model"))
                print(f"🏆 Best Loss: {best_loss:.5f} - Model Saved")

    accelerator.end_training()

if __name__ == "__main__":
    main()
