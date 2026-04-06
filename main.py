import sys
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

import argparse
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from models import DLinear, NLinear, Linear
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='LTSF-Linear Reproduction')
    
    # --- Chọn mô hình ---
    parser.add_argument('--model', type=str, default='DLinear', choices=['DLinear', 'NLinear', 'Linear'], help='Chọn mô hình')
    
    # --- Cấu hình dữ liệu ---
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    
    parser.add_argument('--train_only', action='store_true', default=False, help='train only flag')

    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # --- Cấu hình mô hình ---
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--individual', action='store_true', default=False, help='Channel Independence')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    
    # --- Cấu hình huấn luyện & Early Stopping ---
    parser.add_argument('--train_epochs', type=int, default=100, help='Số vòng lặp tối đa')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Tốc độ học')
    parser.add_argument('--patience', type=int, default=10, help='Dừng sau n vòng nếu Val Loss không giảm')
    
    args = parser.parse_args()

    # 1. Nạp dữ liệu
    print(f"--- Đang nạp dữ liệu cho {args.model} ---")
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    # 2. Khởi tạo mô hình tương ứng
    if args.model == 'NLinear':
        model = NLinear.Model(args)
    else:
        model = DLinear.Model(args)
        
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Checkpoint path
    checkpoint_dir = os.path.join(root_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    best_model_path = os.path.join(checkpoint_dir, f"checkpoint_{args.model}_{args.data}_{args.pred_len}.pth")

    best_val_loss = float('inf')
    early_stop_count = 0

    # ================= 3. TRAINING & VALIDATION =================
    print(f"\n--- Bắt đầu huấn luyện {args.model} (Max: {args.train_epochs} Epochs) ---")
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.float()
            # NLinear và DLinear thường chỉ cần batch_y phần dự đoán
            batch_y = batch_y.float()[:, -args.pred_len:, :]
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = sum(train_loss) / len(train_loss)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()[:, -args.pred_len:, :]
                outputs = model(batch_x)
                val_losses.append(criterion(outputs, batch_y).item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        print(f"Epoch: {epoch + 1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early Stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  [+] Saved best {args.model} model")
        else:
            early_stop_count += 1
            print(f"  [-] Patience: {early_stop_count}/{args.patience}")
            if early_stop_count >= args.patience:
                print(f"\n>>> EARLY STOPPING at epoch {epoch+1}")
                break

    # ================= 4. TESTING =================
    print(f"\n--- Loading best {args.model} for testing ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_mse, test_mae = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float()
            batch_y = batch_y.float()[:, -args.pred_len:, :]
            outputs = model(batch_x)
            
            test_mse.append(criterion(outputs, batch_y).item())
            test_mae.append(mae_metric(outputs, batch_y).item())

    final_mse = sum(test_mse) / len(test_mse)
    final_mae = sum(test_mae) / len(test_mae)
    
    print("\n" + "="*40)
    print(f"RESULTS - Model: {args.model} | Data: {args.data}")
    print(f"MSE: {final_mse:.4f}")
    print(f"MAE: {final_mae:.4f}")
    print("="*40)

    # ================= 5. VISUALIZATION ======================
    print("\n--- Đang vẽ biểu đồ ---")
    # Lấy 1 sample từ đợt test cuối cùng để vẽ
    true_plot = batch_y[0, :, -1].detach().cpu().numpy()
    pred_plot = outputs[0, :, -1].detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(true_plot, label='GroundTruth', color='blue')
    plt.plot(pred_plot, label=f'Prediction ({args.model})', color='red', linestyle='--')
    plt.title(f'{args.model} on {args.data} (Seq: {args.seq_len}, Pred: {args.pred_len})', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    res_dir = os.path.join(root_dir, 'results')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    file_name = f"{args.model}_{args.data}_seq{args.seq_len}_pred{args.pred_len}.png"
    save_path = os.path.join(res_dir, file_name)
    plt.savefig(save_path)
    plt.show()
    print(f"\nGraph saved to: {save_path}")

if __name__ == '__main__':
    main()