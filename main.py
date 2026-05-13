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
from models.Autoformer import Model as AutoformerModel
from models.Informer import Model as InformerModel
from models.Pyraformer import Pyraformer

def run_model(args, model, batch_x, batch_y, batch_x_mark, batch_y_mark, device):
    """Helper: forward pass cho từng model."""
    if args.model == 'Pyraformer':
        predict_token = torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=device)
        batch_x_input = torch.cat([batch_x, predict_token], dim=1)
        batch_x_mark_input = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
        return model(batch_x_input, batch_x_mark_input, batch_y, batch_y_mark, False)

    elif args.model in ('Autoformer', 'Informer'):
        # Decoder input: [label_len steps từ ENCODER (batch_x)] + [zeros cho pred_len]
        # Dùng batch_x thay vì batch_y để tránh data leakage
        dec_inp = torch.zeros(
            batch_y.shape[0], args.pred_len, batch_y.shape[-1]
        ).float().to(device)
        dec_inp = torch.cat([batch_x[:, -args.label_len:, :], dec_inp], dim=1)
        # Time mark: [label_len cuối của x_mark] + [pred_len đầu của y_mark]
        dec_mark = torch.cat([
            batch_x_mark[:, -args.label_len:, :],
            batch_y_mark[:, :args.pred_len, :]
        ], dim=1)
        outputs = model(batch_x, batch_x_mark, dec_inp, dec_mark)
        return outputs

    else:
        return model(batch_x)


def main():
    parser = argparse.ArgumentParser(description='LTSF Reproduction: Linear & Transformers')
    
    # --- 1. Lựa chọn mô hình & dữ liệu ---
    parser.add_argument('--model', type=str, default='Linear', 
                        choices=['DLinear', 'NLinear', 'Linear', 'Pyraformer', 'Autoformer', 'Informer'], help='Chọn mô hình')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--train_only', action='store_true', default=False, help='Chỉ huấn luyện, không test')

    # --- 2. Kích thước chuỗi ---
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # --- 3. Tham số cấu trúc Pyraformer ---
    parser.add_argument('--window_size', type=str, default='[4, 4, 4]', help='window size')
    parser.add_argument('--inner_size', type=int, default=3, help='inner resolution')
    parser.add_argument('--truncate', action='store_true', default=False, help='truncate flag')
    parser.add_argument('--use_tvm', action='store_true', default=False, help='use tvm flag')
    parser.add_argument('--decoder', type=str, default='FC', choices=['FC', 'attention'], help='type of decoder')
    parser.add_argument('--CSCM', type=str, default='Conv_Construct', help='CSCM type')

    # --- Thay đổi/Thêm tham số cho Autoformer ---
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_true', default=True, help='use distilling in Informer encoder')
    
    # --- 4. Tham số Hyperparameters ---
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--d_inner_hid', type=int, default=512, help='dimension of inner hidden layer')
    parser.add_argument('--d_k', type=int, default=128, help='dimension of key')
    parser.add_argument('--d_v', type=int, default=128, help='dimension of value')
    parser.add_argument('--d_bottleneck', type=int, default=128, help='dimension of bottleneck')
    parser.add_argument('--n_head', type=int, default=8, help='num of heads')
    parser.add_argument('--n_layer', type=int, default=2, help='num of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--embed_type', type=str, default='DataEmbedding', help='DataEmbedding or CustomEmbedding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # --- 5. Tham số nạp dữ liệu & Huấn luyện ---
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay') 
    parser.add_argument('--train_epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

    args = parser.parse_args()
    
    # --- Ánh xạ các biến nội bộ ---
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.input_size = args.seq_len
    args.predict_step = args.pred_len
    if isinstance(args.window_size, str):
        args.window_size = eval(args.window_size)
    
    device = args.device

    # 1. Nạp dữ liệu
    print(f"--- Đang nạp dữ liệu cho {args.model} trên {device} ---")
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    # ================= 2. KHỞI TẠO MÔ HÌNH =================
    if args.model == 'Pyraformer':
        model = Pyraformer.Model(args).to(device)
    
    elif args.model == 'Autoformer':
        args.e_layers = args.n_layer
        args.d_layers = 1
        args.n_heads = args.n_head
        args.d_ff = args.d_inner_hid
        args.moving_avg = 25
        args.factor = 3
        args.output_attention = False
        # Autoformer.py expects embed_type as int (0–4), not string
        args.embed_type = 1   # 1 = DataEmbedding (full, with position + time)
        
        model = AutoformerModel(args).to(device) 
        print(f"  [!] Autoformer initialized with Mapping: e_layers={args.e_layers}, d_layers={args.d_layers}")

    elif args.model == 'Informer':
        args.e_layers = args.n_layer
        args.d_layers = 1
        args.n_heads = args.n_head
        args.d_ff = args.d_inner_hid
        args.factor = 3
        args.output_attention = False
        args.embed_type = 1   # integer, same as Autoformer
        # distil=True uses ConvLayer between encoder layers (default for Informer)

        model = InformerModel(args).to(device)
        print(f"  [!] Informer initialized: e_layers={args.e_layers}, d_layers={args.d_layers}, distil={args.distil}")

    elif args.model == 'DLinear':
        model = DLinear.Model(args).to(device)
    else:
        model = Linear.Model(args).to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    best_model_path = os.path.join(root_dir, 'checkpoints', f"checkpoint_{args.model}_{args.data}_{args.pred_len}.pth")
    if not os.path.exists(os.path.dirname(best_model_path)): 
        os.makedirs(os.path.dirname(best_model_path))

    # 4. Training
    best_val_loss = float('inf')
    early_stop_count = 0
    f_dim = -1 if args.features == 'MS' else 0

    print(f"\n--- Bắt đầu huấn luyện {args.model} ---")
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            optimizer.zero_grad()
            batch_x      = batch_x.float().to(device)
            batch_y      = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            outputs = run_model(args, model, batch_x, batch_y, batch_x_mark, batch_y_mark, device)

            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
                batch_x      = batch_x.float().to(device)
                batch_y      = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                outputs = run_model(args, model, batch_x, batch_y, batch_x_mark, batch_y_mark, device)

                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                val_loss.append(criterion(outputs, batch_y).item())
        
        avg_val_loss = sum(val_loss) / len(val_loss)
        scheduler.step()
        print(f"Epoch {epoch+1:02d} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train Loss: {sum(train_loss)/len(train_loss):.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), best_model_path)
            print("  [+] Saved best model")
        else:
            early_stop_count += 1
            if early_stop_count >= args.patience: 
                print("  [!] Early stopping triggered.")
                break

    # 5. Testing
    print("\n--- Testing ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_mse, test_mae = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x      = batch_x.float().to(device)
            batch_y      = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = run_model(args, model, batch_x, batch_y, batch_x_mark, batch_y_mark, device)

            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            test_mse.append(criterion(outputs, batch_y).item())
            test_mae.append(mae_metric(outputs, batch_y).item())

    print(f"\nFINAL RESULTS - MSE: {sum(test_mse)/len(test_mse):.4f} | MAE: {sum(test_mae)/len(test_mae):.4f}")

if __name__ == '__main__':
    main()