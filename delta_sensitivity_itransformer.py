"""
FANTF Delta Sensitivity Analysis Script — FuzzyiTransformer
=============================================================
Purpose : Vary delta in {0.1, 0.3, 0.5, 0.7, 1.0} across
          noise levels {0%, 5%, 10%, 20%} on Exchange Rate
          and ETTh1 datasets using iTransformer+FAN backbone.

Dataset file naming convention (actual paths on disk):
  exchange_rate:
    0%  --> ./dataset/exchange_rate/exchange_rate.csv
    5%  --> ./dataset/exchange_rate/noisy/exchange_rate_noise_5.csv
    10% --> ./dataset/exchange_rate/noisy/exchange_rate_noise_10.csv
    20% --> ./dataset/exchange_rate/noisy/exchange_rate_noise_20.csv
  ETTh1:
    0%  --> ./dataset/ETT-small/ETTh1.csv
    5%  --> ./dataset/ETT-small/noisy/etth1_noise_5.csv
    10% --> ./dataset/ETT-small/noisy/etth1_noise_10.csv
    20% --> ./dataset/ETT-small/noisy/etth1_noise_20.csv
"""

import os
import csv
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FuzzyAttention
from layers.Embed import DataEmbedding_inverted

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DELTA_VALUES  = [0.1, 0.3, 0.5, 0.7, 1.0]
NOISE_LEVELS  = [0, 5, 10, 20]
PRED_LEN      = 96
SEQ_LEN       = 96
BATCH_SIZE    = 32
EPOCHS        = 30          # increased from 10 for visible delta differences
LEARNING_RATE = 1e-4
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────
# DATASET FILE PATH RESOLVER
# ─────────────────────────────────────────────
def get_dataset_path(dataset_name, noise_pct):
    if dataset_name == 'exchange_rate':
        if noise_pct == 0:
            return './dataset/exchange_rate/exchange_rate.csv'
        else:
            return (f'./dataset/exchange_rate/noisy/'
                    f'exchange_rate_noise_{noise_pct}.csv')
    elif dataset_name == 'ETTh1':
        if noise_pct == 0:
            return './dataset/ETT-small/ETTh1.csv'
        else:
            return (f'./dataset/ETT-small/noisy/'
                    f'etth1_noise_{noise_pct}.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ─────────────────────────────────────────────
# FUZZY iTRANSFORMER MODEL
# freeze_fuzziness=True freezes fuzziness_param
# at 1.0 so delta is the sole variable tested
# ─────────────────────────────────────────────
class FuzzyiTransformer(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len,
                 d_model=64, n_heads=4, d_ff=128,
                 e_layers=2, dropout=0.1,
                 embed='timeF', freq='h',
                 factor=5, delta=0.5):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.enc_in   = enc_in

        self.enc_embedding = DataEmbedding_inverted(
            seq_len, d_model, embed, freq, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FuzzyAttention(
                            mask_flag         = False,
                            factor            = factor,
                            attention_dropout = dropout,
                            output_attention  = False,
                            delta             = delta,
                            freeze_fuzziness  = True   # frozen for sensitivity
                        ),
                        d_model, n_heads
                    ),
                    d_model, d_ff,
                    dropout    = dropout,
                    activation = 'gelu'
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        means  = x_enc.mean(1, keepdim=True).detach()
        x_enc  = x_enc - means
        stdev  = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True,
                      unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out    = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out    = self.projection(enc_out)
        dec_out    = dec_out.permute(0, 2, 1)[:, :, :self.enc_in]

        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(
                1, self.pred_len, 1))
        return dec_out


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_dataset(dataset_name, noise_pct, seq_len, pred_len):
    path = get_dataset_path(dataset_name, noise_pct)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFile not found: {path}\n"
            f"Please verify the file exists at this exact path.")

    df = pd.read_csv(path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    data = df.values.astype(np.float32)

    mean = data.mean(axis=0)
    std  = data.std(axis=0) + 1e-8
    data = (data - mean) / std

    total_len = seq_len + pred_len
    X, Y = [], []
    for i in range(len(data) - total_len + 1):
        X.append(data[i           : i + seq_len])
        Y.append(data[i + seq_len : i + total_len])

    X = torch.tensor(np.array(X))
    Y = torch.tensor(np.array(Y))

    split   = int(0.8 * len(X))
    X_train = X[:split];  Y_train = Y[:split]
    X_test  = X[split:];  Y_test  = Y[split:]

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE, shuffle=True,
        worker_init_fn=lambda _: np.random.seed(SEED))

    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=BATCH_SIZE, shuffle=False)

    enc_in = X.shape[-1]
    return train_loader, test_loader, enc_in


# ─────────────────────────────────────────────
# METRICS — 4 decimal places for sensitivity
# ─────────────────────────────────────────────
def compute_metrics(preds, targets):
    mse = torch.mean((preds - targets) ** 2).item()
    mae = torch.mean(torch.abs(preds - targets)).item()
    return round(mse, 4), round(mae, 4)


# ─────────────────────────────────────────────
# TRAINING AND EVALUATION
# ─────────────────────────────────────────────
def train_and_evaluate(model, train_loader, test_loader):
    model     = model.to(DEVICE)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimiser.zero_grad()
            pred = model(x_batch, x_mark_enc=None)
            pred = pred[:, :y_batch.shape[1], :]
            loss = criterion(pred, y_batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:>2}/{EPOCHS} "
                  f"loss={epoch_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred    = model(x_batch, x_mark_enc=None)
            pred    = pred[:, :y_batch.shape[1], :]
            all_preds.append(pred.cpu())
            all_targets.append(y_batch.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return compute_metrics(all_preds, all_targets)


# ─────────────────────────────────────────────
# MAIN SENSITIVITY EXPERIMENT LOOP
# ─────────────────────────────────────────────
def run_sensitivity_analysis():
    results  = []
    datasets = ['exchange_rate', 'ETTh1']

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        for noise_pct in NOISE_LEVELS:
            path = get_dataset_path(dataset_name, noise_pct)
            print(f"\n  Noise level : {noise_pct}%")
            print(f"  File loaded : {path}")
            print(f"  {'-'*50}")

            try:
                train_loader, test_loader, enc_in = load_dataset(
                    dataset_name, noise_pct, SEQ_LEN, PRED_LEN)
            except FileNotFoundError as e:
                print(f"  SKIPPED — {e}")
                continue

            for delta in DELTA_VALUES:
                torch.manual_seed(SEED)
                np.random.seed(SEED)

                model = FuzzyiTransformer(
                    enc_in   = enc_in,
                    seq_len  = SEQ_LEN,
                    pred_len = PRED_LEN,
                    d_model  = 64,
                    n_heads  = 4,
                    d_ff     = 128,
                    e_layers = 2,
                    dropout  = 0.1,
                    embed    = 'timeF',
                    freq     = 'h',
                    factor   = 5,
                    delta    = delta
                )

                print(f"\n  delta = {delta:.1f}")
                mse, mae = train_and_evaluate(
                    model, train_loader, test_loader)

                is_default = (delta == 0.5)
                results.append({
                    'dataset'   : dataset_name,
                    'noise_pct' : noise_pct,
                    'delta'     : delta,
                    'MSE'       : mse,
                    'MAE'       : mae,
                    'default'   : '★' if is_default else ''
                })

                print(f"  --> MSE = {mse:.4f} | MAE = {mae:.4f}"
                      f"{'  <- DEFAULT' if is_default else ''}")

    return results


# ─────────────────────────────────────────────
# SAVE TO CSV
# ─────────────────────────────────────────────
def save_results(results,
                 output_path='delta_sensitivity_itransformer_results.csv'):
    if not results:
        print("No results to save.")
        return
    fieldnames = ['dataset', 'noise_pct', 'delta',
                  'MSE', 'MAE', 'default']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output_path}")


# ─────────────────────────────────────────────
# PRINT FORMATTED SUMMARY TABLE — 4 decimal places
# ─────────────────────────────────────────────
def print_table(results):
    datasets   = ['exchange_rate', 'ETTh1']
    show_noise = [0, 10, 20]

    print("\n" + "="*80)
    print("DELTA SENSITIVITY SUMMARY — FuzzyiTransformer")
    print("(Copy values directly into LaTeX table)")
    print("="*80)

    for dataset_name in datasets:
        print(f"\nDataset: {dataset_name}")
        header = (f"{'delta':<8}"
                  f"{'0% MSE':<12}{'0% MAE':<12}"
                  f"{'10% MSE':<12}{'10% MAE':<12}"
                  f"{'20% MSE':<12}{'20% MAE':<12}")
        print(header)
        print("-" * 78)

        for delta in DELTA_VALUES:
            row = []
            for np_ in show_noise:
                match = [r for r in results
                         if r['dataset']    == dataset_name
                         and r['noise_pct'] == np_
                         and r['delta']     == delta]
                if match:
                    row += [f"{match[0]['MSE']:.4f}",
                            f"{match[0]['MAE']:.4f}"]
                else:
                    row += ['N/A', 'N/A']

            marker = '  ★ default' if delta == 0.5 else ''
            print(f"{delta:<8}"
                  f"{row[0]:<12}{row[1]:<12}"
                  f"{row[2]:<12}{row[3]:<12}"
                  f"{row[4]:<12}{row[5]:<12}"
                  f"{marker}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("FANTF Delta Sensitivity Analysis — FuzzyiTransformer")
    print(f"Device   : {DEVICE}")
    print(f"Deltas   : {DELTA_VALUES}")
    print(f"Noise %%  : {NOISE_LEVELS}")
    print(f"Epochs   : {EPOCHS}  |  Seed: {SEED}")
    print(f"Precision: 4 decimal places")
    print()
    print("Dataset file mapping:")
    for ds in ['exchange_rate', 'ETTh1']:
        for np_ in NOISE_LEVELS:
            print(f"  {ds:>15} {np_:>3}% --> "
                  f"{get_dataset_path(ds, np_)}")

    results = run_sensitivity_analysis()
    save_results(results)
    print_table(results)
