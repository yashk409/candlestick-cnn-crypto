import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import concurrent.futures

def generate_image(window_df, n_days):
    if n_days == 5: height = 32
    elif n_days == 20: height = 64
    else: height = 96
    width = n_days * 3
    
    img = Image.new('L', (width, height), color=0)
    draw = ImageDraw.Draw(img)
    
    if len(window_df) == 0: return np.zeros((1, height, width), dtype=np.float32)
        
    first_close = window_df['Close'].iloc[0]
    if first_close == 0 or np.isnan(first_close): return np.zeros((1, height, width), dtype=np.float32)
    
    prices = window_df[['Open', 'High', 'Low', 'Close']].values / first_close
    ma = window_df['Close'].rolling(window=n_days, min_periods=1).mean().values / first_close
    volumes = window_df['Volume'].values
    
    max_p, min_p = np.nanmax(prices), np.nanmin(prices)
    max_v = np.nanmax(volumes)
    if max_p == min_p: max_p = min_p + 1e-5
    if max_v == 0: max_v = 1e-5
    
    price_h = int(height * 0.8)
    vol_h = height - price_h
    
    def price_to_y(p): return int(np.clip(price_h - 1 - (p - min_p) / (max_p - min_p) * (price_h - 1), 0, price_h - 1))
    def vol_to_y(v): return int(np.clip(height - 1 - (v / max_v) * (vol_h - 1), price_h, height - 1))
    
    ma_points = []
    for i in range(len(window_df)):
        x_center = i * 3 + 1
        o_y = price_to_y(prices[i, 0])
        h_y = price_to_y(prices[i, 1])
        l_y = price_to_y(prices[i, 2])
        c_y = price_to_y(prices[i, 3])
        
        draw.line([(x_center, h_y), (x_center, l_y)], fill=255, width=1)
        draw.line([(x_center-1, o_y), (x_center, o_y)], fill=255, width=1)
        draw.line([(x_center, c_y), (x_center+1, c_y)], fill=255, width=1)
        
        v_y = vol_to_y(volumes[i])
        draw.line([(x_center, height-1), (x_center, v_y)], fill=255, width=1)
        
        ma_y = price_to_y(ma[i])
        ma_points.append((x_center, ma_y))
        
    if len(ma_points) > 1:
        draw.line(ma_points, fill=255, width=1)
        
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def process_ticker(args):
    ticker, t_df, n_days, horizon = args
    t_df = t_df.sort_values('Date').reset_index(drop=True)
    t_df['Target_Ret'] = t_df['Close'].shift(-horizon) / t_df['Close'] - 1
    t_df['Label'] = (t_df['Target_Ret'] > 0).astype(int)
    
    # Vectorized check for NaNs in rolling window
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MOM', 'STR', 'WSTR', 'TREND']
    # Check if any NaN exists in the required columns within the rolling window
    nan_count = t_df[required_cols].isna().rolling(window=n_days).sum().sum(axis=1)
    target_isna = t_df['Label'].isna()
    
    valid_mask = (nan_count == 0) & (~target_isna)
    valid_indices = np.where(valid_mask)[0]
    
    samples = []
    for idx in valid_indices:
        start_idx = idx - n_days + 1
        if start_idx < 0: continue
        
        row = t_df.iloc[idx]
        base = {
            'Date': str(row['Date'].date()),
            'Ticker': ticker,
            'MOM': row['MOM'],
            'STR': row['STR'],
            'WSTR': row['WSTR'],
            'TREND': row['TREND'],
            'Fwd_Ret': row['Target_Ret']
        }
        samples.append({
            'ticker': ticker,
            'start_idx': start_idx,
            'label': int(row['Label']),
            'base': base
        })
    return ticker, t_df, samples

class LazyStockImageDataset(Dataset):
    def __init__(self, df, n_days=20, horizon=20):
        self.n_days = n_days
        self.samples = []
        self.grouped_dfs = {}
        
        args_list = []
        grouped = df.groupby('Ticker')
        for ticker, t_df in grouped:
            args_list.append((ticker, t_df.copy(), n_days, horizon))
            
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(process_ticker, args_list), total=len(args_list), desc=f"Indexing Data (N={n_days}, H={horizon})", leave=False))
            
        for ticker, t_df, t_samples in results:
            self.grouped_dfs[ticker] = t_df
            self.samples.extend(t_samples)
                
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        t_df = self.grouped_dfs[sample['ticker']]
        start = sample['start_idx']
        
        window = t_df.iloc[start:start+self.n_days]
        img = generate_image(window, self.n_days)
        
        return torch.tensor(img), torch.tensor(sample['label'], dtype=torch.long), sample['base']
