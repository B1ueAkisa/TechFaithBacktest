import os
import pandas as pd
import yfinance as yf

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def download_data():
    start_date = "2023-07-11"
    end_date = "2026-07-11"
    
    tickers = {
        "QQQ": "QQQ",
        "TQQQ": "TQQQ",
        "SOXX": "SOXX",
        "SOXL": "SOXL",
        "MU": "MU",
        "MUU": "MUU",
        "WDC": "WDC",
        "SNDK": "SNDK",
        "SNXX": "SNXX",        # 闪迪 2x 做多 ETF (Tradr 2X Long SNDK Daily ETF)
        "EWY": "EWY",
        "KORU": "KORU",
        "688_CHIP": "588200.SS"
    }
    
    print("开始从 Yahoo Finance 下载完整的 OHLCV 数据...")
    data_dict = {}
    
    for name, ticker in tickers.items():
        try:
            print(f"正在下载 {name} ({ticker})...")
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel('Ticker')
                
                cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                if 'Adj Close' in df.columns:
                    df['Close'] = df['Adj Close']
                
                df = df[cols_to_keep].copy()
                data_dict[name] = df
                print(f"成功下载 {name}，共 {len(df)} 行 data")
            else:
                print(f"警告：{name} 数据为空")
        except Exception as e:
            print(f"下载 {name} 失败: {e}")
            
    # 特殊处理闪迪 (SanDisk)：只使用独立上市后的 SNDK 数据，从其实际上市日开始计算
    if "SNDK" in data_dict:
        data_dict["SanDisk"] = data_dict["SNDK"].copy()
        print("直接使用独立上市后的 SNDK 数据作为闪迪历史，起点为其实际上市日 (2025-02-24)")
    elif "WDC" in data_dict:
        print("警告：SNDK 数据不存在，直接使用 WDC 代替闪迪")
        data_dict["SanDisk"] = data_dict["WDC"]
        
    # 保存数据
    os.makedirs("data", exist_ok=True)
    
    final_assets = ["QQQ", "TQQQ", "SOXX", "SOXL", "MU", "MUU", "SanDisk", "SNXX", "EWY", "KORU", "688_CHIP"]
    
    for name in final_assets:
        if name in data_dict:
            df = data_dict[name]
            df = df.ffill().dropna()
            df.to_csv(f"data/{name}.csv")
            print(f"已保存资产 {name} 数据至 data/{name}.csv，共 {len(df)} 行")
            
    print("所有资产的 OHLCV 数据准备完毕。")
    
if __name__ == "__main__":
    download_data()
