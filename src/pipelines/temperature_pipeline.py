import numpy as np
import matplotlib.pyplot as plt
from src.extract import data_loader
from src.config import Config

cfg = Config()

class TemperaturePipeline:
    def __init__(self):
        pass
    
    def load_data(self):
        ds = data_loader.load_meterological(cfg.raw_weather)
        
        df = ds[['t2m', 'd2m']].mean(dim=['latitude', 'longitude']).to_dataframe()
        
        df['temp_c'] = df['t2m'] - 273.15
        df['dew_c'] = df['d2m'] - 273.15
        
        df['svp'] = 0.6108 * np.exp((17.27 * df['temp_c']) / (df['temp_c'] + 237.3))
        
        df['avp'] = 0.6108 * np.exp((17.27 * df['dew_c']) / (df['dew_c'] + 237.3))
        
        df['vpd'] = df['svp'] - df['avp']
        
        return df
    
    def calculate(self, df):
        summer_df = df[df.index.month.isin([5, 6, 7, 8, 9])]
        
        summer_mean = summer_df['vpd'].groupby(summer_df.index.year).mean()
        overall_mean = summer_mean.mean()
        summer_anomaly = summer_mean - overall_mean
        
        colors = ['#b91d47' if val > 0 else '#2b5797' for val in summer_anomaly]
        
        return colors, summer_anomaly
    
    def plot_picture(self, summer_anomaly, colors):
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(summer_anomaly.index, summer_anomaly, color=colors, alpha=0.8, edgecolor='black')
        
        std_dev = summer_anomaly.std()
        plt.axhline(std_dev, color='gray', linestyle='--', alpha=0.7, label=f'+1 Std Dev ({std_dev:.3f} kPa)')
        plt.axhline(-std_dev, color='gray', linestyle='--', alpha=0.7, label=f'-1 Std Dev ({-std_dev:.3f} kPa)')
        
        plt.title('ERA5 Annual Summer Vapor Pressure Deficit Anomalies', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('VPD Anomaly (kPa)', fontsize=12)
        plt.xticks(summer_anomaly.index) 
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("data/processed/vpd_anomalies.png")
        
        plt.show()
        
    def run(self):
        df = self.load_data()
        colors, anomaly = self.calculate(df)
        self.plot_picture(anomaly, colors)