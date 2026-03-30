from src.config import EDA_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 


def execute_eda_pipeline(df):
    df['temp_c'] = df['temp'] - 273.15
    
    fire = df[df['fire'] == 1]
    no_fire = df[df['fire'] == 0]
    cont_vars = ['temp_c', 'vpd', 'precip', 'sm1', 'u10', 'v10', 'dem', 'ndvi',
                 'slope', 'ghm', 'dist_oil_gas', 'pop_density', 'peatland']
    n_vars = len(cont_vars)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, var in enumerate(cont_vars):
        sns.kdeplot(no_fire[var].dropna(), label='No Fire', fill=True, alpha=0.5, ax=axes[i])
        sns.kdeplot(fire[var].dropna(), label='Fire', fill=True, alpha=0.5, ax=axes[i])
        axes[i].set_title(f'Distribution of {var}')
        axes[i].legend()
        
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(Path(EDA_DIR) / 'eda_continuous_distributions.png', dpi=300)
    plt.close()
    
    categorical_vars = ['landcover', 'month']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    landcover_counts = df.groupby(['landcover', 'fire']).size().unstack(fill_value=0)
    landcover_counts.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Fire Occurrence by Land Cover Class')
    axes[0].set_xlabel('Land Cover Class')
    axes[0].set_ylabel('Count')
    axes[0].legend(['No Fire', 'Fire'])
    
    month_counts = df.groupby(['month', 'fire']).size().unstack(fill_value=0)
    month_counts.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Fire Occurrence by Month')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Count')
    axes[1].legend(['No Fire', 'Fire'])
    
    plt.tight_layout()
    plt.savefig(Path(EDA_DIR) / 'eda_categorical_fire.png', dpi=300)
    plt.close()
    
    corr_vars = cont_vars + ['fire']
    corr = df[corr_vars].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Continuous Variables and Fire Occurrence')
    plt.tight_layout()
    plt.savefig(Path(EDA_DIR) / 'eda_correlation_heatmap.png', dpi=300)
    plt.close()
    
    key_vars = ['temp', 'vpd', 'ghm', 'dist_oil_gas', 'sm1', 'u10', 'v10']
    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    axes = axes.flatten()
    
    for i, var in enumerate(key_vars):
        sns.boxplot(x='fire', y=var, data=df, ax=axes[i])
        axes[i].set_title(f'{var} by Fire Occurrence')
        axes[i].set_xlabel('Fire (1) vs. No Fire (0)')
        
    plt.tight_layout()
    plt.savefig(Path(EDA_DIR) / 'eda_boxblots_key_vars.png', dpi=300)