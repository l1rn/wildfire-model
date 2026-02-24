
def temporal_split(df):
    train = df[df["year"] <= 2024]
    test = df[df["year"] >= 2025]   
    return train, test