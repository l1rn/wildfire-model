
def temporal_split(df):
    train = df[df["year"] <= 2022]
    test = df[df["year"] >= 2023]  
    future = df[df["year"] == 2026] 
    return train, test, future