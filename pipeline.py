import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────
RAW_DIR = r"C:\Users\USER\Desktop\EmpowerHer\Data\Raw"
PROCESSED_DIR = r"C:\Users\USER\Desktop\EmpowerHer\Data\Processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Load Women in Agriculture ───────────────────────────
def load_women_agriculture():
    path = os.path.join(RAW_DIR, "women_agriculture.csv")
    df = pd.read_csv(path, skiprows=4)
    
    year_cols = [str(y) for y in range(2010, 2024)]
    available_years = [y for y in year_cols if y in df.columns]
    keep_cols = ["Country Name", "Country Code"] + available_years
    df = df[keep_cols]
    
    df = df.melt(
        id_vars=["Country Name", "Country Code"],
        var_name="Year",
        value_name="Women_Agriculture_Pct"
    )
    df["Year"] = df["Year"].astype(int)
    df.dropna(subset=["Women_Agriculture_Pct"], inplace=True)
    
    print(f"✅ Women Agriculture data loaded: {df.shape[0]} rows")
    return df

# ── Load Food Security ──────────────────────────────────
def load_food_security():
    path = os.path.join(RAW_DIR, "food_security.csv")
    df = pd.read_csv(path)
    
    print("📋 Food Security columns found:", df.columns.tolist())
    
    df = df[["Area", "Year", "Value"]].copy()
    df.columns = ["Country Name", "Year", "Undernourishment_Pct"]
    
    # Extract first 4-digit year from "2020 / 2019-2021" format
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(int)
    # Replace "<2.5" with 2.5 (numerical approximation)
    df["Undernourishment_Pct"] = df["Undernourishment_Pct"].replace("<2.5", 2.5)
    df["Undernourishment_Pct"] = pd.to_numeric(df["Undernourishment_Pct"], errors="coerce")
    df.dropna(subset=["Undernourishment_Pct"], inplace=True)
    
    
    print(f"✅ Food Security data loaded: {df.shape[0]} rows")
    return df

# ── Merge Datasets ──────────────────────────────────────
def merge_datasets(df_women, df_food):
    df = pd.merge(
        df_women,
        df_food,
        on=["Country Name", "Year"],
        how="inner"
    )
    print(f"✅ Merged dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ── Save Processed Data ─────────────────────────────────
def save_processed(df):
    out_path = os.path.join(PROCESSED_DIR, "agrosphere_data.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Processed data saved to: {out_path}")
    return out_path

# ── Run Pipeline ────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting AgroShe Data Pipeline...\n")
    df_women = load_women_agriculture()
    df_food = load_food_security()
    df_merged = merge_datasets(df_women, df_food)
    save_processed(df_merged)
    print("\n🎉 Pipeline complete! Data is ready for the dashboard.")
    print("\n📊 Sample of your data:")
    print(df_merged.head(10))