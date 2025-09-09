from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    _HAS_FA = True
except Exception:
    _HAS_FA = False

# -------------------------
# Helper functions 
# -------------------------

def check_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {', '.join(missing)}")

def winsorize_z3(series: pd.Series) -> pd.Series:
    """Clip a numeric Series to μ ± 3σ."""
    if not pd.api.types.is_numeric_dtype(series):
        return series
    mu = series.mean(skipna=True)
    sd = series.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return series
    lo, hi = mu - 3*sd, mu + 3*sd
    return series.clip(lower=lo, upper=hi)

def zscore_df(df_num: pd.DataFrame) -> pd.DataFrame:
    """Z-score all numeric columns, return a DataFrame with same index/columns."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(df_num.values)
    return pd.DataFrame(X, index=df_num.index, columns=df_num.columns)

def percent_rank_dplyr(x: pd.Series) -> pd.Series:
    """
    dplyr::percent_rank equivalent:
      (rank - 1) / (n - 1) * 100    (ties average; min=0, max=100)
    """
    n = x.shape[0]
    if n <= 1:
        return pd.Series(np.zeros(n), index=x.index)
    r = x.rank(method="average")
    return (r - 1) / (n - 1) * 100.0

def pca_fit_scores(X: pd.DataFrame) -> dict:
    """Fit PCA on a z-scored matrix; return scores, pca object, variance, eigenvalues, loadings."""
    p = PCA(n_components=None, svd_solver="full", whiten=False, random_state=0)
    scores = p.fit_transform(X.values)  # shape (n_samples, n_features)
    # Loadings (correlations variable↔component) for standardized X:
    # loadings = eigenvectors * sqrt(eigenvalues)
    loadings = p.components_.T * np.sqrt(p.explained_variance_)
    return dict(
        pca=p,
        scores=pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(scores.shape[1])]),
        var_ratio=p.explained_variance_ratio_.copy(),
        eigvals=p.explained_variance_.copy(),
        loadings=pd.DataFrame(loadings, index=X.columns, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])
    )

def align_pc1_positive_with(series_scores: pd.Series, ref_feature: pd.Series) -> pd.Series:
    """Flip PC1 scores if correlation with a reference feature is negative."""
    corr = np.corrcoef(series_scores.values, ref_feature.values)[0, 1]
    return -series_scores if np.nan_to_num(corr) < 0 else series_scores

def reconstruction_error_pc1(X: pd.DataFrame, pca: PCA) -> float:
    """
    Reconstruct standardized X using only PC1 and compute MSE.
    Xhat = s1 ⊗ v1    where s1 = X @ v1  and v1 is the first eigenvector
    """
    v1 = pca.components_[0, :]              # shape (n_features,)
    s1 = X.values @ v1                      # shape (n_samples,)
    Xhat = np.outer(s1, v1)                 # (n_samples, n_features)
    return float(np.mean((X.values - Xhat) ** 2))


# -------------------------
# Main functions
# -------------------------

def run_pca_philly_pc1_nomap( # Using PC1 only in our case as it explains >70% of variance
    df2010: pd.DataFrame,
    df2021: pd.DataFrame,
    tract_col: str = "TractNum",
    output_csv: str = "MergedPCA.csv",
    top_n: int = 10,
) -> dict:
    # Summing white-collar industry columns into a single composite variable 
    white_collar = [
        "PercentIndustry_wholesale.trade",
        "PercentIndustry_information",
        "PercentIndustry_finance.and.insurance_real.estate",
        "PercentIndustry_professional_scientific_management_administrative_wastemanagement",
        "PercentIndustry_PublicAdmin",
    ]
    base_vars = ["MedianPropertyValue", "MedianIncome"]
    # 2010 education column is already `PercentEducAttainment_BachelorOrHigher`
    edu_2010 = ["PercentEducAttainment_BachelorOrHigher"]

    need_2010 = [tract_col] + base_vars + white_collar + edu_2010
    need_2021 = [tract_col] + base_vars + white_collar + edu_2010
    check_columns(df2010, need_2010, "df2010")
    check_columns(df21,   need_2021, "df2021")

    df10 = df2010.copy()
    df10["PercentAllIndustry"] = df10[white_collar].sum(axis=1, skipna=True)
    df21["PercentAllIndustry"] = df21[white_collar].sum(axis=1, skipna=True)

    pca_vars = ["MedianPropertyValue", "MedianIncome", "PercentAllIndustry", "PercentEducAttainment_BachelorOrHigher"]

    dat2010 = df10[[tract_col] + pca_vars].dropna(subset=pca_vars).reset_index(drop=True)
    dat2021 = df21[[tract_col] + pca_vars].dropna(subset=pca_vars).reset_index(drop=True)

    # Winsorize transformation and then z-score standardization
    X2010 = dat2010[pca_vars].apply(winsorize_z3).pipe(zscore_df)
    X2021 = dat2021[pca_vars].apply(winsorize_z3).pipe(zscore_df)

    # -------------------------
    # PCA run for 2010 and 2021 years
    # -------------------------
    p10 = pca_fit_scores(X2010)
    p21 = pca_fit_scores(X2021)

    # Variance explained (all PCs)
    vr10 = p10["var_ratio"]; vr21 = p21["var_ratio"]
    print("2010 variance explained:", ", ".join(f"{v*100:.1f}%" for v in vr10))
    print("2021 variance explained:", ", ".join(f"{v*100:.1f}%" for v in vr21))

    # Align PC1 so that higher PC1 proportional to higher income
    pc1_2010 = align_pc1_positive_with(p10["scores"]["PC1"], X2010["MedianIncome"])
    pc1_2021 = align_pc1_positive_with(p21["scores"]["PC1"], X2021["MedianIncome"])

    # Create percentile ranks of PC1 and rank changes
    sc2010 = pd.DataFrame({
        tract_col: dat2010[tract_col].values,
        "PC1_2010": pc1_2010.values,
    })
    sc2010["percent_rank_2010"] = percent_rank_dplyr(sc2010["PC1_2010"])

    sc2021 = pd.DataFrame({
        tract_col: dat2021[tract_col].values,
        "PC1_2021": pc1_2021.values,
    })
    sc2021["percent_rank_2021"] = percent_rank_dplyr(sc2021["PC1_2021"])

    merged = (
        sc2010.merge(sc2021, on=tract_col, how="left")
              .assign(rank_change=lambda d: d["percent_rank_2021"] - d["percent_rank_2010"])
              .sort_values(tract_col, kind="stable")
              .reset_index(drop=True)
    )

    
    merged.to_csv(output_csv, index=False)

    # Print top movers of gentrification according to PCA
    top_inc = merged.nlargest(top_n, "rank_change")[[
        tract_col, "PC1_2010", "percent_rank_2010", "PC1_2021", "percent_rank_2021", "rank_change"
    ]]
    top_dec = merged.nsmallest(top_n, "rank_change")[[
        tract_col, "PC1_2010", "percent_rank_2010", "PC1_2021", "percent_rank_2021", "rank_change"
    ]]

    print("\nTop increases (rank_change):")
    print(top_inc.to_string(index=False))
    print("\nTop decreases (rank_change):")
    print(top_dec.to_string(index=False))

    # -------------------------
    # Diagnostics for PCA performance
    # -------------------------
    # Kaiser rule (eigenvalues > 1) on standardized inputs
    kaiser_2010 = int(np.sum(p10["eigvals"] > 1.0))
    kaiser_2021 = int(np.sum(p21["eigvals"] > 1.0))

    # Communalities for PC1 = squared loadings of PC1
    comm_pc1_2010 = (p10["loadings"]["PC1"] ** 2).rename("communalities_pc1_2010")
    comm_pc1_2021 = (p21["loadings"]["PC1"] ** 2).rename("communalities_pc1_2021")

    # Reconstruction error using only PC1
    rec_err_2010 = reconstruction_error_pc1(X2010, p10["pca"])
    rec_err_2021 = reconstruction_error_pc1(X2021, p21["pca"])

    kmo_bart = {}
    if _HAS_FA:
        chi10, p10_b = calculate_bartlett_sphericity(X2010.values)
        chi21, p21_b = calculate_bartlett_sphericity(X2021.values)
        kmo10_overall, kmo10_per = calculate_kmo(X2010.values)
        kmo21_overall, kmo21_per = calculate_kmo(X2021.values)
        kmo_bart = dict(
            year2010=dict(
                KMO_overall=float(kmo10_overall),
                KMO_per_var=dict(zip(pca_vars, kmo10_per.astype(float))),
                Bartlett_chisq=float(chi10),
                Bartlett_p=float(p10_b),
            ),
            year2021=dict(
                KMO_overall=float(kmo21_overall),
                KMO_per_var=dict(zip(pca_vars, kmo21_per.astype(float))),
                Bartlett_chisq=float(chi21),
                Bartlett_p=float(p21_b),
            )
        )
    else:
        kmo_bart = dict(note="Install `factor-analyzer` to compute KMO and Bartlett tests.")

    # Cross-year PC1 loading correlation (sign-aligned)
    v10 = p10["pca"].components_[0, :]
    v21 = p21["pca"].components_[0, :]
    # align sign
    if np.corrcoef(v10, v21)[0,1] < 0:
        v21 = -v21
    cross_year_loading_corr = float(np.corrcoef(v10, v21)[0,1])

    diagnostics = dict(
        year2010=dict(
            eigenvalues=p10["eigvals"].tolist(),
            variance_proportion=p10["var_ratio"].tolist(),
            kaiser_count=kaiser_2010,
            communalities_pc1=dict(comm_pc1_2010),
            reconstruction_error_pc1=rec_err_2010,
        ),
        year2021=dict(
            eigenvalues=p21["eigvals"].tolist(),
            variance_proportion=p21["var_ratio"].tolist(),
            kaiser_count=kaiser_2021,
            communalities_pc1=dict(comm_pc1_2021),
            reconstruction_error_pc1=rec_err_2021,
        ),
        cross_year=dict(pc1_loading_correlation=cross_year_loading_corr),
    )
    diagnostics.update(kmo_bart)

    return dict(
        merged=merged,
        top_increases=top_inc,
        top_decreases=top_dec,
        variance_2010=vr10,
        variance_2021=vr21,
        diagnostics=diagnostics,
    )


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # df2010 = pd.read_csv("2010Census_data.csv")
    # df2021 = pd.read_csv("2021Census_data.csv")

    # result = run_pca_philly_pc1_nomap(
    #     df2010=df2010,
    #     df2021=df2021,
    #     tract_col="TractNum",
    #     output_csv="MergedPCA_no_neighborhoods.csv",
    #     top_n=10
    # )

    pass
