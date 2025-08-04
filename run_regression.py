import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from io import StringIO


def main():
    #  Load Data 
    df1 = pd.read_csv("df_regression.csv")

    #  Preprocessing 
    df1['pub_year_scaled'] = df1['publication_year'] - df1['publication_year'].min() + 1

    df_disclosed = df1[
        (df1['disclosed_reviewer'] == 1)
        & (df1['total_authors'] > 0)
        & (df1['total_authors'] <= 20)
        & (df1['academic_age'] <= 80)
        & (df1['effective_words'] >= 5)
    ]

    df_anon = df1[
        (df1['disclosed_reviewer'] == 0)
        & (df1['total_authors'] > 0)
        & (df1['total_authors'] <= 20)
        & (df1['academic_age'] <= 80)
        & (df1['effective_words'] >= 5)
    ]

    # Filter full dataset to match disclosed/anon filtering
    df_full = df1[
        (df1['total_authors'] > 0)
        & (df1['total_authors'] <= 20)
        & (df1['academic_age'] <= 80)
        & (df1['effective_words'] >= 5)
    ]

    #  Variable Groups
    tone_vars = [
        'Weighted_Appreciative', 'Weighted_Constructive',
        'Weighted_Questioning', 'Weighted_Critical'
    ]

    base_vars = [
        'gender_male', 'race_white', 'region_west', 'top_100',
        'academic_age', 'avg_c2', 'work_count',
        'source_nature', 'pub_year_scaled',
        'days_received_to_accepted', 'total_authors', 'effective_words'
    ]

    field_vars = [col for col in df1.columns if col.startswith("field_") and col != 'field_Medicine']

    #  Output Path 
    output_folder = "regression_outputs"
    os.makedirs(output_folder, exist_ok=True)

    #  Dataset Dictionary 
    datasets = {
        "disclosed": df_disclosed,
        "anon": df_anon,
        "full": df_full
    }

    #  Run Regressions 
    results_dict = {}
    meta_summary = []
    round_number = "1"

    for group_name, df in datasets.items():
        for dep_var in tone_vars:
            tone_covariates = [t for t in tone_vars if t != dep_var]
            independent_vars = base_vars + field_vars + tone_covariates
            all_vars = independent_vars + [dep_var]

            df_filtered = df[all_vars].dropna()

            y = pd.to_numeric(df_filtered[dep_var], errors='coerce')
            X = df_filtered[independent_vars]
            X = X.astype({col: int for col in X.select_dtypes(bool).columns})
            X = X.apply(pd.to_numeric, errors='coerce')
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit(cov_type='HC1')
            mse = ((model.resid) ** 2).mean()
            rmse = mse ** 0.5

            summary_html = model.summary().tables[1].as_html()
            summary_df = pd.read_html(StringIO(summary_html), header=0, index_col=0)[0]
            summary_df.rename(columns={"z": "robust_z", "P>|z|": "robust_p"}, inplace=True)
            summary_df.columns = [
                col.replace("z", f"{dep_var}_{group_name}_z") if col == "robust_z" else col
                for col in summary_df.columns
            ]

            key = f"{dep_var}_{group_name}"
            results_dict[key] = summary_df

            filename = f"regs_{key}_round{round_number}.csv"
            summary_df.to_csv(os.path.join(output_folder, filename))

            meta_summary.append({
                "Model": key,
                "Round": round_number,
                "Adj_R_squared": round(model.rsquared_adj, 3),
                "F_statistic": round(model.fvalue, 3),
                "N_obs": int(model.nobs),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4)
            })

            print(f"Completed: {key}")

    #  Save Metadata 
    meta_df = pd.DataFrame(meta_summary)
    meta_df.to_csv(os.path.join(output_folder, f"model_summary_metadata_round{round_number}.csv"), index=False)


if __name__ == "__main__":
    main()
    print("Regression analysis completed.")
