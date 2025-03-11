#Please add code here

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
import warnings

def check_normality(data):
    """
    Test for normality using Shapiro-Wilk test
    Returns: bool, dict with test results
    """
    results = {}
    normal = True
    for column in data.columns:
        stat, p_value = stats.shapiro(data[column])
        results[column] = {'statistic': stat, 'p_value': p_value}
        if p_value < 0.05:  # alpha level of 0.05
            normal = False
    return normal, results

def write_results(filename, results_text):
    """Write results to output file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{filename.stem}_results_{timestamp}.txt"
    with open(output_file, 'w') as f:
        f.write(results_text)

def pearson_correlation(file_path):
    """Perform Pearson correlation test"""
    data = pd.read_csv(file_path)
    if len(data.columns) < 2:
        return "Error: Need at least 2 columns for correlation"
    
    normal, norm_results = check_normality(data)
    if not normal:
        warnings.warn("Data may not be normally distributed")
    
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    
    r, p_value = stats.pearsonr(x, y)
    r_squared = r**2
    
    # Calculate confidence interval
    z = np.arctanh(r)
    se = 1/np.sqrt(len(x)-3)
    ci_lower = np.tanh(z - 1.96*se)
    ci_upper = np.tanh(z + 1.96*se)
    
    results = (
        f"Pearson Correlation Results\n"
        f"---------------------------\n"
        f"Correlation coefficient (r): {r:.3f}\n"
        f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
        f"R-squared: {r_squared:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Sample size: {len(x)}\n\n"
        f"Interpretation:\n"
        f"The relationship between {data.columns[0]} and {data.columns[1]} was "
        f"{'strong' if abs(r) > 0.5 else 'moderate' if abs(r) > 0.3 else 'weak'}, "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"and {'positive' if r > 0 else 'negative'} "
        f"(r = {r:.3f} [95% CI {ci_lower:.3f} to {ci_upper:.3f}], P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results)
    return results

def spearman_correlation(file_path):
    """Perform Spearman correlation test"""
    data = pd.read_csv(file_path)
    if len(data.columns) < 2:
        return "Error: Need at least 2 columns for correlation"
    
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    
    rho, p_value = stats.spearmanr(x, y)
    
    results = (
        f"Spearman Correlation Results\n"
        f"----------------------------\n"
        f"Correlation coefficient (ρ): {rho:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Sample size: {len(x)}\n\n"
        f"Interpretation:\n"
        f"The rank correlation between {data.columns[0]} and {data.columns[1]} was "
        f"{'strong' if abs(rho) > 0.5 else 'moderate' if abs(rho) > 0.3 else 'weak'}, "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"and {'positive' if rho > 0 else 'negative'}."
    )
    
    write_results(Path(file_path), results)
    return results

def linear_regression(file_path):
    """Perform bivariate linear regression"""
    data = pd.read_csv(file_path)
    if len(data.columns) < 2:
        return "Error: Need at least 2 columns for regression"
    
    normal, norm_results = check_normality(data)
    if not normal:
        warnings.warn("Data may not be normally distributed")
    
    X = sm.add_constant(data.iloc[:, 0])
    y = data.iloc[:, 1]
    
    model = sm.OLS(y, X).fit()
    
    results = (
        f"Linear Regression Results\n"
        f"-------------------------\n"
        f"{model.summary().as_text()}\n\n"
        f"Interpretation:\n"
        f"R-squared: {model.rsquared:.3f} indicates that {model.rsquared*100:.1f}% "
        f"of the variance in {data.columns[1]} can be explained by {data.columns[0]}."
    )
    
    write_results(Path(file_path), results)
    return results

def independent_ttest(file_path):
    """Perform independent t-test"""
    data = pd.read_csv(file_path)
    group_cols = data['group'].unique()
    if len(group_cols) != 2:
        return "Error: Need exactly 2 groups for independent t-test"
    
    group1 = data[data['group'] == group_cols[0]]['scores']
    group2 = data[data['group'] == group_cols[1]]['scores']
    
    stat, p_value = stats.ttest_ind(group1, group2)
    
    results = (
        f"Independent T-Test Results\n"
        f"--------------------------\n"
        f"t-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Group 1 ({group_cols[0]}) mean: {group1.mean():.2f}\n"
        f"Group 2 ({group_cols[1]}) mean: {group2.mean():.2f}\n"
        f"Sample sizes: n1={len(group1)}, n2={len(group2)}\n\n"
        f"Interpretation:\n"
        f"The difference between groups was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"(t = {stat:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results)
    return results

def paired_ttest(file_path):
    """Perform paired t-test"""
    data = pd.read_csv(file_path)
    before = data['before_treatment']
    after = data['after_treatment']
    
    stat, p_value = stats.ttest_rel(before, after)
    
    results = (
        f"Paired T-Test Results\n"
        f"---------------------\n"
        f"t-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Before mean: {before.mean():.2f}\n"
        f"After mean: {after.mean():.2f}\n"
        f"Sample size: n={len(before)}\n\n"
        f"Interpretation:\n"
        f"The difference between paired measurements was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"(t = {stat:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results)
    return results

def oneway_anova(file_path):
    """Perform one-way ANOVA"""
    data = pd.read_csv(file_path)
    groups = [group for _, group in data.groupby('group')['scores']]
    
    stat, p_value = stats.f_oneway(*groups)
    
    # Post-hoc Tukey test
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey = pairwise_tukeyhsd(data['scores'], data['group'])
    
    results = (
        f"One-way ANOVA Results\n"
        f"---------------------\n"
        f"F-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n\n"
        f"Post-hoc Tukey HSD Test:\n"
        f"{tukey}\n\n"
        f"Interpretation:\n"
        f"The difference between groups was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"(F = {stat:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results)
    return results

def mann_whitney(file_path):
    """Perform Mann-Whitney U test"""
    data = pd.read_csv(file_path)
    group_cols = data['group'].unique()
    
    group1 = data[data['group'] == group_cols[0]]['scores']
    group2 = data[data['group'] == group_cols[1]]['scores']
    
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    results = (
        f"Mann-Whitney U Test Results\n"
        f"--------------------------\n"
        f"U-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Group medians: {group_cols[0]}={group1.median():.2f}, "
        f"{group_cols[1]}={group2.median():.2f}\n\n"
        f"Interpretation:\n"
        f"The difference between groups was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"(U = {stat:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results)
    return results

def wilcoxon(file_path):
    """Perform Wilcoxon signed-rank test"""
    data = pd.read_csv(file_path)
    before = data['before_treatment']
    after = data['after_treatment']
    
    stat, p_value = stats.wilcoxon(before, after)
    
    results = (
        f"Wilcoxon Signed-rank Test Results\n"
        f"--------------------------------\n"
        f"W-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Before median: {before.median():.2f}\n"
        f"After median: {after.median():.2f}\n\n"
        f"Interpretation:\n"
        f"The difference between paired measurements was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"(W = {stat:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results)
    return results

def kruskal_wallis(file_path):
    """Perform Kruskal-Wallis H test"""
    data = pd.read_csv(file_path)
    groups = [group for _, group in data.groupby('group')['scores']]
    
    stat, p_value = stats.kruskal(*groups)
    
    results = (
        f"Kruskal-Wallis H Test Results\n"
        f"-----------------------------\n"
        f"H-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n\n"
        f"Group medians:\n"
        + "\n".join(f"{name}: {group.median():.2f}" 
                   for name, group in data.groupby('group')['scores']) +
        f"\n\nInterpretation:\n"
        f"The difference between groups was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"(H = {stat:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results)
    return results

def friedman(file_path):
    """Perform Friedman test"""
    data = pd.read_csv(file_path)
    
    # Reshape data for Friedman test
    measurements = [data[col] for col in data.columns]
    stat, p_value = stats.friedmanchisquare(*measurements)
    
    results = (
        f"Friedman Test Results\n"
        f"---------------------\n"
        f"Chi-square statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n\n"
        f"Medians:\n"
        + "\n".join(f"{col}: {data[col].median():.2f}" 
                   for col in data.columns) +
        f"\n\nInterpretation:\n"
        f"The difference between repeated measurements was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"(Chi-square = {stat:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results)
    return results

def process_file(file_path):
    """Process file based on filename prefix"""
    file_path = Path(file_path)
    file_name = file_path.name.lower()
    
    test_functions = {
        'pearson': pearson_correlation,
        'spearman': spearman_correlation,
        'regression': linear_regression,
        'ttest_independent': independent_ttest,
        'ttest_paired': paired_ttest,
        'anova': oneway_anova,
        'mannwhitney': mann_whitney,
        'wilcoxon': wilcoxon,
        'kruskal': kruskal_wallis,
        'friedman': friedman
    }
    
    for prefix, func in test_functions.items():
        if file_name.startswith(prefix):
            return func(file_path)
    
    return "Unsupported test type"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = process_file(sys.argv[1])
        print(result)
    else:
        print("Please provide a CSV file path")