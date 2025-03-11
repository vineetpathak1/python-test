import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime
import warnings
import os

# Define default results folder
DEFAULT_RESULTS_FOLDER = Path("c:/Vineet_Learning/python-test/results")

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

def write_results(filename, results_text, output_folder=None):
    """Write results to output file
    Args:
        filename: Path object of input file
        results_text: Text content to write
        output_folder: Optional custom output folder path
    """
    if output_folder is None:
        output_folder = DEFAULT_RESULTS_FOLDER
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f"{filename.stem}_results_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write(results_text)

def pearson_correlation(file_path, output_folder=None):
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
    
    write_results(Path(file_path), results, output_folder)
    return results

def spearman_correlation(file_path, output_folder=None):
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
    
    write_results(Path(file_path), results, output_folder)
    return results

def linear_regression(file_path, output_folder=None):
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
    
    write_results(Path(file_path), results, output_folder)
    return results

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    d = abs(d)
    if d < 0.2: return "trivial"
    elif d < 0.5: return "small"
    elif d < 0.8: return "moderate"
    else: return "large"

def interpret_nonparametric_r(r):
    """Interpret non-parametric r effect size"""
    r = abs(r)
    if r < 0.1: return "trivial"
    elif r < 0.3: return "small"
    elif r < 0.5: return "moderate"
    else: return "large"

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d for independent samples"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return abs(np.mean(group1) - np.mean(group2)) / pooled_se

def calculate_cohens_d_paired(diff):
    """Calculate Cohen's d for paired samples"""
    d = np.mean(diff) / np.std(diff, ddof=1)
    return abs(d)

def calculate_nonparametric_r(statistic, n):
    """Calculate non-parametric r effect size"""
    return abs(statistic) / np.sqrt(n)

def independent_ttest(file_path, output_folder=None):
    """Perform independent t-test"""
    data = pd.read_csv(file_path)
    group_cols = data['group'].unique()
    if len(group_cols) != 2:
        return "Error: Need exactly 2 groups for independent t-test"
    
    group1 = data[data['group'] == group_cols[0]]['scores']
    group2 = data[data['group'] == group_cols[1]]['scores']
    
    stat, p_value = stats.ttest_ind(group1, group2)
    d = calculate_cohens_d(group1, group2)
    effect_size_interp = interpret_cohens_d(d)
    
    results = (
        f"Independent T-Test Results\n"
        f"--------------------------\n"
        f"t-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Effect size (Cohen's d): {d:.3f} ({effect_size_interp} effect)\n"
        f"Group 1 ({group_cols[0]}) mean: {group1.mean():.2f}\n"
        f"Group 2 ({group_cols[1]}) mean: {group2.mean():.2f}\n"
        f"Sample sizes: n1={len(group1)}, n2={len(group2)}\n\n"
        f"Interpretation:\n"
        f"The difference between groups was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"with a {effect_size_interp} effect size "
        f"(t = {stat:.3f}, d = {d:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results, output_folder)
    return results

def paired_ttest(file_path, output_folder=None):
    """Perform paired t-test"""
    data = pd.read_csv(file_path)
    before = data['before_treatment']
    after = data['after_treatment']
    diff = after - before
    
    stat, p_value = stats.ttest_rel(before, after)
    d = calculate_cohens_d_paired(diff)
    effect_size_interp = interpret_cohens_d(d)
    
    results = (
        f"Paired T-Test Results\n"
        f"---------------------\n"
        f"t-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Effect size (Cohen's d): {d:.3f} ({effect_size_interp} effect)\n"
        f"Before mean: {before.mean():.2f}\n"
        f"After mean: {after.mean():.2f}\n"
        f"Sample size: n={len(before)}\n\n"
        f"Interpretation:\n"
        f"The difference between paired measurements was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"with a {effect_size_interp} effect size "
        f"(t = {stat:.3f}, d = {d:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results, output_folder)
    return results

def mann_whitney(file_path, output_folder=None):
    """Perform Mann-Whitney U test"""
    data = pd.read_csv(file_path)
    group_cols = data['group'].unique()
    
    if len(group_cols) != 2:
        return "Error: Need exactly 2 groups for Mann-Whitney U test"
    
    group1 = data[data['group'] == group_cols[0]]['scores']
    group2 = data[data['group'] == group_cols[1]]['scores']
    
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    r = calculate_nonparametric_r(stat, len(group1) + len(group2))
    effect_size_interp = interpret_nonparametric_r(r)
    
    results = (
        f"Mann-Whitney U Test Results\n"
        f"--------------------------\n"
        f"U-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Effect size (r): {r:.3f} ({effect_size_interp} effect)\n"
        f"Group medians: {group_cols[0]}={group1.median():.2f}, "
        f"{group_cols[1]}={group2.median():.2f}\n\n"
        f"Interpretation:\n"
        f"The difference between groups was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"with a {effect_size_interp} effect size "
        f"(U = {stat:.3f}, r = {r:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results, output_folder)
    return results

def wilcoxon(file_path, output_folder=None):
    """Perform Wilcoxon signed-rank test"""
    data = pd.read_csv(file_path)
    before = data['before_treatment']
    after = data['after_treatment']
    
    stat, p_value = stats.wilcoxon(before, after)
    r = calculate_nonparametric_r(stat, len(before))
    effect_size_interp = interpret_nonparametric_r(r)
    
    results = (
        f"Wilcoxon Signed-rank Test Results\n"
        f"--------------------------------\n"
        f"W-statistic: {stat:.3f}\n"
        f"p-value: {p_value:.4f}\n"
        f"Effect size (r): {r:.3f} ({effect_size_interp} effect)\n"
        f"Before median: {before.median():.2f}\n"
        f"After median: {after.median():.2f}\n\n"
        f"Interpretation:\n"
        f"The difference between paired measurements was "
        f"{'significant' if p_value < 0.05 else 'not significant'} "
        f"with a {effect_size_interp} effect size "
        f"(W = {stat:.3f}, r = {r:.3f}, P={p_value:.3f})."
    )
    
    write_results(Path(file_path), results, output_folder)
    return results

def oneway_anova(file_path, output_folder=None):
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
    
    write_results(Path(file_path), results, output_folder)
    return results

def kruskal_wallis(file_path, output_folder=None):
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
    
    write_results(Path(file_path), results, output_folder)
    return results

def friedman(file_path, output_folder=None):
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
    
    write_results(Path(file_path), results, output_folder)
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

def process_folder(folder_path):
    """Process all CSV files in the given folder"""
    folder = Path(folder_path)
    if not folder.is_dir():
        return "Error: Not a valid folder path"
    
    results = []
    for csv_file in folder.glob("*.csv"):
        try:
            result = process_file(csv_file)
            results.append(f"Results for {csv_file.name}:\n{result}\n")
        except Exception as e:
            results.append(f"Error processing {csv_file.name}: {str(e)}\n")
    
    return "\n".join(results) if results else "No CSV files found in the folder"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_dir():
            result = process_folder(path)
        else:
            result = process_file(path)
        print(result)
    else:
        print("Please provide a CSV file path or folder path")