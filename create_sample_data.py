import pandas as pd
import numpy as np
import os

def create_sample_data(data_folder):
    # Create data directory if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    np.random.seed(42)
    
    # 1. Pearson Correlation Data
    pearson_data = pd.DataFrame({
        'running_speed': np.random.normal(10, 2, 30),
        'joint_pronation': np.random.normal(15, 3, 30)
    })
    pearson_data['joint_pronation'] = pearson_data['running_speed'] * 1.5 + np.random.normal(0, 1, 30)
    pearson_data.to_csv(os.path.join(data_folder, 'pearson_running_study.csv'), index=False)
    
    # 2. Spearman Correlation Data
    spearman_data = pd.DataFrame({
        'age': np.random.randint(20, 70, 40),
        'satisfaction_score': np.random.randint(1, 11, 40)
    })
    spearman_data.to_csv(os.path.join(data_folder, 'spearman_satisfaction.csv'), index=False)
    
    # 3. Linear Regression Data
    regression_data = pd.DataFrame({
        'study_hours': np.random.uniform(1, 10, 50),
        'test_score': np.random.uniform(60, 100, 50)
    })
    regression_data['test_score'] = regression_data['study_hours'] * 5 + np.random.normal(60, 5, 50)
    regression_data.to_csv(os.path.join(data_folder, 'regression_study_scores.csv'), index=False)
    
    # 4. Independent t-test Data
    group1 = np.random.normal(75, 10, 30)
    group2 = np.random.normal(70, 10, 30)
    ttest_data = pd.DataFrame({
        'scores': np.concatenate([group1, group2]),
        'group': ['treatment'] * 30 + ['control'] * 30
    })
    ttest_data.to_csv(os.path.join(data_folder, 'ttest_independent.csv'), index=False)
    
    # 5. Paired t-test Data
    before = np.random.normal(80, 15, 25)
    after = before + np.random.normal(5, 2, 25)  # Slight improvement
    paired_data = pd.DataFrame({
        'before_treatment': before,
        'after_treatment': after
    })
    paired_data.to_csv(os.path.join(data_folder, 'ttest_paired.csv'), index=False)
    
    # 6. One-way ANOVA Data
    group_a = np.random.normal(70, 10, 20)
    group_b = np.random.normal(75, 10, 20)
    group_c = np.random.normal(80, 10, 20)
    anova_data = pd.DataFrame({
        'scores': np.concatenate([group_a, group_b, group_c]),
        'group': ['A'] * 20 + ['B'] * 20 + ['C'] * 20
    })
    anova_data.to_csv(os.path.join(data_folder, 'anova_between.csv'), index=False)
    
    # 7. Mann-Whitney U Data
    group1 = np.random.exponential(scale=2.0, size=25)  # Non-normal distribution
    group2 = np.random.exponential(scale=2.5, size=25)
    mannwhitney_data = pd.DataFrame({
        'scores': np.concatenate([group1, group2]),
        'group': ['treatment'] * 25 + ['control'] * 25
    })
    mannwhitney_data.to_csv(os.path.join(data_folder, 'mannwhitney_test.csv'), index=False)
    
    # 8. Wilcoxon Signed-Rank Data
    before = np.random.exponential(scale=2.0, size=30)
    after = before + np.random.exponential(scale=0.5, size=30)
    wilcoxon_data = pd.DataFrame({
        'before_treatment': before,
        'after_treatment': after
    })
    wilcoxon_data.to_csv(os.path.join(data_folder, 'wilcoxon_test.csv'), index=False)
    
    # 9. Kruskal-Wallis Data
    group_a = np.random.exponential(scale=1.0, size=20)
    group_b = np.random.exponential(scale=1.5, size=20)
    group_c = np.random.exponential(scale=2.0, size=20)
    kruskal_data = pd.DataFrame({
        'scores': np.concatenate([group_a, group_b, group_c]),
        'group': ['A'] * 20 + ['B'] * 20 + ['C'] * 20
    })
    kruskal_data.to_csv(os.path.join(data_folder, 'kruskal_test.csv'), index=False)
    
    # 10. Friedman Test Data
    measurements = pd.DataFrame({
        'time1': np.random.normal(10, 2, 25),
        'time2': np.random.normal(11, 2, 25),
        'time3': np.random.normal(12, 2, 25)
    })
    measurements.to_csv(os.path.join(data_folder, 'friedman_test.csv'), index=False)
    
    print(f"Sample data files have been created successfully in '{data_folder}'!")

if __name__ == "__main__":
    # Example usage with relative path
    create_sample_data('data')
    
    # Example usage with absolute path
    # create_sample_data(r'c:\Vineet_Learning\python-test\data')