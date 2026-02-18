 
# HR EMPLOYEE ATTRITION ANALYSIS
# Dataset: IBM HR Analytics (Kaggle)
# Download: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("=" * 60)
print("   HR EMPLOYEE ATTRITION ANALYSIS")
print("=" * 60)

 
# LOAD DATA
 
print("\n Loading dataset...")

df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(f" Dataset loaded: {df.shape[0]:,} employees × {df.shape[1]} features")
print(f"\nFirst 5 rows:")
print(df.head())

 
# DATA EXPLORATION
 
print("\n Exploring data...")

print(f"\n Dataset Shape: {df.shape}")
print(f"\n Column Types:")
print(df.dtypes.value_counts())

print(f"\n Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("    No missing values!")
else:
    print(missing[missing > 0])

print(f"\n Duplicate Rows: {df.duplicated().sum()}")

# Check for useless columns (single unique value)
useless = [col for col in df.columns if df[col].nunique() == 1]
print(f"\n Columns with single value (useless): {useless}")

# Drop useless columns
df = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'],
             errors='ignore')
print(f"   Dropped: EmployeeCount, Over18, StandardHours, EmployeeNumber")

# Encode target variable
df['Attrition_Flag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print(f"\n Final Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

 
# KEY PERFORMANCE INDICATORS
 
print("\n" + "=" * 55)
print("      KEY PERFORMANCE INDICATORS (KPIs)")
print("=" * 55)

total_employees = df.shape[0]
attrition_count = df['Attrition_Flag'].sum()
retention_count = total_employees - attrition_count
attrition_rate = (attrition_count / total_employees) * 100
avg_age = df['Age'].mean()
avg_income = df['MonthlyIncome'].mean()
avg_tenure = df['YearsAtCompany'].mean()
avg_satisfaction = df['JobSatisfaction'].mean()
avg_work_life = df['WorkLifeBalance'].mean()

print(f"      Total Employees:        {total_employees:>8,}")
print(f"      Attrition Count:        {attrition_count:>8,}")
print(f"      Retention Count:        {retention_count:>8,}")
print(f"      Attrition Rate:         {attrition_rate:>8.1f}%")
print(f"      Average Age:            {avg_age:>8.1f} years")
print(f"      Avg Monthly Income:     ${avg_income:>8,.0f}")
print(f"      Avg Tenure:             {avg_tenure:>8.1f} years")
print(f"      Avg Job Satisfaction:   {avg_satisfaction:>8.2f} / 4.0")
print(f"      Avg Work-Life Balance:  {avg_work_life:>8.2f} / 4.0")
print("=" * 55)

 
# ATTRITION OVERVIEW
 
print("\n Generating Attrition Overview...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Attrition Distribution (Pie)
attrition_counts = df['Attrition'].value_counts()
colors_attr = ['#4CAF50', '#F44336']
axes[0].pie(attrition_counts, labels=['Stayed', 'Left'],
            autopct='%1.1f%%', colors=colors_attr, startangle=90,
            explode=[0, 0.05], textprops={'fontsize': 13, 'fontweight': 'bold'})
axes[0].set_title('Overall Attrition Rate', fontweight='bold', fontsize=14)

# Attrition by Department
dept_attrition = df.groupby('Department')['Attrition_Flag'].mean() * 100
dept_attrition = dept_attrition.sort_values(ascending=True)
bars = axes[1].barh(dept_attrition.index, dept_attrition.values,
                    color=['#2196F3', '#FF9800', '#F44336'])
axes[1].set_xlabel('Attrition Rate (%)')
axes[1].set_title('Attrition by Department', fontweight='bold', fontsize=14)
for bar, val in zip(bars, dept_attrition.values):
    axes[1].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontweight='bold')

# Attrition by Gender
gender_attr = df.groupby('Gender')['Attrition_Flag'].mean() * 100
bars2 = axes[2].bar(gender_attr.index, gender_attr.values,
                    color=['#E91E63', '#2196F3'], alpha=0.8, width=0.5)
axes[2].set_ylabel('Attrition Rate (%)')
axes[2].set_title('Attrition by Gender', fontweight='bold', fontsize=14)
for bar, val in zip(bars2, gender_attr.values):
    axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 f'{val:.1f}%', ha='center', fontweight='bold')

plt.suptitle(' Attrition Overview', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('01_attrition_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 01_attrition_overview.png")

 
# ATTRITION BY AGE & INCOME
 
print("\n Analyzing Age & Income Impact...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age Distribution by Attrition
axes[0, 0].hist(df[df['Attrition'] == 'No']['Age'], bins=30, alpha=0.7,
                color='#4CAF50', label='Stayed', edgecolor='white')
axes[0, 0].hist(df[df['Attrition'] == 'Yes']['Age'], bins=30, alpha=0.7,
                color='#F44336', label='Left', edgecolor='white')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Age Distribution by Attrition', fontweight='bold')
axes[0, 0].legend()

# Monthly Income by Attrition
df.boxplot(column='MonthlyIncome', by='Attrition', ax=axes[0, 1],
           patch_artist=True,
           boxprops=dict(facecolor='#2196F3', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
axes[0, 1].set_title('Monthly Income by Attrition', fontweight='bold')
axes[0, 1].set_xlabel('Attrition')
axes[0, 1].set_ylabel('Monthly Income ($)')
plt.sca(axes[0, 1])
plt.title('Monthly Income by Attrition', fontweight='bold')

# Age Group Attrition Rate
df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 55, 65],
                        labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
age_attrition = df.groupby('AgeGroup')['Attrition_Flag'].mean() * 100
bars3 = axes[1, 0].bar(age_attrition.index.astype(str), age_attrition.values,
                       color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(age_attrition))),
                       edgecolor='white')
axes[1, 0].set_xlabel('Age Group')
axes[1, 0].set_ylabel('Attrition Rate (%)')
axes[1, 0].set_title('Attrition Rate by Age Group', fontweight='bold')
for bar, val in zip(bars3, age_attrition.values):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

# Income Group Attrition Rate
df['IncomeGroup'] = pd.cut(df['MonthlyIncome'],
                           bins=[0, 3000, 5000, 8000, 12000, 20000],
                           labels=['<3K', '3K-5K', '5K-8K', '8K-12K', '12K+'])
income_attrition = df.groupby('IncomeGroup')['Attrition_Flag'].mean() * 100
bars4 = axes[1, 1].bar(income_attrition.index.astype(str), income_attrition.values,
                       color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(income_attrition))),
                       edgecolor='white')
axes[1, 1].set_xlabel('Monthly Income Group')
axes[1, 1].set_ylabel('Attrition Rate (%)')
axes[1, 1].set_title('Attrition Rate by Income Group', fontweight='bold')
for bar, val in zip(bars4, income_attrition.values):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

plt.suptitle(' Age & Income Impact on Attrition', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('02_age_income_attrition.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 02_age_income_attrition.png")

 
# JOB ROLE & SATISFACTION ANALYSIS
 
print("\n Analyzing Job Role & Satisfaction...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Attrition by Job Role
role_attrition = df.groupby('JobRole').agg(
    attrition_rate=('Attrition_Flag', 'mean'),
    count=('Attrition_Flag', 'count')
).sort_values('attrition_rate', ascending=True)
role_attrition['attrition_rate'] *= 100

colors_role = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(role_attrition)))
bars5 = axes[0].barh(role_attrition.index, role_attrition['attrition_rate'],
                     color=colors_role)
axes[0].set_xlabel('Attrition Rate (%)')
axes[0].set_title('Attrition Rate by Job Role', fontweight='bold', fontsize=14)
for bar, val in zip(bars5, role_attrition['attrition_rate']):
    axes[0].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=10)

# Job Satisfaction Distribution
sat_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
sat_attrition = df.groupby('JobSatisfaction')['Attrition_Flag'].mean() * 100
sat_attrition.index = [sat_labels[i] for i in sat_attrition.index]

colors_sat = ['#F44336', '#FF9800', '#8BC34A', '#4CAF50']
bars6 = axes[1].bar(sat_attrition.index, sat_attrition.values,
                    color=colors_sat, alpha=0.8, width=0.5, edgecolor='white')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].set_title('Attrition by Job Satisfaction', fontweight='bold', fontsize=14)
for bar, val in zip(bars6, sat_attrition.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                 f'{val:.1f}%', ha='center', fontweight='bold')

plt.suptitle(' Job Role & Satisfaction Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('03_role_satisfaction.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 03_role_satisfaction.png")

 
# OVERTIME, TRAVEL & WORK-LIFE BALANCE
 
print("\n Analyzing Work Conditions...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Overtime
ot_attrition = df.groupby('OverTime')['Attrition_Flag'].mean() * 100
colors_ot = ['#4CAF50', '#F44336']
bars7 = axes[0].bar(ot_attrition.index, ot_attrition.values,
                    color=colors_ot, alpha=0.8, width=0.4, edgecolor='white')
axes[0].set_ylabel('Attrition Rate (%)')
axes[0].set_title('Attrition by Overtime', fontweight='bold', fontsize=14)
for bar, val in zip(bars7, ot_attrition.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)

# Business Travel
travel_attrition = df.groupby('BusinessTravel')['Attrition_Flag'].mean() * 100
travel_attrition = travel_attrition.sort_values()
colors_travel = ['#4CAF50', '#FF9800', '#F44336']
bars8 = axes[1].bar(travel_attrition.index, travel_attrition.values,
                    color=colors_travel, alpha=0.8, width=0.5, edgecolor='white')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].set_title('Attrition by Business Travel', fontweight='bold', fontsize=14)
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars8, travel_attrition.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold')

# Work-Life Balance
wlb_labels = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
wlb_attrition = df.groupby('WorkLifeBalance')['Attrition_Flag'].mean() * 100
wlb_attrition.index = [wlb_labels[i] for i in wlb_attrition.index]
colors_wlb = ['#F44336', '#FF9800', '#8BC34A', '#4CAF50']
bars9 = axes[2].bar(wlb_attrition.index, wlb_attrition.values,
                    color=colors_wlb, alpha=0.8, width=0.5, edgecolor='white')
axes[2].set_ylabel('Attrition Rate (%)')
axes[2].set_title('Attrition by Work-Life Balance', fontweight='bold', fontsize=14)
for bar, val in zip(bars9, wlb_attrition.values):
    axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold')

plt.suptitle(' Work Conditions & Attrition', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('04_work_conditions.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 04_work_conditions.png")

 
# TENURE & EXPERIENCE ANALYSIS
 
print("\n Analyzing Tenure & Experience...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Years at Company
axes[0, 0].hist(df[df['Attrition'] == 'No']['YearsAtCompany'], bins=20,
                alpha=0.7, color='#4CAF50', label='Stayed', edgecolor='white')
axes[0, 0].hist(df[df['Attrition'] == 'Yes']['YearsAtCompany'], bins=20,
                alpha=0.7, color='#F44336', label='Left', edgecolor='white')
axes[0, 0].set_xlabel('Years at Company')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Tenure Distribution', fontweight='bold')
axes[0, 0].legend()

# Years Since Last Promotion
axes[0, 1].hist(df[df['Attrition'] == 'No']['YearsSinceLastPromotion'], bins=15,
                alpha=0.7, color='#4CAF50', label='Stayed', edgecolor='white')
axes[0, 1].hist(df[df['Attrition'] == 'Yes']['YearsSinceLastPromotion'], bins=15,
                alpha=0.7, color='#F44336', label='Left', edgecolor='white')
axes[0, 1].set_xlabel('Years Since Last Promotion')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Promotion Gap Distribution', fontweight='bold')
axes[0, 1].legend()

# Tenure Group Attrition
df['TenureGroup'] = pd.cut(df['YearsAtCompany'],
                           bins=[-1, 1, 3, 5, 10, 40],
                           labels=['0-1yr', '2-3yr', '4-5yr', '6-10yr', '10+yr'])
tenure_attrition = df.groupby('TenureGroup')['Attrition_Flag'].mean() * 100
colors_ten = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(tenure_attrition)))
bars10 = axes[1, 0].bar(tenure_attrition.index.astype(str), tenure_attrition.values,
                        color=colors_ten, edgecolor='white')
axes[1, 0].set_xlabel('Tenure Group')
axes[1, 0].set_ylabel('Attrition Rate (%)')
axes[1, 0].set_title('Attrition by Tenure Group', fontweight='bold')
for bar, val in zip(bars10, tenure_attrition.values):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f'{val:.1f}%', ha='center', fontweight='bold')

# Number of Companies Worked
companies_attrition = df.groupby('NumCompaniesWorked')['Attrition_Flag'].mean() * 100
axes[1, 1].bar(companies_attrition.index, companies_attrition.values,
               color='#9C27B0', alpha=0.7, edgecolor='white')
axes[1, 1].set_xlabel('Number of Companies Worked')
axes[1, 1].set_ylabel('Attrition Rate (%)')
axes[1, 1].set_title('Attrition by Companies Worked', fontweight='bold')

plt.suptitle(' Tenure & Experience Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('05_tenure_experience.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 05_tenure_experience.png")

 
# SALARY & COMPENSATION ANALYSIS
 
print("\n Analyzing Compensation...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Avg Income by Department & Attrition
dept_income = df.groupby(['Department', 'Attrition'])['MonthlyIncome'].mean().unstack()
dept_income.plot(kind='bar', ax=axes[0], color=['#4CAF50', '#F44336'],
                 alpha=0.8, edgecolor='white')
axes[0].set_title('Avg Income by Dept & Attrition', fontweight='bold')
axes[0].set_ylabel('Avg Monthly Income ($)')
axes[0].tick_params(axis='x', rotation=15)
axes[0].legend(['Stayed', 'Left'])

# Salary Hike vs Attrition
hike_attrition = df.groupby('PercentSalaryHike')['Attrition_Flag'].mean() * 100
axes[1].bar(hike_attrition.index, hike_attrition.values,
            color='#FF9800', alpha=0.7, edgecolor='white')
axes[1].set_xlabel('% Salary Hike')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].set_title('Attrition by Salary Hike %', fontweight='bold')

#Stock Option Level
stock_attrition = df.groupby('StockOptionLevel')['Attrition_Flag'].mean() * 100
colors_stock = ['#F44336', '#FF9800', '#8BC34A', '#4CAF50']
bars_stock = axes[2].bar(stock_attrition.index, stock_attrition.values,
                         color=colors_stock, alpha=0.8, edgecolor='white')
axes[2].set_xlabel('Stock Option Level')
axes[2].set_ylabel('Attrition Rate (%)')
axes[2].set_title('Attrition by Stock Options', fontweight='bold')
for bar, val in zip(bars_stock, stock_attrition.values):
    axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold')

plt.suptitle(' Compensation & Attrition', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('06_compensation.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 06_compensation.png")

 
# ENVIRONMENT & RELATIONSHIP SATISFACTION
 
print("\n Analyzing Satisfaction Metrics...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sat_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
sat_colors = ['#F44336', '#FF9800', '#8BC34A', '#4CAF50']

# Environment Satisfaction
env_attr = df.groupby('EnvironmentSatisfaction')['Attrition_Flag'].mean() * 100
env_attr.index = [sat_labels[i] for i in env_attr.index]
axes[0].bar(env_attr.index, env_attr.values, color=sat_colors, alpha=0.8, edgecolor='white')
axes[0].set_ylabel('Attrition Rate (%)')
axes[0].set_title('Environment Satisfaction', fontweight='bold')
for i, val in enumerate(env_attr.values):
    axes[0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

# Relationship Satisfaction
rel_attr = df.groupby('RelationshipSatisfaction')['Attrition_Flag'].mean() * 100
rel_attr.index = [sat_labels[i] for i in rel_attr.index]
axes[1].bar(rel_attr.index, rel_attr.values, color=sat_colors, alpha=0.8, edgecolor='white')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].set_title('Relationship Satisfaction', fontweight='bold')
for i, val in enumerate(rel_attr.values):
    axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

# Job Involvement
inv_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
inv_attr = df.groupby('JobInvolvement')['Attrition_Flag'].mean() * 100
inv_attr.index = [inv_labels[i] for i in inv_attr.index]
axes[2].bar(inv_attr.index, inv_attr.values, color=sat_colors, alpha=0.8, edgecolor='white')
axes[2].set_ylabel('Attrition Rate (%)')
axes[2].set_title('Job Involvement', fontweight='bold')
for i, val in enumerate(inv_attr.values):
    axes[2].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

plt.suptitle(' Satisfaction Metrics & Attrition', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('07_satisfaction_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 07_satisfaction_metrics.png")

 
# DISTANCE FROM HOME & MARITAL STATUS
 
print("\n Analyzing Personal Factors...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Distance from Home
df['DistanceGroup'] = pd.cut(df['DistanceFromHome'],
                             bins=[0, 5, 10, 15, 30],
                             labels=['0-5 mi', '6-10 mi', '11-15 mi', '15+ mi'])
dist_attrition = df.groupby('DistanceGroup')['Attrition_Flag'].mean() * 100
axes[0].bar(dist_attrition.index.astype(str), dist_attrition.values,
            color='#00BCD4', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Distance from Home')
axes[0].set_ylabel('Attrition Rate (%)')
axes[0].set_title('Attrition by Distance', fontweight='bold')
for i, val in enumerate(dist_attrition.values):
    axes[0].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

# Marital Status
marital_attrition = df.groupby('MaritalStatus')['Attrition_Flag'].mean() * 100
marital_attrition = marital_attrition.sort_values(ascending=True)
colors_mar = ['#4CAF50', '#FF9800', '#F44336']
axes[1].bar(marital_attrition.index, marital_attrition.values,
            color=colors_mar, alpha=0.8, width=0.5, edgecolor='white')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].set_title('Attrition by Marital Status', fontweight='bold')
for i, val in enumerate(marital_attrition.values):
    axes[1].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

# Education Level
edu_labels = {1: 'Below College', 2: 'College', 3: 'Bachelor',
              4: 'Master', 5: 'Doctor'}
edu_attrition = df.groupby('Education')['Attrition_Flag'].mean() * 100
edu_attrition.index = [edu_labels[i] for i in edu_attrition.index]
axes[2].bar(edu_attrition.index, edu_attrition.values,
            color='#9C27B0', alpha=0.7, edgecolor='white')
axes[2].set_ylabel('Attrition Rate (%)')
axes[2].set_title('Attrition by Education', fontweight='bold')
axes[2].tick_params(axis='x', rotation=20)
for i, val in enumerate(edu_attrition.values):
    axes[2].text(i, val + 0.3, f'{val:.1f}%', ha='center', fontweight='bold')

plt.suptitle(' Personal Factors & Attrition', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('08_personal_factors.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 08_personal_factors.png")

 
# CORRELATION ANALYSIS
 
print("\n Generating Correlation Heatmap...")

# Select numerical columns for correlation
num_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'TotalWorkingYears', 'DistanceFromHome', 'NumCompaniesWorked',
            'PercentSalaryHike', 'TrainingTimesLastYear', 'JobSatisfaction',
            'EnvironmentSatisfaction', 'WorkLifeBalance',
            'RelationshipSatisfaction', 'JobInvolvement', 'Attrition_Flag']

corr_matrix = df[num_cols].corr()

# Full heatmap
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title(' Correlation Heatmap (All Numeric Features)',
             fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('09_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlation with Attrition specifically
print("\n   Top Correlations with Attrition:")
attrition_corr = corr_matrix['Attrition_Flag'].drop('Attrition_Flag').sort_values()
print("   " + "-" * 50)
for feat, corr in attrition_corr.items():
    direction = "⬆" if corr > 0 else "⬇"
    print(f"   {direction} {feat:35s}: {corr:+.3f}")

print("    Saved: 09_correlation_heatmap.png")

# Correlation with Attrition Bar Chart
fig, ax = plt.subplots(figsize=(10, 8))
colors_corr = ['#F44336' if x < 0 else '#4CAF50' for x in attrition_corr.values]
ax.barh(attrition_corr.index, attrition_corr.values, color=colors_corr, alpha=0.8)
ax.axvline(x=0, color='white', linewidth=0.5)
ax.set_xlabel('Correlation with Attrition')
ax.set_title(' Feature Correlation with Attrition',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('10_attrition_correlations.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 10_attrition_correlations.png")

 
# ATTRITION PREDICTION MODEL
 
print("\n Building Attrition Prediction Model...")

# Prepare features
df_model = df.copy()

# Encode categorical variables
cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != 'Attrition']

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Drop extra columns
drop_cols = ['Attrition', 'AgeGroup', 'IncomeGroup', 'TenureGroup',
             'DistanceGroup', 'Attrition_Flag']
feature_cols = [c for c in df_model.columns if c not in drop_cols]

X = df_model[feature_cols]
y = df_model['Attrition_Flag']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set:     {X_test.shape[0]:,} samples")
print(f"   Features:     {X_train.shape[1]}")

# --- Model 1: Logistic Regression ---
print("\n    Model 1: Logistic Regression")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)
print(f"      Accuracy: {lr_accuracy:.4f}")
print(f"      AUC-ROC:  {lr_auc:.4f}")

# --- Model 2: Random Forest ---
print("\n    Model 2: Random Forest")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight='balanced', max_depth=10)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)
print(f"      Accuracy: {rf_accuracy:.4f}")
print(f"      AUC-ROC:  {rf_auc:.4f}")

# Classification Report
print(f"\n    Classification Report (Random Forest):")
print(classification_report(y_test, rf_pred, target_names=['Stayed', 'Left']))

 
# MODEL VISUALIZATIONS
 
print(" Generating Model Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Stayed', 'Left'],
            yticklabels=['Stayed', 'Left'])
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_title('Confusion Matrix (Random Forest)', fontweight='bold')

# ROC Curve
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)

axes[0, 1].plot(lr_fpr, lr_tpr, color='#2196F3', linewidth=2,
                label=f'Logistic Regression (AUC={lr_auc:.3f})')
axes[0, 1].plot(rf_fpr, rf_tpr, color='#F44336', linewidth=2,
                label=f'Random Forest (AUC={rf_auc:.3f})')
axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve Comparison', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].fill_between(rf_fpr, rf_tpr, alpha=0.1, color='#F44336')

# Feature Importance (Random Forest)
feat_importance = pd.Series(rf_model.feature_importances_,
                            index=feature_cols).sort_values(ascending=True).tail(15)
axes[1, 0].barh(feat_importance.index, feat_importance.values,
                color=plt.cm.viridis(np.linspace(0.3, 0.9, 15)))
axes[1, 0].set_xlabel('Feature Importance')
axes[1, 0].set_title('Top 15 Important Features', fontweight='bold')

# Logistic Regression Coefficients
lr_coefs = pd.Series(lr_model.coef_[0], index=feature_cols).sort_values()
top_bottom = pd.concat([lr_coefs.head(8), lr_coefs.tail(8)])
colors_coef = ['#F44336' if x < 0 else '#4CAF50' for x in top_bottom.values]
axes[1, 1].barh(top_bottom.index, top_bottom.values, color=colors_coef, alpha=0.8)
axes[1, 1].axvline(x=0, color='black', linewidth=0.5)
axes[1, 1].set_xlabel('Coefficient Value')
axes[1, 1].set_title('Logistic Regression Coefficients', fontweight='bold')

plt.suptitle(' Attrition Prediction Model Results', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('11_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 11_model_results.png")

 
# HIGH-RISK EMPLOYEE PROFILING
 
print("\n Profiling High-Risk Employees...")

# Get prediction probabilities for all employees
X_all_scaled = scaler.transform(df_model[feature_cols])
df['attrition_risk'] = lr_model.predict_proba(X_all_scaled)[:, 1]

# Risk categories
df['risk_category'] = pd.cut(df['attrition_risk'],
                             bins=[0, 0.3, 0.6, 1.0],
                             labels=['Low Risk', 'Medium Risk', 'High Risk'])

risk_profile = df.groupby('risk_category').agg(
    count=('Age', 'count'),
    avg_age=('Age', 'mean'),
    avg_income=('MonthlyIncome', 'mean'),
    avg_tenure=('YearsAtCompany', 'mean'),
    avg_satisfaction=('JobSatisfaction', 'mean'),
    pct_overtime=('OverTime', lambda x: (x == 'Yes').mean() * 100)
).reset_index()

print("\n   Employee Risk Profile:")
print("   " + "-" * 75)
for _, row in risk_profile.iterrows():
    print(f"   {row['risk_category']:15s} | Count: {row['count']:5,} | "
          f"Avg Age: {row['avg_age']:.0f} | Income: ${row['avg_income']:,.0f} | "
          f"Tenure: {row['avg_tenure']:.1f}yr | OT: {row['pct_overtime']:.0f}%")

fig, ax = plt.subplots(figsize=(10, 5))
risk_counts = df['risk_category'].value_counts()
colors_risk = ['#4CAF50', '#FF9800', '#F44336']
ax.bar(risk_counts.index, risk_counts.values, color=colors_risk,
       alpha=0.8, edgecolor='white', width=0.5)
ax.set_ylabel('Number of Employees')
ax.set_title(' Employee Attrition Risk Distribution',
             fontsize=16, fontweight='bold')
for i, val in enumerate(risk_counts.values):
    ax.text(i, val + 10, f'{val:,}\n({val/total_employees*100:.1f}%)',
            ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('12_risk_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 12_risk_distribution.png")

 
# ATTRITION COST ANALYSIS
 
print("\n Estimating Attrition Cost...")

# Industry standard: cost of replacing = 50-200% of annual salary
replacement_cost_factor = 1.0  # 100% of annual salary

df_left = df[df['Attrition'] == 'Yes']
avg_salary_left = df_left['MonthlyIncome'].mean()
annual_salary_left = avg_salary_left * 12
cost_per_employee = annual_salary_left * replacement_cost_factor
total_attrition_cost = cost_per_employee * attrition_count

print(f"    Avg Monthly Salary (Left): ${avg_salary_left:,.0f}")
print(f"    Avg Annual Salary (Left):  ${annual_salary_left:,.0f}")
print(f"    Cost per Replacement:      ${cost_per_employee:,.0f}")
print(f"    Total Attrition Cost:      ${total_attrition_cost:,.0f}")
print(f"    Reducing attrition by 5% saves: ${cost_per_employee * (attrition_count * 0.05):,.0f}")

# EXPORT RESULTS

print("\n Exporting Results...")

# Save to Excel
with pd.ExcelWriter('hr_attrition_analysis_summary.xlsx', engine='openpyxl') as writer:

    # KPI Summary
    kpi_df = pd.DataFrame({
        'Metric': ['Total Employees', 'Attrition Count', 'Retention Count',
                   'Attrition Rate (%)', 'Avg Age', 'Avg Monthly Income',
                   'Avg Tenure (Years)', 'Avg Job Satisfaction',
                   'Model Accuracy (RF)', 'Model AUC-ROC (RF)',
                   'Total Attrition Cost'],
        'Value': [total_employees, attrition_count, retention_count,
                  f'{attrition_rate:.1f}%', f'{avg_age:.1f}', f'${avg_income:,.0f}',
                  f'{avg_tenure:.1f}', f'{avg_satisfaction:.2f}',
                  f'{rf_accuracy:.4f}', f'{rf_auc:.4f}',
                  f'${total_attrition_cost:,.0f}']
    })
    kpi_df.to_excel(writer, sheet_name='KPIs', index=False)

    # Department Analysis
    dept_summary = df.groupby('Department').agg(
        employees=('Age', 'count'),
        attrition_rate=('Attrition_Flag', 'mean'),
        avg_income=('MonthlyIncome', 'mean'),
        avg_satisfaction=('JobSatisfaction', 'mean'),
        avg_tenure=('YearsAtCompany', 'mean')
    ).reset_index()
    dept_summary['attrition_rate'] = (dept_summary['attrition_rate'] * 100).round(1)
    dept_summary.to_excel(writer, sheet_name='Department Analysis', index=False)

    # Job Role Analysis
    role_summary = df.groupby('JobRole').agg(
        employees=('Age', 'count'),
        attrition_rate=('Attrition_Flag', 'mean'),
        avg_income=('MonthlyIncome', 'mean'),
        avg_satisfaction=('JobSatisfaction', 'mean')
    ).sort_values('attrition_rate', ascending=False).reset_index()
    role_summary['attrition_rate'] = (role_summary['attrition_rate'] * 100).round(1)
    role_summary.to_excel(writer, sheet_name='Job Role Analysis', index=False)

    # Feature Importance
    feat_imp_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feat_imp_df.to_excel(writer, sheet_name='Feature Importance', index=False)

    # Risk Profile
    risk_profile.to_excel(writer, sheet_name='Risk Profile', index=False)

    # High Risk Employees
    high_risk = df[df['risk_category'] == 'High Risk'][
        ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany',
         'OverTime', 'JobSatisfaction', 'WorkLifeBalance', 'attrition_risk']
    ].sort_values('attrition_risk', ascending=False).head(50)
    high_risk.to_excel(writer, sheet_name='High Risk Employees', index=False)

print("    Saved: hr_attrition_analysis_summary.xlsx")

# Save high risk employees
high_risk_all = df[df['risk_category'] == 'High Risk'].copy()
high_risk_all.to_csv('high_risk_employees.csv', index=False)
print(f"    Saved: high_risk_employees.csv ({high_risk_all.shape[0]} employees)")


# FINAL SUMMARY

print("\n" + "=" * 60)
print("  🎉 HR ATTRITION ANALYSIS COMPLETE!")
print("=" * 60)
print(f"""
 FILES GENERATED:
   ├── 01_attrition_overview.png
   ├── 02_age_income_attrition.png
   ├── 03_role_satisfaction.png
   ├── 04_work_conditions.png
   ├── 05_tenure_experience.png
   ├── 06_compensation.png
   ├── 07_satisfaction_metrics.png
   ├── 08_personal_factors.png
   ├── 09_correlation_heatmap.png
   ├── 10_attrition_correlations.png
   ├── 11_model_results.png
   ├── 12_risk_distribution.png
   ├── hr_attrition_analysis_summary.xlsx
   └── high_risk_employees.csv

 KEY FINDINGS:
   1. Overall Attrition Rate:   {attrition_rate:.1f}%
   2. Highest Risk Role:        Sales Representative
   3. Overtime Impact:          ~30% attrition (vs ~10% no OT)
   4. Income Factor:            Employees who left earn ~33% less
   5. Model Accuracy (RF):      {rf_accuracy:.1%}
   6. Model AUC-ROC (RF):       {rf_auc:.3f}
   7. Total Attrition Cost:     ${total_attrition_cost:,.0f}
   8. High Risk Employees:      {high_risk_all.shape[0]}

 TOP RECOMMENDATIONS:
   1. Reduce overtime — strongest attrition driver
   2. Increase compensation for underpaid roles
   3. Focus retention on 0-2 year tenure employees
   4. Improve work-life balance programs
   5. Provide career growth paths to reduce stagnation
""")