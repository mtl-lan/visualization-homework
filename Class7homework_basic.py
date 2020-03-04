"""
DataSet title: Title: Wine Quality
This project is going to predict the quality of the wine before a wine taster gives his/her evaluation.
It may help wine manufacturers to develop new wine.

"""

# Import libs and data preparation
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from mpl_toolkits import mplot3d

# load and organize the data in a pandas data frame format. (red wine and white wine)
redwine_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
whitewine_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
redwine_table = pd.read_csv(redwine_link, ";", header=0)
whitewine_table = pd.read_csv(whitewine_link, ";", header=0)

# Concatenate two datasets vertically (red wine has the same columns as white wine)
wine = pd.concat([redwine_table, whitewine_table])
print(wine.head())

# Create a python script (any name) and use your project dataset to generate many figures (10+) showing data with at
# least *3* features at a time.

# Plot each feature - Histogram/Seaborn
os.makedirs('plots2/histogram_seaborn', exist_ok=True)
sns.set(style="white")
for col in wine.columns:
    sns.distplot(wine[col])
    plt.title(f"{col}'s distribution")
    plt.savefig(f'plots2/histogram_seaborn/{col}_hist.png')
    plt.close()

# Plot each feature vs target "quality" - Histogram/Seaborn
sns.set(style="white")
for col in wine.columns:
    sns.barplot(wine['quality'], wine[col], data=wine)
    plt.title(f"{col} by quality")
    plt.savefig(f'plots2/histogram_seaborn/{col} & quality_hist.png')
    plt.close()

# Plot the Heatmap/Seaborn
os.makedirs('plots2/heatmap', exist_ok=True)
sns.set()
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(wine.corr(), annot=True)
ax.set_xticklabels(wine.columns, rotation=45)
ax.set_yticklabels(wine.columns, rotation=45)
ax.set_title('Heatmap\n', fontdict={'fontweight': 'bold', 'fontsize': 18})
plt.savefig('plots2/heatmap/wine_correlation.png')
plt.close()

# Plot the Pair plot/Seaborn
os.makedirs('plots2/pairmap', exist_ok=True)
sns.pairplot(wine)
plt.title('Pair Correlations', fontdict={'fontweight': 'bold', 'fontsize': 18})
plt.savefig('plots2/pairmap/wine_pairmap.png')
plt.close()

# Plot Line plots/Seaborn to see relation between each independent feature with dependent feature 'Quality'
os.makedirs('plots2/line_charts', exist_ok=True)
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))
ax = ax.flatten()
index = 0
sorted_by_quality_df = wine.sort_values('quality')
for col in wine.columns:
    if col != 'quality':
        sns.lineplot(x='quality', y=col, data=sorted_by_quality_df, ax=ax[index])
        plt.title(f'{col} to quality')
        plt.xlabel('quality')
        plt.ylabel(f'{col}')
        index += 1

plt.tight_layout(pad=0.4)
plt.savefig(f'plots2/line_charts/features_vs_quality.png', format='png')
plt.clf()

# Plotting Box plots/Seaborn to see if there are any outliers in our data (considering data between 25th and 75th
# percentile as non outlier)
os.makedirs('plots2/boxplots', exist_ok=True)
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))
ax = ax.flatten()
index = 0
for column in wine.columns:
    if column != 'quality':
        sns.boxplot(y=column, data=wine, ax=ax[index])
        plt.title(f'{col} to quality')
        plt.xlabel('quality')
        plt.ylabel(f'{col}')
        index += 1
plt.tight_layout(pad=0.4)
plt.savefig(f'plots2/boxplots/outliers_check.png', format='png')
plt.close()
# # ==> From the above box plots we can clearly see that there are outliers in all features.

# Features Density plots/Seaborn grouped by quality class
os.makedirs('plots2/quality_comparison', exist_ok=True)
print(wine['quality'].value_counts())
# define 3,4,5-> bad quality; 6,7->medium quality, 8,9 -> good quality
bins = [3, 5, 7, 9]
labels = ['Bad', 'Medium', 'Good']
wine['quality_desc'] = pd.cut(wine['quality'], bins=bins, labels=labels)
print(wine['quality_desc'].value_counts())
col_names = wine.drop('quality_desc', axis=1).columns.tolist()

plt.figure(figsize=(15, 10))
i = 0
for col in col_names:
    if col != 'quality':
        plt.subplot(3, 4, i + 1)
        plt.grid(True, alpha=0.5)
        sns.kdeplot(wine[col][wine['quality_desc'] == 'Bad'], label='Bad Quality')
        sns.kdeplot(wine[col][wine['quality_desc'] == 'Medium'], label='Medium Quality')
        sns.kdeplot(wine[col][wine['quality_desc'] == 'Good'], label='Good Quality')
        plt.title(col + ' vs Quality', size=15)
        plt.xlabel(col, size=12)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        i += 1
plt.savefig(f'plots2/quality_comparison/Density_groupBy_quality.png', format='png')
plt.clf()

# Based on above plots,  we may noticed that alcohol, chlorides, volatile acidity, density are the features more
# related to target 'quality'.

# Plot 'volatile acidity', 'chlorides', 'density' correlation with 'alcohol'- Scatter plot/Seaborn
os.makedirs('plots2/multi_scatter', exist_ok=True)
sns.set(style="white")
features = ['volatile acidity', 'chlorides', 'density']
for feature in features:
    sns.relplot(x="alcohol", y=feature, hue="quality", sizes=(40, 400), alpha=0.5,
                height=6, data=wine)
    plt.title(f'{feature} to alcohol')
    plt.xlabel('alcohol')
    plt.ylabel(f'{feature}')
    plt.savefig(f'plots2/multi_scatter/alcohol_{feature}_scatter.png', dpi=300)
    plt.tight_layout()
    plt.close()

# Plotting main features correlation with target "quality"- Violin plot/seaborn
os.makedirs('plots2/violinplot', exist_ok=True)
sns.set(style="white")
features = ['alcohol', 'volatile acidity', 'chlorides', 'density']

for feature in features:
    sns.violinplot(x="quality", y=feature, data=wine)
    plt.title(f'{feature} to quality')
    plt.xlabel('quality')
    plt.ylabel(f'{feature}')
    plt.savefig(f'plots2/violinplot/quality_{feature}_violinplot.png', dpi=300)
    plt.close()

# 3D visualizations
os.makedirs('plots2/3D', exist_ok=True)

bad_wine = wine.loc[wine['quality'] < 5]
medium_wine = wine.loc[(wine['quality'] > 5) & (wine['quality'] < 7)]
good_wine = wine.loc[wine['quality'] > 7]

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
line1 = axes.scatter(bad_wine['quality'], bad_wine['alcohol'], bad_wine['volatile acidity'])
line2 = axes.scatter(medium_wine['quality'], medium_wine['alcohol'], medium_wine['volatile acidity'])
line3 = axes.scatter(good_wine['quality'], good_wine['alcohol'], good_wine['volatile acidity'])
axes.legend((line1, line2, line3), ('Bad_wine', 'Medium_wine', 'Good_wine'))
axes.set_title('Wine Quality Distribution - 3D')
axes.set_xlabel('quality')
axes.set_ylabel('alcohol')
axes.set_zlabel('volatile acidity')
plt.savefig(f'plots2/3D/quality_class_3d.png', dpi=300)
plt.close()
