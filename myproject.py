"""
DataSet title: Title: Wine Quality
This project is going to predict the quality of the wine before a wine taster gives his/her evaluation.
It may help wine manufacturers to develop new wine.

"""
# Import libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# load and organize the data in a pandas data frame format. (red wine and white wine)
redwine_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
whitewine_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
redwine_table = pd.read_csv(redwine_link, ";", header=0)
whitewine_table = pd.read_csv(whitewine_link, ";", header=0)

# Concatenate two datasets vertically (red wine has the same columns as white wine)
wine = pd.concat([redwine_table, whitewine_table])

# Check and make sure the concatenation is correct
assert redwine_table.shape[0] + whitewine_table.shape[0] == wine.shape[0], 'merge error'
assert redwine_table.shape[1] == whitewine_table.shape[1] == wine.shape[1], 'merge error'

# Check how the data is distributed on the combined wine DataFrame
print(f"First 5 rows:\n{wine.head()}")

# Show Information about wine DataFrame (columns and indexes)
print("\nA concise summary of this DataFrame:\n")
wine.info()
# ==> All the data type are numeric numbers. No missing data.

# Show a summary statistics information
print(f"\nThe generate descriptive statistics of whole dataset:\n{wine.describe()}\n")
# ==> mean quality is 5.818378, so we can think the quality over 7 is high quality wine.

# Sort out the wine's quality > 7,8,9 to evaluate their statistics and compare with the whole dataset.
print(f"Statistics for Wine quality 7:\n{wine[wine.quality == 7].describe()}")
print(f"Statistics for Wine quality 8:\n{wine[wine.quality == 8].describe()}")
print(f"Statistics for Wine quality 9:\n{wine[wine.quality == 9].describe()}")
# ==> with the quality is getting higher, the alcohol is higher.

# Show best wine (quality =9)
print(f"Wine quality is 9:\n{wine[wine.quality == 9]}")

# show correlation matrix
pd.set_option('display.max_columns', None)  # to display all the columns in order to show all the correlations.
print("correlation matrix is:\n", wine.corr())
# ==> correlation between quality and alcohol is 0.444319.
# ==> correlation between quality and density is -0.305858.
# ==> correlation between density and residual sugar is 0.552517.
# ==> correlation between alcohol and density is -0.686745.

# Visualize datasets

# newly make a new folder for saving all the plotting charts under this project
os.makedirs('plots/myproject', exist_ok=True)

# show correlation heap map for easy exploring
correlation = wine.corr().round(2)

fig, axes = plt.subplots(1, 1, figsize=(10, 10))
im = axes.imshow(correlation, cmap="coolwarm")
cbar = axes.figure.colorbar(im, ax=axes)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
numrows = len(correlation.iloc[0])
numcolumns = len(correlation.columns)
axes.set_xticks(np.arange(numrows))
axes.set_yticks(np.arange(numcolumns))
axes.set_xticklabels(correlation.columns)
axes.set_yticklabels(correlation.columns)
plt.setp(axes.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
for i in range(numrows):
    for j in range(numcolumns):
        text = axes.text(j, i, correlation.iloc[i, j], ha='center', va='center', color='w', size='medium')
axes.set_title('Heatmap of Correlation of Dimensions', fontdict={'fontsize': 20, 'fontweight': 'bold'})
fig.tight_layout()
plt.savefig(f'plots/myproject/wine_correlation_heatmap.png', format='png')
plt.clf()

# Plotting by some line charts
# The 1st sub line chart to show alcohol & density to quality
sorted_by_quality_df = wine.sort_values('quality')
quality_array = np.unique(sorted_by_quality_df['quality'])
alcohol_list = []
density_list = []

for quality in quality_array:
    alcohol_list.append(sorted_by_quality_df[sorted_by_quality_df['quality'] == quality]['alcohol'].mean())
    density_list.append(sorted_by_quality_df[sorted_by_quality_df['quality'] == quality]['density'].mean())

fig, ax1 = plt.subplots(3, 1, figsize=(21, 21))
ax1[0].plot(quality_array, alcohol_list, 'b.-', label="alcohol")
ax1[0].set_title('Effect of alcohol/density on wine quality', fontdict={'fontsize': 18, 'fontweight': 'bold'})
ax1[0].set_xlabel('quality', fontdict={'fontsize': 16})
ax1[0].set_ylabel('alcohol', fontdict={'fontsize': 16})
ax1[0].set_xticklabels(quality_array)
ax1[0].legend()

# Density is not at the same scale of alcohol, so adding a y-axis label to secondary y-axis
ax2 = ax1[0].twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(quality_array, density_list, 'r.-', label="density")
ax2.set_ylabel("density", fontdict={'fontsize': 16})
ax2.legend()

# Plotting the 2nd sub line chart to show alcohol to density
sorted_by_density_df = wine.sort_values('density')
ax1[1].plot(sorted_by_density_df['density'], sorted_by_density_df['alcohol'], 'b.-', label="alcohol")
ax1[1].set_title('Effect of alcohol on wine density', fontdict={'fontsize': 18, 'fontweight': 'bold'})
ax1[1].set_xlabel('density', fontdict={'fontsize': 16})
ax1[1].set_ylabel('alcohol', fontdict={'fontsize': 16})
ax1[1].legend()

# Plotting the 3rd sub line chart to show residual sugar to density
ax1[2].plot(sorted_by_density_df['density'], sorted_by_density_df['residual sugar'], 'r.-', label="residual sugar")
ax1[2].set_title('Effect of residual sugar on wine density', fontdict={'fontsize': 18, 'fontweight': 'bold'})
ax1[2].set_xlabel('density', fontdict={'fontsize': 16})
ax1[2].set_ylabel('residual sugar', fontdict={'fontsize': 16})
ax1[2].legend()
fig.tight_layout(pad=3.0)
plt.savefig(f'plots/myproject/alcohol_quality_lineChart.png', format='png')
plt.clf()

# Plotting histogram chart
plt.figure(figsize=(12, 8))
sorted_by_quality_df = wine.sort_values('quality')
bins = np.unique(sorted_by_quality_df['quality'])
# display the count over the bar and "+0.4 & +30" is to adjust the coordinates to show text in center of the bar top.
winehist = plt.hist(wine.quality, bins=bins, color='y', rwidth=0.9)
for i in range(len(bins)-1):
    plt.text(winehist[1][i]+0.4, winehist[0][i]+30, str(winehist[0][i]), color='black', size='medium')

plt.xticks(bins)
plt.ylabel('Number of wine', fontdict={'fontsize': 12})
plt.xlabel('wine quality', fontdict={'fontsize': 12})
plt.title('Distribution of wine quality\n', fontdict={'fontsize': 18, 'fontweight': 'bold'})
plt.savefig(f'plots/myproject/wine_quality_histogram.png', format='png')
plt.clf()

# Plotting scatter chart
plt.style.use("bmh")
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
axes.scatter(wine['chlorides'], wine['density'], alpha=0.5, label='density', s=20)
axes.scatter(wine['chlorides'], wine['sulphates'], alpha=0.5, label='sulphates', s=20)
axes.scatter(wine['chlorides'], wine['volatile acidity'], alpha=0.5, label='volatile acidity', s=20)
axes.set_xlabel('chlorides')
axes.set_ylabel('volatile acidity / sulphates / density')
axes.set_title('Chlorides comparison in wine\n', fontdict={'fontsize': 18, 'fontweight': 'bold'})
axes.legend()
plt.savefig(f'plots/myproject/chlorides_comparision_scatter.png', format='png')
plt.clf()

# Plotting pie chart
labels = ['pH<3', '3.5>pH>3', "pH>3.5"]
# google: color picker to get HEX# for customizing colors
colors = ['#abcdef', '#4287f5', '#6f747a']
low_pH = wine.loc[wine['pH'] < 3].count()[0]
medium_pH = wine.loc[(wine['pH'] > 3) & (wine['pH'] < 3.5)].count()[0]
high_pH = wine.loc[wine['pH'] > 3.5].count()[0]

fig, axes = plt.subplots(1, 1, figsize=(8, 8))
axes.pie([low_pH, medium_pH, high_pH], labels=labels, colors=colors, autopct='%.2f %%')
axes.set_title('pH analysis of wine', fontdict={'fontsize': 18, 'fontweight': 'bold'})
axes.legend()
plt.savefig(f'plots/myproject/pH_comparison_pieChart.png', format='png')
plt.clf()
