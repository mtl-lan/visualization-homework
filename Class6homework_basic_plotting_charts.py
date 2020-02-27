"""
Plotting Charts with MatPlotLib

Use the Matplotlib library to generate simple plots from Datasets One of the main libraries for scientifical data
exploration, allows for quick prototyping and has all basic functionalities Was built with the MATLAB interface as a
reference Compatible with Jupyter notebooks, facilitating it’s usage on interactive python clients Has a very rich
gallery page with tons of examples that can be copied and modified Allows users to save figures in high resolution
for printing and supports many different formats Let’s take a look together at matplotlib in the CEBD-1160-code project

"""

# import libs
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# load the data and read basic info.
insurance = pd.read_csv("insurance.csv")
print(insurance.head())
print(insurance.describe())
print(insurance.columns)

# create a folder to save the charts.
os.makedirs('plots/BasicPlotting', exist_ok=True)

# 1. Still using insurance.csv plot the line chart for charges and save it as charges_plot.png
plt.style.use('ggplot')
plt.figure(figsize=(8, 5))
plt.plot(insurance['charges'])
plt.title('Insurance Charges', fontdict={'fontweight': 'bold', 'fontsize': 18})
plt.xlabel('Payer index')
plt.ylabel('USD')
plt.savefig(f'plots/BasicPlotting/charges_plot.png', format='png')
plt.clf()

# 2. plot the histogram for bmi and save it as bmi_hist.png
plt.figure(figsize=(12, 8))
# display the count over the bar and "+2 & +10" is to adjust the coordinates to show text in center of the bar top.
bmiplot = plt.hist(insurance['bmi'], bins=6, color='g', rwidth=0.9)
for i in range(6):
    plt.text(bmiplot[1][i] + 2, bmiplot[0][i] + 10, str(bmiplot[0][i]), color='black', size='medium')

plt.title('BMI Distribution\n', fontdict={'fontweight': 'bold'})
plt.xlabel('BMI')
plt.ylabel('Number of people')
plt.savefig(f'plots/BasicPlotting/bmi_hist.png', format='png')
plt.clf()

# 3. plot the scatter plot for age vs charges and save it as age_charge_scatter.png
plt.style.use('ggplot')
plt.figure(figsize=(8, 5))
plt.scatter(insurance['age'], insurance['charges'], marker='x')
plt.title('Insurance charges distribution by Age\n', fontdict={'fontweight': 'bold'})
plt.xlabel('age', fontdict={'fontsize': 11})
plt.ylabel('USD', fontdict={'fontsize': 11})
plt.savefig(f'plots/BasicPlotting/age_charge_scatter.png', format='png')
plt.clf()

# 4. Do the plots match what we saw with the correlation function.
# Output from last week:
# The correlation between charges and age is : 0.299008. Yes, it shows +correlation
# The correlation between bmi and children is : 0.012759. No, it shows very low correlation


# Let's do a simple method for plotting heat map.
plt.rcParams["axes.grid"] = False
plt.figure(figsize=(10, 10))
corr = insurance.corr().round(4)
names = ['age', 'bmi', 'Children', 'charges']
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, 4, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, fontdict={'fontweight': 'bold'})
ax.set_yticklabels(names, fontdict={'fontweight': 'bold'})
for i in range(0, 4):
    for j in range(0, 4):
        text = ax.text(j, i, corr.iloc[i, j], ha='center', va='center', color='black', size='medium')
ax.set_title('Heat map of Correlation of Dimensions\n', fontdict={'fontweight': 'bold', 'fontsize': 18})
plt.savefig(f'plots/BasicPlotting/corr_map_simple_version.png', format='png')
plt.clf()

# Here is a more general correlation heat map from our instructor.
plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
correlation = insurance.corr().round(4)
im = axes.imshow(correlation, cmap="coolwarm")
cbar = axes.figure.colorbar(im, ax=axes)
cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
numrows = len(correlation.iloc[0])
numcolumns = len(correlation.columns)
axes.set_xticks(np.arange(numrows))
axes.set_yticks(np.arange(numcolumns))
axes.set_xticklabels(correlation.columns, fontdict={'fontweight': 'bold'})
axes.set_yticklabels(correlation.columns, fontdict={'fontweight': 'bold'})
plt.setp(axes.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
for i in range(numrows):
    for j in range(numcolumns):
        text = axes.text(j, i, correlation.iloc[i, j], ha='center', va='center', color='w', size='large')
axes.set_title('Heat map of Correlation of Dimensions\n', fontdict={'fontweight': 'bold', 'fontsize': 18})
fig.tight_layout()
plt.savefig(f'plots/BasicPlotting/corr_map_general_version.png', format='png')
plt.clf()

# Now let's plot a line chart of charges and age.
sorted_by_age_df = insurance.sort_values('age')
plt.figure(figsize=(8, 5))
plt.plot(sorted_by_age_df['age'], sorted_by_age_df['charges'], label='Charges', color='b')
plt.title('Insurance Charges by age', fontdict={'fontweight': 'bold', 'fontsize': 18})
plt.xlabel('age')
plt.ylabel('USD')
plt.savefig(f'plots/BasicPlotting/charges_by_age.png', format='png')
plt.clf()

# Next let's do a box plot chart to explore age groups.

young = insurance.loc[insurance.age < 30]['charges']
middle_age = insurance.loc[(insurance.age > 30) & (insurance.age < 50)]['charges']
old = insurance.loc[insurance.age > 50]['charges']

plt.figure(figsize=(10, 10))
bp = plt.boxplot([young, middle_age, old], labels=['Young', 'Middle_Age', 'Old'], patch_artist=True)
# change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set(color='#7570b3', linewidth=2)
    # change fill color
    box.set(facecolor='#1b9e77')
# change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)

# change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)

# change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a', linewidth=2)

# change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5)
plt.title('Insurance Charges by ageGroup\n', fontdict={'fontweight': 'bold', 'fontsize': 18})
plt.ylabel('USD')
plt.savefig(f'plots/BasicPlotting/ageGroup_boxplot.png', format='png')
plt.clf()

# Plot a scatter chart to see if BMI is related with children number in family.
plt.style.use('ggplot')
plt.figure(figsize=(8, 5))
plt.scatter(insurance['bmi'], insurance['children'])
plt.title('BMI with Children\n', fontdict={'fontweight': 'bold'})
plt.xlabel('bmi')
plt.ylabel('familiy children numbers')
plt.savefig(f'plots/BasicPlotting/BMI_by_children_scatter.png', format='png')
plt.clf()
