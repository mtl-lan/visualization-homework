"""
Visualizing data with Seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Loading dataset
house = pd.read_csv('housing.data', sep='\s+', header=None)

# Setting columns to dataset
house.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                 'MEDV']

print(house.describe())

# 1. Recreate the previous plots from matplotlib using seaborn:
# line plots -> sns.lineplot
os.makedirs('plots3/lineplot', exist_ok=True)
sns.set()
sns.lineplot('MEDV', 'CRIM', data=house)
plt.title('Crime to Boston house price')
plt.xlabel('price')
plt.ylabel('crime rate')
plt.savefig(f'plots3/lineplot/crime_to_price.png')
plt.close()
# --> Plot is showing the house price is going down with the crime rate of town is decreasing.

# scatter plots -> sns.scatterplot or sns.jointplot
os.makedirs('plots3/joinplot', exist_ok=True)
sns.set()
sns.jointplot('MEDV', 'AGE', data=house)
plt.title('Age to house price')
plt.xlabel('price')
plt.ylabel('age')
plt.savefig(f'plots3/joinplot/age_to_price.png')
plt.close()
# --> With the house price up, the rate of house which was built prior to 1940 is dramatically decreasing.

# histograms -> sns.distplot
os.makedirs('plots3/distplot', exist_ok=True)
sns.set()
sns.distplot(house['MEDV'])
plt.title('Boston house price')
plt.xlabel('price')
plt.savefig(f'plots3/distplot/price_hist.png')
plt.close()

# 2. Choose a categorical variable of your dataset and plot it using the following categorical plots:
# sns.countplot or sns.violinplot.

priceClass = []
for price in house['MEDV']:
    if price <= 20:
        priceClass.append('low')
    elif 35 >= price > 20:
        priceClass.append('mid')
    else:
        priceClass.append('high')
house['priceClass'] = priceClass

os.makedirs('plots3/categorical_plot', exist_ok=True)
sns.set()
sns.countplot('priceClass', data=house, order=('low', 'mid', 'high'))
plt.title('Boston house price class count')
plt.savefig(f'plots3/categorical_plot/houseClass_count.png')
plt.close()

# violinplot
os.makedirs('plots3/categorical_plot', exist_ok=True)

sns.set()
sns.violinplot('priceClass', 'CRIM', data=house)
plt.title('Crime rate VS houseClass')
plt.xlabel('houseClass')
plt.ylabel('Crime rate')
plt.savefig(f'plots3/categorical_plot/violinplot.png')
plt.close()

# 3. Create a correlation heatmap using sns.heatmap. Pass in the df.corr() to see the correlation heatmap for all of yours features!
os.makedirs('plots3/heatmap', exist_ok=True)
sns.set()
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(house.corr(), annot=True)
ax.set_xticklabels(house.columns, rotation=45)
ax.set_yticklabels(house.columns, rotation=45)
ax.set_title('Heatmap\n', fontdict={'fontweight': 'bold', 'fontsize': 18})
plt.savefig('plots3/heatmap/house_correlation.png')
plt.close()

# 4. Create the sns.pairplot for your entire dataset
os.makedirs('plots3/pairmap', exist_ok=True)
sns.pairplot(house)
plt.title('Pair Correlations', fontdict={'fontweight': 'bold', 'fontsize': 18})
plt.savefig('plots3/pairmap/house_pairmap.png')
plt.close()