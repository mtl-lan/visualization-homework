#My project: Wine Quality prediction

##Introduction 
This datasets is related to red variants of the Portuguese “Vinho Verde” wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

The input features are as follows:

* fixed acidity - most acids involved with wine or fixed or nonvolatile (do not evaporate readily);
* volatile acidity - the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste;
* citric acid - found in small quantities, citric acid can add ‘freshness’ and flavor to wines;
* residual sugar - the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet;
* chlorides - the amount of salt in the wine;
* free sulfur dioxide - the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine;
* total sulfur dioxide - amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine;
* density - the density of water is close to that of water depending on the percent alcohol and sugar content;
* pH - describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale;
* sulphates - a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidan
* alcohol - the percent alcohol content of the wine;

The output feature is:  
`quality - output variable (based on sensory data, score between 0 and 10);`

###Class7 - Advanced
**The meaningful explorations from different plots:** 
* Histogram and Line plot: showing alcohol/chlorides/citric_acid/density/volatile_acidity are linear-like relations to dependent feature "Quality". Density's value is from 0.992 to 0.995 which has a very low scaling.
* Heatmap: showing only alcohol/chlorides/density/volatile_acidity are more related to dependent feature "Quality" but not any feature has a very high relation to "Quality", the highest is alcohol 0.44. And the 'citric acid' has only 0.086 correlation with 'Quality'. 
* Boxplots: showing there are outliers in all features but there are only few outliers in alcohol & density, chlorides and volatile acidity showing abnormal distribution, we may need log when doing ML. 
* Pairmap and Scatter: showing density & alcohol has linear relation. 
* `Density plots`(most meaningful): showing alcohol, density seems to be good discriminants for the "Quality". And citric acid & total sulfur dioxide are two features omitted because of low correlation to "Quality" but both are good discriminants for the "Quality" showing in density plots. 
There is also an interesting finds: Most of the plots show the ci (confidence interval) of poor quality wine and good quality wine are quite larger than the medium quality wine. I believe that for the sommelier, the quality of the wine is affected by personal preferences, with some uncertainty. 

* 3D plot: approved Density plots. 

##Research Question
Use machine learning to predict wine quality!

##Abstract 

