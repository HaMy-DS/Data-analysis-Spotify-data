# Spotify Data Analysis
## Introduction
In practice, data often faces the problem of imbalanced distribution, where certain target variable values have significantly fewer observations than others. For example, in music, producers may predict if a song will become a hit to adjust its elements. However, hit songs usually have far fewer observations than non-hits, leading to imbalanced data for the hit evaluation variable. Machine Learning and Deep Learning algorithms tend to improve accuracy by minimizing errors, often ignoring the target variable's distribution. This causes models to favor predicting the majority class and misclassify the minority. <br>

Most current methods address imbalance for categorical target variables, with few studies on continuous target variables. In this study, the team applied and combined various methods to handle imbalanced data in the Spotify dataset, where the target variable is continuous. They used traditional resampling techniques like SMOTE, Random Oversampling, and Near Miss, along with modern methods like SMOGN and LDS (Label Distribution Smoothing). After applying Linear Regression and Neural Networks for prediction, they compared the methods’ effectiveness. The findings showed that error rates improved differently across data regions (high, median, low target values), suggesting that the choice of method depends on which data region requires more accurate predictions.

<p align="center">
<img width="600" alt="{DC0227D8-EB25-49B1-9089-5A64FEF9443D}" src="https://github.com/user-attachments/assets/f9d65a36-68c3-46de-a914-b4fd384f38b3">
</p>

## Data Visualization
Data download: https://developer.spotify.com/documentation/web-api/reference/get-audio-features
<p align="center">
<img width="700" alt="{DE2A86F6-C3D6-4553-BD4E-FB8DFE373515}" src="https://github.com/user-attachments/assets/78f171f4-d5af-4fc8-bab9-d645533f7f84">
</p>

<p align="center">
<img width="580" alt="{44FED7AA-D86B-424E-88CF-07F3C98990C1}" src="https://github.com/user-attachments/assets/992d688b-6b65-4b6f-99ad-25c1f8ab1778">
</p>

## Data Analysis
Table of correlation statistics and p-value for each numerical attribute:

| Feature           | Correlation | p-value                      |
|-------------------|-------------|------------------------------|
| acousticness      | -0.370882   | 0                            |
| loudness          | 0.327028    | 0                            |
| energy            | 0.302315    | 0                            |
| instrumentalness  | -0.236487   | 0                            |
| danceability      | 0.1867      | 0                            |
| time_signature    | 0.08675     | 0                            |
| tempo             | 0.07136     | 0                            |
| speechiness       | -0.04735    | 2.067072280290496e-288       |
| valence           | 0.00464     | 0.0003758                    |
| liveness          | -0.04874    | 2.16437141365e-305           |
| duration_ms       | 0.02768     | 8.426127656160351e-100       |

From Table above, the variables ``acousticness``, ``loudness``, and ``energy`` can be selected for the prediction model because they have a correlation at a significant level and a p-value < 0.001, indicating high statistical significance <br>

<p align="center">
<img width="700" alt="{7BF5A1E4-84D9-4E43-8C1F-C6DAD6B4BA3E}" src="https://github.com/user-attachments/assets/b771dc34-92ef-4eea-8794-0f135bbb7e2d">
</p>

| Feature   | F-oneway  | p-value   |
|-----------|-----------|-----------|
| mode      | 665.234   | 1.312e-146|
| explicit  | 27542.2   | 0.0       |
| key       | 335.122   | 0.0       |

The boxplots between the target variable and the variables ``key`` and ``mode`` show a significant amount of overlap. However, the results of the One-way ANOVA analysis yield a large F value and a p-value at a high level of confidence (<0.01), so we cannot conclude that these variables are unimportant for the target variable. Instead, we will retain them for the prediction model. The boxplot of the target variable with the variable ``explicit`` shows less overlap and has a large F-oneway with a high confidence p-value, allowing us to conclude that the explicit variable is important and influences the model.

## Results
To facilitate the evaluation of models on imbalanced data, we divided the target variable space into three regions: the many-shot region (where the target variable has more than 5000 observed samples in the training set), the medium-shot region (with 500 to 5000 observed samples), and the low-shot region (with fewer than 500 observed samples). We report the results based on these regions, as well as the overall training set, using two metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE) for evaluation. In this section, we use the term "VANILLA" to refer to models that do not apply any methods or techniques for imbalanced data, and we do not explicitly mention Random Oversampling, considering it as part of the resampling methods we applied.<br>

<p align="center">
<img width="462" alt="{373034AA-D238-425A-9DED-99044DCC5DD9}" src="https://github.com/user-attachments/assets/7da6e38b-75ca-4bf5-9ccc-c5c58bfbed1d">
</p>


| Metrics                                      | MSE - All | MSE - Many | MSE - Med | MSE - Low | MAE - All | MAE - Many | MAE - Med | MAE - Low |
|----------------------------------------------|-----------|------------|-----------|-----------|-----------|------------|-----------|-----------|
| **VANILLA Linear Regression**                | 272.3     | 204.3     | 732.1     | 1900     | 13.4      | 11.7       | 25.4      | 42.7     |
| **VANILLA Neural Network**                   | 247.13     | 181.9      | 689.1      | 1777      | 12.7      | 11.01      | 24.36      | 41.09     |
| **SMOTE with Linear Regression**             | 478.24    | 503.6     | 270.4     | 713.5     | 17.748    | 18.396     | 12.456    |23.4      |
| **SMOTE with Neural Network**                | 441.89     | 462.7      | 272.8     | 609.9     | 16.913    | 17.4     | 13.025      | 20.8      |
| **SMOTE + NearMiss with Linear Regression**  | 439.032   | 450.7    | 331.8     | 817.3    | 16.8      | 17.066      | 14.375    | 25.2     |
| **SMOTE + NearMiss with Neural Network**     | 397.934     | 403.82     | 333.17     | 842.3     | 16.11    | 16.23      | 14.77      | 25.08     |
| **LDS with Linear Regression**               | 304.882   | 256     | 622.3       | 1782     | 14.34     | 13.06      | 23.151     | 41.4     |
| **LDS with Neural Network**                  | 277.83     | 243.5      | 500.1      | 1329      | 13.51    | 12.61      | 19.62      | 34.6      |
| **Our Best vs VANILLA**                      | +30.7     | +61.6      | -418.7    | -1167     | +0.81     | +1.6       | -11.9     | -20.29    |



<p align="center">
  <img width="1000" alt="{BAB02DDE-82D0-4372-8704-B80009B6719B}" src="https://github.com/user-attachments/assets/f3cd5eae-dc73-403f-b206-e5ea1c2c62ce">
</p>

## References
[3] Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, Dina Katabi, *Delving into Deep Imbalanced Regression*, 2021.

[4] Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, Dina Katabi, *Supplementary Material*, 2021.

[5] Luís Torgo, Bartosz Krawczyk, Paula Branco, Nuno Moniz, *SMOGN: a Preprocessing Approach for Imbalanced Regression*, 2017.

[6] Github Link: [https://github.com/YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression) (accessed on: 14-15/12/2022).

[7] Geeks for Geeks: [https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/](https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/) (accessed on: 14-15/12/2022).


