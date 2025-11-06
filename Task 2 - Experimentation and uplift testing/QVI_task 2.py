import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#### Point the filePath to where you have downloaded the datasets to and 
#### assign the data files to data.tables
data = pd.read_csv('QVI_data.csv')
print(data)

#### Set themes for plots
plt.style.use("ggplot")

## Select control stores
# The client has selected store numbers 77, 86 and 88 as trial stores and want control stores to be established stores that are operational for the entire observation period.
# We would want to match trial stores to control stores that are similar to the trialstore prior to the trial period of Feb 2019 in terms of :
    # - Monthly overall sales revenue
    # - Monthly number of customers
    # - Monthly number of transactions per customer
# Let's first create the metrics of interest and filter to stores that are present throughout the pre-trial period.

#### Calculate these measures over time for each store 
data['DATE'] = pd.to_datetime(data['DATE'])
data['YEAR_MONTH'] = data['DATE'].dt.strftime("%Y-%m")
print(data['YEAR_MONTH'])

#### Next, we define the measure calculations to use during the analysis.
measure_sorted = data.groupby(['YEAR_MONTH', 'STORE_NBR']).nunique()

totSales = measure_sorted.groupby(['YEAR_MONTH', 'STORE_NBR'])['TOT_SALES'].sum()

nCustomer = measure_sorted.groupby(['YEAR_MONTH', 'STORE_NBR'])['LYLTY_CARD_NBR'].sum().rename('nCustomer')

nTxnPerCust = measure_sorted.groupby(['YEAR_MONTH', 'STORE_NBR','LYLTY_CARD_NBR'])['TXN_ID'].count().rename('nTxnPerCust')

nChipsPerTXN = measure_sorted.groupby(['YEAR_MONTH', 'STORE_NBR','TXN_ID'])['PROD_QTY'].sum().rename('nChipsPerTXN')

measure_sorted1 = data.groupby(['YEAR_MONTH', 'STORE_NBR']).agg(
    totSales=('TOT_SALES', 'sum'),  
    totalQty=('PROD_QTY', 'sum'))
avgPricePerUnit = (measure_sorted1['totSales'] / measure_sorted1['totalQty']).rename('avgPricePerUnit')

print(totSales)
print(nCustomer) 
print(nTxnPerCust)
print(nChipsPerTXN)
print(avgPricePerUnit)

measure_total = pd.merge(totSales, nCustomer, on=['YEAR_MONTH', 'STORE_NBR'], how = 'outer')
measure_total1 = pd.merge(measure_total, nTxnPerCust, on=['YEAR_MONTH', 'STORE_NBR'], how = 'outer')
measure_total2 = pd.merge(measure_total1, nChipsPerTXN, on=['YEAR_MONTH', 'STORE_NBR'], how = 'outer')
measureOverTime = pd.merge(measure_total2, avgPricePerUnit, on=['YEAR_MONTH', 'STORE_NBR'], how = 'outer').reset_index()
print(measureOverTime)

#### Filter to the pre-trial period and stores with full observation periods
storesWithFullObs = measureOverTime.groupby('STORE_NBR')['YEAR_MONTH'].nunique()
storesWithFullObs = storesWithFullObs[storesWithFullObs == 12].index

preTrialMeasures = measureOverTime[(measureOverTime['YEAR_MONTH'] < '2019-02') & (measureOverTime['STORE_NBR'].isin(storesWithFullObs))]
print(preTrialMeasures)

# Hàm tính toán mối tương quan giữa các cửa hàng
def calculateCorrelation(input_table, metric_col, store_comparison):
    # Tạo DataFrame rỗng để lưu kết quả
    calc_corr_table = pd.DataFrame(columns=['Store1', 'Store2', 'corr_measure'])
    
    # Lấy danh sách các cửa hàng duy nhất
    store_numbers = input_table['STORE_NBR'].unique()
    
    # Vòng lặp qua từng cửa hàng
    for store in store_numbers:
        # Tính hệ số tương quan giữa cửa hàng so sánh và cửa hàng hiện tại
        store_comparison_data = input_table[input_table['STORE_NBR'] == store_comparison][metric_col]
        store_data = input_table[input_table['STORE_NBR'] == store][metric_col]
        
        # Kiểm tra nếu có đủ dữ liệu để tính tương quan
        if len(store_comparison_data) > 1 and len(store_data) > 1:
            corr_measure = np.corrcoef(store_comparison_data, store_data)[0, 1]
        else:
            corr_measure = np.nan  # Gán giá trị NaN nếu không đủ dữ liệu
        
        # Tạo một hàng mới với kết quả tính toán và thêm vào bảng kết quả
        calculated_measure = pd.DataFrame({
            'Store1': [store_comparison],
            'Store2': [store],
            'corr_measure': [corr_measure]
        })
        
        # Append the result to the calc_corr_table
        calculated_measure = calculated_measure.dropna(axis=1, how='all')  # Loại bỏ cột chứa toàn bộ NaN
        calc_corr_table = pd.concat([calc_corr_table, calculated_measure], ignore_index=True)
    return calc_corr_table

#### Create a function to calculate a standardised magnitude distance for a measure, 
#### looping through each control store 
def calculate_magnitude_distance(input_table, metric_col, store_comparison):
    # Tạo bảng để lưu kết quả
    calc_dist_table = pd.DataFrame(columns=["Store1", "Store2", "YEAR_MONTH", "measure"])
    
    # Lấy danh sách các cửa hàng độc nhất
    store_numbers = input_table['STORE_NBR'].unique()
    
    # Lặp qua từng cửa hàng trong danh sách
    for store in store_numbers:
        # Lọc các dòng liên quan đến storeComparison và store hiện tại
        store_comparison_data = input_table[input_table['STORE_NBR'] == store_comparison]
        store_data = input_table[input_table['STORE_NBR'] == store]
        
        # Tính toán sự khác biệt tuyệt đối cho từng YEARMONTH
        for _, row in store_comparison_data.iterrows():
            yearmonth = row['YEAR_MONTH']
            
            # Tìm giá trị metric của cửa hàng hiện tại cho cùng YEARMONTH
            store_metric_value = store_data[store_data['YEAR_MONTH'] == yearmonth][metric_col]
            if not store_metric_value.empty:
                measure = abs(row[metric_col] - store_metric_value.values[0])
                
                # Thêm kết quả vào bảng calc_dist_table
                new_row = pd.DataFrame({
                    "Store1": [store_comparison],
                    "Store2": [store],
                    "YEAR_MONTH": [yearmonth],
                    "measure": [measure]
                })
                
                calc_dist_table = pd.concat([calc_dist_table, new_row], ignore_index=True)    
    return calc_dist_table

def minMaxDist(input_table):
    min_max_dist = input_table.groupby(['Store1', 'YEAR_MONTH']).agg(
        minDist=('measure', 'min'),
        maxDist=('measure', 'max')
    ).reset_index()

    dist_table = pd.merge(input_table, min_max_dist, on=['Store1', 'YEAR_MONTH'])

    dist_table['magnitudeMeasure'] = 1 - (dist_table['measure'] - dist_table['minDist']) / (dist_table['maxDist'] - dist_table['minDist'])

    final_dist_table = dist_table.groupby(['Store1', 'Store2']).agg(
        mag_measure=('magnitudeMeasure', 'mean')
    ).reset_index()

    return final_dist_table

## Trial store 77
trial_store_77 = 77
corr_nSales_77 = calculateCorrelation(preTrialMeasures, 'TOT_SALES', trial_store_77)
corr_nSales_77_sorted = corr_nSales_77.sort_values(by='corr_measure', ascending=False)
print(corr_nSales_77_sorted)
corr_nCustomers_77 = calculateCorrelation(preTrialMeasures, 'nCustomer', trial_store_77)
corr_nCustomers_77_sorted = corr_nCustomers_77.sort_values(by = 'corr_measure', ascending=False)
print(corr_nCustomers_77_sorted)
#### Then, use the functions for calculating magnitude.
magnitude_nSales_77 = calculate_magnitude_distance(preTrialMeasures, 'TOT_SALES', trial_store_77)
Standardise_magnitude_nSale_77 = minMaxDist(magnitude_nSales_77)
magnitude_nSales_77_sorted = Standardise_magnitude_nSale_77.sort_values(by = 'mag_measure', ascending= True)
print(magnitude_nSales_77_sorted)

magnitude_nCustomers_77 = calculate_magnitude_distance(preTrialMeasures, 'nCustomer', trial_store_77)
Standardise_magnitude_nCustomers_77 = minMaxDist(magnitude_nCustomers_77)
magnitude_nCustomers_77_sorted = Standardise_magnitude_nCustomers_77.sort_values(by = 'mag_measure', ascending= True)
print(magnitude_nCustomers_77_sorted)

merged_nsales_77 = pd.merge(corr_nSales_77_sorted, magnitude_nSales_77_sorted, how='outer')
print(merged_nsales_77)

merged_nCustomers_77 = pd.merge(corr_nCustomers_77_sorted, magnitude_nCustomers_77_sorted, how='outer')
print(merged_nCustomers_77)

### Create a combined score composed of correlation and magnitude, by
#### first merging the correlations table with the magnitude table.
#### A simple average on the scores: 0.5 * corr_measure + 0.5 * mag_measure

corr_weight = 0.5
score_nSales_77 = ((merged_nsales_77['corr_measure'] + merged_nsales_77['mag_measure']) / 2).rename('score_nSales')
score_nsales_77_sorted = pd.concat([merged_nsales_77, score_nSales_77], axis=1)
print(score_nsales_77_sorted)

score_nCustomer_77 = ((merged_nCustomers_77['corr_measure'] + merged_nCustomers_77['mag_measure']) / 2).rename('score_nCustomer')
score_nCustomer_77_sorted = pd.concat([merged_nCustomers_77, score_nCustomer_77], axis=1)
print(score_nCustomer_77_sorted)

#### Combine scores across the drivers by first merging our sales scores and customer scores into a single table
merged_score_control77 = pd.merge(score_nsales_77_sorted, score_nCustomer_77_sorted, on = ['Store1', 'Store2'])
finalControlScore77 =  (score_nSales_77 * 0.5 + score_nCustomer_77 * 0.5).rename('finalControlScore')
score_control77 = pd.concat([merged_score_control77, finalControlScore77], axis = 1)
print(score_control77)

#### Select control stores based on the highest matching store (closest to 1 but
#### not the store itself, i.e. the second ranked highest store)
#### Select the most appropriate control store for trial store 77 by finding the store with the highest
filtered_score_control77 = score_control77[score_control77['Store1'] == trial_store_77]
sorted_score_control77 = filtered_score_control77.sort_values(by = 'finalControlScore', ascending = False)
control_store77 = sorted_score_control77.iloc[1]['Store2']
print(control_store77)

#### Visual checks on trends based on the drivers
measureOverTimeSales77 = measureOverTime.copy()
measureOverTimeSales77['Store_type'] = measureOverTimeSales77['STORE_NBR'].apply(
    lambda x: 'Trial' if x == trial_store_77 else ('Control' if x == control_store77 else 'Other stores'))

pastSales77 = measureOverTimeSales77.groupby(['YEAR_MONTH', 'Store_type'], as_index=False)['TOT_SALES'].mean()
pastSales77['TransactionMonth'] = pd.to_datetime(pastSales77['YEAR_MONTH'], format='%Y-%m')
pastSales77 = pastSales77[pastSales77['YEAR_MONTH'] < '2019-03']
plt.figure(figsize=(10, 6))
for store_type, group_data in pastSales77.groupby('Store_type'):
    plt.plot(group_data['TransactionMonth'], group_data['TOT_SALES'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title="Store type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#### Conduct visual checks on customer count trends by comparing the trial store
#### to the control store and other stores.
measureOverTimeCusts77 = measureOverTime.copy()
measureOverTimeCusts77['Store_type'] = measureOverTimeCusts77['STORE_NBR'].apply(
    lambda x: 'Trial' if x == trial_store_77 else ('Control' if x == control_store77 else 'Other stores'))
numberCustomers77 = measureOverTimeCusts77.groupby(['YEAR_MONTH', 'Store_type'], as_index=False)['nCustomer'].mean()
numberCustomers77['TransactionMonth'] = pd.to_datetime(numberCustomers77['YEAR_MONTH'], format='%Y-%m')
numberCustomers77 = numberCustomers77[numberCustomers77['YEAR_MONTH'] < '2019-03']
plt.figure(figsize=(10, 6))
for store_type, group_data in numberCustomers77.groupby('Store_type'):
    plt.plot(group_data['TransactionMonth'], group_data['nCustomer'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total number of customers")
plt.title("Total number of customers by month")
plt.legend(title="Store type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#### Scale pre-trial control sales to match pre-trial trial store sales
scalingFactorForTrialStore77 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == trial_store_77) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['TOT_SALES'].sum()
scalingFactorForControlStore77 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == control_store77) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['TOT_SALES'].sum()
scalingFactorForControlSales77 = scalingFactorForTrialStore77 / scalingFactorForControlStore77

#### Apply the scaling factor
measureOverTimeSales77 = measureOverTime.copy()
scaledControlSales77 = measureOverTimeSales77[measureOverTimeSales77['STORE_NBR'] == control_store77]
scaledControlSales77['controlSales'] = scaledControlSales77['TOT_SALES'] * scalingFactorForControlSales77

#### Calculate the percentage difference between scaled control sales and trial sales
merged_data77 = pd.merge(scaledControlSales77[['YEAR_MONTH', 'controlSales']],
                       measureOverTime[measureOverTime['STORE_NBR'] == trial_store_77][['TOT_SALES', 'YEAR_MONTH']], # lọc dữ liệu cho trial_store và chỉ lấy hai cột totSales và YEARMONTH
                       on='YEAR_MONTH')
merged_data77['percentageDiff'] = (merged_data77['controlSales'] - merged_data77['TOT_SALES']).abs() / merged_data77['controlSales']
print(merged_data77)

#### As our null hypothesis is that the trial period is the same as the pre-trial
#### period, let's take the standard deviation based on the scaled percentage difference
#### in the pre-trial period
filtered_data77 = merged_data77[merged_data77['YEAR_MONTH'] < '2019-02']
stdDev77 = filtered_data77['percentageDiff'].std()

#### Note that there are 8 months in the pre-trial period
#### hence 8 - 1 = 7 degrees of freedom
degreesOfFreedom = 7

#### We will test with a null hypothesis of there being 0 difference between trial
#### and control stores.
#### Calculate the t-values for the trial months. After that, 
#### find the 95th percentile of the t distribution with the appropriate degrees of freedom
#### to check whether the hypothesis is statistically significant.
#### The test statistic here is (x - u)/standard deviation
merged_data77['tValue'] = (merged_data77['percentageDiff'] - 0) / stdDev77
merged_data77['TransactionMonth'] = pd.to_datetime(merged_data77['YEAR_MONTH'].astype(str), format='%Y-%m')
filtered_data77 = merged_data77[(merged_data77['YEAR_MONTH'] < '2019-05') & (merged_data77['YEAR_MONTH'] > '2019-01')][['TransactionMonth', 'tValue']]
print(filtered_data77)

#### Find the 95th percentile of the t distribution with the appropriate
#### degrees of freedom to compare against
from scipy import stats
t_value = stats.t.ppf(0.95, df=degreesOfFreedom)
print(f"T-value for 95% confidence level: {t_value}")
# We can observe that the t-value is much larger than the 95th percentile value of the t-distribution for March and April - 
# i.e. the increase in sales in the trial store in March and April is statistically greater than in the control store

measureOverTimeSales77 = measureOverTime.copy()

#### Trial and control store total sales
#### Create new variables Store_type, totSales and TransactionMonth in the data table.
measureOverTimeSales77['Store_type'] = measureOverTimeSales77['STORE_NBR'].apply(
    lambda x: 'Trial' if x == trial_store_77 else ('Control' if x == control_store77 else 'Other stores'))
pastSales77 = measureOverTimeSales77.groupby(['YEAR_MONTH', 'Store_type']).agg({'TOT_SALES': 'mean'}).reset_index()
pastSales77['TransactionMonth'] = pd.to_datetime(pastSales77['YEAR_MONTH'].astype(str).str[:4] + 
                                               pastSales77['YEAR_MONTH'].astype(str).str[4:] + '-01', format='%Y-%m-%d')
pastSales77 = pastSales77[pastSales77['Store_type'].isin(['Trial', 'Control'])]

#### Control store 95th percentile
pastSales_Controls95 = pastSales77[pastSales77['Store_type'] == 'Control'].copy()
pastSales_Controls95['TOT_SALES'] = pastSales_Controls95['TOT_SALES'] * (1 + stdDev77 * 2)
pastSales_Controls95['Store_type'] = "Control 95th % confidence interval"

#### Control store 5th percentile
pastSales_Controls5 = pastSales77[pastSales77['Store_type'] == 'Control'].copy()
pastSales_Controls5['TOT_SALES'] = pastSales_Controls5['TOT_SALES'] * (1 - stdDev77 * 2)
pastSales_Controls5['Store_type'] = "Control 95th % confidence interval"

trialAssessment = pd.concat([pastSales77, pastSales_Controls95, pastSales_Controls5], ignore_index=True)

#### Plotting these in one nice grap
rect_data77 = trialAssessment[(trialAssessment['YEAR_MONTH'] < '2019-05') & (trialAssessment['YEAR_MONTH'] > '2019-01')]
plt.figure(figsize=(10, 6))
plt.axvspan(
    pd.to_datetime(rect_data77['TransactionMonth'].min()),  # xmin
    pd.to_datetime(rect_data77['TransactionMonth'].max()),  # xmax
    ymin=0, ymax=1, color='grey', alpha=0.3)
for store_type in trialAssessment['Store_type'].unique():
    subset = trialAssessment[trialAssessment['Store_type'] == store_type]
    plt.plot(subset['TransactionMonth'], subset['TOT_SALES'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title='Store type')
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()

#### This would be a repeat of the steps before for total sales
#### Scale pre-trial control customers to match pre-trial trial store customers
#### Compute a scaling factor to align control store customer counts to our trial store.
#### Then, apply the scaling factor to control store customer counts.
#### Finally, calculate the percentage difference between scaled control store customers and trial cust
scalingFactorForTrialStore77 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == trial_store_77) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['nCustomer'].sum()
scalingFactorForControlStore77 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == control_store77) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['nCustomer'].sum()
scalingFactorForControlCust77 = scalingFactorForTrialStore77 / scalingFactorForControlStore77

#### Apply the scaling factor
measureOverTimeCusts77 = measureOverTime.copy()
scaled_control_customers77 = measureOverTimeCusts77[measureOverTimeCusts77['STORE_NBR'] == control_store77].copy()
scaled_control_customers77['controlCustomers'] = scaled_control_customers77['nCustomer'] * scalingFactorForControlCust77
scaled_control_customers77['Store_type'] = np.where(scaled_control_customers77['STORE_NBR'] == trial_store_77, 
                                                  "Trial", 
                                                  np.where(scaled_control_customers77['STORE_NBR'] == control_store77, 
                                                           "Control", 
                                                           "Other stores"))
measureOverTimeCusts77.loc[measureOverTimeCusts77['STORE_NBR'] == control_store77, 'controlCustomers'] = scaled_control_customers77['controlCustomers']
measureOverTimeCusts77.loc[measureOverTimeCusts77['STORE_NBR'] == control_store77, 'Store_type'] = scaled_control_customers77['Store_type']

trial_customers77 = measureOverTimeCusts77[measureOverTimeCusts77['STORE_NBR'] == trial_store_77][['nCustomer', 'YEAR_MONTH']]
percentage_diff77 = pd.merge(scaled_control_customers77[['YEAR_MONTH', 'controlCustomers']], trial_customers77, on='YEAR_MONTH')

#### Calculate the percentage difference between scaled control sales and trial sales
percentage_diff77['percentageDiff'] = abs(percentage_diff77['controlCustomers'] - percentage_diff77['nCustomer']) / percentage_diff77['controlCustomers']
print(percentage_diff77)

#### As our null hypothesis is that the trial period is the same as the pre-trial
#### period, let's take the standard deviation based on the scaled percentage difference
#### in the pre-trial period
filtered_data77 = merged_data77[merged_data77['YEAR_MONTH'] < '2019-02']
stdDev77 = filtered_data77['percentageDiff'].std()
degreesOfFreedom77 = 7

#### Trial and control store number of customers
pastCustomers77 = measureOverTimeCusts77.groupby(['YEAR_MONTH', 'Store_type']).agg(nCusts=('nCustomer', 'mean')).reset_index()

pastCustomers77['TransactionMonth'] = pd.to_datetime(pastSales77['YEAR_MONTH'].astype(str).str[:4] + 
                                               pastSales77['YEAR_MONTH'].astype(str).str[4:] + '-01', format='%Y-%m-%d')
pastCustomers77 = pastCustomers77[pastCustomers77['Store_type'].isin(['Trial', 'Control'])]

#### Control store 95th percentile
pastCustomers_Controls95 = pastCustomers77[pastCustomers77['Store_type'] == 'Control'].copy()
pastCustomers_Controls95['nCusts'] = pastCustomers_Controls95['nCusts'] * (1 + stdDev77 * 2)
pastCustomers_Controls95['Store_type'] = 'Control 95th % confidence interval'

#### Control store 5th percentile
pastCustomers_Controls5 = pastCustomers77[pastCustomers77['Store_type'] == 'Control'].copy()
pastCustomers_Controls5['nCusts'] = pastCustomers_Controls5['nCusts'] * (1 - stdDev77 * 2)
pastCustomers_Controls5['Store_type'] = 'Control 5th % confidence interval'

trialAssessment77 = pd.concat([pastCustomers77, pastCustomers_Controls95, pastCustomers_Controls5])

#### Plotting these in one nice graph
rect_data77 = trialAssessment77[(trialAssessment77['YEAR_MONTH'] < '2019-05') & (trialAssessment77['YEAR_MONTH'] > '2019-01')]
plt.figure(figsize=(10, 6))
plt.axvspan(
    pd.to_datetime(rect_data77['TransactionMonth'].min()),  # xmin
    pd.to_datetime(rect_data77['TransactionMonth'].max()),  # xmax
    ymin=0, ymax=1, color='grey', alpha=0.3)
for store_type in trialAssessment77['Store_type'].unique():
    subset = trialAssessment77[trialAssessment77['Store_type'] == store_type]
    plt.plot(subset['TransactionMonth'], subset['nCusts'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title='Store type')
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()


## Trial store 86
#### Calculate the metrics below as we did for the first trial store.
#### Use the functions we created earlier to calculate correlations and magnitude for each potential control store
trial_store_86 = 86
corr_nSales_86 = calculateCorrelation(preTrialMeasures, 'TOT_SALES', trial_store_86)
corr_nSales_86_sorted = corr_nSales_86.sort_values(by = 'corr_measure', ascending=False)
print(corr_nSales_86_sorted)
corr_nCustomers_86 = calculateCorrelation(preTrialMeasures, 'nCustomer', trial_store_86)
corr_nCustomers_86_sorted = corr_nCustomers_86.sort_values(by = 'corr_measure', ascending=False)
print(corr_nCustomers_86_sorted)

magnitude_nSales_86 = calculate_magnitude_distance(preTrialMeasures, 'TOT_SALES', trial_store_86)
Standardise_magnitude_nSale_86 = minMaxDist(magnitude_nSales_86)
magnitude_nSales_86_sorted = Standardise_magnitude_nSale_86.sort_values(by = 'mag_measure', ascending= True)
print(magnitude_nSales_86_sorted)

magnitude_nCustomers_86 = calculate_magnitude_distance(preTrialMeasures, 'nCustomer', trial_store_86)
Standardise_magnitude_nCustomers_86 = minMaxDist(magnitude_nCustomers_86)
magnitude_nCustomers_86_sorted = Standardise_magnitude_nCustomers_86.sort_values(by = 'mag_measure', ascending= True)
print(magnitude_nCustomers_86_sorted)

#### Now, create a combined score composed of correlation and magnitude
merged_nsales_86 = pd.merge(corr_nSales_86_sorted, magnitude_nSales_86_sorted, how='outer')
print(merged_nsales_86)

merged_nCustomers_86 = pd.merge(corr_nCustomers_86_sorted, magnitude_nCustomers_86_sorted, how='outer')
print(merged_nCustomers_86)

corr_weight = 0.5
score_nSales_86 = ((merged_nsales_86['corr_measure'] + merged_nsales_86['mag_measure']) / 2).rename('score_nSales')
score_nsales_86_sorted = pd.concat([merged_nsales_86, score_nSales_86], axis=1)
print(score_nsales_86_sorted)

score_nCustomer_86 = ((merged_nCustomers_86['corr_measure'] + merged_nCustomers_86['mag_measure']) / 2).rename('score_nCustomer')
score_nCustomer_86_sorted = pd.concat([merged_nCustomers_86, score_nCustomer_86], axis=1)
print(score_nCustomer_86_sorted)

#### Finally, combine scores across the drivers using a simple average.
merged_score_control86 = pd.merge(score_nsales_86_sorted, score_nCustomer_86_sorted, on = ['Store1', 'Store2'])
finalControlScore86 =  (score_nSales_86 * 0.5 + score_nCustomer_86 * 0.5).rename('finalControlScore1')
score_control86 = pd.concat([merged_score_control86, finalControlScore86], axis = 1)
print(score_control86)

#### Select control stores based on the highest matching store
#### (closest to 1 but not the store itself, i.e. the second ranked highest store)
#### Select control store for trial store 86
filtered_score_control86 = score_control86[score_control86['Store1'] == trial_store_86]
sorted_score_control86 = filtered_score_control86.sort_values(by = 'finalControlScore1', ascending = False)
control_store86 = sorted_score_control86.iloc[1]['Store2']
print(control_store86)

#### Conduct visual checks on trends based on the drivers
measureOverTimeSales86 = measureOverTime.copy()
measureOverTimeSales86['Store_type'] = measureOverTimeSales86['STORE_NBR'].apply(
    lambda x: 'Trial' if x == trial_store_86 else ('Control' if x == control_store86 else 'Other stores'))
pastSales86 = measureOverTimeSales86.groupby(['YEAR_MONTH', 'Store_type'], as_index=False)['TOT_SALES'].mean()
pastSales86['TransactionMonth'] = pd.to_datetime(pastSales86['YEAR_MONTH'], format='%Y-%m')
pastSales86 = pastSales86[pastSales86['YEAR_MONTH'] < '2019-03']

plt.figure(figsize=(10, 6))

for store_type, group_data in pastSales86.groupby('Store_type'):
    plt.plot(group_data['TransactionMonth'], group_data['TOT_SALES'], label=store_type)

plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title="Store type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#### Conduct visual checks on trends based on the drivers
measureOverTimeCusts86 = measureOverTime.copy()
measureOverTimeCusts86['Store_type'] = measureOverTimeCusts86['STORE_NBR'].apply(
    lambda x: 'Trial' if x == trial_store_86 else ('Control' if x == control_store86 else 'Other stores'))
numberCustomers86 = measureOverTimeCusts86.groupby(['YEAR_MONTH', 'Store_type'], as_index=False)['nCustomer'].mean()
numberCustomers86['TransactionMonth'] = pd.to_datetime(numberCustomers86['YEAR_MONTH'], format='%Y-%m')
numberCustomers86 = numberCustomers86[numberCustomers86['YEAR_MONTH'] < '2019-03']

plt.figure(figsize=(10, 6))

for store_type, group_data in numberCustomers86.groupby('Store_type'):
    plt.plot(group_data['TransactionMonth'], group_data['nCustomer'], label=store_type)

plt.xlabel("Month of operation")
plt.ylabel("Total number of customers")
plt.title("Total number of customers by month")
plt.legend(title="Store type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#### Scale pre-trial control sales to match pre-trial trial store sales
scalingFactorForTrialStore86 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == trial_store_86) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['TOT_SALES'].sum()
scalingFactorForControlStore86 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == control_store86) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['TOT_SALES'].sum()
scalingFactorForControlSales86 = scalingFactorForTrialStore86 / scalingFactorForControlStore86

#### Apply the scaling factor
measureOverTimeSales86 = measureOverTime.copy()
scaledControlSales86 = measureOverTimeSales86[measureOverTimeSales86['STORE_NBR'] == control_store86]
scaledControlSales86['controlSales'] = scaledControlSales86['TOT_SALES'] * scalingFactorForControlSales86

#### Calculate the percentage difference between scaled control sales and trial sales
#### When calculating percentage difference, remember to use absolute difference
merged_data86 = pd.merge(scaledControlSales86[['YEAR_MONTH', 'controlSales']],
                       measureOverTime[measureOverTime['STORE_NBR'] == trial_store_86][['TOT_SALES', 'YEAR_MONTH']], # lọc dữ liệu cho trial_store và chỉ lấy hai cột totSales và YEARMONTH
                       on='YEAR_MONTH')
merged_data86['percentageDiff'] = (merged_data86['controlSales'] - merged_data86['TOT_SALES']).abs() / merged_data86['controlSales']
print(merged_data86)

#### As our null hypothesis is that the trial period is the same as the pre-trial
#### period, let's take the standard deviation based on the scaled percentage difference
#### in the pre-trial period
#### Calculate the standard deviation of percentage differences during the pre-trial period
filtered_data86 = merged_data86[merged_data86['YEAR_MONTH'] < '2019-02']
stdDev86 = filtered_data86['percentageDiff'].std()
degreesOfFreedom86 = 7

#### Trial and control store total sales
#### Create a table with sales by store type and month.
#### We only need data for the trial and control store.
measureOverTimeSales86 = measureOverTime.copy()
measureOverTimeSales86['Store_type'] = np.where(measureOverTimeSales86['STORE_NBR'] == trial_store_86, "Trial",
    np.where(measureOverTimeSales86['STORE_NBR'] == control_store86, "Control", "Other stores")) 
measureOverTimeSales86 = measureOverTimeSales86.groupby(['YEAR_MONTH', 'Store_type'], as_index=False).agg({'TOT_SALES': 'mean'})
measureOverTimeSales86['TransactionMonth'] = pd.to_datetime(measureOverTimeSales86['YEAR_MONTH'].astype(str).str[:4] + 
                                            measureOverTimeSales86['YEAR_MONTH'].astype(str).str[4:] + '-01', format='%Y-%m-%d')
pastSales86 = measureOverTimeSales86[measureOverTimeSales86['Store_type'].isin(['Trial', 'Control'])]

#### Calculate the 5th and 95th percentile for control store sales.
#### The 5th and 95th percentiles can be approximated by using two standard deviations away from the mean.
#### Recall that the variable stdDev earlier calculates standard deviation in percentages, and not dollar sales.
#### Control store 95th percentile
pastSales_Controls95 = pastSales86[pastSales86['Store_type'] == 'Control'].copy()
pastSales_Controls95['nCusts'] = pastSales_Controls95['TOT_SALES'] * (1 + stdDev86 * 2)
pastSales_Controls95['Store_type'] = "Control 95th % confidence interval"

#### Control store 5th percentile
pastSales_Controls5 = pastSales86[pastSales86['Store_type'] == 'Control'].copy()
pastSales_Controls5['nCusts'] = pastSales_Controls5['TOT_SALES'] * (1 - stdDev86 * 2)
pastSales_Controls5['Store_type'] = "Control 95th % confidence interval"

trialAssessment86 = pd.concat([pastSales86, pastSales_Controls95, pastSales_Controls5], ignore_index=True)

#### Plotting these in one nice graph
rect_data86 = trialAssessment86[(trialAssessment86['YEAR_MONTH'] < '2019-05') & (trialAssessment86['YEAR_MONTH'] > '2019-01')]
plt.figure(figsize=(10, 6))
plt.axvspan(
    pd.to_datetime(rect_data86['TransactionMonth'].min()),  # xmin
    pd.to_datetime(rect_data86['TransactionMonth'].max()),  # xmax
    ymin=0, ymax=1, color='grey', alpha=0.3)
for store_type in trialAssessment86['Store_type'].unique():
    subset = trialAssessment86[trialAssessment86['Store_type'] == store_type]
    plt.plot(subset['TransactionMonth'], subset['nCusts'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title='Store type')
plt.xticks(rotation=45)  # Xoay nhãn trên trục x nếu cần
plt.tight_layout()  # Đảm bảo bố cục gọn gàng
plt.show()

#### This would be a repeat of the steps before for total sales
#### Scale pre-trial control customers to match pre-trial trial store customers
scalingFactorForTrialStore86 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == trial_store_86) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['nCustomer'].sum()
scalingFactorForControlStore86 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == control_store86) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['nCustomer'].sum()
scalingFactorForControlCust86 = scalingFactorForTrialStore86 / scalingFactorForControlStore86

#### Apply the scaling factor
measureOverTimeCusts86 = measureOverTime.copy()
scaled_control_customers86 = measureOverTimeCusts86[measureOverTimeCusts86['STORE_NBR'] == control_store86].copy()
scaled_control_customers86['controlCustomers'] = scaled_control_customers86['nCustomer'] * scalingFactorForControlCust86
scaled_control_customers86['Store_type'] = np.where(scaled_control_customers86['STORE_NBR'] == trial_store_86, 
                                                  "Trial", 
                                                  np.where(scaled_control_customers86['STORE_NBR'] == control_store86, 
                                                           "Control", 
                                                           "Other stores"))
measureOverTimeCusts86.loc[measureOverTimeCusts86['STORE_NBR'] == control_store86, 'controlCustomers'] = scaled_control_customers86['controlCustomers']
measureOverTimeCusts86.loc[measureOverTimeCusts86['STORE_NBR'] == control_store86, 'Store_type'] = scaled_control_customers86['Store_type']

trial_customers86 = measureOverTimeCusts86[measureOverTimeCusts86['STORE_NBR'] == trial_store_86][['nCustomer', 'YEAR_MONTH']]
percentage_diff86 = pd.merge(scaled_control_customers86[['YEAR_MONTH', 'controlCustomers']], trial_customers86, on='YEAR_MONTH')

#### Calculate the percentage difference between scaled control sales and trial sales
percentage_diff86['percentageDiff'] = abs(percentage_diff86['controlCustomers'] - percentage_diff86['nCustomer']) / percentage_diff86['controlCustomers']
print(percentage_diff86)

#### As our null hypothesis is that the trial period is the same as the pre-trial
#### period, let's take the standard deviation based on the scaled percentage difference
#### in the pre-trial period
filtered_data86 = merged_data86[merged_data86['YEAR_MONTH'] < '2019-02']
stdDev86 = filtered_data86['percentageDiff'].std()
degreesOfFreedom86 = 7

#### Trial and control store number of customers
pastCustomers86 = measureOverTimeCusts86.groupby(['YEAR_MONTH', 'Store_type']).agg(nCusts=('nCustomer', 'mean')).reset_index()
pastCustomers86['TransactionMonth'] = pd.to_datetime(pastSales86['YEAR_MONTH'].astype(str).str[:4] + 
                                               pastSales86['YEAR_MONTH'].astype(str).str[4:] + '-01', format='%Y-%m-%d')
pastCustomers86 = pastCustomers86[pastCustomers86['Store_type'].isin(['Trial', 'Control'])]

#### Control store 95th percentile
pastCustomers_Controls95 = pastCustomers86[pastCustomers86['Store_type'] == 'Control'].copy()
pastCustomers_Controls95['nCusts'] = pastCustomers_Controls95['nCusts'] * (1 + stdDev86 * 2)
pastCustomers_Controls95['Store_type'] = 'Control 95th % confidence interval'

#### Control store 5th percentile
pastCustomers_Controls5 = pastCustomers86[pastCustomers86['Store_type'] == 'Control'].copy()
pastCustomers_Controls5['nCusts'] = pastCustomers_Controls5['nCusts'] * (1 - stdDev86 * 2)
pastCustomers_Controls5['Store_type'] = 'Control 5th % confidence interval'

#### Combine the tables pastSales, pastSales_Controls95, pastSales_Controls5
trialAssessment86 = pd.concat([pastCustomers86, pastCustomers_Controls95, pastCustomers_Controls5])
rect_data86 = trialAssessment86[(trialAssessment86['YEAR_MONTH'] < '2019-05') & (trialAssessment86['YEAR_MONTH'] > '2019-01')]

#### Plotting these in one nice graph
plt.figure(figsize=(10, 6))
plt.axvspan(
    pd.to_datetime(rect_data86['TransactionMonth'].min()),  # xmin
    pd.to_datetime(rect_data86['TransactionMonth'].max()),  # xmax
    ymin=0, ymax=1, color='grey', alpha=0.3)
for store_type in trialAssessment86['Store_type'].unique():
    subset = trialAssessment86[trialAssessment86['Store_type'] == store_type]
    plt.plot(subset['TransactionMonth'], subset['nCusts'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title='Store type')
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()


## Trial store 88
#### Conduct the analysis on trial store 88.
trial_store_88 = 88
corr_nSales_88 = calculateCorrelation(preTrialMeasures, 'TOT_SALES', trial_store_88)
corr_nSales_88_sorted = corr_nSales_88.sort_values(by = 'corr_measure', ascending=False)
print(corr_nSales_88_sorted)
corr_nCustomers_88 = calculateCorrelation(preTrialMeasures, 'nCustomer', trial_store_88)
corr_nCustomers_88_sorted = corr_nCustomers_88.sort_values(by = 'corr_measure', ascending=False)
print(corr_nCustomers_88_sorted)

#### Use the functions from earlier to calculate the magnitude distance of the 
#### sales and number of customers of each potential control store to the trial store
magnitude_nSales_88 = calculate_magnitude_distance(preTrialMeasures, 'TOT_SALES', trial_store_88)
Standardise_magnitude_nSale_88 = minMaxDist(magnitude_nSales_88)
magnitude_nSales_88_sorted = Standardise_magnitude_nSale_88.sort_values(by = 'mag_measure', ascending= True)
print(magnitude_nSales_88_sorted)

magnitude_nCustomers_88 = calculate_magnitude_distance(preTrialMeasures, 'nCustomer', trial_store_88)
Standardise_magnitude_nCustomers_88 = minMaxDist(magnitude_nCustomers_88)
magnitude_nCustomers_88_sorted = Standardise_magnitude_nCustomers_88.sort_values(by = 'mag_measure', ascending= True)
print(magnitude_nCustomers_88_sorted)

#### Create a combined score composed of correlation and magnitude by merging the 
#### correlations table and the magnitudes table, for each driver.
merged_nsales_88 = pd.merge(corr_nSales_88_sorted, magnitude_nSales_88_sorted, how='outer')
print(merged_nsales_88)
merged_nCustomers_88 = pd.merge(corr_nCustomers_88_sorted, magnitude_nCustomers_88_sorted, how='outer')
print(merged_nCustomers_88)

corr_weight = 0.5
score_nSales_88 = ((merged_nsales_88['corr_measure'] + merged_nsales_88['mag_measure']) / 2).rename('score_nSales')
score_nsales_88_sorted = pd.concat([merged_nsales_88, score_nSales_88], axis=1)
print(score_nsales_88_sorted)

score_nCustomer_88 = ((merged_nCustomers_88['corr_measure'] + merged_nCustomers_88['mag_measure']) / 2).rename('score_nCustomer')
score_nCustomer_88_sorted = pd.concat([merged_nCustomers_88, score_nCustomer_88], axis=1)
print(score_nCustomer_88_sorted)

#### Combine scores across the drivers by merging sales scores and customer scores,
#### and compute a final combined score.
merged_score_control88 = pd.merge(score_nsales_88_sorted, score_nCustomer_88_sorted, on = ['Store1', 'Store2'])
finalControlScore88 =  (score_nSales_88 * 0.5 + score_nCustomer_88 * 0.5).rename('finalControlScore1')
score_control88 = pd.concat([merged_score_control88, finalControlScore88], axis = 1)
print(score_control88)

#### Select control stores based on the highest matching store
#### (closest to 1 but not the store itself, i.e. the second ranked highest store)
#### Select control store for trial store 88
filtered_score_control88 = score_control88[score_control88['Store1'] == trial_store_88]
sorted_score_control88 = filtered_score_control88.sort_values(by = 'finalControlScore1', ascending = False)
control_store88 = sorted_score_control88.iloc[1]['Store2']
print(control_store88)

#### Visual checks on trends based on the drivers
#### For the period before the trial, create a graph with total sales of the trial
#### store for each month, compared to the control store and other stores.
measureOverTimeSales88 = measureOverTime.copy()
measureOverTimeSales88['Store_type'] = measureOverTimeSales88['STORE_NBR'].apply(
    lambda x: 'Trial' if x == trial_store_88 else ('Control' if x == control_store88 else 'Other stores'))
pastSales88 = measureOverTimeSales88.groupby(['YEAR_MONTH', 'Store_type'], as_index=False)['TOT_SALES'].mean()
pastSales88['TransactionMonth'] = pd.to_datetime(pastSales88['YEAR_MONTH'], format='%Y-%m')
pastSales88 = pastSales88[pastSales88['YEAR_MONTH'] < '2019-03']
plt.figure(figsize=(10, 6))
for store_type, group_data in pastSales88.groupby('Store_type'):
    plt.plot(group_data['TransactionMonth'], group_data['TOT_SALES'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title="Store type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#### Visual checks on trends based on the drivers
#### For the period before the trial, create a graph with customer counts of the
#### trial store for each month, compared to the control store and other stores.
measureOverTimeCusts88 = measureOverTime.copy()
measureOverTimeCusts88['Store_type'] = measureOverTimeCusts88['STORE_NBR'].apply(
    lambda x: 'Trial' if x == trial_store_88 else ('Control' if x == control_store88 else 'Other stores'))
numberCustomers88 = measureOverTimeCusts88.groupby(['YEAR_MONTH', 'Store_type'], as_index=False)['nCustomer'].mean()
numberCustomers88['TransactionMonth'] = pd.to_datetime(numberCustomers88['YEAR_MONTH'], format='%Y-%m')
numberCustomers88 = numberCustomers88[numberCustomers88['YEAR_MONTH'] < '2019-03']
plt.figure(figsize=(10, 6))
for store_type, group_data in numberCustomers88.groupby('Store_type'):
    plt.plot(group_data['TransactionMonth'], group_data['nCustomer'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total number of customers")
plt.title("Total number of customers by month")
plt.legend(title="Store type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#### Scale pre-trial control store sales to match pre-trial trial store sales
scalingFactorForTrialStore88 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == trial_store_88) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['TOT_SALES'].sum()
scalingFactorForControlStore88 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == control_store88) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['TOT_SALES'].sum()
scalingFactorForControlSales88 = scalingFactorForTrialStore88 / scalingFactorForControlStore88

#### Apply the scaling factor
measureOverTimeSales88 = measureOverTime.copy()
scaledControlSales88 = measureOverTimeSales88[measureOverTimeSales88['STORE_NBR'] == control_store88]
scaledControlSales88['controlSales'] = scaledControlSales88['TOT_SALES'] * scalingFactorForControlSales88

#### Calculate the absolute percentage difference between scaled control sales and trial sales
merged_data88 = pd.merge(scaledControlSales88[['YEAR_MONTH', 'controlSales']],
                       measureOverTime[measureOverTime['STORE_NBR'] == trial_store_88][['TOT_SALES', 'YEAR_MONTH']], # lọc dữ liệu cho trial_store và chỉ lấy hai cột totSales và YEARMONTH
                       on='YEAR_MONTH')
merged_data88['percentageDiff'] = (merged_data88['controlSales'] - merged_data88['TOT_SALES']).abs() / merged_data88['controlSales']
print(merged_data88)

#### As our null hypothesis is that the trial period is the same as the pre-trial period,
#### let's take the standard deviation based on the scaled percentage difference in the pre-trial period 
filtered_data88 = merged_data88[merged_data88['YEAR_MONTH'] < '2019-02']
stdDev88 = filtered_data88['percentageDiff'].std()
degreesOfFreedom88 = 7

#### Trial and control store total sale
measureOverTimeSales88 = measureOverTime.copy()
measureOverTimeSales88['Store_type'] = np.where(measureOverTimeSales88['STORE_NBR'] == trial_store_88, "Trial",
    np.where(measureOverTimeSales88['STORE_NBR'] == control_store88, "Control", "Other stores")) 
measureOverTimeSales88 = measureOverTimeSales88.groupby(['YEAR_MONTH', 'Store_type'], as_index=False).agg({'TOT_SALES': 'mean'})
measureOverTimeSales88['TransactionMonth'] = pd.to_datetime(measureOverTimeSales88['YEAR_MONTH'].astype(str).str[:4] + 
                                            measureOverTimeSales88['YEAR_MONTH'].astype(str).str[4:] + '-01', format='%Y-%m-%d')
pastSales88 = measureOverTimeSales88[measureOverTimeSales88['Store_type'].isin(['Trial', 'Control'])]

#### Control store 95th percentile
pastSales_Controls95 = pastSales88[pastSales88['Store_type'] == 'Control'].copy()
pastSales_Controls95['TOT_SALES'] = pastSales_Controls95['TOT_SALES'] * (1 + stdDev88 * 2)
pastSales_Controls95['Store_type'] = "Control 95th % confidence interval"

#### Control store 5th percentile
pastSales_Controls5 = pastSales88[pastSales88['Store_type'] == 'Control'].copy()
pastSales_Controls5['TOT_SALES'] = pastSales_Controls5['TOT_SALES'] * (1 - stdDev88 * 2)
pastSales_Controls5['Store_type'] = "Control 95th % confidence interval"

trialAssessment = pd.concat([pastSales88, pastSales_Controls95, pastSales_Controls5], ignore_index=True)

#### Plotting these in one nice grap
rect_data88 = trialAssessment[(trialAssessment['YEAR_MONTH'] < '2019-05') & (trialAssessment['YEAR_MONTH'] > '2019-01')]
plt.figure(figsize=(10, 6))
plt.axvspan(
    pd.to_datetime(rect_data88['TransactionMonth'].min()),  # xmin
    pd.to_datetime(rect_data88['TransactionMonth'].max()),  # xmax
    ymin=0, ymax=1, color='grey', alpha=0.3)
for store_type in trialAssessment['Store_type'].unique():
    subset = trialAssessment[trialAssessment['Store_type'] == store_type] 
    plt.plot(subset['TransactionMonth'], subset['TOT_SALES'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title='Store type')
plt.xticks(rotation=45)  
plt.tight_layout() 
plt.show()

#### This would be a repeat of the steps before for total sales
#### Scale pre-trial control store customers to match pre-trial trial store customers
scalingFactorForTrialStore88 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == trial_store_88) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['nCustomer'].sum()
scalingFactorForControlStore88 = preTrialMeasures[(preTrialMeasures['STORE_NBR'] == control_store88) & (preTrialMeasures['YEAR_MONTH'] < '2019-02')]['nCustomer'].sum()
scalingFactorForControlCust88 = scalingFactorForTrialStore88 / scalingFactorForControlStore88

#### Apply the scaling factor
measureOverTimeCusts88 = measureOverTime.copy()
scaled_control_customers88 = measureOverTimeCusts88[measureOverTimeCusts88['STORE_NBR'] == control_store88].copy()
scaled_control_customers88['controlCustomers'] = scaled_control_customers88['nCustomer'] * scalingFactorForControlCust88
scaled_control_customers88['Store_type'] = np.where(scaled_control_customers88['STORE_NBR'] == trial_store_88, 
                                                  "Trial", 
                                                  np.where(scaled_control_customers88['STORE_NBR'] == control_store88, 
                                                           "Control", 
                                                           "Other stores"))
measureOverTimeCusts88.loc[measureOverTimeCusts88['STORE_NBR'] == control_store88, 'controlCustomers'] = scaled_control_customers88['controlCustomers']
measureOverTimeCusts88.loc[measureOverTimeCusts88['STORE_NBR'] == control_store88, 'Store_type'] = scaled_control_customers88['Store_type']
trial_customers88 = measureOverTimeCusts88[measureOverTimeCusts88['STORE_NBR'] == trial_store_88][['nCustomer', 'YEAR_MONTH']]
percentage_diff88 = pd.merge(scaled_control_customers88[['YEAR_MONTH', 'controlCustomers']], trial_customers88, on='YEAR_MONTH')
print(percentage_diff88)

#### Calculate the absolute percentage difference between scaled control sales and trial sales
percentage_diff88['percentageDiff'] = abs(percentage_diff88['controlCustomers'] - percentage_diff88['nCustomer']) / percentage_diff88['controlCustomers']

#### As our null hypothesis is that the trial period is the same as the pre-trial
#### period, let's take the standard deviation based on the scaled percentage difference
#### in the pre-trial period
filtered_data88 = merged_data88[merged_data88['YEAR_MONTH'] < '2019-02']
stdDev881 = filtered_data88['percentageDiff'].std()
degreesOfFreedom88 = 7
# note that there are 8 months in the pre-trial period hence 8 - 1 = 7 degrees of freedom

#### Trial and control store number of customers
pastCustomers881 = measureOverTimeCusts88.groupby(['YEAR_MONTH', 'Store_type']).agg(nCusts=('nCustomer', 'mean')).reset_index()

pastCustomers881['TransactionMonth'] = pd.to_datetime(pastSales88['YEAR_MONTH'].astype(str).str[:4] + 
                                               pastSales88['YEAR_MONTH'].astype(str).str[4:] + '-01', format='%Y-%m-%d')
pastCustomers881 = pastCustomers881[pastCustomers881['Store_type'].isin(['Trial', 'Control'])]

#### Control store 95th percentile
pastCustomers_Controls95 = pastCustomers881[pastCustomers881['Store_type'] == 'Control'].copy()
pastCustomers_Controls95['nCusts'] = pastCustomers_Controls95['nCusts'] * (1 + stdDev88 * 2)
pastCustomers_Controls95['Store_type'] = 'Control 95th % confidence interval'

# Control store 5th percentile
pastCustomers_Controls5 = pastCustomers881[pastCustomers881['Store_type'] == 'Control'].copy()
pastCustomers_Controls5['nCusts'] = pastCustomers_Controls5['nCusts'] * (1 - stdDev88 * 2)
pastCustomers_Controls5['Store_type'] = 'Control 5th % confidence interval'

#### Combine the tables pastSales, pastSales_Controls95, pastSales_Controls5
trialAssessment88 = pd.concat([pastCustomers881, pastCustomers_Controls95, pastCustomers_Controls5])

#### Plotting these in one nice grap
rect_data88 = trialAssessment88[(trialAssessment88['YEAR_MONTH'] < '2019-05') & (trialAssessment88['YEAR_MONTH'] > '2019-01')]
plt.figure(figsize=(10, 6))
plt.axvspan(
    pd.to_datetime(rect_data88['TransactionMonth'].min()),  # xmin
    pd.to_datetime(rect_data88['TransactionMonth'].max()),  # xmax
    ymin=0, ymax=1, color='grey', alpha=0.3)
for store_type in trialAssessment88['Store_type'].unique():
    subset = trialAssessment88[trialAssessment88['Store_type'] == store_type]
    plt.plot(subset['TransactionMonth'], subset['nCusts'], label=store_type)
plt.xlabel("Month of operation")
plt.ylabel("Total sales")
plt.title("Total sales by month")
plt.legend(title='Store type')
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()