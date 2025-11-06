import pandas as pd

d1 = pd.read_excel('QVI_transaction_data.xlsx')


### Task 1: Checking data formats and correcting
## Step 1: check dtypes of each column
print(d1.dtypes)

# Because the "DATE" column is in an integer format. So I will change it into date type
import datetime

d1['DATE'] = pd.to_datetime(d1['DATE'], unit = 'D', origin = '1899-12-30')
print(d1.head())
# unit='D': Chỉ định rằng các giá trị trong cột là số ngày. Nếu unit='s', giá trị sẽ được hiểu là số giây, unit='ms' là mili giây, v.v.
# origin='1899-12-30': Xác định ngày gốc để bắt đầu tính toán (ở đây, số 0 sẽ tương ứng với ngày 1899-12-30).

## Step 2: check the chip products
# Check the all words with digits and special characters such as ‘&’ from our set of product words
d1_product = d1[ ~ d1['PROD_NAME'].str.contains("&")]
print(d1_product['PROD_NAME'])

# There are salsa products in the dataset, so remove it
d1_new = d1_product[ ~ d1_product['PROD_NAME'].str.contains("salsa")]
print(d1_new)


### Task 2: Interpreting high-level summaries of the data
## Step 1: summarise the data
print(d1.describe())

## Step 2: check for nulls 
print(d1_new.isnull().sum())

# Step 3: check for any possible outliers
# in product quantity appears to have an outlier that 200 packets of chips are bought in one transaction
d1_prod_qty = d1_new['PROD_QTY'] == 200
print(d1_new[d1_prod_qty])

# I will remove this transactions
d1_cleaned = d1_new['LYLTY_CARD_NBR'] != 226000
print(d1_new[d1_cleaned])

# Count the number of transactions by date
print(d1_new.groupby('DATE')['TXN_ID'].nunique()) # tính cột TXN_ID dựa trên cột đã nhóm DATE


# Create graph to see clearly
daily_transactions = d1_new.groupby(d1_new['DATE'].dt.date)['TXN_ID'].count()  # tính giao dịch từng ngày cho cả bảng
start_date = pd.to_datetime('2018-12-01').date()
end_date = pd.to_datetime('2018-12-31').date()
daily_transactions_2018 = daily_transactions[(daily_transactions.index >= start_date) & (daily_transactions.index <= end_date)] # chỉ lấy giao dịch từng ngày của năm 2018
print(daily_transactions_2018) 
# As I can see, there is a zero value in '2018-12-25', so let take a closer look in graph

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Tạo dải ngày từ 01/12/2018 đến 31/12/2018
date_range = pd.date_range(start='2018-12-01', end='2018-12-31')

# Gán lại chỉ mục với đầy đủ các ngày trong tháng 12/2018, đặt NaN cho các ngày bị thiếu (bao gồm 25/12)
daily_transactions_2018_full = daily_transactions_2018.reindex(date_range)
daily_transactions_2018_full.loc['2018-12-25'] = np.nan # Đặt NaN (giá trị rỗng) cho ngày 25/12/2018

plt.plot(daily_transactions_2018_full.index, daily_transactions_2018_full.values)
plt.xlabel('Date')
plt.ylabel('Transaction ID')
plt.title('Transaction ID Over Time') 
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
# I can see that the increase in sales occurs in the lead-up to Christmas and that there are zero sales on
# Christmas day itself. This is due to shops being closed on Christmas day.


## Step 4: creating other features such as brand of chips or pack size from PROD_NAME

# lấy số trong 1 cột dữ liệu
import re
d1_new['PACK_SIZE'] = d1_new['PROD_NAME'].str.extract(r'(\d+)')
# r: là một raw string (chuỗi thô)
# \d: đại diện cho một chữ số (digit)
# +: trong regex có nghĩa là "một hoặc nhiều lần
print(d1_new['PACK_SIZE'])

# vẽ đồ thị histogram
plt.hist(x = d1_new['PACK_SIZE'])
plt.show()

# lấy từ đầu tiên trong chuỗi của toàn bộ cột
d1_new['BRAND'] = d1_new['PROD_NAME'].str.split().str[0]
print(d1_new['BRAND'])

## Step 5: Examining customer data
# check out the customer data
d2 = pd.read_csv('QVI_purchase_behaviour.csv')
print(d2)

print(d2.describe())

# tạo một chức năng mới bao gồm length, class và mode:
from statistics import mode
def describe_data(data):
    # length
    length = len(data)

    # class
    data_class = type(data).__name__

    # mode
    try:
        data_mode = mode(data)
    except:
        data_mode = "No mode (or multiple modes)"
    
    # kết quả
    print(f"Length: {length}")
    print(f"Class: {data_class}")
    print(f"Mode: {data_mode}")

print(describe_data(d2['LIFESTAGE']))
print(describe_data(d2['PREMIUM_CUSTOMER']))

# examining the values of lifestage and premium_customer
life_stage = d2.groupby('LIFESTAGE')['LYLTY_CARD_NBR'].count()
print(life_stage)

premium_customer = d2.groupby('PREMIUM_CUSTOMER')['LYLTY_CARD_NBR'].count()
print(premium_customer)

# gộp 2 bảng dữ liệu có chung 1 cột (LYLTY_CARD_NBR) lại với nhau
merge_data = pd.merge(left = d1_new, right = d2, on = 'LYLTY_CARD_NBR', how = 'left')
print(merge_data)

# check for nulls
print(merge_data.isnull().sum())

# In dữ liệu: merge_data.to_csv('QVI_data.csv', index = False)

#### Data analysis on customer segments
## Total sales by LIFESTAGE and PREMIUM_CUSTOMER
df = pd.read_csv('QVI_data.csv')
print(df)

total_sales = df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum()
print(total_sales)

## Vẽ biểu đồ stacked bar plot 
# Loại bỏ khoảng trắng thừa trong cột LIFESTAGE (nếu có)
df['LIFESTAGE'] = df['LIFESTAGE'].str.strip()

# Tính tổng doanh số được nhóm theo LIFESTAGE và PREMIUM_CUSTOMER và chuyển đổi MultiIndex thành DataFrame
total_sales = df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum().unstack() # Sử dụng .unstack() để chuyển đổi MultiIndex thành một DataFrame.
print(total_sales)

# Tính tỷ lệ phần trăm cho mỗi nhóm
# chia mỗi giá trị trong DataFrame total_sales cho tổng của cột tương ứng (trong nhóm PREMIUM_CUSTOMER) và nhân với 100 để ra tỷ lệ phần trăm.
total_sales_pct = total_sales.div(total_sales.sum(axis=1), axis=0) * 100
# axis = 1 có nghĩa là tính tổng theo hàng, tức là tổng doanh số cho mỗi hàng LIFESTAGE
# axis = 0 có nghĩa là chia theo cột. Chia từng giá trị trong mỗi cột của DataFrame total_sales cho giá trị tổng tương ứng của hàng đó.

# Vẽ stacked bar plot
ax = total_sales_pct.plot(kind='bar', stacked=True)

# Hiển thị phần trăm trên mỗi thanh của biểu đồ
for container in ax.containers:
    labels = [f'{v.get_height():.2f}%' if v.get_height() > 0 else '' for v in container]
    ax.bar_label(container, labels=labels, label_type='center')
# [f'{v.get_height():.2f}%' if v.get_height() > 0 else '' for v in container]: Đây là một danh sách kiểu list comprehension, 
    # nơi bạn tạo ra các nhãn cho các thanh trong container hiện tại. Mỗi thanh trong container được đại diện bởi biến v

# v.get_height(): Phương thức này trả về chiều cao của từng thanh (bar) trong biểu đồ. 
    # Với stacked bar plot, chiều cao của mỗi thanh là giá trị đại diện cho phần của tổng của mỗi cột.

# f'{v.get_height():.2f}%': Đây là chuỗi định dạng, dùng để tạo ra nhãn (label) cho từng thanh với giá trị phần trăm. Cụ thể:
    # {v.get_height():.2f}: Định dạng giá trị chiều cao với 2 chữ số thập phân.
    # '%': Thêm dấu phần trăm vào nhãn để hiển thị tỷ lệ phần trăm.

# if v.get_height() > 0 else '': Điều kiện này kiểm tra nếu chiều cao của thanh lớn hơn 0, thì mới tạo nhãn cho nó. Nếu chiều cao là 0 (không có dữ liệu), nhãn sẽ để trống ('').

# ax.bar_label(container, labels=labels, label_type='center')
    # ax.bar_label(container, labels=labels): Phương thức bar_label() được sử dụng để thêm nhãn cho các thanh trong biểu đồ. 
    # Ở đây, container là nhóm thanh hiện tại và labels là danh sách các nhãn bạn đã tạo trong dòng trước.

    # label_type='center': Tham số này xác định vị trí của nhãn trên thanh. 
    # Khi sử dụng 'center', nhãn sẽ được đặt vào giữa thanh (theo chiều cao của thanh).

# Hiển thị biểu đồ
plt.show()


### Higher sales may also be driven by more units of chips being bought per customer. Let’s have a look at this next.
## Average number of units per customer by LIFESTAGE and PREMIUM_CUSTOMER
unique_customer = df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['LYLTY_CARD_NBR'].nunique()
total_customer = df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['PROD_QTY'].sum()
average_customer = total_customer/unique_customer
print(average_customer)

average_customer = average_customer.reset_index()

import seaborn as sns
# Vẽ biểu đồ
plt.figure(figsize=(10, 6))

# Tạo biểu đồ cột với seaborn
sns.barplot(data=average_customer, x='LIFESTAGE', y = 0, hue='PREMIUM_CUSTOMER', palette='Set1')
# y = 0: Ở đây cột có chỉ mục 0 chứa giá trị trung bình số lượng sản phẩm (trong DataFrame sau khi chia).
# hue='PREMIUM_CUSTOMER': Các màu sắc khác nhau đại diện cho các nhóm khách hàng (Budget, Mainstream, Premium).
# palette='Set1': Sử dụng bảng màu mặc định Set1 cho biểu đồ.


# Đặt tiêu đề và nhãn
plt.title('Units per customer', fontsize=14)
plt.xlabel('Lifestage', fontsize=12)
plt.ylabel('Avg units per transaction', fontsize=12)

# Xoay nhãn trên trục x để dễ đọc
plt.xticks(rotation=45, ha='right')

# Hiển thị chú thích
plt.legend(title='PREMIUM_CUSTOMER')

# Căn chỉnh bố cục
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()


### Let's investigate the average price per unit chips bought for each customer segment as this is also a driver of total sales.
## Average price per unit by LIFESTAGE and PREMIUM_CUSTOMER
total_sales_per_unit = df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['TOT_SALES'].sum()
average_price_per_unit = total_sales_per_unit/total_customer
print(average_price_per_unit)

average_price_per_unit = average_price_per_unit.reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=average_price_per_unit, x='LIFESTAGE', y = 0, hue='PREMIUM_CUSTOMER', palette='Set1')

# Đặt tiêu đề và nhãn
plt.title('Price per unit', fontsize=14)
plt.xlabel('Lifestage', fontsize=12)
plt.ylabel('Avg price per unit', fontsize=12)

# Xoay nhãn trên trục x để dễ đọc
plt.xticks(rotation=45, ha='right')

# Hiển thị chú thích
plt.legend(title='PREMIUM_CUSTOMER')

# Căn chỉnh bố cục
plt.tight_layout()
plt.show()
# Mainstream midage and young singles and couples are more willing to pay more per packet of chips compared 
# to their budget and premium counterparts. This may be due to premium shoppers being more likely to
# buy healthy snacks and when they buy chips, this is mainly for entertainment purposes rather than their own
# consumption. This is also supported by there being fewer premium midage and young singles and couples
# buying chips compared to their mainstream counterparts.

# As the difference in average price per unit isn’t large, we can check if this difference is statistically different.
from scipy import stats

df['price'] = df['TOT_SALES'] / df['PROD_QTY']

# Nhóm 1: Khách hàng thuộc Lifestage "YOUNG SINGLES/COUPLES" hoặc "MIDAGE SINGLES/COUPLES" và có phân khúc khách hàng là "Mainstream".
group_mainstream = df[(df['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES", "MIDAGE SINGLES/COUPLES"])) & 
                      (df['PREMIUM_CUSTOMER'] == 'Mainstream')]['price']

# Nhóm 2: Khách hàng thuộc Lifestage "YOUNG SINGLES/COUPLES" hoặc "MIDAGE SINGLES/COUPLES" nhưng không có phân khúc khách hàng là "Mainstream".
group_non_mainstream = df[(df['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES", "MIDAGE SINGLES/COUPLES"])) & 
                          (df['PREMIUM_CUSTOMER'] != 'Mainstream')]['price']

# Thực hiện t-test độc lập (one-tailed)
t_stat, p_value = stats.ttest_ind(group_mainstream, group_non_mainstream, equal_var=False)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# So sánh với mức ý nghĩa alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("Có sự khác biệt có ý nghĩa thống kê giữa giá trung bình của Mainstream và Premium.")
else:
    print("Không có sự khác biệt có ý nghĩa thống kê giữa giá trung bình của Mainstream và Premium.")

# The t-test results in a p-value < 2.2e-16, i.e. the unit price for mainstream, young and mid-age singles and
# couples are significantly higher than that of budget or premium, young and midage singles and couples



#### Deep dive into specific customer segments for insights
### Deep dive into Mainstream, young singles/couples
segment1 = df[(df['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & (df['PREMIUM_CUSTOMER'] == 'Mainstream')]
print(segment1)

other = df[~(df['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & (df['PREMIUM_CUSTOMER'] == 'Mainstream')]
print(other)


### Brand affinity compared to the rest of the population
quantity_segment1 = segment1['PROD_QTY'].sum()
print(quantity_segment1)

quantity_other = other['PROD_QTY'].sum()
print(quantity_other)

quantity_segment1_by_brand = segment1.groupby('BRAND')['PROD_QTY'].sum()/ quantity_segment1
print(quantity_segment1_by_brand)

quantity_other_by_brand = other.groupby('BRAND')['PROD_QTY'].sum()/ quantity_other
print(quantity_other_by_brand)

# Gộp hai bảng để tính chỉ số affinityToBrand
brand_proportions = pd.merge(quantity_segment1_by_brand, quantity_other_by_brand, on='BRAND', suffixes=('_segment1', '_other'))
# suffixes=('_segment1', '_other'): tự động thêm hậu tố _segment1 cho cột của DataFrame đầu tiên (tương ứng với segment1), 
# và thêm hậu tố _other cho cột của DataFrame thứ hai (tương ứng với other).
# để phân biệt giá trị của cột PROD_QTY trong nhóm segment1 với cột PROD_QTY trong nhóm other.

# Tính chỉ số affinityToBrand (độ gắn kết với thương hiệu)
brand_proportions['affinityToBrand'] = brand_proportions['PROD_QTY_segment1'] / brand_proportions['PROD_QTY_other']
print(brand_proportions)
# so sánh tỷ lệ sản phẩm của một thương hiệu được mua bởi nhóm mục tiêu (segment1) 
# với tỷ lệ sản phẩm của cùng thương hiệu được mua bởi nhóm còn lại (other).
# Công thức này đo lường xem thương hiệu có phổ biến hơn ở nhóm mục tiêu segment1 so với nhóm khác không. Cụ thể:
    # Nếu giá trị này lớn hơn 1, nghĩa là nhóm mục tiêu (segment1) có xu hướng mua thương hiệu này nhiều hơn so với nhóm khác.
    # Nếu giá trị này nhỏ hơn 1, nhóm mục tiêu mua thương hiệu này ít hơn so với nhóm còn lại.
    # Nếu bằng 1, thì tỷ lệ mua sản phẩm của thương hiệu đó trong cả hai nhóm là tương đương.

# Sắp xếp theo chỉ số affinityToBrand
brand_proportions_sorted = brand_proportions.sort_values(by='affinityToBrand', ascending=False)
print(brand_proportions_sorted)

# Let’s also find out if our target segment tends to buy larger packs of chips.
### Preferred pack size compared to the rest of the population
quantity_segment1_by_pack = segment1.groupby('PACK_SIZE')['PROD_QTY'].sum()/ quantity_segment1
print(quantity_segment1_by_pack)

quantity_other_by_pack = other.groupby('PACK_SIZE')['PROD_QTY'].sum()/ quantity_other
print(quantity_other_by_pack)

pack_proportion = pd.merge(quantity_segment1_by_pack, quantity_other_by_pack, on = 'PACK_SIZE', suffixes= ('_segment1', '_other'))
pack_proportion['affinityToPack'] = pack_proportion ['PROD_QTY_segment1'] / pack_proportion['PROD_QTY_other']
pack_proportion_sorted = pack_proportion.sort_values(by='affinityToPack', ascending=False)
print(pack_proportion_sorted)

# It looks like Mainstream young singles/couples are 27% more likely to purchase a 270g pack of chips compared 
# to the rest of the population but let’s dive into what brands sell this pack size.

fillterd_brand = df[df['PACK_SIZE'] == 270]
brand_with_pack_size_270 = fillterd_brand['BRAND'].unique()
print(brand_with_pack_size_270)

# Twisties are the only brand offering 270g packs and so this may instead be reflecting a higher likelihood of
# purchasing Twisties.
