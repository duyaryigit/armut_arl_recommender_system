#########################
# Business Problem
#########################

# Armut, which is Turkey's largest online service platform, brings together service providers and those who want to
# receive services. It enables easy access to services such as cleaning, renovation, and transportation with just a few
# touches on a computer or smartphone. An association rule learning-based product recommendation system is desired
# to be created using the dataset containing users who received services and the categories of services they received.

#########################
# Dataset Story
#########################

# The dataset consists of the services customers receive and the categories of these services. Date and time of each
# service received contains information.

# UserId: Customer ID
# ServiceId: They are anonymized services belonging to each category. (Example: Upholstery washing service under the cleaning category)
# A ServiceId can be found under different categories and refers to different services under different categories.
# (Example: Service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: They are anonymized categories. (Example: Cleaning, transportation, renovation category)
# CreateDate: The date the service was purchased

!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

#########################
# Task 1: Prepare and Understand Data (Data Understanding)
#########################

# 1: Read the armut_data.csv dataset.

df_ = pd.read_csv("datasets/armut_data.csv")
df = df_.copy()
df.head()

# Function to get a first look at the data.

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)

check_df(df)

# 2: ServiceID represents a different service for each CategoryID.
# Create a new variable to represent services by combining ServiceID and CategoryID with "_"

df["Services"] = df["ServiceId"].astype(str) + '_' + df["CategoryId"].astype(str)
df.head()

#or

df["Services"] = df[['ServiceId', 'CategoryId']].apply(lambda x: "_".join(x.astype(str)), axis=1)
df.head()

#or

df["Services"] = [str(col[1]) + "_" + str(col[2]) for col in df.values]
df.head()

# Step 3: The dataset consists of services purchased by customers with the date and time of purchase, but there is
# no basket definition (invoice, etc.). In order to apply Association Rule Learning, a basket definition needs to be
# created, which represents the services purchased by each customer on a monthly basis. For example, customer with
# ID 7256 has a basket consisting of services 9_4 and 46_4 purchased in August 2017, and a different basket consisting
# of services 9_4 and 38_4 purchased in October 2017. The baskets should be identified with a unique ID. To do this,
# first create a new date variable that only includes the year and month. Then combine UserId and the new date variable
# using "_" and assign it to a new variable named BasketId.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_date"] = df["CreateDate"].dt.strftime("%Y-%m")
df["BasketID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

#########################
# Task 2: Create Association Rules
#########################

# 1: Create the basket service pivot table as follows.

# Services         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

invoice_product_df = df.groupby(['BasketID', 'Services'])['Services'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
df_t= invoice_product_df
df_t.head()
invoice_product_df.loc[["10591_2018-08"]]

# 2: Create association rules.

frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

rules.sort_values(by="lift",ascending=False)

# 3: Use the arl_recommender function to recommend a service to a user who last received the 2_0 service.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate (sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    #recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0", 2)

# or

def arl_recommender1(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # Sorts the rules in order of lift from largest to smallest. (to catch the most compatible first product)
    # Sortable by confidence also depends on initiative.
    recommendation_list = [] # We create an empty list for recommended products.
    # antecedents: X
    # Returns as frozenset because it is called items. Combines index and service.
    # i: index
    # product: X, that is, the service that asks for suggestions
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product): # Loop through services
            if j == product_id:# if the recommended product is caught:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # You were holding the index information with i. Add the consequents(Y) value in this index information to the recommendation_list.

    # To avoid duplication in the recommendation list:
    # For example, in 2-to-3 combinations, the same product may have dropped to the list again;
    # The unique feature of the dictionary structure is used.

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] # :rec_count Get recommended items up to the desired number.


arl_recommender1(rules,"2_0", 1)

df.loc[df["Services"] == "2_0"].sort_values("CreateDate", ascending=False)

invoice_product_df.loc[["21857_2017-08"]]
