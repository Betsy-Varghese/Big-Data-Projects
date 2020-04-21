# Databricks notebook source
# MAGIC %md |GROUP PROJECT|
# MAGIC |------------|-------------|
# MAGIC |Group Members | <ul>Gutierrez Sara</ul><ul>Olivera Andres</ul><ul>Varghese Betsy</ul> |
# MAGIC |Year |2019 - 2020 |
# MAGIC |Course |Big Data Tools 2|

# COMMAND ----------

#Path
path = "/FileStore/tables/"

# COMMAND ----------

#####The difference between this model and the past one is that this one has less number of units. It passed from epochs=20 to 4. It Improved, the loss decreased and the accuracy grew.
from pyspark.sql.functions import *

#To create dummy variables
import pyspark.sql.functions as F 

#To create dummies
from pyspark.ml.feature import StringIndexer

#For Modelling
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Normalizer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


#For SMOTE to balance the target variable
#from imblearn.over_sampling import SMOTE
#from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split

#For Correlation
from pyspark.mllib.stat import Statistics
import pandas as pd

#For Segmentation
from pyspark.ml.clustering import KMeans

#For PCA
from pyspark.ml.feature import PCA

#For Chi Square
from pyspark.ml.feature import ChiSqSelector

#To extract second value of vector
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# COMMAND ----------

#Reading Data

#Read "Complaints" data
complaints = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Complaints.csv")

#Read "Customers" data
customers = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Customers.csv")

#Read "Delivery" data
delivery = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Delivery.csv")

#Read "Formula" data
formula = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Formula.csv")

#Read "Subscriptions" data
subscriptions = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Subscriptions.csv")

# COMMAND ----------

# DBTITLE 1,TEST SET


# COMMAND ----------

#Subsetting Data for the test set
last_date = '2018-09-01'
#Complaints
complaints = complaints.withColumn("ComplaintDate", to_date(col("ComplaintDate"), "yyyy-MM-dd")).filter(col("ComplaintDate")>=lit(last_date))
complaints = complaints.orderBy(desc('ComplaintDate'))
complaints.show(5)
#Delivery
delivery = delivery.withColumn("DeliveryDate", to_date(col("DeliveryDate"), "yyyy-MM-dd")).filter(col("DeliveryDate")>=lit(last_date))
delivery = delivery.orderBy(desc('DeliveryDate'))
delivery.show(5)
#Subscriptions
subscriptions = subscriptions.withColumn("EndDate", to_date(col("EndDate"), "yyyy-MM-dd")).filter(col("EndDate")>=lit(last_date))
subscriptions = subscriptions.orderBy(desc('EndDate'))
subscriptions.show(5)

# COMMAND ----------

# DBTITLE 1,Data Prep
#COMPLAINTS
#Check schema and first rows
complaints.printSchema() #Schema is ok
complaints.toPandas().head(5)

#Changing column types from string to integer
convert_int = ["ProductID","ComplaintTypeID","SolutionTypeID", "FeedbackTypeID"]

for i in convert_int:
    complaints = complaints.withColumn(i, complaints[i].cast("integer"))

#Check missings data
complaints.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in complaints.columns)) #No missings

#Get the number of complaints by CustomerID
complaints_1 = complaints.groupBy("CustomerID").agg(count("ComplaintID")).withColumnRenamed("CustomerID","1")

#Get the number of complaints per product by CustomerID
complaints_2 = complaints.groupBy("CustomerID").pivot("ProductName").agg(count("ProductName")).withColumnRenamed("NA","NA_Product").withColumnRenamed("CustomerID","2")

#Get the number of complaints per complaint type by Customer ID
complaints_3 = complaints.groupBy("CustomerID").pivot("ComplaintTypeDesc").agg(count("ComplaintTypeDesc")).withColumnRenamed("other","OtherComplaintType").withColumnRenamed("CustomerID","3")

#Get the number of complaints per complaint solution by Customer ID
complaints_4 = complaints.groupBy("CustomerID").pivot("SolutionTypeDesc").agg(count("SolutionTypeDesc")).withColumnRenamed("NA","NA_Solution").withColumnRenamed("other","OtherSolution").withColumnRenamed("CustomerID","4")

#Get the number of complaints per complaint feedback by Customer ID
complaints_5 = complaints.groupBy("CustomerID").pivot("FeedbackTypeDesc").agg(count("FeedbackTypeDesc")).withColumnRenamed("NA","NA_Feedback").withColumnRenamed("other","OtherFeedback").withColumnRenamed("CustomerID","5")

#Merge all tables
join_1 = complaints_1.join(complaints_2, complaints_1["1"] == complaints_2["2"]).drop("2")
join_2 = join_1.join(complaints_3, join_1["1"] == complaints_3["3"]).drop("3")
join_3 = join_2.join(complaints_4, join_2["1"] == complaints_4["4"]).drop("4")
complaints_final = join_3.join(complaints_5, join_3["1"] == complaints_5["5"]).drop("5").withColumnRenamed("1","cIDComplaint")

#Convert nulls into 0's
complaints_final = complaints_final.na.fill(0)
#CUSTOMERS
#Check schema and first rows
customers.printSchema() #Schema is ok
customers.toPandas().head(5)

#Find missings
customers.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in customers.columns)) #No missings

#Renaming the CustomerID column for future joins
customers = customers.withColumnRenamed("CustomerID","cIDCustomer")
#DELIVERY
#Check schema and first rows
delivery.printSchema() #Schema is ok
delivery.toPandas().head(5)

#Find missings
delivery.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in delivery.columns)) #Found 780 missings in the DeliveryClass column

#Treating missing values
delivery = delivery.where(col("DeliveryClass").isNotNull())

#Encoding string columns in "Delivery"
delivery = StringIndexer(inputCol="DeliveryClass", outputCol="DeliveryClass_index").fit(delivery).transform(delivery)
delivery = StringIndexer(inputCol="DeliveryTypeName", outputCol="DeliveryTypeName_index").fit(delivery).transform(delivery)

#Renaming the SubscriptionID column for future joins
delivery = delivery.withColumnRenamed("SubscriptionID","sID_Delivery")
#FORMULA
#Check schema and first rows
formula.printSchema() #Schema is ok
formula.toPandas().head(5)

#Find missings
formula.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in formula.columns)) #No missings

#Renaming the FormulaID column for future joins
formula = formula.withColumnRenamed("FormulaID","fID_Formula")
#SUBSCRIPTIONS
#Find missings
subscriptions.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in subscriptions.columns)) #Found 161 missings in the Price/Discount columns and 20 in the NbrMeans_EXCEP column

#Treating missing values
subscriptions = subscriptions.na.fill(0, subset=["NbrMeals_EXCEP", "GrossFormulaPrice", "NetFormulaPrice", "NbrMealsPrice", "ProductDiscount", "FormulaDiscount","TotalDiscount", "TotalPrice","TotalCredit"])

# COMMAND ----------

#JOINS
joinType="left_outer"

#Merging 'Delivery' and 'Subscriptions' by SubscriptionID
join_1 = subscriptions.join(delivery, delivery["sID_Delivery"] == subscriptions["SubscriptionID"],joinType).drop("sID_Delivery")

#Merging 'Join_1' and 'Formula' by FormulaID
table1 = join_1.join(formula, formula["fID_Formula"] == subscriptions["FormulaID"]).drop("fID_Formula")

# COMMAND ----------

#Changing column types from 'Subscriptions' from string to integers
convert_int = ["NbrMeals_EXCEP", "GrossFormulaPrice", "NetFormulaPrice", "NbrMealsPrice", "ProductDiscount", "FormulaDiscount", "TotalDiscount", "TotalPrice", "TotalCredit"]

for i in convert_int:
    table1 = table1.withColumn(i, table1[i].cast("integer"))

#Changing column types from 'Subscriptions' from string to timestamp
convert_date = ["StartDate","EndDate","RenewalDate","PaymentDate"]
    
for i in convert_date:
  table1 = table1.withColumn(i, table1[i].cast("timestamp"))
  
#Encoding string columns in merged table
table1 = StringIndexer(inputCol = "PaymentStatus", outputCol = "PaymentStatus_index").fit(table1).transform(table1)

#Creating meaningful Time variables
table1 = table1.withColumn("DaysSubscription", datediff(col("EndDate"), col("StartDate")))
table1 = table1.withColumn("MonthsSubscription", months_between(col("EndDate"), col("StartDate")))
table1 = table1.withColumn("Year", year("StartDate"))

# COMMAND ----------

#Feature engineering
#Aggregating variables by CustomerID
subs_totals = table1.groupBy("CustomerID").agg(count("SubscriptionID"), avg("DaysSubscription"), 
                                                  avg("MonthsSubscription"), sum("NbrMeals_REG"), sum("NbrMeals_EXCEP"), 
                                                  min("NbrMealsPrice"), max("NbrMealsPrice"), avg("NbrMealsPrice"), 
                                                  min("ProductDiscount"), max("ProductDiscount"), sum("ProductDiscount"), 
                                                  min("TotalDiscount"), max("TotalDiscount"), sum("TotalDiscount"),
                                                  min("TotalPrice"), max("TotalPrice"), sum("TotalPrice"), 
                                                  min("TotalCredit"), max("TotalCredit"),
                                                  sum("TotalCredit"))

#Aggregating variables by Product Type
subs_products = table1.groupBy("CustomerID").pivot("ProductName").agg(sum("NbrMeals_REG"), sum("NbrMeals_EXCEP"), sum("NbrMealsPrice"),
                                                                      sum("ProductDiscount"), sum("TotalDiscount"), sum("TotalPrice"),
                                                                      sum("TotalCredit")).withColumnRenamed("CustomerID","cIDProduct")

#Aggregating variables by Payment Type
subs_payment_type = table1.groupBy("CustomerID").pivot("PaymentType").agg(sum("TotalPrice"), sum("TotalCredit")).withColumnRenamed("CustomerID","cIDPayment")

#Aggregating variables by Start Year of Subscription
subs_year = table1.groupBy("CustomerID").pivot("Year").agg(count("SubscriptionID")).withColumnRenamed("CustomerID","cIDYear")

#Aggregating variables by Formula Type
subs_formula = table1.groupBy("CustomerID").pivot("FormulaType").agg(count("FormulaType"),sum("Duration"),avg("Duration")).withColumnRenamed("CustomerID","cIDFormula")
#Aggregating variables by DeliveryClass
subs_delivery = table1.groupBy("DeliveryID").pivot("DeliveryClass").agg(count("DeliveryClass"),count("DeliveryTypeName")).withColumnRenamed("DeliveryID","dID_Delivery").drop("null")

# COMMAND ----------

#Finding the latest date across the EndDate column 
MaxDate = subscriptions.groupBy("CustomerID").agg(max("EndDate"), min("StartDate"))
MaxDate.orderBy(desc("max(EndDate)")).show(1)

#Calculate difference between last subscription date in the entire dataset and the last subscription date for each customer
DifferenceDate = MaxDate.withColumn("DaysWithoutSub", datediff(to_date(lit("2019-03-02")), to_date("max(EndDate)","yyyy-MM-dd")))\
                        .withColumn("Customer_Lifetime", datediff(to_date("max(EndDate)", "yyyy-MM-dd"), to_date("min(StartDate)", "yyyy-MM-dd")))

churn = DifferenceDate.withColumnRenamed("CustomerID","cIDChurn")

# COMMAND ----------

#Merging the grouped columns together
joinType="left_outer"

join_1 = subs_totals.join(customers, customers["cIDCustomer"] == subs_totals["CustomerID"],joinType).drop("cIDCustomer")

#Creating dummies for region
categ = join_1.select('Region').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [F.when(F.col('Region') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
join_1 = join_1.select(join_1.columns + exprs)

join_2 = join_1.join(subs_products, subs_products["cIDProduct"] == join_1["CustomerID"],joinType).drop("cIDProduct")
join_3 = join_2.join(subs_payment_type, subs_payment_type["cIDPayment"] == join_2["CustomerID"],joinType).drop("cIDPayment")
join_4 = join_3.join(subs_year, subs_year["cIDYear"] == join_3["CustomerID"],joinType).drop("cIDYear")
join_5 = join_4.join(subs_formula, subs_formula["cIDFormula"] == join_4["CustomerID"],joinType).drop("cIDFormula")
join_6 = join_5.join(complaints_final, complaints_final["cIDComplaint"] == join_5["CustomerID"],joinType).drop("cIDComplaint")

#Merging 'Join_6' and 'Churn' by CustomerID
datamart = join_6.join(churn, churn["cIDChurn"] == join_5["CustomerID"],joinType).drop("cIDChurn")

datamart = datamart.withColumnRenamed('Churn','label')

#Convert nulls into 0's
data_test = datamart.na.fill(0)

# COMMAND ----------

#RENAMING COLUMNS

#For better comprehension

data_test = data_test.withColumnRenamed('count(SubscriptionID)', 'Total_Subscriptions')\
                   .withColumnRenamed('avg(DaysSubscription)', 'Avg_Subscription_Period')\
                   .withColumnRenamed('avg(DaysSubscription)', 'Avg_Subscription_Period_inDays')\
                   .withColumnRenamed('avg(MonthsSubscription)', 'Avg_Subscription_Period_inMonths')\
                   .withColumnRenamed('sum(NbrMeals_REG)', 'Total_meals_Regular')\
                   .withColumnRenamed('sum(NbrMeals_EXCEP)', 'Total_meals_Exceptional')\
                   .withColumnRenamed('sum(NbrMeals_EXCEP)', 'Total_meals_Exceptional')\
                   .withColumnRenamed('avg(NbrMealsPrice)', 'Avg_Meal_Price')\
                   .withColumnRenamed('avg(ProductDiscount)', 'Avg_Product_Discount')\
                   .withColumnRenamed('avg(TotalDiscount)', 'Avg_Total_Discount')\
                   .withColumnRenamed('sum(TotalPrice)', 'Total_Price')\
                   .withColumnRenamed('avg(TotalPrice)', 'Avg_Price')\
                   .withColumnRenamed('sum(TotalCredit)', 'Total_Credit')\
                   .withColumnRenamed('avg(TotalCredit)', 'Avg_Credit')\
                   .withColumnRenamed('Custom Events_sum(CAST(NbrMeals_REG AS BIGINT))', 'Total_NbrMeals_REG_Custom')\
                   .withColumnRenamed('Custom Events_sum(CAST(NbrMeals_EXCEP AS BIGINT))', 'Total_NbrMeals_EXCEP_Custom')\
                   .withColumnRenamed('Custom Events_sum(CAST(NbrMealsPrice AS BIGINT))', 'Total_Custom_NbrMeals_REG')\
                   .withColumnRenamed('Custom Events_sum(CAST(ProductDiscount AS BIGINT))', 'Total_Custom_NbrMeals_EXCEP')\
                   .withColumnRenamed('Custom Events_sum(CAST(TotalDiscount AS BIGINT))', 'Total_Discount_Custom')\
                   .withColumnRenamed('Custom Events_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_Custom')\
                   .withColumnRenamed('Custom Events_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_Custom')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(NbrMeals_REG AS BIGINT))', 'Total_NbrMeals_REG_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(NbrMeals_EXCEP AS BIGINT))', 'Total_NbrMeals_EXCEP_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(NbrMealsPrice AS BIGINT))', 'Total_NbrMealsPrice_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(ProductDiscount AS BIGINT))', 'Total_ProductDiscount_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(TotalDiscount AS BIGINT))', 'Total_Discount_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_GF')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(NbrMeals_REG AS BIGINT))', 'Total_NbrMeals_REG_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(NbrMeals_EXCEP AS BIGINT))', 'Total_NbrMeals_EXCEP_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(NbrMealsPrice AS BIGINT))', 'Total_NbrMealsPrice_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(ProductDiscount AS BIGINT))', 'Total_ProductDiscount_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(TotalDiscount AS BIGINT))', 'Total_Discount_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_GMax')\
                   .withColumnRenamed('Grub Mini_sum(CAST(NbrMeals_REG AS BIGINT))', 'Total_NbrMeals_REG_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(NbrMeals_EXCEP AS BIGINT))', 'Total_NbrMeals_EXCEP_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(ProductDiscount AS BIGINT))', 'Total_ProductDiscount_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(TotalDiscount AS BIGINT))', 'Total_Discount_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(NbrMealsPrice AS BIGINT))', 'Total_NbrMealsPrice_GMin')\
                   .withColumnRenamed('BT_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_BankTransfer')\
                   .withColumnRenamed('BT_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_BankTransfer')\
                   .withColumnRenamed('DD_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_Card')\
                   .withColumnRenamed('DD_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_Card')\
                   .withColumnRenamed('2014', 'Total_Subscriptions_14')\
                   .withColumnRenamed('2015', 'Total_Subscriptions_15')\
                   .withColumnRenamed('2016', 'Total_Subscriptions_16')\
                   .withColumnRenamed('2017', 'Total_Subscriptions_17')\
                   .withColumnRenamed('2018', 'Total_Subscriptions_18')\
                   .withColumnRenamed('2019', 'Total_Subscriptions_19')\
                   .withColumnRenamed('CAM_count(FormulaType)', 'Count_Formula_DirectMail')\
                   .withColumnRenamed('CAM_sum(CAST(Duration AS BIGINT))', 'Total_Formula_Duration_DirectMail')\
                   .withColumnRenamed('CAM_avg(CAST(Duration AS BIGINT))', 'Avg_Formula_Duration_DirectMail')\
                   .withColumnRenamed('REG_count(FormulaType)', 'Count_Formula_Reg')\
                   .withColumnRenamed('REG_sum(CAST(Duration AS BIGINT))', 'Total_Formula_Duration_Reg')\
                   .withColumnRenamed('REG_avg(CAST(Duration AS BIGINT))', 'Avg_Formula_Duration_Reg')\
                   .withColumnRenamed('Grub Flexi (excl. staff)','Grub_Flexi')\
                   .withColumnRenamed('Grub Maxi (incl. staff)','Grub_Maxi')

# COMMAND ----------

display(data_test)

# COMMAND ----------

# DBTITLE 1,TRAIN & VALIDATION
#Reading Data for Train and Validation

#Path
path = "/FileStore/tables/"

#Read "Complaints" data
complaints = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Complaints.csv")

#Read "Customers" data
customers = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Customers.csv")

#Read "Delivery" data
delivery = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Delivery.csv")

#Read "Formula" data
formula = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Formula.csv")

#Read "Subscriptions" data
subscriptions = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "BDT2_1920_Subscriptions.csv")

# COMMAND ----------

#Subsetting Data for Train and Validation

last_date = '2018-08-31'
#Complaints
complaints = complaints.withColumn("ComplaintDate", to_date(col("ComplaintDate"), "yyyy-MM-dd")).filter(col("ComplaintDate")<=lit(last_date))
complaints = complaints.orderBy(desc('ComplaintDate'))
complaints.show(5)
#Delivery
delivery = delivery.withColumn("DeliveryDate", to_date(col("DeliveryDate"), "yyyy-MM-dd")).filter(col("DeliveryDate")<=lit(last_date))
delivery = delivery.orderBy(desc('DeliveryDate'))
delivery.show(5)
#Subscriptions
subscriptions = subscriptions.withColumn("EndDate", to_date(col("EndDate"), "yyyy-MM-dd")).filter(col("EndDate")<=lit(last_date))
subscriptions = subscriptions.orderBy(desc('EndDate'))
subscriptions.show(5)

# COMMAND ----------

#COMPLAINTS
#Check schema and first rows
complaints.printSchema() #Schema is ok
complaints.toPandas().head(5)

#Changing column types from string to integer
convert_int = ["ProductID","ComplaintTypeID","SolutionTypeID", "FeedbackTypeID"]

for i in convert_int:
    complaints = complaints.withColumn(i, complaints[i].cast("integer"))

#Check missings data
complaints.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in complaints.columns)) #No missings

#Get the number of complaints by CustomerID
complaints_1 = complaints.groupBy("CustomerID").agg(count("ComplaintID")).withColumnRenamed("CustomerID","1")

#Get the number of complaints per product by CustomerID
complaints_2 = complaints.groupBy("CustomerID").pivot("ProductName").agg(count("ProductName")).withColumnRenamed("NA","NA_Product").withColumnRenamed("CustomerID","2")

#Get the number of complaints per complaint type by Customer ID
complaints_3 = complaints.groupBy("CustomerID").pivot("ComplaintTypeDesc").agg(count("ComplaintTypeDesc")).withColumnRenamed("other","OtherComplaintType").withColumnRenamed("CustomerID","3")

#Get the number of complaints per complaint solution by Customer ID
complaints_4 = complaints.groupBy("CustomerID").pivot("SolutionTypeDesc").agg(count("SolutionTypeDesc")).withColumnRenamed("NA","NA_Solution").withColumnRenamed("other","OtherSolution").withColumnRenamed("CustomerID","4")

#Get the number of complaints per complaint feedback by Customer ID
complaints_5 = complaints.groupBy("CustomerID").pivot("FeedbackTypeDesc").agg(count("FeedbackTypeDesc")).withColumnRenamed("NA","NA_Feedback").withColumnRenamed("other","OtherFeedback").withColumnRenamed("CustomerID","5")

#Merge all tables
join_1 = complaints_1.join(complaints_2, complaints_1["1"] == complaints_2["2"]).drop("2")
join_2 = join_1.join(complaints_3, join_1["1"] == complaints_3["3"]).drop("3")
join_3 = join_2.join(complaints_4, join_2["1"] == complaints_4["4"]).drop("4")
complaints_final = join_3.join(complaints_5, join_3["1"] == complaints_5["5"]).drop("5").withColumnRenamed("1","cIDComplaint")

#Convert nulls into 0's
complaints_final = complaints_final.na.fill(0)
#CUSTOMERS
#Check schema and first rows
customers.printSchema() #Schema is ok
customers.toPandas().head(5)

#Find missings
customers.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in customers.columns)) #No missings

#Renaming the CustomerID column for future joins
customers = customers.withColumnRenamed("CustomerID","cIDCustomer")
#DELIVERY
#Check schema and first rows
delivery.printSchema() #Schema is ok
delivery.toPandas().head(5)

#Find missings
delivery.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in delivery.columns)) #Found 780 missings in the DeliveryClass column

#Treating missing values
delivery = delivery.where(col("DeliveryClass").isNotNull())

#Encoding string columns in "Delivery"
delivery = StringIndexer(inputCol="DeliveryClass", outputCol="DeliveryClass_index").fit(delivery).transform(delivery)
delivery = StringIndexer(inputCol="DeliveryTypeName", outputCol="DeliveryTypeName_index").fit(delivery).transform(delivery)

#Renaming the SubscriptionID column for future joins
delivery = delivery.withColumnRenamed("SubscriptionID","sID_Delivery")
#FORMULA
#Check schema and first rows
formula.printSchema() #Schema is ok
formula.toPandas().head(5)

#Find missings
formula.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in formula.columns)) #No missings

#Renaming the FormulaID column for future joins
formula = formula.withColumnRenamed("FormulaID","fID_Formula")
#SUBSCRIPTIONS
#Find missings
subscriptions.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in subscriptions.columns)) #Found 161 missings in the Price/Discount columns and 20 in the NbrMeans_EXCEP column

#Treating missing values
subscriptions = subscriptions.na.fill(0, subset=["NbrMeals_EXCEP", "GrossFormulaPrice", "NetFormulaPrice", "NbrMealsPrice", "ProductDiscount", "FormulaDiscount","TotalDiscount", "TotalPrice","TotalCredit"])

# COMMAND ----------

#JOINS
joinType="left_outer"

#Merging 'Delivery' and 'Subscriptions' by SubscriptionID
join_1 = subscriptions.join(delivery, delivery["sID_Delivery"] == subscriptions["SubscriptionID"],joinType).drop("sID_Delivery")

#Merging 'Join_1' and 'Formula' by FormulaID
table1 = join_1.join(formula, formula["fID_Formula"] == subscriptions["FormulaID"]).drop("fID_Formula")

# COMMAND ----------

#Changing column types from 'Subscriptions' from string to integers
convert_int = ["NbrMeals_EXCEP", "GrossFormulaPrice", "NetFormulaPrice", "NbrMealsPrice", "ProductDiscount", "FormulaDiscount", "TotalDiscount", "TotalPrice", "TotalCredit"]

for i in convert_int:
    table1 = table1.withColumn(i, table1[i].cast("integer"))

#Changing column types from 'Subscriptions' from string to timestamp
convert_date = ["StartDate","EndDate","RenewalDate","PaymentDate"]
    
for i in convert_date:
  table1 = table1.withColumn(i, table1[i].cast("timestamp"))
  
#Encoding string columns in merged table
table1 = StringIndexer(inputCol = "PaymentStatus", outputCol = "PaymentStatus_index").fit(table1).transform(table1)

#Creating meaningful Time variables
table1 = table1.withColumn("DaysSubscription", datediff(col("EndDate"), col("StartDate")))
table1 = table1.withColumn("MonthsSubscription", months_between(col("EndDate"), col("StartDate")))
table1 = table1.withColumn("Year", year("StartDate"))

# COMMAND ----------

#Aggregating variables by CustomerID
subs_totals = table1.groupBy("CustomerID").agg(count("SubscriptionID"), avg("DaysSubscription"), 
                                                  avg("MonthsSubscription"), sum("NbrMeals_REG"), sum("NbrMeals_EXCEP"), 
                                                  min("NbrMealsPrice"), max("NbrMealsPrice"), avg("NbrMealsPrice"), 
                                                  min("ProductDiscount"), max("ProductDiscount"), sum("ProductDiscount"), 
                                                  min("TotalDiscount"), max("TotalDiscount"), sum("TotalDiscount"),
                                                  min("TotalPrice"), max("TotalPrice"), sum("TotalPrice"), 
                                                  min("TotalCredit"), max("TotalCredit"),
                                                  sum("TotalCredit"))

#Aggregating variables by Product Type
subs_products = table1.groupBy("CustomerID").pivot("ProductName").agg(sum("NbrMeals_REG"), sum("NbrMeals_EXCEP"), sum("NbrMealsPrice"),
                                                                      sum("ProductDiscount"), sum("TotalDiscount"), sum("TotalPrice"),
                                                                      sum("TotalCredit")).withColumnRenamed("CustomerID","cIDProduct")

#Aggregating variables by Payment Type
subs_payment_type = table1.groupBy("CustomerID").pivot("PaymentType").agg(sum("TotalPrice"), sum("TotalCredit")).withColumnRenamed("CustomerID","cIDPayment")

#Aggregating variables by Start Year of Subscription
subs_year = table1.groupBy("CustomerID").pivot("Year").agg(count("SubscriptionID")).withColumnRenamed("CustomerID","cIDYear")

#Aggregating variables by Formula Type
subs_formula = table1.groupBy("CustomerID").pivot("FormulaType").agg(count("FormulaType"),sum("Duration"),avg("Duration")).withColumnRenamed("CustomerID","cIDFormula")
#Aggregating variables by DeliveryClass
subs_delivery = table1.groupBy("DeliveryID").pivot("DeliveryClass").agg(count("DeliveryClass"),count("DeliveryTypeName")).withColumnRenamed("DeliveryID","dID_Delivery").drop("null")

# COMMAND ----------

#Finding the latest date across the EndDate column 
MaxDate = subscriptions.groupBy("CustomerID").agg(max("EndDate"), min("StartDate"))
MaxDate.orderBy(desc("max(EndDate)")).show(1)

#Calculate difference between last subscription date in the entire dataset and the last subscription date for each customer
DifferenceDate = MaxDate.withColumn("DaysWithoutSub", datediff(to_date(lit("2018-08-31")), to_date("max(EndDate)","yyyy-MM-dd")))\
                        .withColumn("Customer_Lifetime", datediff(to_date("max(EndDate)", "yyyy-MM-dd"), to_date("min(StartDate)", "yyyy-MM-dd")))

#Final dataframe for the target variable [3 months]
churn = DifferenceDate.withColumn("Churn", when(col("DaysWithoutSub") > 180, 1).otherwise(0))
churn = churn.withColumnRenamed("CustomerID","cIDChurn")

# COMMAND ----------

#Merging the grouped columns together
joinType="left_outer"

join_1 = subs_totals.join(customers, customers["cIDCustomer"] == subs_totals["CustomerID"],joinType).drop("cIDCustomer")

#Creating dummies for region
categ = join_1.select('Region').distinct().rdd.flatMap(lambda x:x).collect()
exprs = [F.when(F.col('Region') == cat,1).otherwise(0)\
            .alias(str(cat)) for cat in categ]
join_1 = join_1.select(join_1.columns + exprs)

join_2 = join_1.join(subs_products, subs_products["cIDProduct"] == join_1["CustomerID"],joinType).drop("cIDProduct")
join_3 = join_2.join(subs_payment_type, subs_payment_type["cIDPayment"] == join_2["CustomerID"],joinType).drop("cIDPayment")
join_4 = join_3.join(subs_year, subs_year["cIDYear"] == join_3["CustomerID"],joinType).drop("cIDYear")
join_5 = join_4.join(subs_formula, subs_formula["cIDFormula"] == join_4["CustomerID"],joinType).drop("cIDFormula")
join_6 = join_5.join(complaints_final, complaints_final["cIDComplaint"] == join_5["CustomerID"],joinType).drop("cIDComplaint")

#Merging 'Join_6' and 'Churn' by CustomerID
datamart = join_6.join(churn, churn["cIDChurn"] == join_5["CustomerID"],joinType).drop("cIDChurn")

datamart = datamart.withColumnRenamed('Churn','label')

#Convert nulls into 0's
data_train = datamart.na.fill(0)

# COMMAND ----------

#RENAMING COLUMNS

#For better comprehension

data_train = data_train.withColumnRenamed('count(SubscriptionID)', 'Total_Subscriptions')\
                   .withColumnRenamed('avg(DaysSubscription)', 'Avg_Subscription_Period')\
                   .withColumnRenamed('avg(DaysSubscription)', 'Avg_Subscription_Period_inDays')\
                   .withColumnRenamed('avg(MonthsSubscription)', 'Avg_Subscription_Period_inMonths')\
                   .withColumnRenamed('sum(NbrMeals_REG)', 'Total_meals_Regular')\
                   .withColumnRenamed('sum(NbrMeals_EXCEP)', 'Total_meals_Exceptional')\
                   .withColumnRenamed('sum(NbrMeals_EXCEP)', 'Total_meals_Exceptional')\
                   .withColumnRenamed('avg(NbrMealsPrice)', 'Avg_Meal_Price')\
                   .withColumnRenamed('avg(ProductDiscount)', 'Avg_Product_Discount')\
                   .withColumnRenamed('avg(TotalDiscount)', 'Avg_Total_Discount')\
                   .withColumnRenamed('sum(TotalPrice)', 'Total_Price')\
                   .withColumnRenamed('avg(TotalPrice)', 'Avg_Price')\
                   .withColumnRenamed('sum(TotalCredit)', 'Total_Credit')\
                   .withColumnRenamed('avg(TotalCredit)', 'Avg_Credit')\
                   .withColumnRenamed('Custom Events_sum(CAST(NbrMeals_REG AS BIGINT))', 'Total_NbrMeals_REG_Custom')\
                   .withColumnRenamed('Custom Events_sum(CAST(NbrMeals_EXCEP AS BIGINT))', 'Total_NbrMeals_EXCEP_Custom')\
                   .withColumnRenamed('Custom Events_sum(CAST(NbrMealsPrice AS BIGINT))', 'Total_Custom_NbrMeals_REG')\
                   .withColumnRenamed('Custom Events_sum(CAST(ProductDiscount AS BIGINT))', 'Total_Custom_NbrMeals_EXCEP')\
                   .withColumnRenamed('Custom Events_sum(CAST(TotalDiscount AS BIGINT))', 'Total_Discount_Custom')\
                   .withColumnRenamed('Custom Events_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_Custom')\
                   .withColumnRenamed('Custom Events_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_Custom')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(NbrMeals_REG AS BIGINT))', 'Total_NbrMeals_REG_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(NbrMeals_EXCEP AS BIGINT))', 'Total_NbrMeals_EXCEP_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(NbrMealsPrice AS BIGINT))', 'Total_NbrMealsPrice_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(ProductDiscount AS BIGINT))', 'Total_ProductDiscount_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(TotalDiscount AS BIGINT))', 'Total_Discount_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_GF')\
                   .withColumnRenamed('Grub Flexi (excl. staff)_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_GF')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(NbrMeals_REG AS BIGINT))', 'Total_NbrMeals_REG_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(NbrMeals_EXCEP AS BIGINT))', 'Total_NbrMeals_EXCEP_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(NbrMealsPrice AS BIGINT))', 'Total_NbrMealsPrice_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(ProductDiscount AS BIGINT))', 'Total_ProductDiscount_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(TotalDiscount AS BIGINT))', 'Total_Discount_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_GMax')\
                   .withColumnRenamed('Grub Maxi (incl. staff)_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_GMax')\
                   .withColumnRenamed('Grub Mini_sum(CAST(NbrMeals_REG AS BIGINT))', 'Total_NbrMeals_REG_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(NbrMeals_EXCEP AS BIGINT))', 'Total_NbrMeals_EXCEP_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(ProductDiscount AS BIGINT))', 'Total_ProductDiscount_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(TotalDiscount AS BIGINT))', 'Total_Discount_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_GMin')\
                   .withColumnRenamed('Grub Mini_sum(CAST(NbrMealsPrice AS BIGINT))', 'Total_NbrMealsPrice_GMin')\
                   .withColumnRenamed('BT_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_BankTransfer')\
                   .withColumnRenamed('BT_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_BankTransfer')\
                   .withColumnRenamed('DD_sum(CAST(TotalPrice AS BIGINT))', 'Total_Price_Card')\
                   .withColumnRenamed('DD_sum(CAST(TotalCredit AS BIGINT))', 'Total_Credit_Card')\
                   .withColumnRenamed('2014', 'Total_Subscriptions_14')\
                   .withColumnRenamed('2015', 'Total_Subscriptions_15')\
                   .withColumnRenamed('2016', 'Total_Subscriptions_16')\
                   .withColumnRenamed('2017', 'Total_Subscriptions_17')\
                   .withColumnRenamed('2018', 'Total_Subscriptions_18')\
                   .withColumnRenamed('2019', 'Total_Subscriptions_19')\
                   .withColumnRenamed('CAM_count(FormulaType)', 'Count_Formula_DirectMail')\
                   .withColumnRenamed('CAM_sum(CAST(Duration AS BIGINT))', 'Total_Formula_Duration_DirectMail')\
                   .withColumnRenamed('CAM_avg(CAST(Duration AS BIGINT))', 'Avg_Formula_Duration_DirectMail')\
                   .withColumnRenamed('REG_count(FormulaType)', 'Count_Formula_Reg')\
                   .withColumnRenamed('REG_sum(CAST(Duration AS BIGINT))', 'Total_Formula_Duration_Reg')\
                   .withColumnRenamed('REG_avg(CAST(Duration AS BIGINT))', 'Avg_Formula_Duration_Reg')\
                   .withColumnRenamed('Grub Flexi (excl. staff)','Grub_Flexi')\
                   .withColumnRenamed('Grub Maxi (incl. staff)','Grub_Maxi')

# COMMAND ----------

#Check blance of label
data_train.select("CustomerID", "label").groupBy("label").agg(count("CustomerID")).show()

# COMMAND ----------

#SPLIT THE DATA
#Create train and validation set with 70% train, 30% test split
train, validation = data_train.randomSplit([0.7, 0.3], seed=123)

# COMMAND ----------

#Display to visualize results and download data
display(train)

# COMMAND ----------

#Display to visualize results and download data
display(validation)

# COMMAND ----------

# DBTITLE 1,Feature Selection
#CORRELATION

#Analyze correlations between our target and all variables
from pyspark.mllib.stat import Statistics
import pandas as pd

datamart_features = train.drop('StreetID','DaysWithoutSub','max(EndDate)', 'CustomerID')
col_names = datamart_features.columns
features = datamart_features.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names

# COMMAND ----------

# #Find most correlated variables
corr_df[corr_df["label"]>0.12]

# COMMAND ----------

# #Find most correlated variables
corr_df[corr_df["label"]<-0.12]

# COMMAND ----------

# #Create a list with all the variables found before (drop NA_Feedback)
#Removed "DaysWithoutSub" - it has high correlation with the churn variable becuase the churn was created out of it

correlated = ['Count_Formula_DirectMail', 'Avg_Formula_Duration_DirectMail', 'Total_Subscriptions', 'Avg_Subscription_Period', 'Avg_Subscription_Period_inMonths', 'Total_meals_Regular',                           'Total_meals_Exceptional', 'max(NbrMealsPrice)', 'min(NbrMealsPrice)', 'Avg_Meal_Price', 'min(TotalPrice)', 'max(TotalPrice)', 'Total_Price', 
              'Total_NbrMeals_REG_GMax', 'Total_NbrMeals_EXCEP_GMax', 'Total_NbrMealsPrice_GMax', 'Total_Price_GMax', 
              'Total_Price_BankTransfer', 'Total_Price_Card','Total_Subscriptions_15', 'Total_Subscriptions_16', 'Total_Subscriptions_17', 'Total_Subscriptions_18', 'Total_Subscriptions_19', 
              'Count_Formula_Reg', 'Total_Formula_Duration_Reg', 'Avg_Formula_Duration_Reg']

va = VectorAssembler().setInputCols(correlated).setOutputCol("corrfeatures")
datamart_corrfeatures = va.transform(datamart_features)

# COMMAND ----------

#Pipeline with VA,PCA, Chi^2, Stand and Norm

#Concatenate all numeric features in one big vector
predictors = data_train.drop('StreetID','DaysWithoutSub','max(EndDate)', 'min(StartDate)','CustomerID', 'label')
va = VectorAssembler().setInputCols(predictors.columns).setOutputCol("va_features")

#Feature Reduction using Principal Components Analysis (PCA)
pca = PCA().setInputCol("va_features").setOutputCol("pcafeatures").setK(10)

#Chi Square Selector 
chisq = ChiSqSelector().setFeaturesCol("va_features").setOutputCol("chi_features").setLabelCol("label").setNumTopFeatures(30)


# #Standardization
stand = StandardScaler(inputCol="chi_features",outputCol="features",withStd=True, withMean=True)

# #Normalisation
normalizer =  Normalizer().setInputCol("va_features").setOutputCol("normFeatures").setP(1.0)

#Use first piple when making big benchmark
pipeline_train = Pipeline(stages=[va, pca, chisq, stand, normalizer])
train = pipeline_train.fit(train).transform(train)


#vector assembler for validation and standarizing it 
va_test = VectorAssembler().setInputCols(predictors.columns).setOutputCol("va_features_validation")
stand_test = StandardScaler(inputCol='va_features_validation',outputCol="features",withStd=True, withMean=True)
Pipeline_test = Pipeline(stages=[va_features_test,stand_test])
validation = Pipeline_test.fit(validation).transform(validation)

# COMMAND ----------

#Getting the most relevant features selected by Chi^2 
chisq2 = ChiSqSelector().setFeaturesCol('va_features').setOutputCol("chi_features2").setLabelCol("label").setNumTopFeatures(30)
vvv = chisq2.fit(train)
importantFeatures = vvv.selectedFeatures
variables = train.columns
select_features = []
for i in importantFeatures:
  select_features.append(variables[i])
select_features

# COMMAND ----------

# DBTITLE 1,MODELING


# COMMAND ----------

#Selecting only relevant variables that the models use
train = train.select('features','label')
validation = test.select('features','label')

# COMMAND ----------

#DEFINING THE MODELS AND PARAMETERS

#NOTE: Due to the time Databricks was taking to run and that after 2 hours the clusters detaches, we are keeping only these combinations of hyperparameters in here. The models were run with more combinations of Hyperparameters in a Jupyter Notebook that is attached. 

#NAIVE BAYES MODEL
nb = NaiveBayes()
nbParams = ParamGridBuilder().addGrid(nb.smoothing, [0.01,1]).build()

#LOGISTIC REGRESSION MODEL
lr = LogisticRegression()
lrParams = ParamGridBuilder().addGrid(lr.maxIter, [10, 150]).build()

#RANDOM FOREST MODEL
rfc = RandomForestClassifier()
rfParams = ParamGridBuilder().addGrid(rfc.numTrees, [150, 300]).build()

#DECISION TREE
dt = DecisionTreeClassifier()
dtParams = ParamGridBuilder().addGrid(dt.maxDepth, [4, 10]).build()

#GRADIENT BOOSTING MODEL
gb = GBTClassifier()
gbParams = ParamGridBuilder().addGrid(gb.maxDepth,[2,4]).build()



### Hyperparameters used for the analysis (ran in Jypyter) ###
# #Gradient Boosting
# gb = GradientBoostingClassifier()
# gbParam = {'learning_rate':[0.001, 0.01, 0.1, 0.5,0.8, 1], 'n_estimators':[100,500,1000,1750]}

# #Random Forest
# rfc=RandomForestClassifier()
# rfcParams = {'n_estimators': [50,100,200,300,500],'max_features': ['auto', 'sqrt'],'max_depth' : [4,6,8]}

# #Logistic Regression
# lr = LogisticRegression()
# c_space = np.logspace(-5, 8, 15)
# lrParams = {'C': c_space}

# #KNN
# knn = KNeighborsClassifier()
# knnlrParams = {'n_neighbors':[4,5,7],'leaf_size':[1,3,5],'algorithm':['auto', 'kd_tree']}

# COMMAND ----------

#MODEL EVALUATION

#list of models for loop
models = (rfc, gb, lr, dt)
#list of param for loop
param = (rfParams, gbParams, lrParams, dtParams)

#Creating empty dataframe to store evaluation metrics of models
results = spark.createDataFrame(data=[(0,0,0,0,0,0)],schema=["Model",'AUC','Accuracy','Weighted_Precision','Precision', 'Recall'])

for x in  range(len(models)):  
  pipe_model = Pipeline().setStages([models[x]])
  
  cv = CrossValidator().setEstimator(pipe_model).setEstimatorParamMaps(param[x]).setEvaluator(evaluator).setNumFolds(5)

  cvModel = cv.fit(training) 

  #Get predictions on the test set
  preds = cvModel.transform(validation)
   
  #save predictions 
  if x == 0: 
    pred1 = preds
  elif x == 1:
    pred2 = preds
  else:
    pred3 = preds
  
  # Create a confusion matrix
  preds.groupBy('label', 'prediction').count()

  # Calculate the elements of the confusion matrix
  TN = preds.filter('prediction = 0 AND label = prediction').count()
  TP = preds.filter('prediction = 1 AND label = prediction').count()
  FN = preds.filter('prediction = 0 AND label != prediction').count()
  FP = preds.filter('prediction = 1 AND label != prediction').count()

  # Accuracy measures the proportion of correct predictions
  accuracy = (TN + TP) / (TN + TP + FN + FP)

  # Calculate precision and recall
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  
  # Find weighted precision
  multi_evaluator = MulticlassClassificationEvaluator()
  weighted_precision = multi_evaluator.evaluate(preds, {multi_evaluator.metricName: "weightedPrecision"})

  # Find AUC
  binary_evaluator = BinaryClassificationEvaluator()
  auc = binary_evaluator.evaluate(preds, {binary_evaluator.metricName: "areaUnderROC"})

  #Create a new DataFrame

  #get metrics in data frame
  results_inf = spark.createDataFrame(data=[(str(models[x]), auc, accuracy, weighted_precision, precision, recall)]\
                                      ,schema=["Model",'AUC','Accuracy','Weighted_Precision','Precision', 'Recall'])
  #Append all results in one dataframe
  results = results.union(results_inf)
  results.show()

results.show()

# COMMAND ----------

#Get predictors
predictors2 = datamart.drop('StreetID','DaysWithoutSub','max(EndDate)', 'CustomerID', 'label')
#Assembler object
va = VectorAssembler().setInputCols(predictors2.columns).setOutputCol("all_features")
#Scaler object
stand = StandardScaler(inputCol="all_features",outputCol="features",withStd=True, withMean=True)
#Creating pipleline
pipeline_datapreds = Pipeline(stages=[va, stand])
#Adding a variable name features with all predictors standarized
data_preds = pipeline_datapreds.fit(datamart).transform(datamart)

# COMMAND ----------

#Getting only current customers
test = test.filter(col('label')==0)
#Droping label
test = test.drop('label')

# COMMAND ----------

#Getting churning probability of current customers
final_preds = model1.transform(test).select('probability','CustomerID','prediction')

# COMMAND ----------

#Visualizing probabilities oupput
final_preds.select('probability','CustomerID','prediction').toPandas().head(10)

# COMMAND ----------

#Extracting second value of vector
second_element = udf(lambda v:float(v[1]),FloatType())
latest_predictions = final_preds.select('CustomerID','prediction', second_element('probability'))
#Changing name
latest_predictions = latest_predictions.withColumnRenamed("<lambda>(probability)", 'probability to churn')

# COMMAND ----------

#sort by probability 
latest_predictions = predi_subset.sort('probability to churn')
display(predi_subset

# COMMAND ----------

# DBTITLE 1,SEGMENTIATION


# COMMAND ----------

#Reading Predictions
latest_predictions = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "Latest_Predictions.csv")

# COMMAND ----------

#Reading full datamart
datamart = spark.read.format("csv").option("header","true").option("inferSchema","true").load(path + "Datamart_to_2019.csv")

# COMMAND ----------

#Joining full datamart with predictions
data = datamart.withColumnRenamed("CustomerID", "c_id").drop("label")
preds_join = latest_predictions.join(data, data["c_id"] == latest_predictions["CustomerID"],joinType).drop("c_id")
preds_join = preds_join.withColumnRenamed("prediction", "label")
preds_join = preds_join.where(col("label") == 1)

# COMMAND ----------

#SEGMENTATION

segment = preds_join
segment = segment.withColumnRenamed("prediction", "churn_prediction")

features = ('Total_Subscriptions', 'Customer_Lifetime','Avg_Subscription_Period', 'Avg_Subscription_Period_inMonths', 'Total_meals_Regular', 'Total_meals_Exceptional', 'Avg_Meal_Price',
            'sum(ProductDiscount)', 'sum(TotalDiscount)', 'Total_Price', 'Avg_Price', 'Avg_Credit', 'Total_Credit', 'Total_Subscriptions_14', 'Total_Subscriptions_15', 'Total_Subscriptions_16',
            'Total_Subscriptions_17', 'Total_Subscriptions_18', 'Total_Subscriptions_19', 'Count_Formula_DirectMail', 'Total_Formula_Duration_DirectMail', 'Count_Formula_Reg',
            'Total_Formula_Duration_Reg', 'count(ComplaintID)')

segment_join =  VectorAssembler()\
                            .setInputCols(features)\
                            .setOutputCol("features")\
                            .transform(segment)

#Selecting the features column for clustering
segment_features = segment_join.select("features")

#Creating the kmeans clustering model
kmeans = KMeans().setK(3).setSeed(1)
KM_model = kmeans.fit(segment_join)
clusters = KM_model.clusterCenters()
churn_segmentation = KM_model\
                        .transform(segment_join).select('CustomerID', 'Total_Subscriptions', 'Customer_Lifetime', 'Avg_Subscription_Period',                                                                                           'Avg_Subscription_Period_inMonths', 'Total_meals_Regular','Total_meals_Exceptional', 'Avg_Meal_Price', 'sum(ProductDiscount)',                                                                         'sum(TotalDiscount)', 'Total_Price', 'Total_Credit', 'Region','Total_Subscriptions_14','Total_Subscriptions_15', 'Total_Subscriptions_16',                                                             'Total_Subscriptions_17', 'Total_Subscriptions_18', 'Total_Subscriptions_19', 'Count_Formula_DirectMail',                                                                                             'Total_Formula_Duration_DirectMail', 'Count_Formula_Reg', 'Total_Formula_Duration_Reg', 'Avg_Price', 'Avg_Credit',
                                'count(ComplaintID)', col("prediction").alias("clusters"))


# COMMAND ----------

#Visualizing clusters
display(final_segmentation)

# COMMAND ----------

#Aggregating variables by Product Type
churn_analysis = churn_segmentation.groupBy("clusters").agg(count("CustomerID"), avg("Total_Subscriptions"), avg("Avg_Subscription_Period"), avg("Avg_Subscription_Period_inMonths"),                                                                          avg("Total_meals_Regular"), avg("Total_meals_Exceptional"), avg("Avg_Meal_Price"), avg("sum(TotalDiscount)"), avg("Avg_Price"),                                                                      avg("Avg_Credit"), avg("Total_Subscriptions_14"), avg("Total_Subscriptions_15"), avg("Total_Subscriptions_16"),                                                                                      avg("Total_Subscriptions_17"), avg("Total_Subscriptions_18"), avg("Total_Subscriptions_19"),  avg("Count_Formula_DirectMail"),                                                                        avg("Total_Formula_Duration_DirectMail"), avg("Count_Formula_Reg"), avg("Total_Formula_Duration_Reg"), avg("count(ComplaintID)"), 
                                                         avg("Customer_Lifetime"))
                                                                     

# COMMAND ----------

#Visualize Churn Analysis
display(churn_analysis)
