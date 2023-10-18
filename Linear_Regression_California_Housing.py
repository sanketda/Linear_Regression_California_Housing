#!/usr/bin/env python
# coding: utf-8

# # LINEAR REGRESSION
In this we'll make use of the California Housing data set. Note, of course, that this is actually 'small' data and that using Spark in this context might be overkill; This notebook is for educational purposes only and is meant to give us an idea of how we can use PySpark to build a machine learning model.
# # 1. Understanding the Data Set

# The California Housing data set appeared in a 1997 paper titled Sparse Spatial Autoregressions, written by Pace, R. Kelley and Ronald Barry and published in the Statistics and Probability Letters journal. The researchers built this data set by using the 1990 California census data.
# 
# The data contains one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). In this sample a block group on average includes 1425.5 individuals living in a geographically compact area.
# 
# These spatial data contain 20,640 observations on housing prices with 9 economic variables:
Longitude:refers to the angular distance of a geographic place north or south of the earth’s equator for each block group
Latitude :refers to the angular distance of a geographic place east or west of the earth’s equator for each block group

Housing Median Age:is the median age of the people that belong to a block group. Note that the median is the value that lies at the midpoint of a frequency distribution of observed values

Total Rooms:is the total number of rooms in the houses per block group
Total Bedrooms:is the total number of bedrooms in the houses per block group

Population:is the number of inhabitants of a block group
Households:refers to units of houses and their occupants per block group

Median Income:is used to register the median income of people that belong to a block group
Median House Value:is the dependent variable and refers to the median house value per block group
# What's more, we also learn that all the block groups have zero entries for the independent and dependent variables have been excluded from the data.
# 
# The Median house value is the dependent variable and will be assigned the role of the target variable in our ML model.

# In[1]:


get_ipython().system('pip install pyspark')


# In[2]:


import os
import pandas as pd
import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# Visualization
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

from matplotlib import rcParams
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})
rcParams['figure.figsize'] = 18,4

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[5]:


# setting random seed for notebook reproducability
rnd_seed=23
np.random.seed=rnd_seed
np.random.set_state=rnd_seed


# # 2. Creating the Spark Session

# In[6]:


spark = SparkSession.builder.master("local[2]").appName("Linear-Regression-California-Housing").getOrCreate()


# In[7]:


spark


# In[8]:


sc = spark.sparkContext
sc


# In[9]:


sqlContext = SQLContext(spark.sparkContext)
sqlContext


# # 3. Load The Data From a File Into a Dataframe

# In[10]:


HOUSING_DATA = r'E:\PG_DBDA\New_project_pyspark_housing\cal_housing.data'


# Specifying the schema when loading data into a DataFrame will give better performance than schema inference.

# In[11]:


# define the schema, corresponding to a line in the csv data file.
schema = StructType([
    StructField("long", FloatType(), nullable=True),
    StructField("lat", FloatType(), nullable=True),
    StructField("medage", FloatType(), nullable=True),
    StructField("totrooms", FloatType(), nullable=True),
    StructField("totbdrms", FloatType(), nullable=True),
    StructField("pop", FloatType(), nullable=True),
    StructField("houshlds", FloatType(), nullable=True),
    StructField("medinc", FloatType(), nullable=True),
    StructField("medhv", FloatType(), nullable=True)]
)


# In[12]:


# Load housing data
housing_df = spark.read.csv(path=HOUSING_DATA, schema=schema).cache()


# In[13]:


# Inspect first five rows
housing_df.take(5)


# In[14]:


housing_df.show(5)


# In[15]:


# show the dataframe columns
housing_df.columns


# In[16]:


# show the schema of the dataframe
housing_df.printSchema()


# # 4. Data Exploration

# In[17]:


# run a sample selection
housing_df.select('pop','totbdrms').show(10)


# 4.1 Distribution of the median age of the people living in the area:

# In[18]:


# group by housingmedianage and see the distribution
result_df = housing_df.groupBy("medage").count().sort("medage", ascending=False)


# In[19]:


result_df.show(10)


# In[20]:


result_df.toPandas().plot.bar(x='medage',figsize=(14, 6))


# Most of the residents are either in their youth or they settle here during their senior years. Some data are showing median age < 10 which seems to be out of place

# 4.2 Summary Statistics:
# Spark DataFrames include some built-in functions for statistical processing. The describe() function performs summary statistics calculations on all numeric columns and returns them as a DataFrame.

# In[21]:


(housing_df.describe().select(
                    "summary",
                    F.round("medage", 4).alias("medage"),
                    F.round("totrooms", 4).alias("totrooms"),
                    F.round("totbdrms", 4).alias("totbdrms"),
                    F.round("pop", 4).alias("pop"),
                    F.round("houshlds", 4).alias("houshlds"),
                    F.round("medinc", 4).alias("medinc"),
                    F.round("medhv", 4).alias("medhv"))
                    .show())


# Look at the minimum and maximum values of all the (numerical) attributes. We see that multiple attributes have a wide range of values: we will need to normalize your dataset.

# # 5. Data Preprocessing

# 
# With all this information that we gathered from our small exploratory data analysis, we know enough to preprocess our data to feed it to the model.
# 
# we shouldn't care about missing values; all zero values have been excluded from the data set.
# We should probably standardize our data, as we have seen that the range of minimum and maximum values is quite big.
# There are possibly some additional attributes that we could add, such as a feature that registers the number of bedrooms per room or the rooms per household.
# Our dependent variable is also quite big; To make our life easier, we'll have to adjust the values slightly.

# 5.1 Preprocessing The Target Values¶
# First, let's start with the medianHouseValue, our dependent variable. To facilitate our working with the target values, we will express the house values in units of 100,000. That means that a target such as 452600.000000 should become 4.526:

# In[22]:


# Adjust the values of `medianHouseValue`
housing_df = housing_df.withColumn("medhv", col("medhv")/100000)


# In[23]:


# Show the first 2 lines of `df`
housing_df.show(2)


# We can clearly see that the values have been adjusted correctly when we look at the result of the show() method:

# # 6. Feature Engineering

# 
# Now that we have adjusted the values in medianHouseValue, we will now add the following columns to the data set:
# 
# 1)Rooms per household which refers to the number of rooms in households per block group;
# 2)Population per household, which basically gives us an indication of how many people live in households per block group; And
# 3)Bedrooms per room which will give us an idea about how many rooms are bedrooms per block group;
# 
# As we're working with DataFrames, we can best use the select() method to select the columns that we're going to be working with, namely totalRooms, households, and population. Additionally, we have to indicate that we're working with columns by adding the col() function to our code. Otherwise, we won't be able to do element-wise operations like the division that we have in mind for these three variables

# In[24]:


housing_df.columns


# In[25]:


# Add the new columns to `df`
housing_df = (housing_df.withColumn("rmsperhh", F.round(col("totrooms")/col("houshlds"), 2))
                       .withColumn("popperhh", F.round(col("pop")/col("houshlds"), 2))
                       .withColumn("bdrmsperrm", F.round(col("totbdrms")/col("totrooms"), 2)))


# In[26]:


# Inspect the result
housing_df.show(5)

We can see that, for the first row, there are about 6.98 rooms per household, the households in the block group consist of about 2.5 people and the amount of bedrooms is quite low with 0.14:

Since we don't want to necessarily standardize our target values, we'll want to make sure to isolate those in our data set. Note also that this is the time to leave out variables that we might not want to consider in our analysis. In this case, let's leave out variables such as longitude, latitude, housingMedianAge and totalRooms.

In this case, we will use the select() method and passing the column names in the order that is more appropriate. In this case, the target variable medianHouseValue is put first, so that it won't be affected by the standardization
# In[27]:


# Re-order and select columns
housing_df = housing_df.select("medhv", 
                              "totbdrms", 
                              "pop", 
                              "houshlds", 
                              "medinc", 
                              "rmsperhh", 
                              "popperhh", 
                              "bdrmsperrm")

6.1 Feature Extraction

Now that we have re-ordered the data, we're ready to normalize the data. We will choose the features to be normalized.
# In[28]:


featureCols = ["totbdrms", "pop", "houshlds", "medinc", "rmsperhh", "popperhh", "bdrmsperrm"]


# Use a VectorAssembler to put features into a feature vector column

# In[29]:


# put features into a feature vector column
assembler = VectorAssembler(inputCols=featureCols, outputCol="features") 


# In[30]:


assembled_df = assembler.transform(housing_df)


# In[31]:


assembled_df.show(10, truncate=False)


# All the features have transformed into a Dense Vector.
6.2 Standardization

Next, we can finally scale the data using StandardScaler. The input columns are the features, and the output column with the rescaled that will be included in the scaled_df will be named "features_scaled":
# In[32]:


# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")


# In[33]:


# Fit the DataFrame to the scaler
scaled_df = standardScaler.fit(assembled_df).transform(assembled_df)


# In[34]:


# Inspect the result
scaled_df.select("features", "features_scaled").show(10, truncate=False)


# # 7. Building A Machine Learning Model With Spark ML

# 
# With all the preprocessing done, it's finally time to start building our Linear Regression model! Just like always, we first need to split the data into training and test sets. Luckily, this is no issue with the randomSplit() method:

# In[35]:


# Split the data into train and test sets
train_data, test_data = scaled_df.randomSplit([.8,.2], seed=rnd_seed)

We pass in a list with two numbers that represent the size that we want your training and test sets to have and a seed, which is needed for reproducibility reasons.

Note that the argument elasticNetParam corresponds to  α
  or the vertical intercept and that the regParam or the regularization paramater corresponds to  λ
 .
# In[36]:


train_data.columns


# Create an ElasticNet model:
# 
# ElasticNet is a linear regression model trained with L1 and L2 prior as regularizer. This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge. We control the convex combination of L1 and L2 using the l1_ratio parameter.
# 
# Elastic-net is useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
# 
# A practical advantage of trading-off between Lasso and Ridge is it allows Elastic-Net to inherit some of Ridge’s stability under rotation.
# 
# The objective function to minimize is in this case:
# minw12nsamples∥Xw−y∥22+αλ∥Xw−y∥1+α(1−λ)2∥w∥22
#  
# http://scikit-learn.org/stable/modules/linear_model.html#elastic-net

# In[37]:


# Initialize `lr`
lr = (LinearRegression(featuresCol='features_scaled', labelCol="medhv", predictionCol='predmedhv', 
                               maxIter=10, regParam=0.3, elasticNetParam=0.8, standardization=False))


# In[38]:


# Fit the data to the model
linearModel = lr.fit(train_data)


# # 8. Evaluating the Model

# With our model in place, we can generate predictions for our test data: use the transform() method to predict the labels for our test_data. Then, we can use RDD operations to extract the predictions as well as the true labels from the DataFrame.

# 8.1 Inspect the Model Co-efficients

# In[39]:


# Coefficients for the model
linearModel.coefficients


# In[40]:


featureCols


# In[41]:


# Intercept for the model
linearModel.intercept


# In[42]:


coeff_df = pd.DataFrame({"Feature": ["Intercept"] + featureCols, "Co-efficients": np.insert(linearModel.coefficients.toArray(), 0, linearModel.intercept)})
coeff_df = coeff_df[["Feature", "Co-efficients"]]


# In[43]:


coeff_df


# 8.2 Generating Predictions

# In[44]:


# Generate predictions
predictions = linearModel.transform(test_data)


# In[45]:


# Extract the predictions and the "known" correct labels
predandlabels = predictions.select("predmedhv", "medhv")


# In[46]:


predandlabels.show()


# 8.3 Inspect the Metrics

# Looking at predicted values is one thing, but another and better thing is looking at some metrics to get a better idea of how good your model actually is.
# 
# Using the LinearRegressionModel.summary attribute:
# 
# Next, we can also use the summary attribute to pull up the rootMeanSquaredError and the r2.

# In[47]:


# Get the RMSE
print("RMSE: {0}".format(linearModel.summary.rootMeanSquaredError))


# In[48]:


print("MAE: {0}".format(linearModel.summary.meanAbsoluteError))


# In[49]:


# Get the R2
print("R2: {0}".format(linearModel.summary.r2))


# The RMSE measures how much error there is between two datasets comparing a predicted value and an observed or known value. The smaller an RMSE value, the closer predicted and observed values are.
# 
# The R2 ("R squared") or the coefficient of determination is a measure that shows how close the data are to the fitted regression line. This score will always be between 0 and a 100% (or 0 to 1 in this case), where 0% indicates that the model explains none of the variability of the response data around its mean, and 100% indicates the opposite: it explains all the variability. That means that, in general, the higher the R-squared, the better the model fits our data.

# # Using the RegressionEvaluator from pyspark.ml package:

# In[50]:


evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='rmse')
print("RMSE: {0}".format(evaluator.evaluate(predandlabels)))


# In[51]:


evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='mae')
print("MAE: {0}".format(evaluator.evaluate(predandlabels)))


# In[52]:


evaluator = RegressionEvaluator(predictionCol="predmedhv", labelCol='medhv', metricName='r2')
print("R2: {0}".format(evaluator.evaluate(predandlabels)))


# There's definitely some improvements needed to our model! If we want to continue with this model, we can play around with the parameters that we passed to your model, the variables that we included in your original DataFrame.

# In[53]:


spark.stop()


# In[ ]:




