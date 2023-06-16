import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def corr_matrix(df,mode = '') :

    ''' 
    Create a correlation matrix for the top 10 selected features 

    '''

    correlation_matrix = df.corr(mode)
    selected_features = correlation_matrix['shares'].abs().nlargest(11).index.tolist()[1:]
    selected_features.insert(0,"shares")
    correlation = df[selected_features].corr(mode)
    plt.figure(figsize = (15,15))
    # Plot the correlation matrix as a heatmap
    sns.heatmap(correlation, annot=True, cmap='coolwarm')

    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()



def datachannel_correlation(datachannel,df):

    ''' Analyzing the different correlation for each distinct data channel '''

    channel_group= df.groupby(datachannel).get_group(1)
    #channel_group.drop(datachannel, axis = 1, inplace = True)
    

    correlation_matrix = channel_group.corr(method='spearman')
    top_10_features = correlation_matrix['shares'].abs().nlargest(11).index.tolist()[1:]
    top_10_values = correlation_matrix['shares'].abs().nlargest(11)[1:]
    print(f'top 10 features correlated with shares fo news with datachannel {datachannel}\n', top_10_values)
    corr_matrix(channel_group,'spearman')
    
    correlation_matrix = df.corr(method='spearman')
    top_10_values = correlation_matrix[datachannel].abs().nlargest(11)[1:]
    print('top 10 features correlated with the datachannel analyzed', top_10_values)

    

def find_negative_columns(df):

    ''' Analyzing negative data '''

    negative_columns = []
    for column in df.columns:
        if df[column].dtype.kind in 'biufc' and (df[column] < 0).any():
            negative_columns.append(column)
    return negative_columns

def random_forest_regressor(X,mode,y, model):
    X['shares'] = y
    correlation_matrix = df.corr(mode)
    top_10_features = correlation_matrix['shares'].abs().nlargest(11).index.tolist()[1:]
    X_t = X[top_10_features]
    X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    rf_regressor = model
    # Fit the model on the training data
    rf_regressor.fit(X_train, y_train)

    # Predict on the testing data
    y_pred = rf_regressor.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   
    print("Root Mean Squared Error:", rmse)


def deleting_outliers(df,columns):
    for feature in columns:

        numerical_columns = [feature]

        # box plot to visualize the distribution and identify outliers
        #sns.boxplot(data=df[numerical_columns])

        # IQR for each numerical column
        Q1 = df[numerical_columns].quantile(0.25)
        Q3 = df[numerical_columns].quantile(0.75)
        IQR = Q3 - Q1

        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
            
        for col in numerical_columns:
            df[col] = np.where(df[col] < lower_bound[col], lower_bound[col], df[col])
            df[col] = np.where(df[col] > upper_bound[col], upper_bound[col], df[col])


def removingOutlierColumn(col,df,fact):
  #fact usually should be 1.5
  q1 = df[col].quantile(0.25)    # First Quartile
  q3 = df[col].quantile(0.75)    # Third Quartile
  IQR = q3 - q1                            # Inter Quartile Range

  llimit = q1 - fact*IQR                       # Lower Limit
  ulimit = q3 + fact*IQR                        # Upper Limit

  outliers = df[(df[col] < llimit) | (df[col] > ulimit)]

  df.drop(outliers.index, axis = 0, inplace = True)


  print('Number of outliers in "' + col + ' : ' + str(len(outliers)))
  print(llimit)
  print(ulimit)
  print(IQR)

def findImportance(df):
  X = df
  y = df['shares']
  df.drop('shares', axis = 1, inplace = True)
  reg = RandomForestRegressor(100, random_state=42)
  reg.fit(X, y)
  df["shares"] = y
  feature_dict = dict(sorted(zip(df.columns, reg.feature_importances_), key=lambda x: x[1],reverse=True))
  feature_list = list(feature_dict.keys())
  return feature_dict,feature_list  

def plot_hist(feature,df):
    fig, axs = plt.subplots(dpi = 120)
    plt.hist(df[feature], bins= 100, edgecolor='black', color = 'mediumpurple')
    plt.grid(alpha = 0.2)
    plt.title(f"{feature}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    print("minimum value",max(df[feature]))
    print("maximum value",min(df[feature]))

def plot_log_hist(feature,df):
    fig, axs = plt.subplots(dpi = 120)
   
    plt.hist(np.log(df[feature]+1.0001), bins= 100, edgecolor='black', color = 'mediumpurple')
    plt.grid(alpha = 0.2)
    plt.title(f"log({feature})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    print("minimum value",min(np.log(df[feature]+1)))
    print("maximum value",max(np.log(df[feature]+1)))

def removingOutlierColumn(col,df,fact = 1.5):
  #fact usually should be 1.5
  q1 = df[col].quantile(0.25)    # First Quartile
  q3 = df[col].quantile(0.75)    # Third Quartile
  IQR = q3 - q1                            # Inter Quartile Range

  llimit = q1 - fact*IQR                       # Lower Limit
  ulimit = q3 + fact*IQR                        # Upper Limit

  outliers = df[(df[col] < llimit) | (df[col] > ulimit)]

  df.drop(outliers.index, axis = 0, inplace = True)


  print('Number of outliers in "' + col + ' : ' + str(len(outliers)))
  print(llimit)
  print(ulimit)
  print(IQR)

def findImportance(df):
  X = df
  y = df['shares']
  feature_list = []
  df.drop('shares', axis = 1, inplace = True)
  reg = RandomForestRegressor(100, random_state=42)
  reg.fit(X, y)
  df["shares"] = y
  feature_dict = dict(sorted(zip(df.columns, reg.feature_importances_), key=lambda x: x[1],reverse=True))
  temp = feature_dict.keys()
  for key in temp:
    feature_list.append(key)
  return feature_dict,feature_list

# Utils functions

def plot_hist(feature,df):
    fig, axs = plt.subplots(dpi = 120)
    plt.hist(df[feature], bins= 100, edgecolor='black', color = 'mediumpurple')
    plt.grid(alpha = 0.2)
    plt.title(f"{feature}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    print("minimum value",min(df[feature]))
    print("maximum value",max(df[feature]))

def plot_log_hist(feature,df):
    fig, axs = plt.subplots(dpi = 120)
   
    plt.hist(np.log(df[feature]+1.0001), bins= 100, edgecolor='black', color = 'mediumpurple')
    plt.grid(alpha = 0.2)
    plt.title(f"log({feature})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    print("minimum value",min(np.log(df[feature]+1)))
    print("maximum value",max(np.log(df[feature]+1)))


def triangle_corr_matrix_paper(df,method):
    fig, axs = plt.subplots(dpi = 120, figsize = (12,12))
    cor=df.corr(method = method)
    df_lt = cor.where(np.tril(np.ones(cor.shape)).astype(bool))
    print(df_lt.shape)
    sns.heatmap(df_lt,cmap='Blues', annot = True, annot_kws={'fontsize': 27})
    axs.tick_params(labelsize=30)
    plt.setp(axs.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    #plt.title(f"{method} correlation matrix", fontsize = 40)
    plt.savefig('./images/spearmancorr.svg', format='svg')

    plt.show()