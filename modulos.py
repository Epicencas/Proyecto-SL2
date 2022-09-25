import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os
import scipy.stats as stats


### ============ Obtener los tipo de dato de las variables ========

def getColumnsDataTypes(df):
    """
    Auto: Preng Biba
    Version: 1.0.0
    Descripción: Función para obtener los tipos de datos de cada columna de un dataframe.
    """

    categoric_vars = []
    discrete_vars = []
    continues_vars = []

    for colname in df.columns:
        if(df[colname].dtype == 'object'):
            categoric_vars.append(colname)
        else:
            cantidad_valores = len(df[colname].value_counts())
            if(cantidad_valores <= 30):
                discrete_vars.append(colname)
            else:
                continues_vars.append(colname)

    return categoric_vars, discrete_vars, continues_vars


### ============ Series temporales ========

def plot_serie_temporal(df, continues_vars,y):
    
    for col in continues_vars:
        
        dfr_st = df[[y,col]]
        dfr_st = dfr_st.set_index(y)
        dfr_st.sort_index(inplace=True)
        dfr_st.plot(figsize = (15,6))
    
    
### ============ Graficacion de Histogramas de las variables numéricas ========


def plot_density_variable(df, variable):
    
    plt.figure(figsize = (15,6))
    plt.subplot(121)
    df[variable].hist(bins=30)
    plt.title(variable)
    
    plt.subplot(122)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()

### ============ Inspección de Outliers ========

    
def inspect_outliers(df, variable):
    
    plt.figure(figsize = (15,6))
    
    plt.subplot(131)
    sns.distplot(df[variable], bins=30)
    plt.title("Densisd-Histograma: " + variable)
    
    plt.subplot(132)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.title("QQ-Plot: " + variable)
    
    plt.subplot(133)
    sns.boxplot(y=df[variable])
    plt.title("Boxplot: " + variable)
    
    plt.show()

### ============ Detección de Outliers ========
    
def detect_outliers(df, variable, factor):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    LI = df[variable].quantile(0.25) - (IQR*factor)
    LS = df[variable].quantile(0.75) + (IQR*factor)
    #print(LI,LS)
    #return LI, LS
    
    
### ============ Graficación de Variabel Categóricas ========

def plotCategoricalVals(df, categoric_vars, y):
    """
    Auto: Erick Picén
    Version: 1.0.0
    Descripción: Función para desplegar variables categoricas.
    """

    for column in categoric_vars:
        plt.figure(figsize=(12,6))
        plot = sns.countplot(x=df[column], hue=df[y])
        plt.show()

### ============ Graficación de Variables Numéricas Continuas ========

        
def plotVars(df, vars):
    """
    Auto: Preng Biba
    Version: 1.0.0
    Descripción: Función para desplegar variables categoricas.
    """

    for column in vars:
        plt.figure(figsize=(12,6))
        sns.histplot(df[column])
        plt.title(df[column].name)
        plt.show()

        
def graphs (df, continues_vars):
    """
    Auto: Erick Picén
    Version: 1.0.0
    Descripción: Función para desplegar variables Continuas.
    """
    for col in continues_vars:
        sns.set_theme(); np.random.seed(0)
        sns.set_color_codes()
        x = df[col]
        ax = sns.distplot(x,  color="k")
        rcParams['figure.figsize'] = 15,6
        plt.title("Histograma Variable "+ col, fontsize =20)
        plt.show()
    
    """
    sns.set_theme(); np.random.seed(0)
    sns.set_color_codes()
    x = df[col]
    ax = sns.distplot(x,  color="k")
    rcParams['figure.figsize'] = 15,6
    plt.title("Histograma Variable "+ col, fontsize =20)
    plt.show()
    """
### ============ Obtener solo variables numéricas ========


def getNumColNames(df):
    colnames = df.columns
    cols_num = []
    for col in colnames:
        if((df[col].dtypes == 'int64') | (df[col].dtypes == 'float64')):
            cols_num.append(col)
    return cols_num

### ============ Obtener solo variables categoricas ========

def getCatColNames(df):
    colnames = df.columns
    cols_cat = []
    for col in colnames:
        if(df[col].dtypes == 'object'):
            cols_cat.append(col)
    return cols_cat

### ============ Obtener variables numéricas con nan ========

def getNumNanColNames(df):
    colnames = df.columns
    cols_num_con_na = []
    for col in colnames:
        if((df[col].isnull().sum() > 0) & (df[col].dtypes != 'object')):
            cols_num_con_na.append(col)
    return cols_num_con_na


  ### ============ obtener variables con cantidad aceptable de nan ========
                  
            
def getNanGoodColsNames(df, rate = 0.2):
    cols_procesables = []
    for col in df.columns:
        if((df[col].isnull().mean() < rate)):
            cols_procesables.append(col)
    return cols_procesables


### ============ obtener variables categoricas con nan ========

def getCatNanColNames(df):
    colnames = df.columns
    cols_cat_con_na = []
    for col in colnames:
        if((df[col].isnull().sum() > 0) & (df[col].dtypes == 'object')):
            cols_cat_con_na.append(col)
    return cols_cat_con_na


### ============ Imputación de Variabel Numéricas ========

    
def getCategoryVars(df):
    colnames = df.columns
    cat_cols = []
    for col in colnames:
        if(df[col].dtype == 'object'):
            cat_cols.append(col)
    return cat_cols

### ============ Imputación de Variabel Numéricas ========


def getContinuesCols(df):
    colnames = df.columns
    numeric_continues_vars = []
    for col in colnames:
        unique_values =len (df[col].unique())
        if((df[col].dtype != 'object') and (unique_values > 10)):
            numeric_continues_vars.append(col)
    return numeric_continues_vars


### ========== Gráficos de barras Apiladas =============

#def bar_chart(feature):
#    survived = train[train['Survived']==1
                     
##https://www.youtube.com/watch?v=3eTSVGY_fIE


### ========== Función Para Obtener Estadísticas  =============


def stats (df, vars):
    for col in vars:
        promedio = round(df[col].mean(),2)
        maximo = round(df[col].max(),2)
        minimo = round(df[col].min(),2)
        desviacion = round(df[col].std(),2)
        q25, q75 =df[col].quantile([.25,.75])
        rango = round(q75 - q25)
        #lista = (promedio, maximo, minimo, desviacion, rango)
        dicc = {'Media':promedio, 'Max':maximo, 'Min':minimo, 'Desv':desviacion, 'Rango':rango}
        df1 = pd.DataFrame([[key, dicc[key]] for key in dicc.keys()], columns=['Estadística', 'Valor'])
        print(f'Estadísticas de {df[col].name} {dicc}')