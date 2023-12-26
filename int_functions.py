from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import pandas as pd

# define function to generate custom color maps
#define color map
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]
def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

# define function to return accuracy, specificity, sensitivity
def evaluation_score(y_test, y_pred_proba, thresh = 0.5):
    '''
    Compute the confusion matrix and return the accuracy, specificity and sensitivity score
    '''
    y_pred = [1 if x >= thresh else 0  for x in y_pred_proba]
    cm      = confusion_matrix(y_test,y_pred)
    n       = cm.sum()
    tn,fp,fn,tp      = tuple(cm.reshape((1,4))[0])
    # fp      = cm[0,1]
    # fn      = cm[1,0]
    # tp      = cm[1,1]

    acc     = (tp+tn)/n
    spe     = tn / (tn + fp)
    sen     = tp / (tp + fn)
    print("Accuracy:{:.2f}, Specifity:{:.2f} ,Sensitivity:{:.2f}".format(acc,spe,sen))

    # plot the confusion matrix (normalised terms)
    cm_norm = np.array([[spe,1-spe],
                    [1-sen,sen]
    ])
    # set norm (cmap)
    fig, ax = plt.subplots()
    sns.heatmap(cm_norm, annot = cm,fmt = "d",
                vmin=1, vmax=0 , cmap = 'Blues',ax = ax)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    return acc, spe,sen, cm


def census_request(n = 57,query = "NAME",table = "",year = "2020"):
    '''return a Pandas Dataframe of the given query over the state codes up to n state fips code.'''
    # api request
    df = []
    api_request = ["https://api.census.gov/data/{year}/{table}?get={query}&for=tract:*&in=state:{:0>2d}&in=county:*".format(x,query = query,table = table,year = year) for x in range(1,n)]
    print(f"Calling {len(api_request)} API requests")
    for link in api_request:
        try:   
            r = requests.get(link)
            response = r.json()
            df.extend(response[1:])
            print("retrieved {}".format(link.split(':')[3][0:2]))
        except:
            print("FAILED {}: {}".format(link.split(':')[3][0:2],link))

    # extract column name
    columns = response[0]

    # contrust Pandas DataFrame
    df = pd.DataFrame(df,columns = columns)
    df.dropna(axis = 'columns',thresh = len(df.index), inplace = True)
    
    df['CensusTract'] = df['state'] + df['county'] + df['tract']  
    df.drop(columns= ['state','county','tract'], inplace = True)

    try:
        df.drop(columns= ['GEO_ID'], inplace = True)
    except:
        pass
    
    df.replace("(X)",np.nan,inplace = True)
    df.replace("-888888888",np.nan,inplace = True)
    df.dropna(axis = "columns",inplace  = True)

    return df