# importing libraries
import streamlit as st

import SessionState
import random
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.gridspec as gridspec
from scipy import stats
import missingno as msno
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server


# Will load css style.css
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")

# This is created using session state solution provided in the link below
# https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "x" : {},
            "y" : {},
            "X_train" : {},
            "X_test" : {},
            "y_train" : {},
            "y_test" : {},
            "id_col": None,
            "target": None, 
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state
# Some useful lists

ls = []
lsn = []
cat_feats = []
num_feats = []
lsnid = []
lsnt = []
lsnidnt = []
lsnidac = []
lsac = []
lsnidntac = []
num_feats_nid = []
cat_feats_nid = []
num_feats_nt = []

#EDA functions
@st.cache(allow_output_mutation=True)
def load_data(data):
    dfx = pd.read_csv(data)
    return dfx
@st.cache(allow_output_mutation=True)
def load_data1(data,na_values):
    dfx = pd.read_csv(data, na_values = na_values)
    return dfx

def unique_non_null(s):
    return s.dropna().unique()

def show_df(df):
    num = st.number_input("Enter number of rows",min_value = 0, max_value = df.shape[0],value = 5,format = "%d",step = 1,key = "ni1")
    num = int(num)
    st.dataframe(df.head(num))
def show_selcted_cols(df,selected_columns):
    num = st.number_input("Enter number of Rows",min_value = 0, max_value = df.shape[0],value = 5,format = "%d",step = 1,key = "ni2")
    num = int(num)
    st.dataframe(df.head(num))

def show_df_highly_cust(df):
    num_feats = []
    for i in df.columns:
        if df[i].dtype != "object":
            num_feats.append(i)

    
    cat_feats = []
    for i in df.columns:
        if df[i].dtype == "object":
            cat_feats.append(i)
    
    cols_num = st.multiselect("Select numerical columns.",num_feats, key = "fm1")
    num_dict = {}
    k =9999
    for i in cols_num:
        num_dict[i] = [0,0,0]
        if df[i].min() != df[i].max():
            (num_dict[i][0],num_dict[i][1]) = st.slider("Select min and max values of {}".format(i), float(df[i].min()), float(df[i].max()),(float(df[i].min()), float(df[i].max())),key = k)
            
        else:
            st.write("{} column have only value. So, no slider :)".format(i))
            num_dict[i][0] = df[i].min()
            num_dict[i][1] = df[i].min()
        nan_choice = st.radio("To include the NaN values of {} column select 'YES' else 'NO'".format(i), ("YES","NO"), key = k+2)
        if nan_choice == "YES":
                num_dict[i][2] = 1
        k += 3
    
    cols_cat = st.multiselect("Select categorical columns.",cat_feats, key = "fm2")
    cat_dict = {}
    
    for i in cols_cat:
        cat_dict[i] = st.multiselect("Select {}".format(i),df[i].unique(),key = k)
        k +=1
    
    session_state = SessionState.get(name="3", button_sent=False)
    button_sent =  st.button("Show Dataframe",key = 'fb1')
    if button_sent:
        session_state.button_sent = True
    if session_state.button_sent:
        temp_df = df.copy()
        list_num = list(num_dict.keys())
        list_cat = list(cat_dict.keys())
        for i in list_num:
            if num_dict[i][2] == 1:
                temp_df1 = temp_df[temp_df[i].isna()]
                temp_df = temp_df[(temp_df[i] >= num_dict[i][0]) & (temp_df[i] <= num_dict[i][1])]
                temp_df = pd.concat([temp_df,temp_df1], axis = 0)
            else : 
                temp_df = temp_df[(temp_df[i] >= num_dict[i][0]) & (temp_df[i] <= num_dict[i][1])]
                
        for i in list_cat:
            lsx = list(cat_dict[i])
            temp_df1 = pd.DataFrame(columns = temp_df.columns)
            for j in lsx:
                try:
                    if np.isnan(j):
                        temp_df1 = pd.concat([temp_df1,temp_df[temp_df[i].isnull()]], axis = 0)
                except:
                    temp_df1 = pd.concat([temp_df1,temp_df[temp_df[i] == j]] ,axis = 0)
            temp_df = temp_df1
        st.dataframe(temp_df)
    
    
def show_shape(df):
    st.write("Number of rows : ", df.shape[0])
    st.write("Number of columns : ", df.shape[1])

def show_dtypes_and_num_missing_vals(df):
    df1 = df.isnull().sum()
    df1 = df1.to_frame()
    df1.rename(columns = {0:"Number of missing values"},inplace = True)
    df2 = df.dtypes
    df2 = df2.to_frame()
    df2.rename(columns = {0:"Data Type"},inplace = True)
    df3 = pd.concat([df1,df2], axis = 1)
    st.dataframe(df3)

def show_unique_nuni(df,selected_columns):
    for i in selected_columns:
        st.write("Column name {}  : {} unique values".format(i,df[i].nunique()))
        st.write("The unique values are as follows : ", df[i].unique())
        st.write("The value count of unique values : ")
        st.write(df[i].value_counts())
    
def change_df(temp_df):
    global df
    df = temp_df
 
# functions to make plots

def plot_count(df,i):
    fig = plt.figure(figsize = (8,5))
    fig = sns.countplot(x = i,data = df,  palette = 'rainbow')
    fig_copy = fig.get_figure()
    st.pyplot(fig_copy)

def plot_pie(df,i):
    ran_floats = np.random.rand(len(unique_non_null(df[i]))) * (0.2-0)
    explode = ran_floats.tolist()
    l1 = unique_non_null(df[i])
    l2 = df[i].value_counts()
    dict1 = dict(zip(l1, l2))
    dict2 = {}
    sor = sorted(dict1.values())
    for s in sor:
        for key,value in dict1.items():
            if value == s:
                dict2[key] = value
    fig,ax = plt.subplots()
    labels = list(dict2.keys())
    explode.sort(reverse = True)
    #plt.legend(labels, loc='upper right')
    patches, texts,a = plt.pie(x = dict2.values(),
                     autopct='%1.1f%%',shadow = True, 
                     startangle  = 90,explode = explode)
    plt.legend(patches, labels, loc="best")
    st.pyplot(fig)
    
def plot_box(df,i):
    fig = plt.figure(figsize = (3,3))
    sns.boxplot(data = df[i], orient= 'v',palette = 'rainbow')
    st.pyplot(fig)

def plot_distribution(df,i):
    fig = plt.figure(figsize = (5,3))
    sns.distplot(df[i])
    st.pyplot(fig)
    st.write("Skewness: " + str(df[i].skew()))
    st.write("Kurtosis: " + str(df[i].kurt()))

def plot_QQ(df,i):
    fig = plt.figure(figsize = (5,3))
    ax1 = fig.add_subplot()
    stats.probplot(df.loc[:,i] , plot = ax1)
    st.pyplot(fig)

def plot_scatter(df,X,Y):
    fig = plt.figure(figsize = (5,3))
    ax1 = fig.add_subplot()
    df.plot(kind='scatter',
        x=X,
        y=Y,
        alpha=0.5,
        color="orange",
        ax = ax1
       )
    st.pyplot(fig)
    
def plot_reg(df,X_,Y_):
    fig = plt.figure(figsize = (6,4))
    sns.regplot(x = X_,y = Y_,data = df,color = "purple")
    st.pyplot(fig)


def plot_count_hue(df,x_,hue_):
    fig = plt.figure(figsize = (6,4))
    sns.countplot(x = x_,hue = hue_ ,data = df,  palette = 'rainbow')
    st.pyplot(fig)

def plot_scatter_wtf(df,X,Y,f3):
    fig = plt.figure(figsize = (5,3))
    ax1 = fig.add_subplot()
    lsx = unique_non_null(df[f3])
    for i in lsx:
        df[df[f3] == i].plot(kind='scatter',
            x=X,
            y=Y,
            alpha=0.6,
            color=np.random.rand(3,),
            label = i,
            ax = ax1
           )
    st.pyplot(fig)

def make_corrmat(df,cols_corr):
    corrmat=df[cols_corr].corr()
    fig = plt.figure(figsize=(20,12)) 
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corrmat, 
                cmap='rainbow', 
                mask = mask, 
                annot=True, 
                center = 0, 
               )
    st.pyplot(fig)

def plot_msno_matrix(df):
    if st.button("Display only missing",key = "fb2"):
        ls = df.columns[df.isna().any()].tolist()
        fig1 = msno.matrix(df[ls],labels = True)
        fig1_copy = fig1.get_figure()
        st.pyplot(fig1_copy)
        
    if st.button("Display all",key = "fb3"):
        fig2 = msno.matrix(df,labels = True)
        fig2_copy = fig2.get_figure()
        st.pyplot(fig2_copy)

def plot_msno_bar(df):
    if st.button("Display only missing",key = "fb4"):
        ls = df.columns[df.isna().any()].tolist()
        fig1 = msno.bar(df[ls],labels = True)
        fig1_copy = fig1.get_figure()
        st.pyplot(fig1_copy)
        
    if st.button("Display all",key = "fb5"):
        fig2 = msno.bar(df,labels = True)
        fig2_copy = fig2.get_figure()
        st.pyplot(fig2_copy)



#functions for handling missing values
def show_missing_percent(df,cols):
    df1 = df[cols].isnull().sum()
    df1 = df1.to_frame()
    df1.rename(columns = {0:"Number of missing values"},inplace = True)
    df2 = df1.copy()
    df2.loc[:,'Number of missing values']  = (df2.loc[:,'Number of missing values']/df.shape[0])
    df2.rename(columns = {"Number of missing values":"Percent"},inplace = True)
    df3 = pd.concat([df1,df2], axis = 1)
    df3 = df3[df3["Number of missing values"] > 0]
    df3.sort_values("Number of missing values",ascending = False,inplace = True)
    st.dataframe(df3)

def fill_manually(df):
    index_num = st.number_input("Enter index number",key = 'ni3')
    index_num = int(index_num)
    d_type = st.selectbox("Select data tyoe of column ",['int','float','object'])
    col = st.selectbox("Select column",df.columns)
    
    if d_type == 'int':
        val = st.number_input("Enter value",value = 0,format = "%d",step = 1,key = "ni4")
        df.loc[index_num,col] = val
        st.write("Successfully updated")
    elif d_type == 'float':
        val = st.number_input("Enter value",key = "ni5")
        df.loc[index_num,col] = val
        st.write("Successfully updated")
    else :
        val = st.text_input("Enter text",key = 123)
        df.loc[index_num,col] = val
        st.write("Successfully updated")
    
    return df
                 
def fill_manually_w_const(df):
    d_type = st.selectbox("Select data tyoe of column ",['int','float','object'])
    col = st.selectbox("Select column",df.columns)
    if d_type == 'int':
        val = st.number_input("Enter value",value = 0,format = "%d",step = 1,key = "ni4")
        if st.button("Press to continue",key = "fb6"):
            df[col].fillna(val,axis = 0,inplace = True)
            st.write("Successfully updated")
    elif d_type == 'float':
        val = st.number_input("Enter value",key = "ni5")
        if st.button("Press to continue",key = "fb7"):
            df[col].fillna(val,axis = 0,inplace = True)
            st.write("Successfully updated")
    else :
        val = st.text_input("Enter text",key = 123)
        if st.button("Press to continue",key = "fb8"):
            df[col].fillna(val,axis = 0,inplace = True)
            st.write("Successfully updated")
    
    return df
    
 
def fill_using_simple(df,num_feats,cat_feats):
    lsx = list(df.columns)
    lsx.append("ALL_NUM")
    lsx.append("ALL_CAT")
    strat = st.selectbox("Select startegy",["most_frequent","mean","median"])
    dict_opt1 = {}
    dict_opt1["To select all numerical columns select"] = "ALL_NUM"
    dict_opt1["To select all categorical columns select"] = "ALL_CAT"
    st.write(dict_opt1)
    cols = st.multiselect("Select columns",lsx, key = "fm3")
    if cols == ["ALL_NUM"]:
        cols = num_feats
    elif cols == ["ALL_CAT"]:
        cols = cat_feats
    choice = st.radio("Select YES to continue with the new dataframe else NO",("WAIT","YES","NO"),key = 1)  
    if choice == 'YES':
        temp_df = df[cols]
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy = strat)
        imputer.fit(temp_df)
        temp_df = imputer.transform(temp_df)
        temp_df = pd.DataFrame(data =temp_df,columns = cols)
        st.write("The upadted part of dataframe")
        st.dataframe(temp_df)
        if st.button("Press to continue",key = "fb9"):
            df[cols] = temp_df
            st.dataframe(temp_df)
            return df
    elif choice == 'NO':
        return df
            
               

def apply_scaler(X_train, X_test,scaler):
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    state.X_train = X_train
    state.X_test = X_test

#Model Building Functions

def Lasso_model(X_train, X_test, y_train, y_test,x,y):
    alp = st.number_input("Enter the value of alp",key = "ni9")
    if st.button("Confirm",key = 'b32'):
        lasso = linear_model.Lasso(alpha=alp)
        lasso.fit(X_train,y_train)
        score_test = lasso.score(X_test,y_test)
        score_train = lasso.score(X_train,y_train)
        st.write("Score on train set : ",score_train)
        st.write("Score on test set : ",score_test)
            

def Ridge_model(X_train, X_test, y_train, y_test,x,y):
    alp = st.number_input("Enter the value of alp",key = "ni10")
    if st.button("Confirm",key = 'b33'):
        ridge = linear_model.Ridge(alpha=alp)
        ridge.fit(X_train,y_train)
        score_test = ridge.score(X_test,y_test)
        score_train = ridge.score(X_train,y_train)
        st.write("Score on train set : ",score_train)
        st.write("Score on test set : ",score_test)

def ElasticNet_model(X_train, X_test, y_train, y_test,x,y):
    alp = st.number_input("Enter the value of alp",key = "ni11")
    l1_rat = st.number_input("Enter the value of alp",key = "ni12")
    if st.button("Confirm",key = 'b34'):
        elasnet = linear_model.ElasticNet(alpha=alp, l1_ratio = l1_rat)
        elasnet.fit(X_train,y_train)
        score_test = elasnet.score(X_test,y_test)
        score_train = elasnet.score(X_train,y_train)
        st.write("Score on train set : ",score_train)
        st.write("Score on test set : ",score_test)
        
     
    
def SVR_model(X_train, X_test, y_train, y_test,x,y):
    svr = SVR()
    svr.fit(X_train,y_train)
    y_pred_test = svr.predict(X_test)
    y_pred_train = svr.predict(X_train)
    st.write('MSE train data prediction %.2f'%mean_squared_error(y_test, y_pred_test))
    st.write('MSE test data prediction %.2f'%mean_squared_error(y_train, y_pred_train))

def RandomForestRegressor_model(X_train, X_test, y_train, y_test,x,y):
    depth = st.number_input("Enter max_depth ",min_value = 0, max_value = 10,value = 0,format = "%d",step = 1,key = "ni16")
    n_est = st.number_input("Enter n_estimators ",min_value = 0, max_value = 1000,value = 0,format = "%d",step = 1,key = "ni17")
    if st.button("Confirm",key = 'b35'):
        reg = RandomForestRegressor(max_depth=depth,n_estimators = n_est)
        reg.fit(X_train,y_train)
        y_pred_test = reg.predict(X_test)
        y_pred_train = reg.predict(X_train)
        st.write('MSE train data prediction %.2f'%mean_squared_error(y_train, y_pred_train))
        st.write('MSE test data prediction %.2f'%mean_squared_error(y_test, y_pred_test))
    
 
    
# download dataframe
# the two functions below are copied from this link -
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/11
# it was by streamlit user 'Bernardo_tod'
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc

widget_values = {}

def make_recording_widget(f):
    """Return a function that wraps a streamlit widget and records the
    widget's values to a global dictionary.
    """
    def wrapper(label, *args, **kwargs):
        widget_value = f(label, *args, **kwargs)
        widget_values[label] = widget_value
        return widget_value

    return wrapper
                                     
dict_opt = {'To select all columns ':'ALL_COL', 'To select all numerical columns': 'NUM_COL',
            'To select all categorical columns': 'CAT_COL'}



state = get_state()
def main():
    """Machine learning web app for Regressionn problems"""
    html1 = '''<body style>
    <h1 style="font-family:georgia;color:#FFD700;font-size:60px;">SAM<i style="color:white;"> (Semi-Auto-Ml)</i></h1>
    </body>'''
    st.markdown(html1,unsafe_allow_html = True)
    activities = ["How to use","Regression Analysis","About"]  
    choice = st.sidebar.selectbox("Select Activitiy",activities)

    if choice == 'Regression Analysis':
        if state.data is None:
            st.markdown("## Upload your dataset")
    
            data = st.file_uploader("Upload dataset", type=["csv"])
            if data is not None:
                na_choice = st.radio("Select YES if you want read file with default na_values else select NO to enter your own data" , ("HOLD","YES","NO"),key = "r1")
                if na_choice == 'YES':
                    df = load_data(data)
                    state.data = df
                elif na_choice == 'NO':
                    butx = SessionState.get(name="1st button", button_sent=False)
                    DONE = st.button("Done",key = 'b1')
                    if DONE:
                        butx.DONE = True
                    if butx.DONE:
                        na_val = st.text_input("Enter na_values seprated by space.")
                        na_values = na_val.split(' ')
                        st.write("Entered na_values are : ",na_values)
                        if st.button("Press to Continue", key = 'b2'):
                            df = load_data1(data,na_values)
                            state.data = df
        if state.data is not None:
            df = state.data
            st.markdown("## Exploratory Data Analysis(EDA)")
            if st.checkbox("Show dataframe"):
                show_df(df)
            if st.checkbox("Show Selected Columns"):
                temp2 = df.columns
                selected_columns = st.multiselect("Select Columns",temp2,key = 'm2')
                new_df = df[selected_columns]
                show_selcted_cols(new_df,selected_columns)
                
            if st.checkbox("Show dataframe (Customisable view)"):
                show_df_highly_cust(df)
            if st.checkbox("Display Shape of the dataset."):
                show_shape(df)

            if st.checkbox("Display number of missing values and dtype of each column."):
                show_dtypes_and_num_missing_vals(df)
                
            ls = list(df.columns)
            lsn = ["NONE"] + ls
            for i in df.columns:
                if df[i].dtype != "object":
                    num_feats.append(i)
            # num_feats with ALL
            num_all = num_feats[:]
            num_all.append('ALL')
            #list of all categorical features
            
            for i in df.columns:
                if df[i].dtype == "object":
                    cat_feats.append(i)
            cat_all = cat_feats[:]
            cat_all.append('ALL')
            
            if st.checkbox("Display list of all the categorical features."):
                st.write("{}".format(cat_feats))
            
            if st.checkbox("Display list of all the numerical features."):
                st.write("{}".format(num_feats))
            
            if st.checkbox("Display descriptive statics of the dataframe."):
                st.write(df.describe())
            
            id_col = "X"
            target = "X"
            if state.target == None:
                if st.checkbox("Select target feature. Select 'NONE' if no such column"):
                    target = st.selectbox("Select 'NONE' if no such column",lsn,key = 1)
                    if st.button("Confirm",key = 42):
                        state.target = target
            else : 
                target = state.target
             
            
            try:
                if target == 'NONE':
                    lsnt = ls[:]
                elif target != 'X':
                    lsnt = ls[:]
                    lsnt.remove(id_col)
            except:
                pass
                
            
            if state.id_col == None:
                if st.checkbox("Select ID column. Select NONE' if no such column "):
                    id_col = st.selectbox("Select 'NONE' if no such column",lsn,key = 2)
                    if st.button("Confirm",key = 52):
                        state.id_col = id_col
            
            else : 
                id_col = state.id_col
            
            try:
                if id_col == 'NONE':
                    lsnid = ls[:]
                elif id_col != 'X':
                    lsnid = ls[:]
                    lsnid.remove(id_col)
            except:
                st.write("Please select id column")
            
            try:
                lsnidnt = lsnid[:]
            except:
                pass
            try:
                lsnidnt.remove(target)
            except:
                pass
            try:
                lsac = ls[:]
                lsac.append('ALL_COL')
                lsac.append('NUM_COL')
                lsac.append('CAT_COL')
                
                lsnidac = lsnid[:]
                lsnidac.append('ALL_COL')
                lsnidac.append('NUM_COL')
                lsnidac.append('CAT_COL')
                
                lsnidntac = lsnidnt[:]
                lsnidntac.append('ALL_COL')
                lsnidntac.append('NUM_COL')
                lsnidntac.append('CAT_COL')
            except:
                pass

            if st.checkbox("Display Unique values and there value counts for selected columns."):
                try:
                    st.write(dict_opt)
                    selected_columns = st.multiselect("Select Columns.",lsnidntac,key = 'm1')
                    if selected_columns == ['ALL_COL']:
                        selected_columns = lsnidnt[:]
                    elif selected_columns == ['NUM_COL']:
                        selected_columns = num_feats[:]
                    elif selected_columns == ['CAT_COL']:
                        selected_columns = cat_feats[:]
                    show_unique_nuni(df,selected_columns)
                except :
                    st.write("Please select Target/ID columns")
                    
                
            if st.checkbox("Change datatype of a column"):
                col = st.multiselect("Select Columns.",ls,key = '_m1')
                to_type = st.text_input("Enter the new datatype")
                if st.buton("Apply") :
                    try:
                        df[col] = df[col].astype(to_type)
                    except:
                        st.write("An error occurred try again")
                
            st.subheader("Data Visualisation")
            if st.checkbox("Make count plots of columns"):
                st.write(dict_opt)
                cols = st.multiselect("Select Columns.Select ALL_COL to select all columns",lsnidac, key = "m3")
                session_state = SessionState.get(name="1", button_sent=False)
                button_sent = st.button("PLOT",key = 'b3')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    if cols == ['ALL_COL']:
                        cols = lsnidnt[:]
                    elif cols == ['NUM_COL']:
                        cols = num_feats[:]
                    elif cols == ['CAT_COL']:
                        cols = cat_feats[:]
                    for i in cols:
                        plot_count(df,i)
                        
            
            if st.checkbox("Make pie plots"):
                st.write(dict_opt)
                cols1 = st.multiselect("Select Columns.",lsnidac,key = "m4")
                session_state = SessionState.get(name="12", button_sent=False)
                button_sent = st.button("PLOT",key = 'b4')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    if cols1 == ['ALL_COL']:
                        cols1 = lsnidnt[:]
                    
                    for i in cols1:
                        plot_pie(df,i)
            
            if st.checkbox("Make box plots"):
                st.write(dict_opt)
                cols = st.multiselect("Select Columns.",lsnidac, key = "m5")
                session_state = SessionState.get(name="13", button_sent=False)
                button_sent = st.button("PLOT",key = 'b5')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    if cols == ['ALL_COL']:
                        cols = lsnidnt[:]
                    for i in cols:
                        plot_box(df,i)
            
            if st.checkbox("Make distribution plots"):
                cols = st.multiselect("Select Columns.Select ALL to select all the numeric columns.",num_all, key = "m6")
                session_state = SessionState.get(name="14", button_sent=False)
                button_sent = st.button("PLOT",key = 'b6')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    if cols == ['ALL']:
                        cols = num_feats[:]
                    for i in cols:
                        plot_distribution(df,i)
                
            if st.checkbox("Make QQ plots"):
                cols = st.multiselect("Select Columns.Select ALL to select all the numeric columns.",num_all, key = "m7")
                if cols == ['ALL']:
                    cols = num_feats[:]
                session_state = SessionState.get(name="15", button_sent=False)
                button_sent = st.button("PLOT",key = 'b7')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    for i in cols:
                        plot_QQ(df,i)
            
            if st.checkbox("Make scatter plots"):
                X = st.selectbox("Select x", ls)
                Y = st.selectbox("Select y", ls)
                session_state = SessionState.get(name="16", button_sent=False)
                button_sent = st.button("PLOT",key = 'b8')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    plot_scatter(df,X,Y)
                
            if st.checkbox("Make area plots"):
                st.write(dict_opt)
                cols = st.multiselect("Select Columns",lsnidac, key = "m8")
                if cols == ['ALL_COL']:
                    cols = lsnid [:]
                elif cols == ['NUM_COL']:
                    cols = num_feats[:]
                elif cols == ['CAT_COL']:
                    cols = cat_feats[:]
                session_state = SessionState.get(name="17", button_sent=False)
                button_sent = st.button("PLOT",key = 'b9')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    for i in cols:
                        st.area_chart(df[i])
                
            
            if st.checkbox("Make bar charts"):
                st.write(dict_opt)
                cols = st.multiselect("Select Columns",lsnidac, key = "m9")
                if cols == ['ALL_COL']:
                    cols = lsnid [:]
                elif cols == ['NUM_COL']:
                    cols = num_feats[:]
                elif cols == ['CAT_COL']:
                    cols = cat_feats[:]
                session_state = SessionState.get(name="18", button_sent=False)
                button_sent = st.button("PLOT",key = 'b10')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    source = df
                    for i in cols:
                        chart = alt.Chart(source).mark_bar().encode(
                            alt.X(i, bin=True),
                            y='count()',
                        )
                        st.altair_chart(chart)
                    
            if st.checkbox("Make line charts"):
                st.write(dict_opt)
                cols = st.multiselect("Select Columns",lsnidac, key = "m78")
                if cols == ['ALL_COL']:
                    cols = lsnid[:]
                elif cols == ['NUM_COL']:
                    cols = num_feats[:]
                elif cols == ['CAT_COL']:
                    cols = cat_feats[:]
                session_state = SessionState.get(name="19", button_sent=False)
                button_sent = st.button("PLOT",key = 'b10')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent: 
                    st.line_chart(df[cols])
                
            if st.checkbox("Make a regression plot"):
                X_ = st.selectbox("Select X",df.columns)
                Y_ = st.selectbox("Select Y",df.columns)
                session_state = SessionState.get(name="20", button_sent=False)
                button_sent = st.button("PLOT",key = '25')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    plot_reg(df,X_,Y_)
            
            if st.checkbox("Make count plot with hue"):
                x_ = st.selectbox("Select X",ls,key = 1)
                hue_ = st.selectbox("Select hue",ls,key = 2)
                session_state = SessionState.get(name="21", button_sent=False)
                button_sent = st.button("PLOT",key = 'b11')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    plot_count_hue(df,x_,hue_)
            
            if st.checkbox("Make scatter plots for classifying X,Y based on a third feature"):
                X = st.selectbox("Select X", ls)
                Y = st.selectbox("Select Y", ls)
                f3 = st.selectbox("Select feature", ls)
                session_state = SessionState.get(name="22", button_sent=False)
                button_sent = st.button("PLOT",key = 'b12')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    plot_scatter_wtf(df,X,Y,f3)
            
            if st.checkbox("Make Heat-Map of correlation matrix"):
                cols_corr = st.multiselect("Select Columns",num_all, key = "m10")
                if cols_corr == ['ALL']:
                    cols_corr = num_feats[:]
                session_state = SessionState.get(name="23", button_sent=False)
                button_sent = st.button("Generate Heat-Map",key = 'b13')
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                   make_corrmat(df,cols_corr)
            
            btn = st.button("Press to see baloons go brrr",key = 'b14')
            if btn:
                st.balloons()

            
            st.markdown("## Handling missing values and outliers")
            if st.checkbox("See the percentage and number of missing values"):
                st.write(dict_opt)
                cols = st.multiselect("Select Columns",lsnidac, key = "m11")
                if cols == ['ALL_COL']:
                    cols = lsnid[:]
                elif cols == ['NUM_COL']:
                    cols = num_feats[:]
                elif cols == ['CAT_COL']:
                    cols = cat_feats[:]

                if st.button("Display",key = 'b15'):
                    show_missing_percent(df,cols)
                    
            if st.checkbox("Plot msno matrix"):
                plot_msno_matrix(df)
            
            if st.checkbox("Plot msno bar chart"):
                plot_msno_bar(df)
            
            if st.checkbox("Drop columns"):
                cols = st.multiselect("Select Columns",ls, key = "m12")
                st.write("These columns will be removed from the dataset")
                st.write(cols)
                if st.button("DROP",key = 'b16'):
                    df.drop(labels = cols,axis = 1,inplace = True)
                    state.data = df
                    st.write("Columns dropped")
            
            if st.checkbox("Drop all rows with missing values"):
                ls = list(df.columns)
                lsx = ls[:]
                lsx.append('ALL_COL')
                cols = st.multiselect("Select Columns. Select ALL_COL to select all the columns",lsx, key = "m13")
                if cols == ['ALL_COL']:
                    cols = ls[:]
                if st.button("DROP",key = 'b17'):
                    df[cols].dropna(inplace = True,axis = 0)
                    state.data = df
                    st.write("rows dropped")
                            
                            
            if st.checkbox("Drop rows manually using row number"):
                rows = st.text_input("Enter the iloc values seprated by spaces")
                st.write("Selected rows : ",rows)
                move = st.radio("Select Coninue to continue" , ("Continue","Wait"),index = 1,key = "r2")
                if move == 'Continue':
                    try : 
                        rows = rows.strip("")
                        rows = rows.split(" ")
                        rowsx = []
                        for i in rows:
                            rowsx.append(int(i))
                        temp_df = df.drop(index = rowsx,axis = 0)
                        st.write("Displaying the new dataframe")
                        show_df(temp_df)
                        if st.button("Press to confirm",key = 'b20'):
                            df.drop(index = rowsx,axis = 0,inplace = True)
                            state.data = df
                    except:
                        st.write("Rows already Removed")
                    
            if st.checkbox("Remove outliers using z_score"):
                cols = st.multiselect("Select Columns.Select ALL to select all the numeric columns.",num_all, key = "m17")
                df1 = df.copy()
                if cols == ['ALL']:
                    num_feats_nid = num_feats[:]
                    try:
                        num_feats_nid.remove(id_col)
                    except:
                        pass
                    cols = num_feats_nid[:]
                    
                st.write("Selected columns are",cols)    
                if st.button("Press to continue",key = 'b22'):
                    temp_df = stats.zscore(df[cols])
                    temp_df = pd.DataFrame(data = temp_df, columns = cols)
                    st.write("z_score values of selected dataframe")
                    st.dataframe(temp_df)
                    for i in cols:
                        df1[i] = df[i].loc[temp_df[i].abs()<=3]
                    st.write("Updated dataframe after outlier removal")
                    st.dataframe(df1)
                    state.data = df
                    if st.button("Press to confirm",key = 'b18'):
                        df = df1
                    
                    
            if st.checkbox("Remove outliers using IQR"):
                lsx = ls[:]
                lsx.append('ALL_COL')
                cols = st.multiselect("Select Columns. Select ALL_COL to select all the columns",lsx, key = "m14")
                if cols == ['ALL_COL']:
                    cols = lsx
                Q1 = df[cols].quantile(0.25)
                Q3 = df[cols].quantile(0.75)
                IQR = Q3 - Q1
                temp_df = df.copy()
                temp_df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
                st.write("Displaying updated dataframe")
                st.dataframe(temp_df)
                if st.button("CONFIRM",key = 'b23'):
                    df[df.columns] = temp_df[df.columns]
                    df.dropna(how = 'all',axis = 0,inplace = True)
                    state.data = df
                    st.dataframe(df)
                    st.write("Dataframe updated successfully!")
            
            st.markdown("### Handling missing values")
            if st.checkbox("Manually fill by row number missing value"):
                df = fill_manually(df)
                state.data = df

            if st.checkbox("Fill missing values with a constant"):
                df = fill_manually_w_const(df)
                state.data = df

            if st.checkbox("Fill missing values using SimpleImputer"):
                df = fill_using_simple(df,num_feats,cat_feats)
                state.data = df
                
            if st.checkbox("Apply transformation on your data"):
                trans = st.selectbox("Select the transformation you want to apply",["NONE", "boxcox1p","log1p"])
                col = st.selectbox("Select column",num_feats)
                if trans == "boxcox1p":
                    st.write("Distribution before transfowmation")
                    plot_distribution(df,col)
                    st.write("Distribution after applying boxcoxip")
                    fig = plt.figure(figsize = (5,3))
                    sns.distplot(boxcox1p(df[col], boxcox_normmax(df[col] + 1)))
                    st.pyplot(fig)
                    st.write("Skewness: " + str(df[col].skew()))
                    st.write("Kurtosis: " + str(df[col].kurt()))
                    confirm = False
                    choice = st.selectbox("Select YES to save changes else select NO",["WAIT","YES","NO"])
                    
                    if choice == "YES":
                        df[i] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))
                        state.data = df
                        confirm = True
                    
                    elif choice == "NO":
                        if confirm:
                            st.write("Changes were already made.")
                        else :
                            st.write("Changes not saved")
            
                
            if st.checkbox("Encode data using Dummies"):
                cols = st.multiselect("Select Columns",df.columns, key = "m15")
                st.write("Selected columns are : {}".format(cols))
                session_state = SessionState.get(name="1", button_sent=False)
                button_sent = st.button("Confirm",key = 1231)
                if button_sent:
                    session_state.button_sent = True
                if session_state.button_sent:
                    temp_df = pd.get_dummies(df[cols])
                    st.write("Dummied dataframe")
                    st.dataframe(temp_df.head())
                    choice = st.selectbox("Select YES to save changes else select NO",["WAIT","YES","NO"])
                    session_state.welcome = "YES"
                    session_state.bye = "NO"
                    if choice == session_state.welcome:
                        df.drop(cols,axis = 1,inplace = True)
                        df = pd.concat([df,temp_df],axis = 1)
                        state.data = df
                        st.write("Changes saved")
                    elif choice == session_state.bye:
                        st.write("Not saved")
                        
            
            if st.checkbox("Use KNNImputer"):
                lsx = lsnid[:]
                lsx.append("ALL")
                cols = st.multiselect("Select Columns.Select ALL to select all columns",lsx, key = "m16")
                if cols == ["ALL"]:
                    cols = lsnid[:]
                st.write("Selected columns are : {}".format(cols))
                temp_df = df[cols]
                lsxn = cols[:]
                lsxn.append("ALL")
                colmult = st.multiselect("Select columns in which you want to apply Standard scalar.Select ALL to select all",lsxn, key = "m16")
                if colmult == ["ALL"]:
                    colmult = cols[:]
                choice = st.selectbox("Select YES to apply else select NO",["YES","NO"],index = 1)
                confirm = False
                if choice == "YES":
                    scaler = MinMaxScaler()
                    temp_df[colmult] = pd.DataFrame(scaler.fit_transform(temp_df[colmult].copy()), columns = df.columns)
                    st.write("Applied MinMaxScaler")
                    confirm = True
                elif choice == "NO":
                    if confirm:
                        st.write("Changes were already made.")
                    else :
                        st.write("Scaler note applied")
                
                if st.checkbox("Apply KNNImputer"):
                    neigh = st.number_input("Enter the value of n_neighbors",min_value = 0, max_value = df.shape[0],value = 0,format = "%d",step = 1,key = "ni19")
                    session_state1 = SessionState.get(name="1", button_sent=False)
                    button_sent = st.button("Press to impute",key = 'b20')
                    if button_sent:
                        session_state1.button_sent = True
                    if session_state1.button_sent:
                        imputer = KNNImputer(n_neighbors=neigh, weights='uniform', metric='nan_euclidean')
                        imputer.fit(temp_df)
                        temp_ls = temp_df.columns
                        temp_df = imputer.transform(temp_df)
                        temp_df = pd.DataFrame(data = temp_df,columns = temp_ls) 
                        st.write("New dataframe")
                        st.dataframe(temp_df)
                        choice1 = st.selectbox("Select YES to save changes else select NO",["WAIT","YES","NO"],index = 1)
                        session_state1.welcome = "YES"
                        session_state1.bye = "NO"
                        if choice1 == session_state1.welcome:
                            df = df.drop(cols,axis = 1)
                            df = pd.concat([df,temp_df], axis = 1)
                            state.data = df
                            df = state.data
                            st.write("Changes saved")
                        elif choice1 == session_state1.bye:
                            st.write("Changes not saved")
                                
            st.markdown("## Training model")
            
            
            if st.checkbox("Make train test split"):
                t_size = st.number_input("Enter size of test split",key = "ni20")
                if st.button("Confirm",key = 'b24'):
                    try :
                        x = df.drop([id_col,target],axis = 1)
                        state.x = x
                    except :
                        x = df.drop([target],axis = 1)
                        state.x = x 
                    y = df[target]
                    state.y = y
                    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size = t_size, random_state = 0)
                    state.X_train = X_train
                    state.X_test = X_test
                    state.y_train = y_train
                    state.y_test = y_test
            
            if st.checkbox("Display shape of train and test splits"):
                st.write("Size of X_train : {}".format(state.X_train.shape))
                st.write("Size of y_train : {}".format(state.y_train.shape))
                st.write("Size of X_test : {}".format(state.X_test.shape))
                st.write("Size of y_test : {}".format(state.y_test.shape))
            
            if st.checkbox("Apply transformation on data"):
                opt = st.selectbox("Select transformer", ["MinMaxScaler","RobustScaler","StandardScaler"])
                if opt == 'MinMaxScaler':
                    mmscaler = MinMaxScaler()
                    apply_scaler(state.X_train, state.X_test,mmscaler)
                elif opt == 'RobustScaler':
                    robscaler = RobustScaler()
                    apply_scaler(state.X_train, state.X_test,robscaler)
                elif opt == 'StandardScaler':
                    standscaler = StandardScaler()
                    apply_scaler(state.X_train, state.X_test,standscaler)
            
            if st.checkbox("Select Model"):
                avail_models = ["Lasso","Ridge","ElasticNet","SVR","RandomForestRegressor"]
                st.write("List of available models")
                st.write(avail_models)
                modeL = st.selectbox("Select model",avail_models)
                session_state2 = SessionState.get(name="1", button_sent=False)
                button_sent = st.button("Press to continue",key = 'b21')
                if button_sent:
                    session_state2.button_sent = True
                if session_state2.button_sent:
                    if modeL == 'Lasso':
                        Lasso_model(state.X_train,state.X_test, state.y_train, state.y_test,state.x,state.y)
                    elif modeL == 'Ridge':
                        Ridge_model(state.X_train,state.X_test, state.y_train, state.y_test,state.x,state.y)
                    elif modeL == 'ElasticNet':
                        ElasticNet_model(state.X_train,state.X_test, state.y_train, state.y_test,state.x,state.y)
                    elif modeL == 'SVR':
                        SVR_model(state.X_train,state.X_test, state.y_train, state.y_test,state.x,state.y)
                    elif modeL == 'RandomForestRegressor':
                        RandomForestRegressor_model(state.X_train,state.X_test, state.y_train, state.y_test,state.x,state.y)
            
            
            if st.checkbox("Download dataframe(s)"):
                st.markdown("### Download dataframe")
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
                if state.X_train is not None:
                    X_train = pd.DataFrame(state.X_train)
                    y_train = pd.DataFrame(state.y_train)
                    X_test = pd.DataFrame(state.X_test)
                    y_test = pd.DataFrame(state.y_test)
                    st.markdown("#### Donwload X_train ")
                    st.markdown(get_table_download_link(X_train), unsafe_allow_html=True)
                    st.markdown("#### Donwload y_train ")
                    st.markdown(get_table_download_link(y_train), unsafe_allow_html=True)
                    st.markdown("#### Donwload X_test ")
                    st.markdown(get_table_download_link(X_test), unsafe_allow_html=True)
                    st.markdown("#### Donwload y_test ")
                    st.markdown(get_table_download_link(y_test), unsafe_allow_html=True)

    elif choice == "How to use":
        st.markdown("## How to use")
        st.markdown("Hi welcome to this application.")
        st.markdown("* Upload your data via drag and drop or select the file from your system.")
        st.markdown("* Click on selectbox to use that option.")
        st.markdown("* Remember to select target and id columns in your dataset. If their is no such column just select NONE (it will be available as one of the options).")
        st.markdown("* The page is divided into different section to help you in analysing your data.")
        st.markdown("* If you have made some changes to your uploaded data. You can download the updated dataframe as a csv file.")
        st.markdown("* If you have split your data into train and test splits you can also download them.")
        st.markdown("* If you come across any errors please mail them to developer(sheoran26800@gmail.com)")
    elif choice == "About":
        st.write("Hi! My name is Himalaya Sheoran. I created SAM(semi-auto-ml) to help users to take a quick peek at their datasets.")
        st.write("I have added a lot options to  asist you in your data analysis.")
        st.write("Currently only regression analysis option is available but I'll add more soon.")
        st.write("Please mail me your feedback.")
        st.write("Also if you have ideas about some more features. Please mail them to me. I'll be happy to add them :)")
        st.write("email - sheoran26800@gmail.com")
        st.write("Linkedin - https://www.linkedin.com/in/himalaya-singh-5747061b0/")

if __name__ == '__main__':
    main()
            

        
