import sdv
from sdv.evaluation.single_table import evaluate_quality
from rdt import HyperTransformer
from rdt.transformers.categorical import LabelEncoder
import pandas as pd
import numpy as np
def hybrid(real_data,df1,df2,metadata,col=None):
    if col is None:
        col=df1.columns
    print("First dataset quality score:")
    quality_report1 = evaluate_quality(
    real_data,
    df1,
    metadata)
    print("Second dataset quality score:")
    quality_report2 = evaluate_quality(
    real_data,
    df2,
    metadata)
    tab1=quality_report1.get_details('Column Shapes')
    tab2=quality_report2.get_details('Column Shapes')
    print("--------------First dataset report:-------------")
    print(tab1)
    print("--------------Second dataset report:------------")
    print(tab2)
    print("------------- Combined Quality score------------")
    tab1['Quality Score_1']=tab1['Quality Score']
    tab1['Quality Score_2']=tab2['Quality Score']
    tab1=tab1.drop('Quality Score',axis=1)
    print(tab1)
    #conversion of data
    
    ht = HyperTransformer()
    ht.detect_initial_config(data=df1)
    ht.remove_transformers_by_sdtype(sdtype='numerical')
    
    #Transformaer update(only for those variable that are not of good quality)
    
    col1 =[i for i in col if tab1.loc[tab1['Column']==i, 'Quality Score_1'].values[0] < tab1.loc[tab1['Column']==i, 'Quality Score_2'].values[0]]  
    # do not transform the credit_card or age columns
    ht.remove_transformers(column_names=col1)
    # do not transform any categorical columns in the dataset
    ht.remove_transformers_by_sdtype(sdtype='numerical')
    config = ht.get_config()
    

    print("Name of columns for which transformation have  been used:")
    col=[i for i in col if i not in col1]
    print(col)
    for feature in col:
        if config["transformers"][feature].__class__.__name__ == 'FrequencyEncoder':
            config["transformers"][feature] = LabelEncoder()

    print(config)
    ht.set_config(config)
    
    config = ht.get_config()
    print("Details of transformers used for all variables:")
    print(config)
    #merging datasets 
    
    
    ht.fit(df2)
    transformed_df2 = ht.transform(df2)
    ht.fit(df1)
    transformed_df1 = ht.transform(df1)
    
    dff2=transformed_df2
    dff1=transformed_df1
    for i in col:
        if tab1.loc[tab1['Column']==i, 'Quality Score_1'].values[0] > tab1.loc[tab1['Column']==i, 'Quality Score_2'].values[0]:
            dff2=dff2.sort_values(by=[i])
            dff1=dff1.sort_values(by=[i])
            dff2[i]=dff1[i]
    #reversing conversion
    
    reversed_dff = ht.reverse_transform(dff2)
    
    
    print("Hybrid quality score")
    quality_report4 = evaluate_quality(
    real_data,
    reversed_dff,
    metadata) 
    
    #qualityscore
    tab3=quality_report4.get_details('Column Shapes')
    print("Hybrid dataset report:")
    print(tab3)
    print(" Combined Quality score")
    tab1['Quality Score_3']=tab3['Quality Score']
    print(tab1)
    hybrid_data=reversed_dff
    
    return hybrid_data