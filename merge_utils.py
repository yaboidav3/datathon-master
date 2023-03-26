import pandas as pd

# these functions are used to generate "fin_data.csv" and "ult_data.csv" which are the training and starting dataset combined with 
# selected columns from the macro dataset.

def append_macro_to_training_data():
    ## read csvs
    data = pd.read_csv("training_data.csv")
    macro_data = pd.read_csv('macro_data.csv')

    macro2 = macro_data.loc[223:246] # select rows from the date range we are interested in

    macro2.to_csv('macro2.csv') ## export as csv to use in excel

    ## used Excel to convert date format in 'Mnemonic' column from mm/dd/yyyy to yyyymm and renamed to 'mth_code' to 
    ## make merging with training dataset easier

    macro3 = pd.read_csv('macro2.csv')
    macro3 = macro3[['mth_code','M_FUNI.IUSA', 'M_FLBR.IUSA', 'M_FHPNR.IUSA', 'M_FGDP.IUSA', 'M_FRFED.IUSA', 'M_FCBC.IUSA']]
    macro3
    ## selected only the variables we deemed significant from the macro dataset

    ult_data = pd.merge(data, macro3, how='inner', on='mth_code') ##merge the two datasets with 'mth_code' being common

    ult_data.to_csv('ult_data.csv') ## download csv

def append_macro_to_starting_data():
    newforecast = pd.read_csv("forecast_starting_data.csv")
    macro_data = pd.read_csv('macro_data.csv')
    onelast_macro = macro_data.loc[247:259] ## selecting the data from the date range we will be forcasting

    onelast_macro.to_csv('onelast_macro.csv') ## export as csv to use in excel 

    ## used Excel to convert date format in 'Mnemonic' column from mm/dd/yyyy to yyyymm and renamed to 'mth_code' to 
    ## make merging with forcast starting dataset easier

    fin_marco = pd.read_csv('onelast_macro.csv')
    fin_marco = fin_marco[['mth_code','M_FUNI.IUSA', 'M_FLBR.IUSA', 'M_FHPNR.IUSA', 'M_FGDP.IUSA', 'M_FRFED.IUSA', 'M_FCBC.IUSA']]
    fin_marco

    ## selected only the variables we deemed significant from the macro dataset

    fin_data = pd.merge(newforecast, fin_marco, how='inner', on='mth_code') ##merge the two datasets with 'mth_code' being common

    fin_data.to_csv('fin_data.csv') ## download csv
