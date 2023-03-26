import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter 
##### lifelines version 0.26.0 ##### 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
from merge_utils import * 

# main
def main():
    disable_warnings()
    
    #Appended dataset for training
    df = pd.read_csv('ult_data.csv')
    clean(df)

    cph = cox_p_h(df)
    
    #Appended dataset for predicting
    newdf = pd.read_csv('fin_data.csv')
    clean(newdf)
    predict_new(cph, newdf)

# util
def disable_warnings():
    warnings.filterwarnings('ignore', message='Columns \(\d+\) have mixed types.*')
    pd.options.mode.chained_assignment = None

# encode categorical columns to floats
def encode_categorical(df, col):
    df_encoded = df.copy()
    
    le = LabelEncoder()
    le.fit(df_encoded[col])
    df_encoded[col] = le.transform(df[col])
  
    return df_encoded

# wrapper for encoding the whole dataframe
def to_numeric(df):
    for col in df.columns:
        try:
            df[col].astype(float)
        except ValueError:
            df = encode_categorical(df, col)
    return df

# wrapper for data preprocessing
def clean(df):
    df.fillna(0,inplace=True)
    df.drop("writeoff_date",inplace=True, axis = 1) # cannot encode
    df = to_numeric(df)

# util for getting the month_count variable from fomatted yyyymm
def parse_month_diff(m1, m2):
    return np.abs((m2//100 - m1//100) * 12 + (m2%100 - m1%100))

# combining and encoding columns for the snapshot jan2018
def cox_p_h_prep_jan2018(df):
    dfjan = df[df["snapshot"] == 201801]
    dfjan["monthcount"] = parse_month_diff(dfjan["mth_code"], dfjan["snapshot"])
    # df = dfjan[["bank_fico_buckets_20","credit_limit_amt","monthcount","charge_off"]]
    dfjan = dfjan[["net_payment_behaviour_tripd","promotion_flag","variable_rate_index",
                   "bank_fico_buckets_20","mob","ever_delinquent_flg",
                   "nbr_mths_due", "credit_limit_pa", "monthcount","charge_off","industry",
                   "M_FUNI.IUSA","M_FLBR.IUSA","M_FHPNR.IUSA","M_FGDP.IUSA","M_FRFED.IUSA","M_FCBC.IUSA"]]
    return dfjan

# combining and encoding columns for the whole dataframe
def cox_p_h_prep(df):
    df["monthcount"] = parse_month_diff(df["mth_code"], df["snapshot"])
    df = df[["net_payment_behaviour_tripd","promotion_flag","variable_rate_index",
                   "bank_fico_buckets_20","mob","ever_delinquent_flg",
                   "nbr_mths_due", "credit_limit_pa", "monthcount","charge_off","industry",
                   "M_FUNI.IUSA","M_FLBR.IUSA","M_FHPNR.IUSA","M_FGDP.IUSA","M_FRFED.IUSA","M_FCBC.IUSA"]]
    return df

# fit and evaluate coxph
def cox_p_h(df):
    # model_training code

    # mydf = to_numeric(cox_p_h_prep_jan2018(df))
    # train_df, test_df = train_test_split(mydf, test_size=0.2)
    # cph = CoxPHFitter()
    # # cph.fit(train_df, duration_col="monthcount", event_col="charge_off", fit_options = {'step_size' : 0.2, 'max_steps': 1000})
    # cph.fit(train_df, duration_col="monthcount", event_col="charge_off", step_size= 0.2)
    # evaluate_cox_ph(cph, test_df)

    mydf = to_numeric(cox_p_h_prep(df))
    train_df, test_df = train_test_split(mydf, test_size=0.2)
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="monthcount", event_col="charge_off", step_size= 0.2)

    test_covariates = test_df.drop(["monthcount", "charge_off"], axis=1)
    partial_hazards = cph.predict_partial_hazard(test_covariates)
    from lifelines.utils import concordance_index
    c_index = concordance_index(test_df["monthcount"], -partial_hazards, test_df["charge_off"])
    print('Concordance index based on test_df:', c_index)    

    return cph

# use fitted coxph to predict based on months
def predictions(df, cph, num_months):
    # Calculate survival probabilities and hazard rates
    surv_func = cph.predict_survival_function(df)
    hazard_func = -np.log(surv_func)
    pred_events = np.zeros(num_months)

    # Loop over months and calculate predicted event count
    for month in range(1, num_months):
        total = df.shape[0]
        each = total // num_months
        total_hazard = hazard_func.loc[month, :each].sum()
        pred_events[month] = total_hazard * month

    return pred_events

# extract actual charge_off counts
def actuals(test_df, num_months):
    actual_nums = np.zeros(num_months)
    for index, row in test_df.iterrows():
        if row["charge_off"] == 1:
            actual_nums[int(row["monthcount"])] += 1
    return actual_nums

# plots mean survival rate
def plot_cph(num_months, mean_probs):
    plt.plot(np.arange(1, num_months+1), mean_probs)
    plt.title('Average Survival Probability, based on training_data')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.savefig('myplot.png')

# evualuate model with comparison and concordance index
def evaluate_cox_ph(cph, test_df):
    sf = cph.predict_survival_function(test_df)
    mean_probs = sf.values.mean(axis=1)
    num_months = len(mean_probs)
    plot_cph(num_months, mean_probs)
    actual = actuals(test_df, num_months)
    preds = predictions( test_df, cph, num_months)
    for i in range(len(actual)):
        print("For month {}: actual chgoff = {}, predicted chgoff = {}".format(i, actual[i], preds[i]))
    
    test_covariates = test_df.drop(["monthcount", "charge_off"], axis=1)
    partial_hazards = cph.predict_partial_hazard(test_covariates)
    from lifelines.utils import concordance_index
    c_index = concordance_index(test_df["monthcount"], -partial_hazards, test_df["charge_off"])
    print('Concordance index based on test_df:', c_index)    

# column selections using AIC and Pvalue
def column_selection(cph, data, timecolname, eventcolname):
    while True:
    #log_likelihood of the current model
        aic_current = cph.AIC_partial_
        aic = float('inf')
        to_remove = None #column to remove
        #find which variable we should remove
        for predictor in data.columns:
            if predictor in [timecolname, eventcolname]: #skip the columns representing "time" and "event" -> response variables
                continue
            #possible predictor candidates for the next iteration
            candidate_columns = [col for col in data.columns if col != predictor]
            candidate_data = data[candidate_columns]
            candidate_cph = CoxPHFitter()
            #fit the model again
            candidate_cph.fit(candidate_data, duration_col=timecolname, event_col=eventcolname, step_size = 0.2)
            #new AIC without the current predictor
            candidate_aic = candidate_cph.AIC_partial_
            #If the new AIC without the current predictor is better (smaller than 'aic')...
            if candidate_aic < aic:
                aic = candidate_aic 
                to_remove = predictor #this is the predictor we want to delete
        #We break out the while loop when no predictor can be removed to improve the AIC score
        if to_remove is None or aic >= aic_current:
            break
        #delete the selected predictor
        del data[to_remove]
        #re-fit the model
        cph.fit(data, duration_col=timecolname, event_col=eventcolname, step_size = 0.2)
        print("refitting")
        cph.print_summary()
    return data.columns
  
# generate 2020 monthly predictions
def predict_new(cph, newdf):
    df = to_numeric(cox_p_h_prep(newdf))
    # print(len(df))
    out = predictions(df, cph, 13)
    print("End Result:")
    for i in range(len(out)):
        print("In the {}th month after 2020 01, {} charge-offs are predicted".format(i, out[i]))
    plt.bar(np.arange(0,13),out,align='center', alpha=0.5)
    plt.ylabel('Number of Charge-off Accounts')
    monthnames = ["2020/01","2020/02","2020/03","2020/04","2020/05","2020/06","2020/07",
                "2020/08","2020/09","2020/10","2020/11","2020/12","2021/01"]
    plt.xticks(np.arange(0,13), monthnames, rotation=45)
    plt.title('Predicted num of charge off per month over time')
    plt.savefig("results.pdf")

main()