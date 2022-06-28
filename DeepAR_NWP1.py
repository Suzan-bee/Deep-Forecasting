#!pip install --upgrade mxnet-cu101==1.6.0.post0

##importing necessary libraries
import pandas as pd
import numpy as np
import mxnet as mx
import random
import json
from tqdm import tqdm,tqdm_notebook
import matplotlib.pyplot as plt
from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.model import deepar
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from typing import Dict, Tuple, Union
from gluonts.model.forecast import Forecast
import matplotlib.dates as mdates


## MAAPE evaluator class
class MyEvaluator(Evaluator):

    @staticmethod
    def maape(target, forecast):
        denominator = np.abs(target)
        flag = denominator == 0

        maape = np.mean(
            np.arctan((np.abs(target - forecast) * (1 - flag)) / (denominator + flag))
        )
        return maape
#     def mae(target, forecast):
#         denominator = np.abs(target)
#         flag = denominator == 0

#         mae = np.mean((np.abs(target - forecast)))
           
#         return mae

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None]]:
        metrics = super().get_metrics_per_ts(time_series, forecast)

        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        pred_target = np.ma.masked_invalid(pred_target)
        median_fcst = forecast.quantile(0.5)

        metrics["MAAPE"] = self.maape(
            pred_target, median_fcst
        ) 
#         metrics["MAE"]=self.mae(
#              pred_target, median_fcst)
        return metrics

    def get_aggregate_metrics(
        self, metric_per_ts: pd.DataFrame
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        totals, metric_per_ts = super().get_aggregate_metrics(metric_per_ts)

        agg_funs = {
            "MAAPE": "mean",
#             "MAE":"mean",
        }
        assert (
            set(metric_per_ts.columns) >= agg_funs.keys()
        ), "Some of the requested item metrics are missing."
        my_totals = {
            key: metric_per_ts[key].agg(agg) for key, agg in agg_funs.items()
        }

        totals.update(my_totals)
        return totals, metric_per_ts

## DATA
    
dfm = pd.read_csv('fmi_helsinki_101004_meteopv.txt',sep=';',decimal=',')
dfp = pd.read_csv('fmi_helsinki_pvproduction.csv',sep=';', skiprows=14, decimal=',')
dfm.drop(columns=['#fmisid','stationname'],inplace=True, axis=1)
#dfm.drop('stationname', inplace=True, axis=1)
dfp = dfp.rename(columns = {'#prod_time': 'prod_time'}, inplace = False)
dfp.drop(0,axis=0,inplace=True)

dfm['prod_time'] = pd.to_datetime(dfm['prod_time'])
dfm=dfm.set_index('prod_time')
dfp['prod_time'] = pd.to_datetime(dfp['prod_time'])
dfp=dfp.set_index('prod_time')
dfm=dfm.astype(float)
cols=['TA_PT1M_AVG','WS_PT10M_AVG','WD_PT10M_AVG']
dfm[cols]=dfm[cols].interpolate(method='linear')

dfm= dfm.resample('10T').mean()
dfm= dfm.resample('1H').mean()
dfp=dfp.astype(float)
dfp= dfp.resample('1H').mean()

dfm=dfm.reset_index()
dfp=dfp.reset_index()
dfm['prod_time'] = pd.to_datetime(dfm['prod_time'])
dfp['prod_time'] = pd.to_datetime(dfp['prod_time'])
df =pd.merge(dfm, dfp, on='prod_time',how = 'outer')
cols=['pv_inv_out','pv_inv_in','pv_str_1','pv_str_2']
df[cols]=df[cols].interpolate(method='linear')
df.head()

df['prod_time'] = pd.to_datetime(df['prod_time'])
df=df.set_index('prod_time')
df=df.fillna(0)

# Time feature
from gluonts.time_feature import (
        DayOfMonth,
        DayOfWeek,
        DayOfYear,
        HourOfDay,
        MinuteOfHour,
        MonthOfYear,
        WeekOfYear
        
    )

time_features = [DayOfMonth,
        DayOfWeek,
        DayOfYear,
        HourOfDay,
        MinuteOfHour,
        MonthOfYear,
        WeekOfYear]

## Setting target value and weather data

columns=['pv_inv_out']
target_cols = list(set(df[columns]))
#,'DIFF_PT1M_AVG','DIR_PT1M_AVG','TTECH_PT1M_AVG(:32)','TTECH_PT1M_AVG(:33)','TA_PT1M_AVG','WS_PT10M_AVG','WD_PT10M_AVG',,'TTECH_PT1M_AVG(:31)','TA_PT1M_AVG'
#GLOB_PT1M_AVG: global radiation on horizontal surface
#GLOBA_PT1M_AVG(:31): global radiation on inclined PV plane-of-array 
feature_cols=['GLOB_PT1M_AVG','DIR_PT1M_AVG','TA_PT1M_AVG']
f=list(set(df[feature_cols]))


# Splitting Data 

df_train=df.iloc[1:42086,:]
df_test=df.iloc[42086:51000,:]


## Setting model hyperparameter

freq="1H"
start_train = pd.Timestamp("2015-12-21  12:00:00", freq=freq)
start_test = pd.Timestamp("2019-10-20 14:00:00", freq=freq)
prediction_length=1

## DeepAR_Estimator

estimator = DeepAREstimator(freq=freq, 
                            context_length=24,
                            prediction_length=prediction_length,
                            num_layers=2,
                            num_cells=50,
                            cell_type='lstm',
                            distr_output=NegativeBinomialOutput(),
                            dropout_rate=0.1,
                            use_feat_dynamic_real=True,
                            trainer=Trainer(epochs=100,
                                            learning_rate=1e-3,
                                            batch_size=32,
                                            #patience=20,
                                                                                       
                                          ))


 # train_dataset and test_dataset
    
train_ds = ListDataset([
    {
        FieldName.TARGET: df_train[:-prediction_length][target_cols].to_numpy().reshape(-1,),
        FieldName.START: start_train,
        #FieldName.FEAT_STATIC_CAT: fsc,
         FieldName.FEAT_DYNAMIC_REAL:df_train[:-prediction_length][f].to_numpy().reshape(3,-1),
        FieldName.FEAT_TIME:time_features
    }
    
], freq=freq)
test_ds = ListDataset([
    {
        FieldName.TARGET: df_test[target_cols].to_numpy().reshape(-1,),
        FieldName.START: start_test,
        #FieldName.FEAT_STATIC_CAT: fsc,
         FieldName.FEAT_DYNAMIC_REAL:df_test[f].to_numpy().reshape(3,-1),
        FieldName.FEAT_TIME:time_features
    }
    
], freq=freq)

## Model training Preditor

# predictor = estimator.train(training_data=train_ds)


# save the trained model in tmp/
from pathlib import Path
# predictor.serialize(Path("/suzan_yemane/work/suzan_thesis/newexp/DeepAR_NWP1/"))

# loads it back
from gluonts.model.predictor import Predictor
predictor = Predictor.deserialize(Path("/suzan_yemane/work/suzan_thesis/newexp/DeepAR_NWP1/"))


 
#generating rolling dataset to make all prediction in the test_dataset    

from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset,
) 
dataset_rolled = generate_rolling_dataset(
        dataset=test_ds,
        start_time=pd.Timestamp("2019-10-24 14:00:00", freq="1H"),
        end_time=pd.Timestamp("2020-10-25 23:00:00", freq="1H"),
        strategy=StepStrategy(
            prediction_length=prediction_length, step_size=24
        ),
    )


##Evaluating the forecasting

forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset_rolled,  
    predictor=predictor,  
    num_samples=len(dataset_rolled), 
)
forecasts = list(forecast_it)
tss = list(ts_it)

## Print the first forecast value information

print(f"Number of sample paths: {forecasts[0].num_samples}")
print(f"Dimension of samples: {forecasts[0].samples.shape}")
print(f"Start date of the forecast window: {forecasts[0].start_date}")
print(f"Frequency of the time series: {forecasts[0].freq}")


## Plotting best/worst/average
# item_metrics.sort_values('MASE', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)
# x=item_metrics['MSE']
# np.sqrt(x[49])


## Plotting based on the tss and forecast index 


plot_length = 150
prediction_intervals = (80.0, 90.0)
legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
fig = plt.figure()
fig = plt.figure(figsize=(9,5))
ax = plt.gca()
plt.plot(tss[0][-plot_length:])  # plot the time series
forecasts[0].plot(prediction_intervals=prediction_intervals, color='g')
formatter = mdates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(formatter)
plt.xlabel("Day")
plt.ylabel("Solar energy production (in W)")
plt.grid(which="both")
plt.legend(legend, loc="upper left")
plt.show()
# plt.savefig("Image5_f.jpg")

def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = prediction_length
    prediction_intervals = (80.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig = plt.figure()
    fig = plt.figure(figsize=(9,5))
    ax = plt.gca()
    plt.plot(tss[0][-plot_length:])  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    formatter = mdates.DateFormatter('%H:%M')
    # months = mdates.MonthLocator()
    # ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(formatter)
    plt.xlabel("Time(h)")
    plt.ylabel("Solar energy production (in W)")
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()
#     plt.savefig("Image5_par.jpg")
for i in tqdm(range(1)):
    ts_entry = tss[i]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry)   

#evaluating the forecating performance  
    
evaluator = MyEvaluator(quantiles=[0.1, 0.8, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset_rolled))

#printing the results
print(json.dumps(agg_metrics, indent=4))

#saving the results
# with open('Image5.txt', 'w') as outfile:
#     json.dump(agg_metrics, outfile)
