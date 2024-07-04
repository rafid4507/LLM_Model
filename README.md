# LLM_Model
### Title: Using Large Language Models for Predicting Bitcoin Prices on Binance

Trying to develop a predictive model using Large Language Models (LLMs) to forecast Bitcoin (BTC) price movements on Binance. Evaluate the model's performance with the Sharpe Ratio, utilizing both backtesting and out-of-sample analysis.

## Task Description:


Extracting and preparing historical BTC price data from Binance for analysis, focusing on key metrics like open, high, low, close, and volume. I have also added sentiment score analyzing the wikipedia Bitcoin page comments to improve the prediction.

1. At first I am collecting the historical_data by "data_collection.ipynb" file in the csv format.
2. "data_with_sentiment.ipynb" file is getting sentiment score from the wikipedia bitcoin comments and analysing them using Textblob and saving the data in the merged_data.csv file for future usage
3. "preparing_data.ipynb" is used to get the merged data and preprocess for traning the transformer model, saved in the btc_data.csv file


## Model Development:

Utilized a LLM, specifically from the Transformers library, to analyze trends and predict future BTC prices.

I have used the following parameters to train the model:
#### parameters:
input_dim = 10  # number of features
num_heads = 1  # input_dim must be divisible by num_heads
num_layers = 2
hidden_dim = 128

I did not go for hyper parameter tuning. And used only 50 epochs due to the computational complexity.

I have trained and tested the model using segregated datasets to ensure robustness.

## Development Environment:

Implemented the project in Google Colab for access to free GPUs, facilitating efficient model training and experimentation.
Leverage Python libraries such as pandas for data manipulation and pytorch for model training.

## Backtesting and Evaluation:

Conducted backtesting of the model’s predictions to simulate trading strategies and their outcomes using historical data.
Evaluated the model’s performance by calculating the Sharpe Ratio and comparing both in-sample and out-of-sample results.

