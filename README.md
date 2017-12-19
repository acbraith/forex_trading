# ML Forex Tests

Trying out applying some ML approaches to Forex trading. First I mark success / failure of trades with fixed stop loss and take profit (preprocessing). Next add some technical indicators, then see if I can train a ML model on this and print some data about the performance of the model.

I'm using data from [here](http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/EURUSD); minute OHLCV data is put into a data/raw/ directory (a data/preprocessed/ directory should be made too).

Run using `python3 model_testing.py`. This will preprocess your files (in `/data`), then use an autoregressive model on OHLC data to try and predict trade success / failure. Cross validation is done and results are displayed.

So far this doesn't manage to trade with a profit.

`model_testing.py` can be changed to use a technical indicator based model, which appears to have some more success. Making these changes easy to make is somewhere on my todo list.
