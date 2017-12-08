# ML Forex Tests

Trying out applying some ML approaches to Forex trading. First I mark success / failure of trades with fixed stop loss and take profit (preprocessing). Next add some technical indicators, then see if I can train a ML model on this and print some data about the performance of the model.

I'm using data from [here](http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/EURUSD); minute OHLCV data is put into a data/raw/ directory (a data/preprocessed/ directory should be made too).
