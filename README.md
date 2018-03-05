# ML Forex Trading

Here I'm experimenting with strategies for trading Forex using some ML approaches.

`RL Approach` contains an archive of a predious attempt, involving tagging winning and losing trades, training a classifier on these trades, then trying to use this to trade successfully.

The more recent attempts have been looking into creating a Signal based model. This model contains four sub-models, each producing *entry* and *exit* signals for *long* and *short* positions.

Various approaches have been experimented with for training these sub-models, including some traditional supervised learning approaches as well as effective searches over the parameter space; random search, evolutionary search and simulated annealing are all implemented here.

The code first preprocesses tick data from [here](http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/EURUSD), placed in a `EURUSD` directory (see source code for filenames etc). This is then chunked up and used for train/test.

The model preprocesses data it is given by taking diffs between OHLCs, and by using some technical indicators (the parameters for these are included in the model parameter search).

So far I am still not happy with the results.