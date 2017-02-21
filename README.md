# Bayesian Norm Detection Experimental Code

## Installation and Running
This package depends on the [ThinkBayes](http://www.thinkbayes.com/) code package. We already include the parts of the package (as of 2015) that we use in this repository in the [thinkbayes.py](thinkbayes.py), but you may wish to update it to the latest version of it from [ThinkBayes' Github repository](https://github.com/AllenDowney/ThinkBayes). 

Once you have this installed, you can either try to run your own tests using the instructions below, or run the [Benchmarks](#benchmarks)

### Software packages

For purposes of testing, we encapsulate norm detection algorithms in the [norm_detector](norm_detector.py) class, which is then subclassed to implement different norm detection strategies, namely:

- Bayesian Norm Detection in the [bayesian_norm_detector](bayesian_norm_detector.py) class
- Oren and Meneguzzi's algorithms from COIN
   - A basic norm detector in the [basic_norm_detector](oren_meneguzzi_norm_detector.py) class
   - A threshold norm detector in the [treshold_norm_detector](oren_meneguzzi_norm_detector.py) class

### Creating a Norm Suite

### Introducing observations

## Benchmarks

Benchmarks were originally implemented as stand-alone scripts, including [aamas_experiments.py](aamas_experiments.py) and [journal_experiments.py](journal_experiments.py)
