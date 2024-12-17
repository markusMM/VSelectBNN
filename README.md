# VSelectBNN
Bayes-by-Backprop using probabilistic variable selection sampling and coefficients.

This repositiory basically reproduces Bayes by Backprop in a minimalistic way and uses optimization in form of creeping (unparameterized) samples. This, especially in the variable selection problem, assuming a Bernoulli prior, provides a marginal distribution over many possible frozen states of the system.

Additionally, the mean field posterior will be replaced by a hierachical approximate posterior depending on whether using variable selection and which kind of distribution was sampled in the previous layer(s).

Another unclear condition is the actual I|O likelihood, which, given other sources, seems to not fully represent the posterior distribution rather than trying to create differnt error metrics.

However this modules even includes 1D convolutional layers and Simple Recurrent Networks (SRNs) as well as a purely recurrent layers. 
Unfortunately those layers currently only accept input data with its time window dimension at the first place.

## uncertainty

The main idea beyond all these Baysian statistics, is almost certain, the measurement of a mathematical model, based on observation and probability, to be confident about its decisions. 
The principle was long ago already well know as Markov Chain Monte Carlo, where the sampling was the most important part to retrieve the closest approximate posterior.
With scalable and fast paste variational inference however, the , former intractable, posterior of different solutions in algorithms, like EM, MLE, MAP and ADVI, became computable, even with less resources and due to variational sampling and preselection.

## variable selection

We emphasis on the sparse probit model of a selective prior $p(h|\Theta) = Ber(h|\pi_h)$ with $h \in [1;0]^{N,H}$, H is the length of our parameter space and N the numper of data points.
This preselection refers to many previouse researches and works shrinking the problem of large and complex parameter spaces to computational tractable mappings without loosing computational recovery of the parameter space.

## requirements

for the modules:
  PyTorch (latest)
  Numpy (latest)
  math

for the test script:
  da_test.csv
  Pandas (latest)

Backend:
  YAML (latest) ~ sorry no better format jet---
