# Data generation for benchmarking RSL algorithms


This technique is presented in the paper *Towards Principled Synthetic
Benchmarks for Explainable Rule Set Learning Algorithms* presented at the
*Evolutionary Computing and Explainable Artificial Intelligence* (ECXAI)
workshop taking place as part of the 2023 GECCO conference.


The general idea is to generate many learning tasks (i.e. data-generating
models) that have the same form as models created by rule set learning
algorithms (and most importantly, metaheuristic rule set learners such as
Learning Classifier Systems or Artificial Immune Systems) and then store both
the data-generating model as well as training and test data to NPZ files. This
allows for a different kind of analysis for rule set learning algorithms since
not only goodness-of-fit to the data can be estimated but also how close the
learnt model is to the data-generating (“ground truth”) model. In particular,
the behaviour of the metaheuristics used by metaheuristic rule set learners can
be investigated and explained more directly than with mere train/test data–based
approaches since the ground truth model provides a clear known target for the
optimization process. For more details, see above-mentioned paper and the code.


## Generating learning tasks and data sets


First, enter a development shell.

```
nix develop
```


Then, generate many (e.g. 30) data sets, e.g. with input space dimension 20, 10
model components and 5000 training data points (and 50000 testing data points),
and prefix the generated NPZ files with `data/rsl` (i.e. store them in the
`data` directory):


```
python gen_data.py genmany -d 20 -K 10 --startseed=0 --endseed=29 5000 data/rsl
```


## Generating data sets as the ones used in the ECXAI paper


To generate many data sets in parallel, you can use GNU parallel. We used this
command to generate the data for the above-mentioned paper:


```
nix develop --command \
    parallel \
    python gen_data.py genmany \
    --startseed=0 \
    --endseed=9 \
    -d '{1//}' \
    -K '{2}' '{1/}' \
    data/rsl \
    ::: "1/300" "3/500" "5/1000" "10/2000" "20/5000" \
    ::: 5 10 20
```
