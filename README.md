# BO_DoE
## Bayesian Optimisation for Design of Experiment

In synthesis of multiple-linker Metal-Organic Frameworks (MOF), selection of compositions in experiment is challenging.
One can approximate the field of the outcomes as a Gaussian process, which can be optimised via Bayesian Optimisation.
The latter can suggest the batches of experimental points for synthesis. The process of synthesis and optimisation of experiments can be performed iteratively, starting from the initial batch of experiment.


Usage: Collect the data from the first experiment in the form of a table (See example .csv file)
Modify the bottom simple_BO.py for your needs and select possible ranges for variables.
Run `python simple_BO.py`

