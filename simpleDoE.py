import sys
import time
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization

class Experiment:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.get_float_numbers()
        self.best = self.new_phase()
        self.variables = [self.df[f].values for f in self.df.columns[:-1]]
        self.ranges = [list(set(var)) for var in self.variables]
        self.references = np.array(self.variables).T
        self.objective = self.df['outcome'].values.reshape(-1, 1)
        self.X_init = np.array([np.array(s) for s in self.references])
        self.nex = []

    def get_float_numbers(self):
        for f in self.df.columns:
           self.df[f] = self.df[f].apply(lambda x: float(x))

    def f(self, x):
        """ objective function - to mimic XRD signal in compostional space"""
        return -abs(np.dot(np.where(np.all(self.references == x, axis=1), 1, 0), self.objective))

    def generate_new_points(self, ranges=None):
        if isinstance(ranges, list):
            self.candidates = np.array([np.array(p) for p in product(*ranges)])
        else:
            self.candidates = np.array([np.array(p) for p in product(*self.ranges)])

    def get_timelog(self):
        t = time.localtime()
        #timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        timestamp = time.strftime('%b-%d', t)
        return f"logfile-{timestamp}-N{len(self.candidates)}" 

    def plot_3d(self, image):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        seeds = self.references
        new = self.nex
        best = self.best
        ax.scatter(seeds[:,0], seeds[:,1], seeds[:,2], marker='^', label='Seeds')
        if len(new):
            ax.scatter(new[:,0], new[:,1], new[:,2], marker='o', label='BO suggested')
        if len(best):
            ax.scatter(best[:,0], best[:,1], best[:,2], s=80, marker='*', label='New phase')
        ax.set_xlabel(f'{self.df.columns[0]}')
        ax.set_ylabel(f'{self.df.columns[1]}')
        ax.set_zlabel(f'{self.df.columns[2]}')
        fig.legend()
        plt.show()
        fig.savefig(image)

    def new_phase(self):
        outcome = self.df.columns[-1]
        bf = self.df[self.df[outcome] > 100]
        return np.array([bf[f].values for f in self.df.columns[:-1]]).T

    def mixed_domain(self):
        domain = []
        for name, irange in zip(self.df.columns, self.ranges):
            domain.append({'name': name, 'type': 'continuous', 'domain': (min(irange), max(irange))})
        return domain

    def get_densepoints(self, midpoints, n_points, i, b):
        """ get densification ranges """

        if b - midpoints[i] > min(self.ranges[i]):
            hitmin = b - midpoints[i] 
        else: 
            hitmin = min(self.ranges[i])
        if b + midpoints[i] < max(self.ranges[i]):
            hitmax = b + midpoints[i]
        else:
            hitmax = max(self.ranges[i])
        return list(np.linspace(hitmin, hitmax, n_points[i]))

    def densify_ranges(self, zoom=4):
        ''' densify grid around successful experiments '''
        self.best = self.new_phase()
        n_points = [zoom*len(irange) for irange in self.ranges]
        midpoints = [(max(i) - min(i)) / zoom for i in self.ranges]
        dense = [[] for i in n_points]
        for best in self.best:
             for i, b in enumerate(best):
                 densier = self.get_densepoints(midpoints, n_points, i, b)
                 dense[i] += [round(i, 2) for i in densier]
        self.ranges = [[round(i, 2) for i in list(set(r + d))] for r, d in zip(self.ranges, dense)] 
        self.generate_new_points()
        print('Best:', self.best)
        print('Dense grid:', dense)

    def run_optimisation(self, iterations=1, batch_size=50):
        """ Run bayesian optimisation iterations times """
        logfile = self.get_timelog()
        nexx = {}
        for iteration in range(iterations):
            bo = BayesianOptimization(f=self.f,
                       domain=self.domain,
                       X = self.X_init,
                       Y = self.objective,
                       acquisition_type = 'EI',
                       #acquisition_type = 'MPI',
                       evaluator_type = 'thompson_sampling',
                       batch_size = batch_size,
                       exact_feval = True,
                       de_duplication = False)
            nex = bo.suggest_next_locations()
            for i in nex:
                if ','.join(map(str, i)) not in nexx:
                    nexx[','.join(map(str, i))] = 1
                else:
                    nexx[','.join(map(str, i))] += 1
            print("iteration: ", iteration, len(nexx), ' points')

        nexx = {n:i for n,i in sorted(nexx.items(), key=lambda x: x[1], reverse=True)}
        self.nex = np.array([[float(i) for i in point.split(',')] for point in list(nexx.keys())[:batch_size]])
        for n,i in nexx.items():
             print(n,',',i, file=open(f'{logfile}', 'a')) 

    def optimise_doe(self, batch_size=50, iterations=1, mixed_domain=False, zoomin=0, ranges=None):
        """ Set domain, generate candidate points (zoomin the best regions), suggest best points """
        if zoomin:
            print(f'Zooming in: {zoomin}')
            self.densify_ranges(zoom=zoomin)
        else:
            self.generate_new_points(ranges)

        print('N candidates:', len(self.candidates))

        if mixed_domain:
            self.domain = self.mixed_domain()
        else:
            # Discrete domain
            self.domain = [{'name': 'var_1', 'type': 'bandit', 'domain': self.candidates}]

        # Run optimisation multiple times
        self.run_optimisation(iterations=iterations, batch_size=batch_size)


# =================================  MODIFY BELOW =============================== 

if __name__ == '__main__':
    # prepare data in 'input.csv' file in a table format: 'var1, var2, var3, output'
    # where var1, var2, var3 (or more than 3) are parameters of the experiment
    # 'output' is a numerical result of the experiment, e.g., XRD intensity:
    # 
    #----------------------------- example 'input.csv' ---------------------------  
    # ZrOCl2,Fumaric,FA:Zr,outcome      # names of the columns are arbitrary 
    # 0.33,0.25,167,0                   # 0 in the last column represents 'Gel Product' 
    # 1.0,0.25,167,50                   # 50 in the last column represents 'Solid Product'
    # 1.0,0.67,334,1000                 # 1000 in the last column represents 'New Phase'
    #-----------------------------------------------------------------------------  

    data1 = Experiment('input_KA.csv')
    
    # choose ranges of values for variables of experiment if you like:
    ranges = None    # default. The ranges will remain the same as in the input.csv
    #----------------------------- example ranges  --------------------------- 
    range_ZrOCl2 = [round(i,2) for i in np.linspace(0.6, 1.40, 10)] # min, max, N points
    range_Fumari = [round(i,2) for i in np.linspace(0.5, 0.70, 5)] # or equivalently explicitly [0.5, 0.54, 0.58, 0.62, 0.66, 0.7] 
    range_FAtoZr = [round(i,2) for i in np.linspace(251, 418, 10)] 

    range_ZrOCl2 = [round(i,2) for i in np.linspace(0.33, 2, 10)]
    range_Fumari = [round(i,2) for i in np.linspace(0.25, 0.67, 10)]
    range_FAtoZr = [round(i,2) for i in np.linspace(167, 501, 10)]
    ranges = [range_ZrOCl2, range_Fumari, range_FAtoZr]   # pack ranges together


    # choose parameters of optimisation
    batch_size = 12  # default. Number of points to suggest
    iterations = 10  # default. Number of optimisation runs
    zoomin = 2       # default: 0. If zoomin > 0: densify the grid around the best experiments:
                     # each variable will have additional points in the area 
                     #  +/- (max(range) - min(range)) / zoom 
                     # around the best experiments. This increases an initial number 
                     # of points along each axis by the (zoom x number of points along this axis in input.csv)

    # run optimisation
    data1.optimise_doe(zoomin=zoomin, batch_size=batch_size, iterations=iterations, ranges=ranges)

    # the results - the suggested parameters of the new experiments and the frequencies
    # of these suggestions, if 'iterations' > 1  - are stored in the logfile-{timestamp}  

    # plot the results if you like, choose the name of a file to store the image
    data1.plot_3d('image_doe.png')
