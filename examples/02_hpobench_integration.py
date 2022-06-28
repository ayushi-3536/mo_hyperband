"""
This script runs a Multi-Objective Hyperparameter Optimisation using MOhb on adult benchmark in HPOBENCH:
'https://github.com/automl/HPOBench/blob/mo_experiments/hpobench/benchmarks/mo/adult_benchmark.py'
Installation:

git clone -b development  https://github.com/automl/HPOBench.git
cd HPOBench
pip install .
cd ..
pip install pygmo~=2.6

"""

import argparse
import os
import pickle
import time
import numpy as np
from mo_hyperband import MOHB
from hpobench.benchmarks.mo.adult_benchmark import AdultBenchmark
from distributed import Client
from pygmo import hypervolume

adult_benchmark = AdultBenchmark()
search_space = adult_benchmark.get_configuration_space()
objectives = adult_benchmark.get_objective_names()


def objective_function(config, budget, **kwargs):
    """ The target function to minimize for HPO"""

    res = adult_benchmark.objective_function(configuration=config, fidelity={"budget": int(budget)})
    res = {
        'function_value': {'error': 1 - res['function_value']['accuracy'],
                           'DSP': res['function_value']['accuracy']},
        "cost": res['cost'],
        "info": res['info']
    }
    return res


def input_arguments():
    parser = argparse.ArgumentParser(description='Optimizing MNIST in PyTorch using mohb.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--refit_training', action='store_true', default=False,
                        help='Refit with incumbent configuration on full training data and budget')
    parser.add_argument('--min_budget', type=float, default=1,
                        help='Minimum budget (epoch length)')
    parser.add_argument('--max_budget', type=float, default=200,
                        help='Maximum budget (epoch length)')
    parser.add_argument('--eta', type=int, default=3,
                        help='Parameter for Hyperband controlling early stopping aggressiveness')
    parser.add_argument('--output_path', type=str, default="./adult_rw",
                        help='Directory for mohb to write logs and outputs')
    parser.add_argument('--scheduler_file', type=str, default=None,
                        help='The file to connect a Dask client with a Dask scheduler')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPU workers for MOHB to distribute function evaluations to')
    parser.add_argument('--single_node_with_gpus', default=False, action="store_true",
                        help='If True, signals the MOHB run to assume all required GPUs are on '
                             'the same node/machine. To be specified as True if no client is '
                             'passed and n_workers > 1. Should be set to False if a client is '
                             'specified as a scheduler-file created. The onus of GPU usage is then'
                             'on the Dask workers created and mapped to the scheduler-file.')
    parser.add_argument('--verbose', action="store_true", default=True,
                        help='Decides verbosity of MOHB optimization')
    parser.add_argument('--runtime', type=float, default=6,
                        help='Total time in seconds as budget to run MOHB')
    args = parser.parse_args()
    return args


def main():
    args = input_arguments()

    # Some insights into Dask interfaces to mohb and handling GPU devices for parallelism:
    # * if args.scheduler_file is specified, args.n_workers need not be specifed --- since
    #    args.scheduler_file indicates a Dask client/server is active
    # * if args.scheduler_file is not specified and args.n_workers > 1 --- the mohb object
    #    creates a Dask client as at instantiation and dies with the associated mohb object
    # * if args.single_node_with_gpus is True --- assumes that all GPU devices indicated
    #    through the environment variable "CUDA_VISIBLE_DEVICES" resides on the same machine

    # Dask checks and setups
    single_node_with_gpus = args.single_node_with_gpus
    if args.scheduler_file is not None and os.path.isfile(args.scheduler_file):
        client = Client(scheduler_file=args.scheduler_file)
        # explicitly delegating GPU handling to Dask workers defined
        single_node_with_gpus = False
    else:
        client = None

    ###########################
    # mohb optimisation block #
    ###########################
    np.random.seed(args.seed)

    # The scalarization algorithm to be used and its parameters are provided as
    # a dictionary. There are three different scalarization algorithms available.
    random_weights_options = {
        "algorithm": "random_weights",
        "num_weights": 100
    }

    parego_options = {
        "algorithm": "parego",
        "num_weights": 100,
        "rho": 0.05,
    }

    golovin_options = {
        "algorithm": "golovin",
        "num_weights": 100,
    }
    mohb = MOHB(f=objective_function, cs=search_space, min_budget=args.min_budget,
                max_budget=args.max_budget, eta=args.eta, output_path=args.output_path,
                objectives=['error', 'DSP'], mo_strategy=random_weights_options,
                # if client is not None and of type Client, n_workers is ignored
                # if client is None, a Dask client with n_workers is set up
                client=client, n_workers=args.n_workers)
    traj, runtime, history = mohb.run(total_cost=args.runtime, verbose=args.verbose,
                                      # arguments below are part of **kwargs shared across workers
                                      single_node_with_gpus=single_node_with_gpus)
    name = time.strftime("%x %X %Z", time.localtime(mohb.start))
    name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
    mohb.logger.info("Saving optimisation trace history...")
    with open(os.path.join(args.output_path, "history_{}.pkl".format(name)), "wb") as f:
        pickle.dump(history, f)
    pareto = mohb.pareto_trials
    acc = [trial.get_fitness() for trial in pareto]
    mohb.logger.info(f"pareto fitness:{acc}")
    with open(args.output_path + '/pareto_front.txt', 'a+')as f:
        np.savetxt(f, acc)

    hv = hypervolume(acc)
    mohb.logger.info(f"hypervolume obtained:{hv.compute([1.0, 1.0])}")

    # end of HB optimisation


if __name__ == "__main__":
    main()
