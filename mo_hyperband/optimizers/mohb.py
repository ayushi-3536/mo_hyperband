import os
import sys
import json
import time
import pickle
import numpy as np
from typing import List
from copy import deepcopy
from loguru import logger
from distributed import Client
from mo_hyperband.optimizers import Trial
from mo_hyperband.utils import SHBracketManager
logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])
_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB"
}



class MOHB:
    def __init__(self, cs=None, f=None, min_budget=None,
                 max_budget=None, eta=3, min_clip=None, max_clip=None, n_workers=None, client=None, **kwargs):

        self.iteration_counter = -1
        self.cs = cs
        self.f = f
        self.mo_params = {
            "cs": self.cs,
            "f": f
        }

        # Hyperband related variables
        self.min_budget = min_budget
        self.max_budget = max_budget
        assert self.max_budget > self.min_budget, "only (Max Budget > Min Budget) supported!"
        self.eta = eta
        self.min_clip = min_clip
        self.max_clip = max_clip

        # Precomputing budget spacing and number of configurations for HB iterations
        self.max_SH_iter = None
        self.budgets = None
        if self.min_budget is not None and \
                self.max_budget is not None and \
                self.eta is not None:
            self.max_SH_iter = -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta)) + 1
            self.budgets = self.max_budget * np.power(self.eta,
                                                      -np.linspace(start=self.max_SH_iter - 1,
                                                                   stop=0, num=self.max_SH_iter))

        # Miscellaneous
        self.output_path = kwargs['output_path'] if 'output_path' in kwargs else './'
        os.makedirs(self.output_path, exist_ok=True)
        self.logger = logger
        log_suffix = time.strftime("%x %X %Z")
        log_suffix = log_suffix.replace("/", '-').replace(":", '-').replace(" ", '_')
        self.logger.add(
            "{}/mohb_{}.log".format(self.output_path, log_suffix),
            **_logger_props
        )
        self.log_filename = "{}/mohb_{}.log".format(self.output_path, log_suffix)
        # Updating MO parameter list
        self.mo_params.update({"output_path": self.output_path})

        self.inc_score = np.inf
        self.inc_config = None
        self.history = []
        self.hb_bracket_tracker = {}
        self._max_pop_size = None
        self.active_brackets = []  # list of SHBracketManager objects
        self.traj = []
        self.runtime = []
        self.history = []
        self.start = None

        # Dask variables
        if n_workers is None and client is None:
            raise ValueError("Need to specify either 'n_workers'(>0) or 'client' (a Dask client)!")
        if client is not None and isinstance(client, Client):
            self.client = client
            self.n_workers = len(client.ncores())
        else:
            self.n_workers = n_workers
            if self.n_workers > 1:
                self.client = Client(
                    n_workers=self.n_workers, processes=True, threads_per_worker=1, scheduler_port=0
                )  # port 0 makes Dask select a random free port
            else:
                self.client = None
        self.futures = []
        self.shared_data = None

        # Misc.
        self.available_gpus = None
        self.gpu_usage = None
        self.single_node_with_gpus = None

    def reset(self):
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.traj = []
        self.runtime = []
        self.history = []
        self.logger.info("\n\nRESET at {}\n\n".format(time.strftime("%x %X %Z")))

    def get_next_iteration(self, iteration):
        '''Computes the Successive Halving spacing

        Given the iteration index, computes the budget spacing to be used and
        the number of configurations to be used for the SH iterations.

        Parameters
        ----------
        iteration : int
            Iteration index
        clip : int, {1, 2, 3, ..., None}
            If not None, clips the minimum number of configurations to 'clip'

        Returns
        -------
        ns : array
        budgets : array
        '''
        # number of 'SH runs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # budget spacing for this iteration
        budgets = self.budgets[(-s - 1):]
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
        if self.min_clip is not None and self.max_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
        elif self.min_clip is not None:
            ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def __getstate__(self):
        """ Allows the object to picklable while having Dask client as a class attribute.
        """
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        return d

    def __del__(self):
        """ Ensures a clean kill of the Dask client and frees up a port.
        """
        if hasattr(self, "client") and isinstance(self, Client):
            self.client.close()

    def f_objective(self, config, budget=None, **kwargs):
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        # if self.encoding:
        #     x = self.map_to_original(x)
        # converts [0, 1] vector to a ConfigSpace object

        if budget is not None:  # to be used when called by multi-fidelity based optimizers
            res = self.f(config, budget=budget, **kwargs)
        else:
            res = self.f(config, **kwargs)
        assert "fitness" in res
        assert "cost" in res
        return res

    def _f_objective(self, job_info):
        """ Wrapper to call MO's objective function.
        """
        # check if job_info appended during job submission self.submit_job() includes "gpu_devices"
        if "gpu_devices" in job_info and self.single_node_with_gpus:
            # should set the environment variable for the spawned worker process
            # reprioritising a CUDA device order specific to this worker process
            os.environ.update({"CUDA_VISIBLE_DEVICES": job_info["gpu_devices"]})

        config, budget = job_info['config'], job_info['budget']
        bracket_id = job_info['bracket_id']
        kwargs = job_info["kwargs"]
        res = self.f_objective(config, budget, **kwargs)
        info = res["info"] if "info" in res else dict()
        run_info = {
            'fitness': res["fitness"],
            'cost': res["cost"],
            'config': config,
            'budget': budget,
            'trial': job_info['trial'],
            'bracket_id': bracket_id,
            'info': info
        }

        if "gpu_devices" in job_info:
            # important for GPU usage tracking if single_node_with_gpus=True
            device_id = int(job_info["gpu_devices"].strip().split(",")[0])
            run_info.update({"device_id": device_id})
        return run_info

    def _create_cuda_visible_devices(self, available_gpus: List[int], start_id: int) -> str:
        """ Generates a string to set the CUDA_VISIBLE_DEVICES environment variable.

        Given a list of available GPU device IDs and a preferred ID (start_id), the environment
        variable is created by putting the start_id device first, followed by the remaining devices
        arranged randomly. The worker that uses this string to set the environment variable uses
        the start_id GPU device primarily now.
        """
        assert start_id in available_gpus
        available_gpus = deepcopy(available_gpus)
        available_gpus.remove(start_id)
        np.random.shuffle(available_gpus)
        final_variable = [str(start_id)] + [str(_id) for _id in available_gpus]
        final_variable = ",".join(final_variable)
        return final_variable

    def distribute_gpus(self):
        """ Function to create a GPU usage tracker dict.

        The idea is to extract the exact GPU device IDs available. During job submission, each
        submitted job is given a preference of a GPU device ID based on the GPU device with the
        least number of active running jobs. On retrieval of the result, this gpu usage dict is
        updated for the device ID that the finished job was mapped to.
        """
        try:
            available_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            available_gpus = available_gpus.strip().split(",")
            self.available_gpus = [int(_id) for _id in available_gpus]
        except KeyError as e:
            print("Unable to find valid GPU devices. "
                  "Environment variable {} not visible!".format(str(e)))
            self.available_gpus = []
        self.gpu_usage = dict()
        for _id in self.available_gpus:
            self.gpu_usage[_id] = 0



    def reset(self):
        super().reset()
        if self.n_workers > 1 and hasattr(self, "client") and isinstance(self.client, Client):
            self.client.restart()
        else:
            self.client = None
        self.futures = []
        self.shared_data = None
        self.iteration_counter = -1
        self._max_pop_size = None
        self.start = None
        self.active_brackets = []
        self.traj = []
        self.runtime = []
        self.history = []
        self._get_pop_sizes()
        self.available_gpus = None
        self.gpu_usage = None


    def clean_inactive_brackets(self):
        """ Removes brackets from the active list if it is done as communicated by Bracket Manager
        """
        if len(self.active_brackets) == 0:
            return
        self.active_brackets = [
            bracket for bracket in self.active_brackets if ~bracket.is_bracket_done()
        ]
        return

    def _update_trackers(self, traj, runtime, history):
        self.traj.append(traj)
        self.runtime.append(runtime)
        self.history.append(history)

    def _update_incumbents(self, config, score, info):
        self.inc_config = config
        self.inc_score = score
        self.inc_info = info

    def _get_pop_sizes(self):
        """Determines maximum pop size for each budget
        """
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self.get_next_iteration(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = max(
                    n[j], self._max_pop_size[r_j]
                ) if r_j in self._max_pop_size.keys() else n[j]

    def _start_new_bracket(self):
        """ Starts a new bracket based on Hyperband
        """
        # start new bracket
        self.iteration_counter += 1  # iteration counter gives the bracket count or bracket ID
        n_configs, budgets = self.get_next_iteration(self.iteration_counter)
        logger.debug(f'n_configs:{n_configs}, budgets:{budgets}')
        bracket = SHBracketManager(
            n_configs=n_configs, budgets=budgets, bracket_id=self.iteration_counter
        )
        self.active_brackets.append(bracket)
        return bracket

    def _get_worker_count(self):
        if isinstance(self.client, Client):
            return len(self.client.ncores())
        else:
            return 1

    def is_worker_available(self, verbose=False):
        """ Checks if at least one worker is available to run a job
        """
        if self.n_workers == 1 or self.client is None or not isinstance(self.client, Client):
            # in the synchronous case, one worker is always available
            return True
        # checks the absolute number of workers mapped to the client scheduler
        # client.ncores() should return a dict with the keys as unique addresses to these workers
        # treating the number of available workers in this manner
        workers = self._get_worker_count()  # len(self.client.ncores())
        if len(self.futures) >= workers:
            # pause/wait if active worker count greater allocated workers
            return False
        return True

    def _acquire_config(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # identify lower budget/fidelity to transfer information from

        assert budget in bracket.budgets

        if budget not in bracket.trials:

            # init population randomly for base rung
            if budget == bracket.budgets[0]:
                pop_size = bracket.n_configs[0]
                population = self.cs.sample_configuration(size=pop_size)
                trials = [Trial(individual) for individual in population]

            #Promote candidates from lower budget for next rung
            elif budget not in bracket.trials:
                lower_budget, n_configs = bracket.get_lower_budget_promotions(budget)
                candidate_trials = bracket.trials[lower_budget]
                fitness = [trial.get_fitness() for trial in candidate_trials]
                logger.debug(f'trials fitness:{fitness}')
                sorted_index = np.argsort(fitness)
                logger.debug(f'sorted index:{sorted_index}')
                trials = np.array(candidate_trials)[sorted_index][:n_configs]
                logger.debug(f'trial promoted from budget:{lower_budget} to budget:{budget}:{trials}')

            bracket.trials[budget] = trials

        logger.debug(f'budget:{budget}')
        pending_trials = bracket.get_pending_trials(budget)
        logger.debug(f'pending trials:{pending_trials}')
        return pending_trials[0]


    def _get_next_job(self):
        """ Loads a configuration and budget to be evaluated next by a free worker
        """
        bracket = None
        if len(self.active_brackets) == 0 or \
                np.all([bracket.is_bracket_done() for bracket in self.active_brackets]):
            # start new bracket when no pending jobs from existing brackets or empty bracket list
            bracket = self._start_new_bracket()
        else:
            for _bracket in self.active_brackets:
                # check if _bracket is not waiting for previous rung results of same bracket
                # _bracket is not waiting on the last rung results
                # these 2 checks allow HB to have a "synchronous" Successive Halving
                if not _bracket.previous_rung_waits() and _bracket.is_pending():
                    # bracket eligible for job scheduling
                    bracket = _bracket
                    break
            if bracket is None:
                # start new bracket when existing list has all waiting brackets
                bracket = self._start_new_bracket()
        # budget that the SH bracket allots
        budget = bracket.get_next_job_budget()
        trial = self._acquire_config(bracket, budget)
        # notifies the Bracket Manager that a single config is to run for the budget chosen
        job_info = {
            "config": trial.config,
            "budget": budget,
            "trial": trial,
            "bracket_id": bracket.bracket_id
        }
        return job_info

    def _get_gpu_id_with_low_load(self):
        candidates = []
        for k, v in self.gpu_usage.items():
            if v == min(self.gpu_usage.values()):
                candidates.append(k)
        device_id = np.random.choice(candidates)
        # creating string for setting environment variable CUDA_VISIBLE_DEVICES
        gpu_ids = self._create_cuda_visible_devices(
            self.available_gpus, device_id
        )
        # updating GPU usage
        self.gpu_usage[device_id] += 1
        self.logger.debug("GPU device selected: {}".format(device_id))
        self.logger.debug("GPU device usage: {}".format(self.gpu_usage))
        return gpu_ids

    def submit_job(self, job_info, **kwargs):
        """ Asks a free worker to run the objective function on config and budget
        """
        job_info["kwargs"] = self.shared_data if self.shared_data is not None else kwargs
        # submit to to Dask client
        if self.n_workers > 1 or isinstance(self.client, Client):
            if self.single_node_with_gpus:
                # managing GPU allocation for the job to be submitted
                job_info.update({"gpu_devices": self._get_gpu_id_with_low_load()})
            self.futures.append(
                self.client.submit(self._f_objective, job_info)
            )
        else:
            # skipping scheduling to Dask worker to avoid added overheads in the synchronous case
            self.futures.append(self._f_objective(job_info))

        # pass information of job submission to Bracket Manager
        for bracket in self.active_brackets:
            if bracket.bracket_id == job_info['bracket_id']:
                # registering is IMPORTANT for Bracket Manager to perform SH
                bracket.register_job(job_info['budget'], job_info['trial'])
                break

    def _fetch_results_from_workers(self):
        """ Iterate over futures and collect results from finished workers
        """
        if self.n_workers > 1 or isinstance(self.client, Client):
            done_list = [(i, future) for i, future in enumerate(self.futures) if future.done()]
        else:
            # Dask not invoked in the synchronous case
            done_list = [(i, future) for i, future in enumerate(self.futures)]
        if len(done_list) > 0:
            self.logger.debug(
                "Collecting {} of the {} job(s) active.".format(len(done_list), len(self.futures))
            )
        for _, future in done_list:
            if self.n_workers > 1 or isinstance(self.client, Client):
                run_info = future.result()
                if "device_id" in run_info:
                    # updating GPU usage
                    self.gpu_usage[run_info["device_id"]] -= 1
                    self.logger.debug("GPU device released: {}".format(run_info["device_id"]))
                future.release()
            else:
                # Dask not invoked in the synchronous case
                run_info = future
            # update bracket information
            fitness, cost = run_info["fitness"], run_info["cost"]
            info = run_info["info"] if "info" in run_info else dict()
            budget = run_info["budget"]
            config = run_info["config"]
            bracket_id = run_info["bracket_id"]
            logger.debug(f'run info:{run_info}')
            for bracket in self.active_brackets:
                if bracket.bracket_id == bracket_id:
                    # bracket job complete
                    bracket.complete_job(budget, run_info["trial"], fitness)  # IMPORTANT to perform synchronous SH


            # updating incumbents
            if fitness < self.inc_score:
                self._update_incumbents(
                    config=config,
                    score=fitness,
                    info=info
                )
            # book-keeping
            self._update_trackers(
                traj=self.inc_score, runtime=cost, history=(
                    config.get_dictionary, float(fitness), float(cost), float(budget), info
                )
            )
        # remove processed future
        self.futures = np.delete(self.futures, [i for i, _ in done_list]).tolist()

    def _is_run_budget_exhausted(self, fevals=None, brackets=None, total_cost=None):
        """ Checks if the DEHB run should be terminated or continued
        """
        delimiters = [fevals, brackets, total_cost]
        delim_sum = sum(x is not None for x in delimiters)
        if delim_sum == 0:
            raise ValueError(
                "Need one of 'fevals', 'brackets' or 'total_cost' as budget for DEHB to run."
            )
        if fevals is not None:
            if len(self.traj) >= fevals:
                return True
        elif brackets is not None:
            if self.iteration_counter >= brackets:
                for bracket in self.active_brackets:
                    # waits for all brackets < iteration_counter to finish by collecting results
                    if bracket.bracket_id < self.iteration_counter and \
                            not bracket.is_bracket_done():
                        return False
                return True
        else:
            if time.time() - self.start >= total_cost:
                return True
            if len(self.runtime) > 0 and self.runtime[-1] - self.start >= total_cost:
                return True
        return False

    def _save_incumbent(self, name=None):
        if name is None:
            name = time.strftime("%x %X %Z", time.localtime(self.start))
            name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
        try:
            res = dict()
            config = self.inc_config
            res["config"] = config.get_dictionary()
            res["score"] = self.inc_score
            res["info"] = self.inc_info
            with open(os.path.join(self.output_path, "incumbent_{}.json".format(name)), 'w') as f:
                json.dump(res, f)
        except Exception as e:
            self.logger.warning("Incumbent not saved: {}".format(repr(e)))

    def _save_history(self, name=None):
        if name is None:
            name = time.strftime("%x %X %Z", time.localtime(self.start))
            name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
        try:
            with open(os.path.join(self.output_path, "history_{}.pkl".format(name)), 'wb') as f:
                pickle.dump(self.history, f)
        except Exception as e:
            self.logger.warning("History not saved: {}".format(repr(e)))

    def _verbosity_debug(self):
        for bracket in self.active_brackets:
            self.logger.debug("Bracket ID {}:\n{}".format(
                bracket.bracket_id,
                str(bracket)
            ))

    def _verbosity_runtime(self, fevals, brackets, total_cost):
        if fevals is not None:
            remaining = (len(self.traj), fevals, "function evaluation(s) done")
        elif brackets is not None:
            _suffix = "bracket(s) started; # active brackets: {}".format(len(self.active_brackets))
            remaining = (self.iteration_counter + 1, brackets, _suffix)
        else:
            elapsed = np.format_float_positional(time.time() - self.start, precision=2)
            remaining = (elapsed, total_cost, "seconds elapsed")
        self.logger.info(
            "{}/{} {}".format(remaining[0], remaining[1], remaining[2])
        )

    @logger.catch
    def run(self, fevals=None, brackets=None, total_cost=None, single_node_with_gpus=False,
            verbose=False, debug=False, save_intermediate=True, save_history=True, **kwargs):
        """ Main interface to run optimization by DEHB

        This function waits on workers and if a worker is free, asks for a configuration and a
        budget to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low):
        1) Number of function evaluations (fevals)
        2) Number of Successive Halving brackets run under Hyperband (brackets)
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_cost)
        """
        # checks if a Dask client exists
        if len(kwargs) > 0 and self.n_workers > 1 and isinstance(self.client, Client):
            # broadcasts all additional data passed as **kwargs to all client workers
            # this reduces overload in the client-worker communication by not having to
            # serialize the redundant data used by all workers for every job
            self.shared_data = self.client.scatter(kwargs, broadcast=True)

        # allows each worker to be mapped to a different GPU when running on a single node
        # where all available GPUs are accessible
        self.single_node_with_gpus = single_node_with_gpus
        if self.single_node_with_gpus:
            self.distribute_gpus()

        self.start = time.time()
        if verbose:
            print("\nLogging at {} for optimization starting at {}\n".format(
                os.path.join(os.getcwd(), self.log_filename),
                time.strftime("%x %X %Z", time.localtime(self.start))
            ))
        if debug:
            logger.configure(handlers=[{"sink": sys.stdout}])
        while True:
            if self._is_run_budget_exhausted(fevals, brackets, total_cost):
                break
            if self.is_worker_available():
                job_info = self._get_next_job()
                if brackets is not None and job_info['bracket_id'] >= brackets:
                    # ignore submission and only collect results
                    # when brackets are chosen as run budget, an extra bracket is created
                    # since iteration_counter is incremented in _get_next_job() and then checked
                    # in _is_run_budget_exhausted(), therefore, need to skip suggestions
                    # coming from the extra allocated bracket
                    # _is_run_budget_exhausted() will not return True until all the lower brackets
                    # have finished computation and returned its results
                    pass
                else:
                    if self.n_workers > 1 or isinstance(self.client, Client):
                        self.logger.debug("{}/{} worker(s) available.".format(
                            self._get_worker_count() - len(self.futures), self._get_worker_count()
                        ))
                    # submits job_info to a worker for execution
                    self.submit_job(job_info, **kwargs)
                    if verbose:
                        budget = job_info['budget']
                        self._verbosity_runtime(fevals, brackets, total_cost)
                        self.logger.info(
                            "Evaluating a configuration with budget {} under "
                            "bracket ID {}".format(budget, job_info['bracket_id'])
                        )
                        self.logger.info(
                            "Best score seen/Incumbent score: {}".format(self.inc_score)
                        )
                    self._verbosity_debug()
            self._fetch_results_from_workers()
            if save_intermediate and self.inc_config is not None:
                self._save_incumbent()
            if save_history and self.history is not None:
                self._save_history()
            self.clean_inactive_brackets()
        # end of while

        if verbose and len(self.futures) > 0:
            self.logger.info(
                "MOHB optimisation over! Waiting to collect results from workers running..."
            )
        while len(self.futures) > 0:
            self._fetch_results_from_workers()
            if save_intermediate and self.inc_config is not None:
                self._save_incumbent()
            if save_history and self.history is not None:
                self._save_history()
            time.sleep(0.05)  # waiting 50ms

        if verbose:
            time_taken = time.time() - self.start
            self.logger.info("End of optimisation! Total duration: {}; Total fevals: {}\n".format(
                time_taken, len(self.traj)
            ))
            self.logger.info("Incumbent score: {}".format(self.inc_score))
            self.logger.info("Incumbent config: ")
            if self.configspace:
                config = self.vector_to_configspace(self.inc_config)
                for k, v in config.get_dictionary().items():
                    self.logger.info("{}: {}".format(k, v))
            else:
                self.logger.info("{}".format(self.inc_config))
        self._save_incumbent()
        self._save_history()
        return np.array(self.traj), np.array(self.runtime), np.array(self.history, dtype=object)
