import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm
from datetime import datetime
import random

from ema_workbench import (MultiprocessingEvaluator, ema_logging, RealParameter, ScalarOutcome, Constant,
                           Model, HypervolumeMetric, save_results)
from ema_workbench.em_framework.optimization import (GenerationalBorg, epsilon_nondominated, to_problem, ArchiveLogger,
                                                     EpsilonProgress, Hypervolume)
#from ema_workbench import (GenerationalDistanceMetric, EpsilonIndicatorMetric, InvertedGenerationalDistanceMetric,
                            #SpacingMetric,)

# Because everything outside the main statement runs 8 times due to Multiprocessing evaluator, we first check cwd
if not os.path.exists("../src"):
    os.chdir("../../src")
else:
    os.chdir('../src')

sys.path.append("..")
sys.path.append(".")

cwd_initial = os.getcwd()
print("cwd line 26 is: ", cwd_initial)

from model_zambezi_OPT_hyd import ModelZambezi

ZambeziProblem = ModelZambezi()

def model_wrapper(**kwargs):
    input = [kwargs['v' + str(i)] for i in range(len(kwargs))]
    Hydropower, Environment, Irrigation, HydropowerITT, HydropowerKGU, HydropowerKA, HydropowerCB, HydropowerKGL = tuple(ZambeziProblem.evaluate(np.array(input)))
    return Hydropower, Environment, Irrigation, HydropowerITT, HydropowerKGU, HydropowerKA, HydropowerCB, HydropowerKGL

# specify model
model = Model('zambeziproblem', function=model_wrapper)

# levers
model.levers = [RealParameter('v' + str(i), -1, 1) for i in range(ZambeziProblem.Nvar)]

# specify outcomes
model.outcomes = [ScalarOutcome('Hydropower', ScalarOutcome.MINIMIZE),  # Minimize, because deficits
                  ScalarOutcome('Environment', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('Irrigation', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('HydropowerITT', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('HydropowerKGU', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('HydropowerKA', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('HydropowerCB', ScalarOutcome.MINIMIZE),
                  ScalarOutcome('HydropowerKGL', ScalarOutcome.MINIMIZE),
                  ]

if __name__ == '__main__':
    print('within main statement')
    project_dir = os.getcwd()

    ######################################################################################
    ################################# RUN SETTINGS #######################################
    ######################################################################################

    # Specify the nfe and add a comment for the run save name
    nfe = 200000 #150000 1 seed; 200000 5 seeds in HPC
    number_of_seeds = 5
    epsilon_list = [0.9, 1, 0.9, 1.2, 1.2, 1.2, 1.2, 1.2]
    seeds_list = [17, 42, 63, 188, 1234]
    run_comment = 'pseudo'  # add a comment to recognize the run output

    ######################################################################################

    run_label = f"HYD_{run_comment}_{nfe}nfe_{number_of_seeds}seed" # BC = BaseCase (3 objectives)
    dir_runs = f"{cwd_initial}/../runs"

    # Check if the directory already exists and create it if it doesn't
    output_dir = f"{dir_runs}/{run_label}"  # the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' has been created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    print("working directory main statement is", os.getcwd())
    if os.getcwd() != output_dir:
        os.chdir(output_dir)

    # move directory creators to utils?

    ema_logging.LOG_FORMAT = "[%(name)s/%(levelname)s/%(processName)s] %(message)s"
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Create a list for storing the results and convergences of the different seeds
    results_list = []
    convergences = []

    # Construct the archives directory path for this run
    archives_dir_path = f"{output_dir}/archives"
    if not os.path.exists(archives_dir_path): # Check if the directory already exists and create it if it doesn't
        os.makedirs(archives_dir_path)
        print(f"Archives directory '{archives_dir_path}' has been created.")
    else:
        print(f"Archives directory '{archives_dir_path}' already exists.")

        # Remove '/tmp' if it exists
        path = os.path.join(archives_dir_path, "tmp")
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
            print(f"tmp has been removed successfully")

    ############
    # Evaluator
    ############
    before = datetime.now()
    print("time before is", before)

    with MultiprocessingEvaluator(model) as evaluator:
        for i in tqdm(range(number_of_seeds)):  # for every seed
            s = seeds_list[i]
            np.random.seed(int(s))
            random.seed(int(s))
            print("working directory within evaluator is", os.getcwd())
            # we create 2 convergence tracker metrics
            # the archive logger writes the archive to disk for every x nfe
            # the epsilon progress tracks during runtime
            convergence_metrics = [
                ArchiveLogger(
                    "./archives",
                    [lever.name for lever in model.levers],
                    [outcome.name for outcome in model.outcomes],
                    base_filename=f"{i}.csv",
                ),
                EpsilonProgress(),
                #Hypervolume(minimum=[-1e9] * len(model.outcomes), maximum=[1e9] * len(model.outcomes)),
            ]

            results, convergence = evaluator.optimize(
                algorithm=GenerationalBorg,
                nfe=nfe,
                searchover="levers",
                epsilons=epsilon_list,
                convergence=convergence_metrics,
            )

            # Print some information
            print("results type", type(results))
            print("results", results)
            print("run name is", run_label)

            # Save the results of this seed
            results_file_name = f"results_seed{i}.csv"
            convergence_file_name = f"convergence{i}.csv"
            print('results_file_name is', results_file_name)

            cwd = os.getcwd()
            results.to_csv(os.path.join(cwd, results_file_name), index=False)
            convergence.to_csv(os.path.join(cwd, convergence_file_name), index=False)
            print("cwd at to_csv:", cwd)

            # Append the result (df) for each seed to 'results' (list)
            results_list.append(results)
            # Append the convergence per seed to 'convergences'
            convergences.append(convergence)

    after = datetime.now()
    print(f"Time after is {after}")
    print(f"It took {after - before} time to do {nfe} nfes")

    print("results_list", results_list)
    print("results_list type", type(results_list))

    #############
    # MERGE SEEDS
    #############

    if cwd != output_dir:
        print(f'Current directory is {cwd}')
        os.chdir(output_dir)
        print(f'Changed current directory to {output_dir}')

    # Merge the 5 runs of the optimization
    problem = to_problem(model, searchover="levers")
    epsilons = [1.1] * len(model.outcomes)
    merged_results = epsilon_nondominated(results_list, epsilons, problem)

    print('merged_results', merged_results, 'saved to: ', os.getcwd())

    # Save the results
    merged_results_name = 'merged_results.csv'
    merged_results.to_csv(os.path.join(cwd, merged_results_name), index=False)

