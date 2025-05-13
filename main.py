import os
from TestIdentificator import TestIdentificator  # Import TestIdentificator class
from Instance import Instance  # Import Instance class
from Constants import Constants
from Solver import Solver
from Evaluator import Evaluator
from EvaluatorIdentificator import EvaluatorIdentificator
import numpy as np
import argparse
import platform
import sys

def CreateRequiredDir():
    if Constants.Debug:
        print("\n We are in 'CreateRequiredDir' Function.")
    requireddir = [
        "./Test",
        "./GurobiLog",
        "./Test/Statistic",
        "./Test/Bounds",
        "./Test/SolveInfo",
        "./Solutions",
        "./Evaluations",
        "./Temp"
    ]
    for dir in requireddir:
        os.makedirs(dir, exist_ok=True)

# Function to generate random parameters
def generate_random_parameters(I, J, S, seed):
    """
    Generate random parameters for the problem.
    :param I: Number of facilities.
    :param J: Number of customer locations.
    :param S: Number of scenarios.
    :param seed: Random seed for reproducibility.
    :return: Dictionary containing generated parameters.
    """
    np.random.seed(seed)
    parameters = {
        "fixed_costs": np.random.randint(100, 500, size=I),
        "facility_capacity": np.random.randint(100, 400, size=I),
        "transport_costs": np.random.uniform(10, 10, size=(I, J)),
        "penalty_cost": 200,
        "demands": np.random.randint(50, 150, size=(J, S)),
    }
    return parameters

# Parse command-line arguments
def parseArguments():
    parser = argparse.ArgumentParser(description="Stochastic Location Allocation Model")

    if platform.system() == "Linux":
        parser.add_argument("--Action", help="Action to perform", type=str, choices=["GenerateInstances", "Solve"], required=True)
        parser.add_argument("--Instance", help="Instance name", type=str, required=True)
        parser.add_argument("--Model", help="Stochastic model type", type=str, choices=["Average", "2Stage"], required=True)
        parser.add_argument("--Solver", help="Solver type", type=str, choices=["MIP", "ALNS", "PHA", "BBC"], required=True)
        parser.add_argument("--NrScenario", help="The number of scenarios used for optimization (all10 ...)", type=str, required=True)
        parser.add_argument("--PHAObj", help="Obj. function of PHA either Quadratic or Linear", type=str, choices=["Q", "L"], required=True)
        parser.add_argument("--PHAPenalty", help="Penalty Parameter (rho) in PHA, Static, Dynamic, dynamic Learning", type=str, choices=["S", "D", "DL"], required=True)
        parser.add_argument("--ALNSRL", help="Whether we use RL in ALNS or not", type=int, choices=[0, 1], required=True)
        parser.add_argument("--ALNSRL_DeepQ", help="The type of RL we used in ALNS if (ALNSRL==1), Deep Q-Learning(1) or Q-Learning(0)", type=int, choices=[0, 1], required=True)        
        parser.add_argument("-c", "--bbcsetting", help="Enhancements?", choices=["NE", "JM", "NM", "JS", "NS", "JW", "NW", "JL", "NL", "AE"], required=True)
        parser.add_argument("--ScenarioGeneration", help="Which Type of Sampling?", type=str, choices=["MC","RQMC", "QMC"], required=True)
        parser.add_argument("-Cluster", "--ClusteringMethod", help="The method used for Clustering Scenarios?", type=str, choices=["NoC", "KM", "KMPP", "SOM", "DB"], required=True) 

    else:

        # Mandatory arguments
        parser.add_argument("--Action", help="Action to perform", type=str, choices=["GenerateInstances", "Solve"], default = "GenerateInstances")
        parser.add_argument("--Instance", help="Instance name", type=str, default="4_31_4_60_3_1_CRP") 
        parser.add_argument("--Model", help="Stochastic model type", type=str, choices=["Average", "2Stage"], default = "2Stage")
        parser.add_argument("--Solver", help="Solver type", type=str, choices=["MIP", "ALNS", "PHA", "BBC"], default = "ALNS")
        parser.add_argument("--NrScenario", help="The number of scenarios used for optimization (all10 ...)", type=str, default = "50")
        parser.add_argument("--PHAObj", help="Obj. function of PHA either Quadratic or Linear", type=str, choices=["Q", "L"], default = "Q")
        parser.add_argument("--PHAPenalty", help="Penalty Parameter (rho) in PHA, Static, Dynamic, dynamic Learning", type=str, choices=["S", "D", "DL"], default = "S")
        parser.add_argument("--ALNSRL", help="Whether we use RL in ALNS or not", type=int, choices=["0", "1"], default = 1)
        parser.add_argument("--ALNSRL_DeepQ", help="The type of RL we used in ALNS if (ALNSRL==1), Deep Q-Learning(1) or Q-Learning(0)", type=int, choices=["0", "1"], default = 0)
        parser.add_argument("-c", "--bbcsetting", help="Enhancements?", choices=["NE: NoEnhancement", "JM: JustMultiCut", "NM: NoMultiCut", "JS: JustStrongCut", "NS: NoStrongCut", "JW: JustWarmUp", "NW: NoWarmUp", "JL: JustLBF", "NL: NoLBF", "AE: AllEnhancement"], default="AE")
        parser.add_argument("--ScenarioGeneration", help="Which Type of Sampling?", type=str, choices=["MC","RQMC", "QMC"], default="RQMC")
        parser.add_argument("-Cluster", "--ClusteringMethod", help="The method used for Clustering Scenarios? DB: Decisional-Based", type=str, choices=["NoC", "KM", "KMPP", "SOM", "DB"], default = "DB") 



    # Optional arguments
    parser.add_argument("-p", "--policy", help="NearestNeighbor", type=str, default="_")    
    parser.add_argument("-n", "--nrevaluation", help="nr scenario used for evaluation.", type=int, default = 500)   
    parser.add_argument("-t", "--timehorizon", help="the time horizon used in shiting window.", type=int, default = 1)
    parser.add_argument("-a", "--allscenario", help="generate all possible scenario.", type=int, default = 0)
    parser.add_argument("-s", "--ScenarioSeed", help="The seed used for scenario generation", type=int, default=-1)
    parser.add_argument("--RLSelectionMethod", help="Which type of selection Method?", type=str, choices=["e-greedy", "softmax"], default="e-greedy")


    # Parse arguments
    args = parser.parse_args()

    global TestIdentifier
    global EvaluatorIdentifier
    global Action
    policygeneration = args.policy
    Action = args.Action

    TestIdentifier = TestIdentificator(instance_name = args.Instance,
                                       model = args.Model,
                                       solver = args.Solver,
                                       nrScenario = args.NrScenario,
                                       seed = Constants.SeedArray[args.ScenarioSeed],
                                       sampling = args.ScenarioGeneration,                                  
                                       phaobj = args.PHAObj,                                  
                                       phapenalty = args.PHAPenalty,                                  
                                       alnsRL = args.ALNSRL,                                
                                       alnsRL_DeepQ = args.ALNSRL_DeepQ,                               
                                       rlSelectionMethod = args.RLSelectionMethod,
                                       bbcsetting = args.bbcsetting,                             
                                       clustering = args.ClusteringMethod)
    
    EvaluatorIdentifier = EvaluatorIdentificator(policygeneration,  args.nrevaluation, args.timehorizon, args.allscenario)

    # Check for incompatible options
    if args.Model == "Average" and not (args.Solver == "MIP"):
        print("Error: 'Average' model cannot be used with The type of Solver you selected (It is only compatible with MIP) to findout the value of considering uncertainty in the model!")
        sys.exit(1)  # Exit the program with an error status
    # new check: forbid Average with DB clustering
    if args.Model == "Average" and args.ClusteringMethod == "DB":
        print("Error: 'Average' model cannot be used with DB (Decisional‑Based) clustering.")
        sys.exit(1)
        
# Generate instances
def generate_instances():
    print("Generating instances...")

    for t in range(4, 5, 1):            ## Set it No more than 20 time periods!
        for i in range(31, 32, 5):
            for h in range(4, 5, 5):
                for l in range(94, 95, 30):
                    for m in range(3, 4, 1):
                        for instance_number in range(1, 6, 1):
                            instance_name = f"{t}_{i}_{h}_{l}_{m}_{instance_number}_CRP"

                            instance = Instance(instance_name)
                            instance.NrTimeBucket = t
                            instance.NrACFs = i
                            instance.NrHospitals = h
                            instance.NrMedFacilities = (i + h)
                            instance.NrDisasterAreas = l
                            instance.NrRescueVehicles = m       ## m=0:Advanced Ambulances, m=1: Basic Ambulances; m=2: Ambuses
                            instance.NrInjuries = 2         ## Do not CHANGE IT!
                            
                            instance.ComputeIndices()

                            instance.build_J_u()            ## Creates a [J*U] binary matrix accordingly
                            instance.build_J_m()            ## Creates a [J*M] binary matrix accordingly
                            # Assign a percentage of ACFs as aerial-equipped
                            instance.assign_aerial_acfs(percentage=0.5, seed=Constants.SeedArray[0])
                            instance.assign_backup_hospitals(BackupPercentage=0.8, seed=Constants.SeedArray[0])

                            RandomSeed_InstanceGeneration = Constants.SeedArray[0] + instance_number
                            
                            if Constants.Case_Study_Data_Generation == 0:
                                instance.Generate_Data(RandomSeed_InstanceGeneration)
                            else:
                                print("-------------- Generating Data for Case Study -------------")
                                instance.Generate_Data_CaseStudy(RandomSeed_InstanceGeneration)


                            instance.SaveInstanceToPickle()
                            if Constants.Debug: instance.Print_Attributes()
                            print(f"Instance {instance_name} generated and saved.")

def Solve(instance):
    if Constants.Debug: print("We are in Solve Method - main.py Class")
    global LastFoundSolution
    
    solver = Solver(instance, TestIdentifier)
    solution = solver.Solve()
    
    LastFoundSolution = solution
    Constants.Evaluation_Part = True
    #Constants.ClusteringMethod = 'NoC'
    if Constants.Evaluation_Part:

        evaluator = Evaluator(instance, TestIdentifier, EvaluatorIdentifier, solver)

        evaluator.RunEvaluation()

        if Constants.LauchEvalAfterSolve and EvaluatorIdentifier.NrEvaluation>0:
            evaluator.GatherEvaluation()

# Main execution
if __name__ == "__main__":

    try:
        CreateRequiredDir()
        parseArguments()

        ## For PHA Settings
        if TestIdentifier.PHAObj == "L":
            Constants.Quadratic_to_Linear_PHA = True
        if TestIdentifier.PHAPenalty == "D":
            Constants.Dynamic_rho_PenaltyParameter = True
        if TestIdentifier.PHAPenalty == "DL":
            Constants.Dynamic_Learning_rho_PenaltyParameter = True

        ################ For BBC Settings and Accelerators
        if TestIdentifier.BBCSetting == "JS": #Just_StrongCut
            Constants.GenerateStrongCut = True
            Constants.UseLBF = False
            Constants.UseWarmUp = False
            Constants.UseMultiCut = False
        if TestIdentifier.BBCSetting == "JL":  #Just_LBF
            Constants.GenerateStrongCut = False
            Constants.UseLBF = True
            Constants.UseWarmUp = False
            Constants.UseMultiCut = False
        if TestIdentifier.BBCSetting == "JW":   #Just_WarmUp
            Constants.GenerateStrongCut = False
            Constants.UseLBF = False
            Constants.UseWarmUp = True
            Constants.UseMultiCut = False
        if TestIdentifier.BBCSetting == "JM": #Just_Multi-Cut
            Constants.GenerateStrongCut = False
            Constants.UseLBF = False
            Constants.UseWarmUp = False
            Constants.UseMultiCut = True
        if TestIdentifier.BBCSetting == "NE": #No_Enhancement
            Constants.GenerateStrongCut = False
            Constants.UseLBF = False
            Constants.UseWarmUp = False
            Constants.UseMultiCut = False
        if TestIdentifier.BBCSetting == "NS":   #No_StrongCut
            Constants.GenerateStrongCut = False
            Constants.UseLBF = True
            Constants.UseWarmUp = True
            Constants.UseMultiCut = True
        if TestIdentifier.BBCSetting == "NL":   #No_LBF
            Constants.GenerateStrongCut = True
            Constants.UseLBF = False
            Constants.UseWarmUp = True
            Constants.UseMultiCut = True            
        if TestIdentifier.BBCSetting == "NW": #No_WarmUp
            Constants.GenerateStrongCut = True
            Constants.UseLBF = True
            Constants.UseWarmUp = False
            Constants.UseMultiCut = True
        if TestIdentifier.BBCSetting == "NM": #No_MultiCut
            Constants.GenerateStrongCut = True
            Constants.UseLBF = True
            Constants.UseWarmUp = True
            Constants.UseMultiCut = False
        if TestIdentifier.BBCSetting == "AE":   #All_Enhancement
            Constants.GenerateStrongCut = True
            Constants.UseLBF = True
            Constants.UseWarmUp = True
            Constants.UseMultiCut = True 
        ######################### Clustering      
        Constants.ClusteringMethod = TestIdentifier.Clustering                      

    except KeyError:
        print(KeyError.message)
        print("This instance does not exist.")

    if Action == "GenerateInstances":
        generate_instances()
    elif Action == "Solve":
        # Load instance or create one if not found
        try:
            instance = Instance(TestIdentifier.InstanceName)
            instance.LoadInstanceFromPickle(TestIdentifier.InstanceName)
        except FileNotFoundError:
            print(f"Instance {TestIdentifier.InstanceName} not found. Generating a new instance...")
            instance = Instance(TestIdentifier.InstanceName)
            instance.Generate_Data(TestIdentifier.Seed)
            instance.SaveInstanceToPickle()

        Solve(instance)

        print("****************************** WE ARE DONE *************************************") 
        
    elif Action == "DebugLPFile":
        print("Debugging LP file...")

    print("****************************** WE ARE DONE *************************************")  