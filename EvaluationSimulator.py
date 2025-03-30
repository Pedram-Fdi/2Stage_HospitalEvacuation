#This class provide a framework to evaluate the performance of the method through a simulation
#over a large number of scenarios.
from __future__ import absolute_import, division, print_function
import pandas as pd
from MIPSolver import MIPSolver
from ScenarioTree import ScenarioTree
from Constants import Constants
import time
import math
from datetime import datetime
import csv
from scipy import stats
import numpy as np
import copy
import itertools
import pickle

class EvaluationSimulator(object):

    #Constructor
    def __init__(self, 
                 instance, 
                 solutions=[], 
                 testidentificator = [], 
                 evaluatoridentificator =[], 
                 treestructure=[], 
                 model="XFix"):
        if Constants.Debug: print("\n We are in 'EvaluationSimulator' Class -- Constructor")
        self.Instance = instance
        self.Solutions = solutions

        self.TestIdentificator = testidentificator
        self.EvalatorIdentificator = evaluatoridentificator

        self.NrSolutions = len(self.Solutions)

        self.Policy = evaluatoridentificator.PolicyGeneration
        self.StartSeedResolve = Constants.SeedArray[0]

        self.ScenarioGenerationResolvePolicy = self.TestIdentificator.ScenarioSampling

        self.MIPResolveTime = [None for t in instance.TimeBucketSet]
        self.IsDefineMIPResolveTime = [False for t in instance.TimeBucketSet]

        self.ReferenceTreeStructure = treestructure
        self.EvaluateAverage = Constants.IsDeterministic(self.TestIdentificator.Model)
        self.Model = model
        if Constants.Debug: print(f"ReferenceTreeStructure: {treestructure}, EvaluateAverage: {self.EvaluateAverage}, Model: {model}")

    #This function evaluate the performance of a set of solutions obtain with the same method (different solutions due to randomness in the method)
    def EvaluateXFixSolution(self, saveevaluatetab=False, filename=""):
        if Constants.Debug: print("\n We are in 'EvaluationSimulator' Class -- EvaluateYQFixSolution")

        # Compute the average value of the demand
        nrscenario = self.EvalatorIdentificator.NrEvaluation
        allscenario = self.EvalatorIdentificator.AllScenario
        start_time = time.time()
        
        if Constants.Debug: print(f"Number of scenarios: {nrscenario}, All scenarios: {allscenario}")

        Evaluated = [-1 for e in range(nrscenario)]
        Probabilities = [-1 for e in range(nrscenario)]

        OutOfSampleSolution = None
        mipsolver = None
        firstsolution = True
        nrerror = 0

        for n in range(self.NrSolutions):
                if Constants.Debug: print(f"Evaluating solution {n+1} / {self.NrSolutions}")

                sol = None

                sol = self.Solutions[n]
                if Constants.Debug: print(f"Using direct solution for scenario {n+1}")

                evaluatoinscenarios, scenariotrees = self.GetScenarioSet(Constants.EvaluationScenarioSeed, nrscenario, allscenario)
                                
                firstscenario = True
                self.IsDefineMIPResolveTime = [False for t in self.Instance.TimeBucketSet]

                average = 0
                totalproba = 0
                for indexscenario in range(nrscenario):

                    scenario = evaluatoinscenarios[indexscenario]
                    scenariotree = scenariotrees[indexscenario]

                    givenACFestablishments, givenlandRescueVehicles, givenBackupHospitals = self.GetDecisionFromSolutionForScenario(sol, scenario)
   
                    if Constants.Debug: print(f"ACF Establishments:\n", givenACFestablishments)
                    if Constants.Debug: print(f"Land Rescue Vehicles:\n", givenlandRescueVehicles)
                    if Constants.Debug: print(f"Backup Hospitals:\n", givenBackupHospitals)
                    if Constants.Debug: print(f"CasualtyDemand:\n", scenario.CasualtyDemand)
                    if Constants.Debug: print(f"HospitalDisruption:\n", scenario.HospitalDisruption)
                    if Constants.Debug: print(f"PatientDemand:\n", scenario.PatientDemand)
                    if Constants.Debug: print(f"PatientDischargedPercentage:\n", scenario.PatientDischargedPercentage)

                    #Defin the MIP
                    mipsolver = MIPSolver(  instance = self.Instance, 
                                            model = Constants.Two_Stage, 
                                            scenariotree = scenariotree,
                                            nrscenario = scenario.TreeStructure[1],
                                            evaluatesolution=True,
                                            givenACFEstablishment=givenACFestablishments,
                                            givenNrLandRescueVehicle=givenlandRescueVehicles,
                                            givenBackupHospital=givenBackupHospitals,
                                            )                       
                    mipsolver.BuildModel()

                    mipsolver.LocAloc.Params.Method = 2  # Use barrier method for linear programming
                    mipsolver.LocAloc.Params.FeasibilityTol = 0.01  # Set feasibility tolerance

                    solution = mipsolver.Solve()  # Assuming Solve() method is appropriately defined for Gurobi

                    #GRB should always find a solution due to complete recourse
                    if solution is None:
                        if Constants.Debug:
                            mipsolver.LocAloc.write("LocAloc_Evaluation.lp")  # Save the model to an LP file
                            # Raising an error with a custom message
                            raise NameError(f"Error at seed {indexscenario} with given quantity {givenvartrans}")
                            nrerror += 1
                    else:
                        if Constants.Debug: print(f"Solution found: Total Cost = {solution.TotalCost}")
                        Evaluated[indexscenario] = solution.TotalCost

                        if allscenario == 0:
                            scenario.Probability = 1.0 / float(nrscenario)
                            if Constants.Debug: print(f"Set scenario probability: {scenario.Probability}")
                        Probabilities[indexscenario] = scenario.Probability
                        average += solution.TotalCost * scenario.Probability
                        if Constants.Debug: print(f"Updated average: {average}")
                        totalproba += scenario.Probability
                        if Constants.Debug: print(f"Updated total probability: {totalproba}")
                        
                        #Record the obtain solution in an MRPsolution  OutOfSampleSolution
                        if firstsolution:
                            if firstscenario:
                                if Constants.Debug: print("Recording first solution.")
                                OutOfSampleSolution = solution
                            else:
                                if Constants.Debug: print("Merging solution.")
                                OutOfSampleSolution.Merge(solution)

                        firstscenario = False

                    # if firstsolution:
                    #     print("OutOfSampleSolution.Scenarioset: ", OutOfSampleSolution.Scenarioset)
                    #     print("OutOfSampleSolution.ScenarioTree: ", OutOfSampleSolution.ScenarioTree)
                    #     if Constants.Debug: print(f"Adjusting probabilities for {(OutOfSampleSolution.ScenarioTree)} scenarios.")
                    #     for s in range(OutOfSampleSolution.scenario):
                    #         s.Probability = 1.0 / (OutOfSampleSolution.scenario.Probability)
                    #         if Constants.Debug: print(f"Scenario {s} probability adjusted to {s.Probability}")


        OutOfSampleSolution.ComputeStatistics()

        duration = time.time() - start_time

        if Constants.Debug: print("Duration of evaluation (sec): %r, \nOutofSample cost:%r \ntotal proba:%r" % (duration, average, totalproba)) # %r"%( duration, Evaluated )
        
        self.EvaluationDuration = duration

        KPIStat = OutOfSampleSolution.PrintStatistics(self.TestIdentificator, "OutOfSample", indexscenario, nrscenario, duration, False, self.Policy )

        #Save the evaluation result in a file (This is used when the evaluation is parallelized)
        if saveevaluatetab:
                with open(filename+"_Evaluator.txt", "wb") as fp:
                    pickle.dump(Evaluated, fp)

                with open(filename + "_Probabilities.txt", "wb") as fp:
                    pickle.dump(Probabilities, fp)

                with open(filename+"_KPIStat.txt", "wb") as fp:
                    pickle.dump(KPIStat, fp)

        if Constants.PrintDetailsExcelFiles:
            namea = self.TestIdentificator.GetAsString()
            nameb = self.EvalatorIdentificator.GetAsString()
            OutOfSampleSolution.PrintToExcel(namea + nameb)

    def ComputeStatistic(self, Evaluated, Probabilities, KPIStat, nrerror):
        if Constants.Debug: print("\n We are in 'EvaluationSimulator' Class -- ComputeStatistic")

        # Flatten the Evaluated list to a single list of values
        all_evaluated_values = [val for sublist in Evaluated for val in sublist]

        # Number of scenarios (KK) and evaluations per scenario (MM)
        KK = len(Evaluated)
        MM = self.EvalatorIdentificator.NrEvaluation

        # Calculate mean
        mean = np.mean(all_evaluated_values)
        
        # Calculate variance
        variance_New = np.var(all_evaluated_values, ddof=0)  # Population variance (ddof=0)
        
        # Calculate standard deviation (STD)
        std_New = np.sqrt(variance_New)
        
        # Calculate 95% confidence interval
        z = 1.96  # Critical value for 95% confidence
        margin_of_error = z * (std_New / math.sqrt(KK * MM))
        LB = max(0, mean - margin_of_error)
        UB = mean + margin_of_error
        
        d = datetime.now()
        date = d.strftime('%m_%d_%Y_%H_%M_%S')

        EvaluateInfo = self.ComputeInformation(Evaluated, self.EvalatorIdentificator.NrEvaluation)

        MinAverage = 0
        MaxAverage = 0
              
        if Constants.PrintDetailsExcelFiles:
            if Constants.Debug: print("self.TestIdentificator.GetAsStringList():\n ", self.TestIdentificator.GetAsStringList())
            if Constants.Debug: print("self.EvalatorIdentificator.GetAsStringList():\n ", self.EvalatorIdentificator.GetAsStringList())

            general = self.TestIdentificator.GetAsStringList() + self.EvalatorIdentificator.GetAsStringList() + [mean, variance_New, LB, UB, MinAverage, MaxAverage, nrerror]

            columnstab = ["Instance", "Model", "Method", "ScenarioGeneration", "NrScenario", "ScenarioSeed",  "EVPI", "NrForwardScenario", 
                          "mipsetting", "SDDPSetting", "HybridPHSetting", "MLLocalSearchSetting",
                           "Policy Generation", "NrEvaluation", "Time Horizon", "All Scenario",  
                           "Mean", "Variance", "LB", "UB", "Min Average", "Max Average", "nrerror"]
            #myfile = open(r'./Test/Bounds/TestResultOfEvaluated_%s_%r.csv' % (self.TestIdentificator.GetAsStringList(), self.EvalatorIdentificator.GetAsStringList()), 'w', newline='', encoding='utf-8')  # 'newline' to avoid extra newlines in Windows
            
            test_identificator_str = '_'.join(self.TestIdentificator.GetAsStringList())
            evaluator_identificator_str = '_'.join(self.EvalatorIdentificator.GetAsStringList())
            # Combine the strings with an underscore and format them into the file path
            # Ensure to replace empty strings or undesired characters if necessary
            filename = f"./Test/Bounds/TestResultOfEvaluated_{test_identificator_str}_{evaluator_identificator_str}.csv"

            myfile = open(filename, 'w', newline='', encoding='utf-8')
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(columnstab)  # Write the column headers
            wr.writerow(general)  # Write the data
            myfile.close()

        #KPIStat = KPIStat[6:] #The first values in KPIStats are not interesting for out of sample evalution (see MRPSolution::PrintStatistics)
        EvaluateInfo = [mean, LB, UB, MinAverage, MaxAverage, nrerror] + KPIStat

        if Constants.Debug: print("EvaluateInfo: ", EvaluateInfo)

        return EvaluateInfo
    #This function return the FixedTrans decision and VarTrans decisions for the scenario given in argument
    def GetDecisionFromSolutionForScenario(self, sol,  scenario):
        if Constants.Debug: print("\n We are in 'EvaluationSimulator' Class -- GetDecisionFromSolutionForScenario")

        ################# ACF Establishment
        givenACFestablishments = [0 for i in self.Instance.ACFSet]
        givenACFestablishments = [(sol.ACFEstablishment_x_wi[0][i] ) 
                                    for i in self.Instance.ACFSet]
        if Constants.Debug: print("givenACFestablishments: ", givenACFestablishments)   
        
        ################# Land Rescue Vehicles
        givenlandRescueVehicles = [[0 for m in self.Instance.RescueVehicleSet] 
                                    for i in self.Instance.ACFSet]
        givenlandRescueVehicles = [[(sol.LandRescueVehicle_thetaVar_wim[0][i][m] ) 
                                    for m in self.Instance.RescueVehicleSet]
                                    for i in self.Instance.ACFSet]
        if Constants.Debug: print("givenlandRescueVehicles: ", givenlandRescueVehicles)
    
        ################# Backup Hospital Vehicles
        givenBackupHospitals = [[0 for hprime in self.Instance.HospitalSet] 
                                    for h in self.Instance.HospitalSet]
        givenBackupHospitals = [[(sol.BackupHospital_W_whhPrime[0][h][hprime] ) 
                                    for hprime in self.Instance.HospitalSet]
                                    for h in self.Instance.HospitalSet]
        if Constants.Debug: print("givenBackupHospitals: ", givenBackupHospitals)
    
        return givenACFestablishments, givenlandRescueVehicles, givenBackupHospitals

    def ComputeInformation(self, Evaluation, nrscenario):
        if Constants.Debug: print("\n We are in 'EvaluationSimulator' Class -- ComputeInformation")
        if Constants.Debug: print("Evaluation: ", Evaluation)
        if Constants.Debug: print("nrscenario: ", nrscenario)

        Sum = sum(Evaluation[s][sol] for s in range(nrscenario) for sol in range(self.NrSolutions))
        if Constants.Debug: print("Sum: ", Sum)
        Average = Sum/nrscenario
        if Constants.Debug: print(f"Average: {Average}")

        sumdeviation = sum(math.pow((Evaluation[s][sol] - Average), 2) for s in range(nrscenario) for sol in range(self.NrSolutions))
        std_dev = math.sqrt((sumdeviation / nrscenario))
        if Constants.Debug: print(f"Standard Deviation: {std_dev}")

        EvaluateInfo = [nrscenario, Average, std_dev]

        return EvaluateInfo

    def GetScenarioSet(self, solveseed, nrscenario, allscenarios):
        if Constants.Debug: print("\n We are in 'EvaluationSimulator' Class -- GetScenarioSet")
        scenarioset = []
        treeset = []
        # Use an offset in the seed to make sure the scenario used for evaluation are different from the scenario used for optimization
        offset = solveseed + 999323

        #Uncoment to generate all the scenario if a  distribution with smallll support is used
        if allscenarios == 1:
            if Constants.Debug: print("Generate all the scenarios")

            scenariotree = ScenarioTree(self.Instance, [1] + [1]*self.Instance.NrTimeBucketWithoutUncertaintyBefore + [8, 8, 8, 8, 0], offset,
                                         scenariogenerationmethod=Constants.All,
                                         model=Constants.ModelMulti_Stage)
            scenarioset = scenariotree.GetAllScenarios(False)

            for s in range(len(scenarioset)):
                tree = ScenarioTree(self.Instance, [1, 1, 1, 1, 1, 1, 1, 1, 0], offset,
                                    model=Constants.ModelMulti_Stage, givenfirstperiod=scenarioset[s].Demands)
                treeset.append(tree)
        else:
            for seed in range(offset, nrscenario + offset, 1):
                # Generate a random scenario
                ScenarioSeed = seed
                # Evaluate the solution on the scenario
                #treestructure = [1] + [1] * self.Instance.NrTimeBucket + [0]
                treestructure = [1] * self.Instance.NrTimeBucket

                if Constants.Debug: print("treestructure: ", treestructure)
                Constants.ClusteringMethod = 'NoC'      #3 To Generate all out-of-sample scenarios uniquely
                scenariotree = ScenarioTree(instance = self.Instance, 
                                            tree_structure = treestructure, 
                                            scenario_seed = ScenarioSeed, 
                                            evaluationscenario=True, 
                                            scenariogenerationmethod="MC")
                if(Constants.Debug): print("CasualtyDemand:\n", scenariotree.CasualtyDemand) 
                if(Constants.Debug): print("HospitalDisruption:\n", scenariotree.HospitalDisruption) 
                if(Constants.Debug): print("PatientDemand:\n", scenariotree.PatientDemand) 
                if(Constants.Debug): print("PatientDischargedPercentage:\n", scenariotree.PatientDischargedPercentage) 
                
                scenario = scenariotree.GetAllScenarioSet()[0]
                #scenario = scenariotree

                scenarioset.append(scenario)
                treeset.append(scenariotree)


        return scenarioset, treeset

