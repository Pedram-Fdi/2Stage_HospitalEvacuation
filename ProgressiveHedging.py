# This class contains the attributes and methods allowing to define the progressive hedging algorithm.

from ScenarioTree import ScenarioTree
from Constants import Constants
from MIPSolver import MIPSolver
from Solution import Solution
import gurobipy as gp
from gurobipy import *
from Scenario import Scenario
import copy
import time
import math
import os
import numpy as np
from collections import deque
import random
from types import SimpleNamespace
from Tool import Tool

# Define the directory path relative to the current script location
directory = "./PH_Model_lp"
try:
    os.makedirs(directory, exist_ok=True)
except OSError as e:
    if e.errno != os.errno.EEXIST:
        raise

#This class give the methods for the classical Progressive Hedging approach
class ProgressiveHedging(object):

    def __init__(self, 
                 instance, 
                 testidentifier, 
                 treestructure, 
                 scenariotree=None, 
                 givenACFestablishments=[],
                 givenlandRescueVehicles=[],
                 givenBackupHospitals=[]
                 ):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- Constructor")

        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure

        self.LastTwoSolutionsPerMIP = {}  # Store the last two solutions

        ########## The Following block of code is only for the when that we terminate PH earlier than its convergence. As a result, it may lead to infeasible solutions!
        # That is why we do the following calculations, to prevent infeasibility! However, note that, using this way cause having not good solutions at the end of the day!
        if len(givenACFestablishments) > 0:
            givenACFestablishments = [[min(round(x), 1) for x in row] for row in givenACFestablishments]

        if len(givenlandRescueVehicles) > 0:
            givenlandRescueVehicles = [[[round(x) for x in row] for row in sublist] for sublist in givenlandRescueVehicles]
        
        if len(givenBackupHospitals) > 0:
            givenBackupHospitals = [[[min(round(x), 1) for x in row] for row in sublist] for sublist in givenBackupHospitals]



        self.GivenACFEstablishment = givenACFestablishments
        self.GivenNrLandRescueVehicle = givenlandRescueVehicles
        self.GivenBackupHospital = givenBackupHospitals

        self.SolveWithfixedACFEstablishment = len(self.GivenACFEstablishment) > 0

        self.Evaluation = False

        self.GenerateScenarios(scenariotree)

        self.rho_PenaltyParameter = 0
        self.CurrentImplementableSolution = None

        self.TraceFileName = "./Temp/PHtrace_%s_Evaluation_%s.txt" % (self.TestIdentifier.GetAsString(), Constants.Evaluation_Part)
        
        ####################### ACFEstablishment
        self.LagrangianACFEstablishment = [[0   for i in self.Instance.ACFSet]
                                                for w in self.ScenarioNrSet]

        self.lambda_LinearLagACFEstablishment = [[0    for i in self.Instance.ACFSet]
                                                        for w in self.ScenarioNrSet]
        
        ####################### LandRescueVehicle
        self.LagrangianLandRescueVehicle = [[[0     for m in self.Instance.RescueVehicleSet]
                                                    for i in self.Instance.ACFSet]
                                                    for w in self.ScenarioNrSet]

        self.lambda_LinearLagLandRescueVehicle = [[[0     for m in self.Instance.RescueVehicleSet]
                                                            for i in self.Instance.ACFSet]
                                                            for w in self.ScenarioNrSet]
        
        ####################### BackupHospital
        self.LagrangianBackupHospital = [[[0    for hprime in self.Instance.HospitalSet]
                                                for h in self.Instance.HospitalSet]
                                                for w in self.ScenarioNrSet]

        self.lambda_LinearLagBackupHospital = [[[0      for hprime in self.Instance.HospitalSet]
                                                        for h in self.Instance.HospitalSet]
                                                        for w in self.ScenarioNrSet]


        self.CurrentIteration = 0
        self.StartTime = time.time()
        
        if Constants.Dynamic_Learning_rho_PenaltyParameter:
            # Parameters for Dynamic Learning rho
            self.tauHistory_x = deque([1, 1], maxlen=2)  # Start with the first elements as 1
            self.sigmaHistory_x = deque([1 , 1], maxlen=2)  # Start with the first elements as 1
            self.alphaHistory_x = deque([1 , 1], maxlen=2)  # Start with the first elements as 1
            self.betaHistory_x = deque([0.5 , 0.5], maxlen=2)  # Start with the first elements as 0.5

            # Parameters for Dynamic Learning rho
            self.tauHistory_thetaVar = deque([1, 1], maxlen=2)  # Start with the first elements as 1
            self.sigmaHistory_thetaVar = deque([1 , 1], maxlen=2)  # Start with the first elements as 1
            self.alphaHistory_thetaVar = deque([1 , 1], maxlen=2)  # Start with the first elements as 1
            self.betaHistory_thetaVar = deque([0.5 , 0.5], maxlen=2)  # Start with the first elements as 0.5

            # Parameters for Dynamic Learning rho
            self.tauHistory_w = deque([1, 1], maxlen=2)  # Start with the first elements as 1
            self.sigmaHistory_w = deque([1 , 1], maxlen=2)  # Start with the first elements as 1
            self.alphaHistory_w = deque([1 , 1], maxlen=2)  # Start with the first elements as 1
            self.betaHistory_w = deque([0.5 , 0.5], maxlen=2)  # Start with the first elements as 0.5

            self.PreviousOptimalityGap = None  # Initialize PreviousOptimalityGap as None

        self.BuildMIPs2()

        self.duration = 0

    def BuildMIPs2(self):
        #Build the mathematicals models (1 per scenarios)
        #mipset = [0]
        mipset = range(self.NrMIPBatch)

        treestructure = [1] * (self.Instance.NrTimeBucket)

        self.MIPSolvers = [MIPSolver(instance = self.Instance, 
                                     model = Constants.Two_Stage, 
                                     scenariotree = self.SplitedScenarioTree[w],
                                     nrscenario = treestructure[1],
                                     givenACFEstablishment = self.GivenACFEstablishment,  
                                     givenNrLandRescueVehicle = self.GivenNrLandRescueVehicle,  
                                     givenBackupHospital = self.GivenBackupHospital,  
                                     logfile="NO")
                                        for w in mipset]
        for w in mipset:
            self.MIPSolvers[w].BuildModel()

            if Constants.Debug:
                # Define the path for the LP file within the newly created (or existing) directory
                file_path = os.path.join(directory, f"PH_MathematicalModel_w_{w}.lp")
                # Write the model to the file
                self.MIPSolvers[w].LocAloc.write(file_path)

    def SplitScenrioTree2(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- SplitScenrioTree2")

        #batchsize = 1030
        batchsize = 1
        self.NrMIPBatch = int(math.ceil(len(self.ScenarioNrSet)/(batchsize)))
        if Constants.Debug: print("Total number of batches:", self.NrMIPBatch)
        self.Indexscenarioinbatch = [None for m in range(self.NrMIPBatch)]
        self.Scenarioinbatch = [None for m in range(self.NrMIPBatch)]
        self.SplitedScenarioTree = [None for m in range(self.NrMIPBatch)]
        self.BatchofScenario = [int(math.floor(w/batchsize)) for w in self.ScenarioNrSet]
        self.NewIndexOfScenario = [ w % batchsize for w in self.ScenarioNrSet]

        if Constants.Debug: print("Batch of each scenario:", self.BatchofScenario)
        if Constants.Debug: print("New index of each scenario within its batch:", self.NewIndexOfScenario)

        for m in range(self.NrMIPBatch):

            firstscenarioinbatch = m * batchsize
            lastscenarioinbatch = min((m+1) * batchsize, len(self.ScenarioNrSet))
            nrscenarioinbatch = lastscenarioinbatch - firstscenarioinbatch
            
            if Constants.Debug: print("\nProcessing Batch #", m+1)
            if Constants.Debug: print("First scenario index in batch:", firstscenarioinbatch)
            if Constants.Debug: print("Last scenario index in batch:", lastscenarioinbatch)
            if Constants.Debug: print("Number of scenarios in batch:", nrscenarioinbatch)

            self.Indexscenarioinbatch[m] = range(firstscenarioinbatch, lastscenarioinbatch)
            self.Scenarioinbatch[m] = [self.ScenarioSet[w] for w in self.Indexscenarioinbatch[m]]

            if Constants.Debug: print("Scenarios in Batch #", m+1, ":", self.Scenarioinbatch[m])

            #treestructure = [1] * (self.Instance.NrTimeBucket - 1) + [0]
            treestructure = [1] * (self.Instance.NrTimeBucket)

            if Constants.Debug: print("Tree structure for Batch #", m+1, ":", treestructure)

            self.SplitedScenarioTree[m] = ScenarioTree(instance = self.Instance, 
                                                        tree_structure = treestructure, 
                                                        scenario_seed = 0,
                                                        givenscenarioset=self.Scenarioinbatch[m],
                                                        CopyscenariofromMulti_Stage = True,
                                                        scenariogenerationmethod=self.TestIdentifier.ScenarioSampling)    
              

        if Constants.Debug: print("self.SplitedScenarioTree: ", self.SplitedScenarioTree) 

    #This function creates the scenario tree
    def GenerateScenarios(self, scenariotree=None):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- GenerateScenarios")
        
        # Build the scenario tree
        if Constants.Debug: print(self.TreeStructure)
        if scenariotree is None:
            self.ScenarioTree = ScenarioTree(instance=self.Instance,
                                            tree_structure=self.TreeStructure,
                                            scenario_seed=self.TestIdentifier.ScenarioSeed,
                                            scenariogenerationmethod=self.TestIdentifier.ScenarioSampling)
        else:
            self.ScenarioTree = scenariotree

        self.ScenarioSet = self.ScenarioTree.GetAllScenarioSet()

        self.ScenarioNrSet = range(len(self.ScenarioSet))
        if Constants.Debug: print("self.ScenarioNrSet: ", self.ScenarioNrSet)
        
        self.SplitScenrioTree2()

    def InitTrace(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- InitTrace")
        if Constants.PrintPHATrace:
            self.TraceFile = open(self.TraceFileName, "w")
            self.TraceFile.write("Start the Progressive Hedging algorithm \n")
            self.TraceFile.close()

    def ComputeConvergenceX(self):

        difference = 0
        for w in self.ScenarioNrSet:
            nw = self.NewIndexOfScenario[w]
            mm = self.BatchofScenario[w]
            for i in self.Instance.ACFSet:    
                difference += self.ScenarioSet[w].Probability \
                                * math.pow(self.CurrentSolution[mm].ACFEstablishment_x_wi[nw][i] - 
                                            self.CurrentImplementableSolution.ACFEstablishment_x_wi[w][i], 2)

        convergence = math.sqrt(difference)

        return convergence
    
    def ComputeConvergenceThetaVar(self):

        difference = 0
        for w in self.ScenarioNrSet:
            nw = self.NewIndexOfScenario[w]
            mm = self.BatchofScenario[w]
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    difference += self.ScenarioSet[w].Probability \
                                  * math.pow(self.CurrentSolution[mm].LandRescueVehicle_thetaVar_wim[nw][i][m] - 
                                             self.CurrentImplementableSolution.LandRescueVehicle_thetaVar_wim[w][i][m], 2)

        convergence = math.sqrt(difference)

        return convergence
    
    def ComputeConvergenceW(self):

        difference = 0
        for w in self.ScenarioNrSet:
            nw = self.NewIndexOfScenario[w]
            mm = self.BatchofScenario[w]
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:
                    difference += self.ScenarioSet[w].Probability \
                                  * math.pow(self.CurrentSolution[mm].BackupHospital_W_whhPrime[nw][h][hprime] - 
                                             self.CurrentImplementableSolution.BackupHospital_W_whhPrime[w][h][hprime], 2)

        convergence = math.sqrt(difference)

        return convergence

    def Compute_OptimalityGap_X(self):
        """
        Compute the gap based on the difference between the last two solutions.
        """
        # Initialize the convergence gap
        difference = 0

        for m, solution_deque in self.LastTwoSolutionsPerMIP.items():
            # Ensure we have at least two solutions to compare
            if len(solution_deque) < 2:
                if Constants.Debug:
                    print(f"Skipping MIP {m}, not enough solutions in deque")
                continue

            # Retrieve the last two solutions
            last_solution = solution_deque[-1]
            second_last_solution = solution_deque[-2]

            # Iterate over the scenarios in the current batch
            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[m]):
                nw = self.NewIndexOfScenario[scenario]
                for i in self.Instance.ACFSet:
                    # Compute the squared difference between the two solutions for each variable
                    difference += self.ScenarioSet[scenario].Probability * \
                                math.pow(
                                    last_solution.ACFEstablishment_x_wi[nw][i] -
                                    second_last_solution.ACFEstablishment_x_wi[nw][i],
                                    2)

        # Compute the convergence gap as the square root of the accumulated differences
        convergence = math.sqrt(difference)

        return convergence
    
    def Compute_OptimalityGap_ThetaVar(self):
        """
        Compute the gap based on the difference between the last two solutions.
        """
        # Initialize the convergence gap
        difference = 0

        for m, solution_deque in self.LastTwoSolutionsPerMIP.items():
            # Ensure we have at least two solutions to compare
            if len(solution_deque) < 2:
                if Constants.Debug:
                    print(f"Skipping MIP {m}, not enough solutions in deque")
                continue

            # Retrieve the last two solutions
            last_solution = solution_deque[-1]
            second_last_solution = solution_deque[-2]

            # Iterate over the scenarios in the current batch
            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[m]):
                nw = self.NewIndexOfScenario[scenario]
                for i in self.Instance.ACFSet:
                    for m in self.Instance.RescueVehicleSet:
                        # Compute the squared difference between the two solutions for each variable
                        difference += self.ScenarioSet[scenario].Probability * \
                                math.pow(
                                    last_solution.LandRescueVehicle_thetaVar_wim[nw][i][m] -
                                    second_last_solution.LandRescueVehicle_thetaVar_wim[nw][i][m],
                                    2)

        # Compute the convergence gap as the square root of the accumulated differences
        convergence = math.sqrt(difference)

        return convergence
    
    def Compute_OptimalityGap_W(self):
        """
        Compute the gap based on the difference between the last two solutions.
        """
        # Initialize the convergence gap
        difference = 0

        for m, solution_deque in self.LastTwoSolutionsPerMIP.items():
            # Ensure we have at least two solutions to compare
            if len(solution_deque) < 2:
                if Constants.Debug:
                    print(f"Skipping MIP {m}, not enough solutions in deque")
                continue

            # Retrieve the last two solutions
            last_solution = solution_deque[-1]
            second_last_solution = solution_deque[-2]

            # Iterate over the scenarios in the current batch
            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[m]):
                nw = self.NewIndexOfScenario[scenario]
                for h in self.Instance.HospitalSet:
                    for hprime in self.Instance.HospitalSet:
                        # Compute the squared difference between the two solutions for each variable
                        difference += self.ScenarioSet[scenario].Probability * \
                                math.pow(
                                    last_solution.BackupHospital_W_whhPrime[nw][h][hprime] -
                                    second_last_solution.BackupHospital_W_whhPrime[nw][h][hprime],
                                    2)

        # Compute the convergence gap as the square root of the accumulated differences
        convergence = math.sqrt(difference)

        return convergence

    def GetLinearPenaltyForScenario(self, w):
        #if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- GetLinearPenaltyForScenario")

        nw = self.NewIndexOfScenario[w]
        mm = self.BatchofScenario[w]
        linterm = 0

        linterm = linterm + sum(self.LagrangianACFEstablishment[w][i] \
                                * (self.CurrentSolution[mm].ACFEstablishment_x_wi[nw][i])
                                for i in self.Instance.ACFSet)
        
        linterm = linterm + sum(self.LagrangianLandRescueVehicle[w][i][m] \
                                * (self.CurrentSolution[mm].LandRescueVehicle_thetaVar_wim[nw][i][m])
                                for m in self.Instance.RescueVehicleSet
                                for i in self.Instance.ACFSet)
        
        linterm = linterm + sum(self.LagrangianBackupHospital[w][h][hprime] \
                                * (self.CurrentSolution[mm].BackupHospital_W_whhPrime[nw][h][hprime])
                                for hprime in self.Instance.HospitalSet
                                for h in self.Instance.HospitalSet)

        return linterm
        
    def GetLinearPenalty(self):

        result = sum(self.ScenarioSet[w].Probability * (self.GetLinearPenaltyForScenario(w))
                     for w in self.ScenarioNrSet)

        return result

    def GetQuadraticPenaltyForScenario(self, w):
        #if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- GetQuadraticPenaltyForScenario")
        
        nw = self.NewIndexOfScenario[w]
        mm = self.BatchofScenario[w]
        quadterm = 0

        quadterm = quadterm + sum(Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter \
                                    * math.pow((self.CurrentSolution[mm].ACFEstablishment_x_wi[nw][i]), 2)
                                    for i in self.Instance.ACFSet)

        quadterm = quadterm + sum(Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter \
                                    * math.pow((self.CurrentSolution[mm].LandRescueVehicle_thetaVar_wim[nw][i][m]), 2)
                                    for m in self.Instance.RescueVehicleSet
                                    for i in self.Instance.ACFSet)        

        quadterm = quadterm + sum(Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter \
                                    * math.pow((self.CurrentSolution[mm].BackupHospital_W_whhPrime[nw][h][hprime]), 2)
                                    for hprime in self.Instance.HospitalSet
                                    for h in self.Instance.HospitalSet)  
        
        return quadterm

    def GetQuadraticPenalty(self):

        result = sum(self.ScenarioSet[w].Probability * self.GetQuadraticPenaltyForScenario(w)
                     for w in self.ScenarioNrSet)

        return result

    def RateQuadLinear(self):


        result = self.GetQuadraticPenalty()/ (self.Getlambda_LinearLagrangianterm())

        return result

    def Getlambda_LinearLagrangianterm(self):

        result = sum( self.ScenarioSet[w].Probability * \
                      (self.GetLinearPenaltyForScenario(w) + self.CurrentSolution[self.BatchofScenario[w]].TotalCost)
                        for w in self.ScenarioNrSet)
        return result

    def WriteInTraceFile(self, string):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- WriteInTraceFile")

        if Constants.PrintPHATrace:
            self.TraceFile = open(self.TraceFileName, "a")
            self.TraceFile.write(string)
            self.TraceFile.close()

    def GetPrimalConvergenceIndice(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- GetPrimalConvergenceIndice")

        result = 0

        result = result + sum(self.ScenarioSet[w].Probability \
                                * math.pow(self.CurrentImplementableSolution.ACFEstablishment_x_wi[w][i]
                                - self.PreviousImplementableSolution.ACFEstablishment_x_wi[w][i], 2)
                                for i in self.Instance.ACFSet
                                for w in self.ScenarioNrSet)

        result = result + sum(self.ScenarioSet[w].Probability \
                                * math.pow(self.CurrentImplementableSolution.LandRescueVehicle_thetaVar_wim[w][i][m]
                                - self.PreviousImplementableSolution.LandRescueVehicle_thetaVar_wim[w][i][m], 2)
                                for m in self.Instance.RescueVehicleSet
                                for i in self.Instance.ACFSet
                                for w in self.ScenarioNrSet)

        result = result + sum(self.ScenarioSet[w].Probability \
                                * math.pow(self.CurrentImplementableSolution.BackupHospital_W_whhPrime[w][h][hprime]
                                - self.PreviousImplementableSolution.BackupHospital_W_whhPrime[w][h][hprime], 2)
                                for hprime in self.Instance.HospitalSet
                                for h in self.Instance.HospitalSet
                                for w in self.ScenarioNrSet)                
        return result

    def GetDualConvergenceIndice(self):

        result = 0

        result = result + sum(self.ScenarioSet[w].Probability \
                                * math.pow(self.CurrentSolution[self.BatchofScenario[w]].ACFEstablishment_x_wi[self.NewIndexOfScenario[w]][i]
                                - self.CurrentImplementableSolution.ACFEstablishment_x_wi[w][i], 2)
                                for i in self.Instance.ACFSet
                                for w in self.ScenarioNrSet)

        result = result + sum(self.ScenarioSet[w].Probability \
                                * math.pow(self.CurrentSolution[self.BatchofScenario[w]].LandRescueVehicle_thetaVar_wim[self.NewIndexOfScenario[w]][i][m]
                                - self.CurrentImplementableSolution.LandRescueVehicle_thetaVar_wim[w][i][m], 2)
                                for m in self.Instance.RescueVehicleSet
                                for i in self.Instance.ACFSet
                                for w in self.ScenarioNrSet)

        result = result + sum(self.ScenarioSet[w].Probability \
                                * math.pow(self.CurrentSolution[self.BatchofScenario[w]].BackupHospital_W_whhPrime[self.NewIndexOfScenario[w]][h][hprime]
                                - self.CurrentImplementableSolution.BackupHospital_W_whhPrime[w][h][hprime], 2)
                                for hprime in self.Instance.HospitalSet
                                for h in self.Instance.HospitalSet
                                for w in self.ScenarioNrSet)
                        
        return result

    def GetDistance(self, solution):

        result = 0

        result = result + sum(self.ScenarioSet[w].Probability \
                            * math.pow(solution.ACFEstablishment_x_wi[w][i], 2)
                            for i in self.Instance.ACFSet
                            for w in self.ScenarioNrSet)
        
        result = result + sum(self.ScenarioSet[w].Probability \
                            * math.pow(solution.LandRescueVehicle_thetaVar_wim[w][i][m], 2)
                            for m in self.Instance.RescueVehicleSet
                            for i in self.Instance.ACFSet
                            for w in self.ScenarioNrSet)

        result = result + sum(self.ScenarioSet[w].Probability \
                            * math.pow(solution.BackupHospital_W_whhPrime[w][h][hprime], 2)
                            for hprime in self.Instance.HospitalSet
                            for h in self.Instance.HospitalSet
                            for w in self.ScenarioNrSet)
                        
        return result
            
    def RateLargeChangeInImplementable(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- RateLargeChangeInImplementable")

        primalcon = self.GetPrimalConvergenceIndice()
        divider = max(self.GetDistance(self.CurrentImplementableSolution),
                      self.GetDistance(self.PreviousImplementableSolution))

        result =(primalcon / divider)
        return result

    def RatePrimalDual(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- RatePrimalDual")

        primalcon = self.GetPrimalConvergenceIndice()
        dualcon = self.GetDualConvergenceIndice()
        divider = max(1,dualcon)

        result =(primalcon-dualcon / divider)
        return result

    def RateDualPrimal(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- RateDualPrimal")

        primalcon = self.GetPrimalConvergenceIndice()
        dualcon = self.GetDualConvergenceIndice()
        divider = max(1, primalcon)

        result = (dualcon - primalcon / divider)
        return result
                    
    def CheckStopingCriterion(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- CheckStopingCriterion")      
        gapX = Constants.Infinity       
        gapThetaVar = Constants.Infinity       
        gapW = Constants.Infinity       

        if self.CurrentIteration > 0:
            gapX = self.ComputeConvergenceX()
            gapThetaVar = self.ComputeConvergenceThetaVar()
            gapW = self.ComputeConvergenceW()

        convergencereached = ((gapX < Constants.PHConvergenceTolerence) and 
                              (gapThetaVar < Constants.PHConvergenceTolerence) and 
                              (gapW < Constants.PHConvergenceTolerence))
        
        self.duration = time.time() - self.StartTime
        timelimitreached = self.duration > Constants.AlgorithmTimeLimit
        iterationlimitreached = self.CurrentIteration > Constants.PHIterationLimit
        result = convergencereached or timelimitreached or iterationlimitreached

        if Constants.PrintPHATrace and self.CurrentIteration > 0:
            #self.CurrentImplementableSolution.ComputeInventory()
            self.CurrentImplementableSolution.ComputeCost()

            dualconv = -1
            primconv = -1
            lpenalty = self.GetLinearPenalty()
            qpenalty = self.GetQuadraticPenalty()
            ratequad_lin = self.RateQuadLinear()
            ratechangeimplem = -1
            ratedualprimal = -1
            rateprimaldual = -1
            if self.CurrentIteration > 1:
                primconv = self.GetPrimalConvergenceIndice()
                dualconv = self.GetDualConvergenceIndice()
                ratechangeimplem = self.RateLargeChangeInImplementable()
                rateprimaldual = self.RatePrimalDual()
                ratedualprimal = self.RateDualPrimal()

            trace_message = (
                "Iteration: %r, Duration: %.2f, GapX: %.2f, gapThetaVar: %.2f, gapW: %.2f, UB: %.2f, linear penalty: %.2f, "
                "quadratic penalty: %.2f, Multiplier: %.6f, primal conv: %.2f, dual conv: %.2f, "
                "Rate Large Change(l): %.2f, rate quad_lin(s): %.2f, rateprimaldual(l<-): %.2f, "
                "ratedualprimal(l->): %.2f\n"
                % (
                    self.CurrentIteration,
                    self.duration,
                    gapX,
                    gapThetaVar,
                    gapW,
                    self.CurrentImplementableSolution.TotalCost,
                    lpenalty,
                    qpenalty,
                    self.rho_PenaltyParameter,
                    primconv,
                    dualconv,
                    ratechangeimplem,
                    ratequad_lin,
                    rateprimaldual,
                    ratedualprimal,
                )
            )
            self.WriteInTraceFile(trace_message)

        return result

    def UpdatePenaltyParameter_rho_x(self):

        print("Iteration: ", self.CurrentIteration)
        # Compute the Non-Anticipativity Gap (NA Gap) and Partial Optimality Gap
        NAGap_g_x = self.ComputeConvergenceX()
        print("NAGap_g_x: ", NAGap_g_x)
        
        Partial_Optimality_Gap = self.Compute_OptimalityGap_X()
        print("Partial_Optimality_Gap: ", Partial_Optimality_Gap)

        # Calculate the Optimality Gap
        OptimalityGap = NAGap_g_x + Partial_Optimality_Gap
        print("OptimalityGap: ", OptimalityGap)

        # Calculate tau
        if self.PreviousOptimalityGap is not None:
            tau = OptimalityGap / self.PreviousOptimalityGap  # Calculate the new tau
        else:
            tau = self.tauHistory_x[-1]  # Use the most recent tau if no previous optimality gap exists
        # Update the deque with the new tau
        self.tauHistory_x.append(tau)
        print("tauHistory_x: ", list(self.tauHistory_x))

        # Update gamma based on the calculated tau
        gamma = max(0.1, min(0.9, self.tauHistory_x[-1] - 0.6))
        print("gamma: ", gamma)

        # Update sigma based on the calculated gamma and tau and previosusigma
        sigma = (1-gamma)*self.sigmaHistory_x[-1] + gamma * self.tauHistory_x[-1]
        self.sigmaHistory_x.append(sigma)
        print("sigmaHistory_x: ", list(self.sigmaHistory_x))

        # Update zeta based on sigma
        zeta = np.sqrt(1.1 * self.sigmaHistory_x[-1])
        print("zeta: ", zeta)

        # Update alpha based on previous alpha, NAGap_g_x, and OptimalityGap
        alpha = 0.8 * self.alphaHistory_x[-1] + 0.2 * (NAGap_g_x/OptimalityGap)
        self.alphaHistory_x.append(alpha)
        print("alphaHistory_x: ", list(self.alphaHistory_x))
        
        # Update beta based on previous beta, new alpha
        beta = 0.98 * self.betaHistory_x[-1] + 0.02 * self.alphaHistory_x[-1]
        self.betaHistory_x.append(beta)
        print("betaHistory_x: ", list(self.betaHistory_x))

        # Update c (contraction rate) based on new beta
        c = max(0.95, ((1 - (2 * self.betaHistory_x[-1]))/(1 - self.betaHistory_x[-1])))
        print("c: ", c)

        # Update h based on new beta
        h = max(c + (((1 - c) / (self.betaHistory_x[-1])) * self.alphaHistory_x[-1]), 1 + ((self.alphaHistory_x[-1] - self.betaHistory_x[-1]) / (1 - self.betaHistory_x[-1])))
        print("h: ", h)
        
        # Update q based on zeta and h and current iteration
        q = pow(max(zeta , h) , (1 / (1 + 0.01 * (self.CurrentIteration - 2))))
        #q = min(1.5, q)        If your rho is increased too fast, just activate this part and try to control the increagin rho!
        print("q: ", q)

        print("Previous rho: ", self.rho_PenaltyParameter)
        new_rho = max(0.01, min(100, q * self.rho_PenaltyParameter))
        self.rho_PenaltyParameter = new_rho
        print("New rho: ", self.rho_PenaltyParameter)
        # Update the PreviousOptimalityGap to the current value
        self.PreviousOptimalityGap = OptimalityGap
        print("------")

    def UpdatePenaltyParameter_rho(self): 
        # Compute the Non-Anticipativity Gap for x and y combined.
        # For example, you could compute:
        NAGap_g_x = self.ComputeConvergenceX()
        NAGap_g_ThetaVar = self.ComputeConvergenceThetaVar()
        NAGap_g_W = self.ComputeConvergenceW()  
        NAGap_g = NAGap_g_x + NAGap_g_ThetaVar + NAGap_g_W  # Or use a weighted sum if needed
        
        # Compute the Partial Optimality Gap for x and y combined.
        Partial_Optimality_Gap_x = self.Compute_OptimalityGap_X()
        Partial_Optimality_Gap_ThetaVar = self.Compute_OptimalityGap_ThetaVar() 
        Partial_Optimality_Gap_W = self.Compute_OptimalityGap_W() 
        Partial_Optimality_Gap = Partial_Optimality_Gap_x + Partial_Optimality_Gap_ThetaVar + Partial_Optimality_Gap_W

        # Calculate the Overall Optimality Gap
        OptimalityGap = NAGap_g + Partial_Optimality_Gap
        print("OptimalityGap: ", OptimalityGap)

        # Calculate tau: If a previous optimality gap exists, update tau; otherwise, use the latest stored tau.
        if self.PreviousOptimalityGap is not None:
            tau = OptimalityGap / self.PreviousOptimalityGap
        else:
            tau = self.tauHistory_x[-1]  # Here, you might rename your history to reflect x and y combined
        self.tauHistory_x.append(tau)

        # Update gamma based on the calculated tau
        gamma = max(0.1, min(0.9, self.tauHistory_x[-1] - 0.6))
        print("gamma: ", gamma)

        # Update sigma based on the calculated gamma and tau, and previous sigma
        sigma = (1 - gamma) * self.sigmaHistory_x[-1] + gamma * self.tauHistory_x[-1]
        self.sigmaHistory_x.append(sigma)

        # Update zeta based on sigma
        zeta = np.sqrt(1.1 * self.sigmaHistory_x[-1])

        # Update alpha based on previous alpha and the ratio NAGap/OptimalityGap
        alpha = 0.8 * self.alphaHistory_x[-1] + 0.2 * (NAGap_g / OptimalityGap)
        self.alphaHistory_x.append(alpha)
        
        # Update beta based on previous beta and new alpha
        beta = 0.98 * self.betaHistory_x[-1] + 0.02 * self.alphaHistory_x[-1]
        self.betaHistory_x.append(beta)

        # Update contraction rate c based on new beta
        c = max(0.95, ((1 - (2 * self.betaHistory_x[-1])) / (1 - self.betaHistory_x[-1])))
        print("c: ", c)

        # Update h based on new beta and alpha
        h = max(c + (((1 - c) / self.betaHistory_x[-1]) * self.alphaHistory_x[-1]),
                1 + ((self.alphaHistory_x[-1] - self.betaHistory_x[-1]) / (1 - self.betaHistory_x[-1])))
        print("h: ", h)
        
        # Update q based on zeta, h, and current iteration using a cooling factor
        q = pow(max(zeta, h), (1 / (1 + 0.01 * (self.CurrentIteration - 2))))
        print("q: ", q)

        print("Previous rho: ", self.rho_PenaltyParameter)
        new_rho = max(0.01, min(100, q * self.rho_PenaltyParameter))
        self.rho_PenaltyParameter = new_rho
        print("New rho: ", self.rho_PenaltyParameter)
        
        # Update the previous optimality gap for next iteration's computation
        self.PreviousOptimalityGap = OptimalityGap
        print("------")


    def SolveScenariosIndependently(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- SolveScenariosIndependently")

        #For each scenario
        for m in range(self.NrMIPBatch):
            print("self.CurrentIteration: ", self.CurrentIteration)

            #Update the coeffient in the objective function
            if Constants.Quadratic_to_Linear_PHA:
                self.UpdateLagrangianCoeff_LinearVersion(m)
            else:
                self.UpdateLagrangianCoeff(m)
            mip = self.MIPSolvers[m]
            mip.ModifyMipForScenarioTree(self.SplitedScenarioTree[m])

            if Constants.Quadratic_to_Linear_PHA:
                if self.CurrentImplementableSolution:
                    mip.ModifyMipForACF_LinearPHA(self.CurrentImplementableSolution.ACFEstablishment_x_wi)
                    mip.ModifyMipForLandRescueVehicle_LinearPHA(self.CurrentImplementableSolution.LandRescueVehicle_thetaVar_wim)
                    mip.ModifyMipForBackupHospital_LinearPHA(self.CurrentImplementableSolution.BackupHospital_W_whhPrime)

            #Solve the model.
            new_solution = mip.Solve(True)

            # Dynamically initialize the deque for this MIP if it doesn't exist
            if m not in self.LastTwoSolutionsPerMIP:
                self.LastTwoSolutionsPerMIP[m] = deque(maxlen=2)

            # Save the current solution for this MIP and keep the last two
            self.LastTwoSolutionsPerMIP[m].append(new_solution)  # Update the deque for this MIP
            self.CurrentSolution[m] = new_solution  # Current solution remains as it is

            #compute the cost for the penalty update strategy
            self.CurrentSolution[m].ComputeCost()

    def UpdateLagrangianCoeff(self, batch):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- UpdateLagrangianCoeff")

        mipsolver = self.MIPSolvers[batch]

        # Initialize or retrieve storage for quadratic expressions
        if not hasattr(mipsolver, 'quadratic_terms'):
            mipsolver.quadratic_terms = {}

        # Store the last value of rho_PenaltyParameter for this specific solver
        if not hasattr(mipsolver, 'last_rho_PenaltyParameter'):
            mipsolver.last_rho_PenaltyParameter = None

        # Check if rho_PenaltyParameter has changed for this specific solver
        rho_changed = self.rho_PenaltyParameter != mipsolver.last_rho_PenaltyParameter
        if rho_changed:
            mipsolver.last_rho_PenaltyParameter = self.rho_PenaltyParameter  # Update to the current value

        # Function to handle linear and quadratic term updates
        def update_variable(variable, new_coeff):
            variable.setAttr(GRB.Attr.Obj, new_coeff)
            mipsolver.LocAloc.update()
            var_name = variable.VarName
            
            # Add quadratic terms only if rho_PenaltyParameter has changed or not yet added
            if rho_changed or var_name not in mipsolver.quadratic_terms:
                if var_name in mipsolver.quadratic_terms:
                    old_quad_expr = mipsolver.quadratic_terms[var_name]
                    mipsolver.LocAloc.setObjective(mipsolver.LocAloc.getObjective() - old_quad_expr)
                    mipsolver.LocAloc.update()
                # Create the quadratic term
                quad_expr = (Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter *  variable * variable)
                mipsolver.quadratic_terms[var_name] = quad_expr

                # Add the quadratic term to the objective
                mipsolver.LocAloc.setObjective(mipsolver.LocAloc.getObjective() + quad_expr)
                mipsolver.LocAloc.update()
        
        # Update coefficients for the current batch
        for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):

            ###### x Variable
            for i in self.Instance.ACFSet:
                x_var_index = mipsolver.GetIndexACFEstablishmentVariable(scenario_index, i)
                variable = mipsolver.ACFEstablishment_Var[x_var_index]
                new_coeff = (
                    mipsolver.GetACFestablishmentCoeff_Obj(i) +
                    self.LagrangianACFEstablishment[scenario][i]
                )
                update_variable(variable, new_coeff)

            ###### thetaVar Variable
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    thetaVar_var_index = mipsolver.GetIndexLandRescueVehicleVariable(scenario_index, i, m)
                    variable = mipsolver.LandRescueVehicle_Var[thetaVar_var_index]
                    new_coeff = (
                        mipsolver.GetlandRescueVehicleCoeff(i, m) +
                        self.LagrangianLandRescueVehicle[scenario][i][m]
                    )
                    update_variable(variable, new_coeff)

            ###### w Variable
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:
                    w_var_index = mipsolver.GetIndexBackupHospitalVariable(scenario_index, h, hprime)
                    variable = mipsolver.BackupHospital_Var[w_var_index]
                    new_coeff = (
                        mipsolver.GetbackupHospitalCoeff(h, hprime) +
                        self.LagrangianBackupHospital[scenario][h][hprime]
                    )
                    update_variable(variable, new_coeff)

        # Set all variables to continuous for a QP problem if necessary
        if self.SolveWithfixedACFEstablishment:
            for var in mipsolver.LocAloc.getVars():
                var.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)
            if Constants.Debug: print("All variables set to continuous for a QP problem.")

        if Constants.Debug: print("Objective and problem type updated.")

    def UpdateLagrangianCoeff_LinearVersion(self, batch):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- UpdateLagrangianCoeff")

        mipsolver = self.MIPSolvers[batch]

        # Store the last value of rho_PenaltyParameter for this specific solver
        if not hasattr(mipsolver, 'last_rho_PenaltyParameter'):
            mipsolver.last_rho_PenaltyParameter = None

        # Check if rho_PenaltyParameter has changed for this specific solver
        rho_changed = self.rho_PenaltyParameter != mipsolver.last_rho_PenaltyParameter
        if rho_changed:
            mipsolver.last_rho_PenaltyParameter = self.rho_PenaltyParameter  # Update to the current value

        # Function to handle linear and quadratic term updates
        def update_variable(variable, new_coeff):
            variable.setAttr(GRB.Attr.Obj, new_coeff)
            mipsolver.LocAloc.update()
            var_name = variable.VarName
            if Constants.Debug: print("var_name: ", var_name)
        
        ########### Update x coefficients for the current batch
        for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
            for i in self.Instance.ACFSet:
                x_var_index = mipsolver.GetIndexACFEstablishmentVariable(scenario_index, i)
                variable = mipsolver.ACFEstablishment_Var[x_var_index]
                new_coeff = (mipsolver.GetACFestablishmentCoeff_Obj(i) +
                            self.lambda_LinearLagACFEstablishment[scenario][i])
                update_variable(variable, new_coeff)

        # Update z+ and z- coefficients only (if rho_changed) for the current batch
        if rho_changed:
            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
                for i in self.Instance.ACFSet:
                    zPlus_var_index = mipsolver.GetIndex_PHA_ZPlus_ACFEstablishmentVariable(scenario_index, i)
                    variable = mipsolver.PHA_ZPlus_ACFEstablishment_Var[zPlus_var_index]
                    new_coeff = (0 + Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter)
                    update_variable(variable, new_coeff)

            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
                for i in self.Instance.ACFSet:
                    zMinus_var_index = mipsolver.GetIndex_PHA_ZMinus_ACFEstablishmentVariable(scenario_index, i)
                    variable = mipsolver.PHA_ZMinus_ACFEstablishment_Var[zMinus_var_index]
                    new_coeff = (0 + Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter)
                    update_variable(variable, new_coeff)

        ########### Update thetaVar coefficients for the current batch
        for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    thetaVar_var_index = mipsolver.GetIndexLandRescueVehicleVariable(scenario_index, i, m)
                    variable = mipsolver.LandRescueVehicle_Var[thetaVar_var_index]
                    new_coeff = (mipsolver.GetlandRescueVehicleCoeff(i, m) +
                                self.lambda_LinearLagLandRescueVehicle[scenario][i][m])
                    update_variable(variable, new_coeff)

        # Update z+ and z- coefficients only (if rho_changed) for the current batch
        if rho_changed:
            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
                for i in self.Instance.ACFSet:
                    for m in self.Instance.RescueVehicleSet:
                        zPlus_var_index = mipsolver.GetIndex_PHA_ZPlus_LandRescueVehicleVariable(scenario_index, i, m)
                        variable = mipsolver.PHA_ZPlus_LandRescueVehicle_Var[zPlus_var_index]
                        new_coeff = (0 + Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter)
                        update_variable(variable, new_coeff)

            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
                for i in self.Instance.ACFSet:
                    for m in self.Instance.RescueVehicleSet:
                        zMinus_var_index = mipsolver.GetIndex_PHA_ZMinus_LandRescueVehicleVariable(scenario_index, i, m)
                        variable = mipsolver.PHA_ZMinus_LandRescueVehicle_Var[zMinus_var_index]
                        new_coeff = (0 + Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter)
                        update_variable(variable, new_coeff)

        ########### Update w coefficients for the current batch
        for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:
                    w_var_index = mipsolver.GetIndexBackupHospitalVariable(scenario_index, h, hprime)
                    variable = mipsolver.BackupHospital_Var[w_var_index]
                    new_coeff = (mipsolver.GetbackupHospitalCoeff(h, hprime) +
                                self.lambda_LinearLagBackupHospital[scenario][h][hprime])
                    update_variable(variable, new_coeff)

        # Update z+ and z- coefficients only (if rho_changed) for the current batch
        if rho_changed:
            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
                for h in self.Instance.HospitalSet:
                    for hprime in self.Instance.HospitalSet:
                        zPlus_var_index = mipsolver.GetIndex_PHA_ZPlus_BackupHospitalVariable(scenario_index, h, hprime)
                        variable = mipsolver.PHA_ZPlus_BackupHospital_Var[zPlus_var_index]
                        new_coeff = (0 + Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter)
                        update_variable(variable, new_coeff)

            for scenario_index, scenario in enumerate(self.Indexscenarioinbatch[batch]):
                for h in self.Instance.HospitalSet:
                    for hprime in self.Instance.HospitalSet:
                        zMinus_var_index = mipsolver.GetIndex_PHA_ZMinus_BackupHospitalVariable(scenario_index, h, hprime)
                        variable = mipsolver.PHA_ZMinus_BackupHospital_Var[zMinus_var_index]
                        new_coeff = (0 + Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter)
                        update_variable(variable, new_coeff)

    def CreateImplementableSolution(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- CreateImplementableSolution")

        ###########
        solACFEstablishment = [[-1 for i in self.Instance.ACFSet]
                                        for w in self.ScenarioNrSet]
        ###########
        sollandRescueVehicle = [[[-1 for m in self.Instance.RescueVehicleSet]
                                        for i in self.Instance.ACFSet]
                                        for w in self.ScenarioNrSet]
        ###########
        solbackupHospital = [[[-1 for hprime in self.Instance.HospitalSet]
                                        for h in self.Instance.HospitalSet]
                                        for w in self.ScenarioNrSet]
        
        ###########
        solcasualtyTransfer = [[[[[[-1      for m in self.Instance.RescueVehicleSet]
                                            for u in self.Instance.MedFacilitySet]
                                            for l in self.Instance.DisasterAreaSet]
                                            for j in self.Instance.InjuryLevelSet]
                                            for t in self.Instance.TimeBucketSet]
                                            for w in self.ScenarioNrSet]
        ###########
        solunsatisfiedCasualties = [[[[-1      for l in self.Instance.DisasterAreaSet]
                                                for j in self.Instance.InjuryLevelSet]
                                                for t in self.Instance.TimeBucketSet]
                                                for w in self.ScenarioNrSet]
        ###########
        soldischargedPatients = [[[[-1      for u in self.Instance.MedFacilitySet]
                                            for j in self.Instance.InjuryLevelSet]
                                            for t in self.Instance.TimeBucketSet]
                                            for w in self.ScenarioNrSet]
        ###########
        sollandEvacuatedPatients = [[[[[[-1      for m in self.Instance.RescueVehicleSet]
                                                for u in self.Instance.MedFacilitySet]
                                                for h in self.Instance.HospitalSet]
                                                for j in self.Instance.InjuryLevelSet]
                                                for t in self.Instance.TimeBucketSet]
                                                for w in self.ScenarioNrSet]
        ###########
        solaerialEvacuatedPatients = [[[[[[[-1      for m in self.Instance.RescueVehicleSet]
                                                    for hprime in self.Instance.HospitalSet]
                                                    for i in self.Instance.ACFSet]
                                                    for h in self.Instance.HospitalSet]
                                                    for j in self.Instance.InjuryLevelSet]
                                                    for t in self.Instance.TimeBucketSet]
                                                    for w in self.ScenarioNrSet]
        
        ###########
        solunevacuatedPatients = [[[[-1      for h in self.Instance.HospitalSet]
                                                for j in self.Instance.InjuryLevelSet]
                                                for t in self.Instance.TimeBucketSet]
                                                for w in self.ScenarioNrSet]
        
        ###########
        solavailableCapFacility = [[[-1      for u in self.Instance.MedFacilitySet]
                                            for t in self.Instance.TimeBucketSet]
                                            for w in self.ScenarioNrSet]
                    
        # First-Stage Variables
        ####################################### ACF Establishment
        ACFestablishment = [(sum(self.ScenarioSet[w].Probability * 
                                self.CurrentSolution[self.BatchofScenario[w]].ACFEstablishment_x_wi[self.NewIndexOfScenario[w]][i] 
                                for w in range(len(self.ScenarioSet)))\
                                / 1)  
                                #Here, exceptionally we devide the final value over 1 because it is the first-stage variable and it should be the same for all scenarios
                                for i in self.Instance.ACFSet]
        if Constants.Debug: print("\nCurrent Value of ACFEstablishment at node:")
        for w in range(len(self.ScenarioSet)):
            if Constants.Debug: print(self.CurrentSolution[self.BatchofScenario[w]].ACFEstablishment_x_wi[self.NewIndexOfScenario[w]])
        if Constants.Debug: print(f"Implementable Value of Facility Establishments at node: \n", ACFestablishment)
        for w in range(len(self.ScenarioSet)):
            for i in self.Instance.ACFSet:
                solACFEstablishment[w][i] = ACFestablishment[i]                    
        
        ####################################### Land Rescue Vehicle
        landRescueVehicle = [[(sum(self.ScenarioSet[w].Probability * 
                                self.CurrentSolution[self.BatchofScenario[w]].LandRescueVehicle_thetaVar_wim[self.NewIndexOfScenario[w]][i][m] 
                                for w in range(len(self.ScenarioSet)))\
                                / 1)  
                                #Here, exceptionally we devide the final value over 1 because it is the first-stage variable and it should be the same for all scenarios
                                for m in self.Instance.RescueVehicleSet]
                                for i in self.Instance.ACFSet]
        if Constants.Debug: print("\nCurrent Value of FacilityEstablishment at node:")
        for w in range(len(self.ScenarioSet)):
            if Constants.Debug: print(self.CurrentSolution[self.BatchofScenario[w]].LandRescueVehicle_thetaVar_wim[self.NewIndexOfScenario[w]])
        if Constants.Debug: print(f"Implementable Value of Facility Establishments at node: \n", landRescueVehicle)
        for w in range(len(self.ScenarioSet)):
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    sollandRescueVehicle[w][i][m] = landRescueVehicle[i][m]                   
        
        ####################################### Backup Hospital
        backupHospital = [[(sum(self.ScenarioSet[w].Probability * 
                                self.CurrentSolution[self.BatchofScenario[w]].BackupHospital_W_whhPrime[self.NewIndexOfScenario[w]][h][hprime] 
                                for w in range(len(self.ScenarioSet)))\
                                / 1)  
                                #Here, exceptionally we devide the final value over 1 because it is the first-stage variable and it should be the same for all scenarios
                                for hprime in self.Instance.HospitalSet]
                                for h in self.Instance.HospitalSet]
        if Constants.Debug: print("\nCurrent Value of FacilityEstablishment at node:")
        for w in range(len(self.ScenarioSet)):
            if Constants.Debug: print(self.CurrentSolution[self.BatchofScenario[w]].BackupHospital_W_whhPrime[self.NewIndexOfScenario[w]])
        if Constants.Debug: print(f"Implementable Value of Facility Establishments at node: \n", backupHospital)
        for w in range(len(self.ScenarioSet)):
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:
                    solbackupHospital[w][h][hprime] = backupHospital[h][hprime]                   


        ####################################### solcasualtyTransfer
        for w in range(len(self.ScenarioSet)):
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for l in self.Instance.DisasterAreaSet:
                        for u in self.Instance.MedFacilitySet:
                            for m in self.Instance.RescueVehicleSet:
                                solcasualtyTransfer[w][t][j][l][u][m] = self.CurrentSolution[self.BatchofScenario[w]].CasualtyTransfer_q_wtjlum[self.NewIndexOfScenario[w]][t][j][l][u][m]
        if Constants.Debug: print("$$$$$$$ solcasualtyTransfer:\n ", solcasualtyTransfer)

        ####################################### solunsatisfiedCasualties
        for w in range(len(self.ScenarioSet)):
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for l in self.Instance.DisasterAreaSet:
                        solunsatisfiedCasualties[w][t][j][l] = self.CurrentSolution[self.BatchofScenario[w]].UnsatisfiedCasualties_mu_wtjl[self.NewIndexOfScenario[w]][t][j][l]
        if Constants.Debug: print("$$$$$$$ solunsatisfiedCasualties:\n ", solunsatisfiedCasualties)

        ####################################### soldischargedPatients
        for w in range(len(self.ScenarioSet)):
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for u in self.Instance.MedFacilitySet:
                        soldischargedPatients[w][t][j][u] = self.CurrentSolution[self.BatchofScenario[w]].DischargedPatients_sigmaVar_wtju[self.NewIndexOfScenario[w]][t][j][u]
        if Constants.Debug: print("$$$$$$$ soldischargedPatients:\n ", soldischargedPatients)

        ####################################### sollandEvacuatedPatients
        for w in range(len(self.ScenarioSet)):
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        for u in self.Instance.MedFacilitySet:
                            for m in self.Instance.RescueVehicleSet:
                                sollandEvacuatedPatients[w][t][j][h][u][m] = self.CurrentSolution[self.BatchofScenario[w]].LandEvacuatedPatients_u_L_wtjhum[self.NewIndexOfScenario[w]][t][j][h][u][m]
        if Constants.Debug: print("$$$$$$$ sollandEvacuatedPatients:\n ", sollandEvacuatedPatients)

        ####################################### solaerialEvacuatedPatients
        for w in range(len(self.ScenarioSet)):
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        for i in self.Instance.ACFSet:
                            for hprime in self.Instance.HospitalSet:
                                for m in self.Instance.RescueVehicleSet:
                                    solaerialEvacuatedPatients[w][t][j][h][i][hprime][m] = self.CurrentSolution[self.BatchofScenario[w]].AerialEvacuatedPatients_u_A_wtjhihPrimem[self.NewIndexOfScenario[w]][t][j][h][i][hprime][m]
        if Constants.Debug: print("$$$$$$$ solaerialEvacuatedPatients:\n ", solaerialEvacuatedPatients)

        ####################################### solunevacuatedPatients
        for w in range(len(self.ScenarioSet)):
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        solunevacuatedPatients[w][t][j][h] = self.CurrentSolution[self.BatchofScenario[w]].UnevacuatedPatients_Phi_wtjh[self.NewIndexOfScenario[w]][t][j][h]
        if Constants.Debug: print("$$$$$$$ solunevacuatedPatients:\n ", solunevacuatedPatients)

        ####################################### solavailableCapFacility
        for w in range(len(self.ScenarioSet)):
            for t in self.Instance.TimeBucketSet:
                for u in self.Instance.MedFacilitySet:
                    solavailableCapFacility[w][t][u] = self.CurrentSolution[self.BatchofScenario[w]].AvailableCapFacility_zeta_wtu[self.NewIndexOfScenario[w]][t][u]
        if Constants.Debug: print("$$$$$$$ solavailableCapFacility:\n ", solavailableCapFacility)


        solution = Solution(instance=self.Instance, 
                            solACFEstablishment_x_wi = solACFEstablishment, 
                            solLandRescueVehicle_thetaVar_wim = sollandRescueVehicle, 
                            solBackupHospital_W_whhPrime = solbackupHospital, 
                            solCasualtyTransfer_q_wtjlum = solcasualtyTransfer, 
                            solUnsatisfiedCasualties_mu_wtjl = solunsatisfiedCasualties, 
                            solDischargedPatients_sigmaVar_wtju = soldischargedPatients, 
                            solLandEvacuatedPatients_u_L_wtjhum = sollandEvacuatedPatients, 
                            solAerialEvacuatedPatients_u_A_wtjhihPrimem = solaerialEvacuatedPatients, 
                            solUnevacuatedPatients_Phi_wtjh = solunevacuatedPatients, 
                            solAvailableCapFacility_zeta_wtu = solavailableCapFacility, 
                            Final_ACFEstablishmentCost = 0, 
                            Final_LandRescueVehicleCost = 0, 
                            Final_BackupHospitalCost = 0, 
                            Final_CasualtyTransferCost = 0, 
                            Final_UnsatisfiedCasualtiesCost = 0, 
                            Final_DischargedPatientsCost = 0, 
                            Final_LandEvacuatedPatientsCost = 0, 
                            Final_AerialEvacuatedPatientsCost = 0, 
                            Final_UnevacuatedPatientsCost = 0, 
                            Final_AvailableCapFacilityCost = 0, 
                            scenarioset = self.ScenarioSet, 
                            scenariotree = self.ScenarioTree, 
                            partialsolution = False)

        return solution

    def UpdateLagragianMultipliers(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- UpdateLagragianMultipliers")

        for w in self.ScenarioNrSet:
            mm = self.BatchofScenario[w]
            nw = self.NewIndexOfScenario[w] 

            ############################# ACF Establishment
            for i in self.Instance.ACFSet:
                self.lambda_LinearLagACFEstablishment[w][i], self.LagrangianACFEstablishment[w][i] = \
                    self.ComputeLagrangian(self.lambda_LinearLagACFEstablishment[w][i],
                                            self.CurrentSolution[mm].ACFEstablishment_x_wi[nw][i],
                                            self.CurrentImplementableSolution.ACFEstablishment_x_wi[w][i])
            
            ############################# Land Rescue Vehicle
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    self.lambda_LinearLagLandRescueVehicle[w][i][m], self.LagrangianLandRescueVehicle[w][i][m] = \
                        self.ComputeLagrangian(self.lambda_LinearLagLandRescueVehicle[w][i][m],
                                                self.CurrentSolution[mm].LandRescueVehicle_thetaVar_wim[nw][i][m],
                                                self.CurrentImplementableSolution.LandRescueVehicle_thetaVar_wim[w][i][m])
            ############################# Facility Establishment
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:
                    self.lambda_LinearLagBackupHospital[w][h][hprime], self.LagrangianBackupHospital[w][h][hprime] = \
                        self.ComputeLagrangian(self.lambda_LinearLagBackupHospital[w][h][hprime],
                                                self.CurrentSolution[mm].BackupHospital_W_whhPrime[nw][h][hprime],
                                                self.CurrentImplementableSolution.BackupHospital_W_whhPrime[w][h][hprime])
        if(Constants.Debug): 
            print("lambda_Linear Lagrangian (x):\n", np.round(self.lambda_LinearLagACFEstablishment, 3))
            print("The coefficient of x after combining the Quadratic and Linear penalty::\n", np.round(self.LagrangianACFEstablishment, 3))
            print("----------------------")
            print("lambda_Linear Lagrangian (thetaVar):\n", np.round(self.lambda_LinearLagLandRescueVehicle, 3))
            print("The coefficient of thetaVar after combining the Quadratic and Linear penalty::\n", np.round(self.LagrangianLandRescueVehicle, 3))
            print("----------------------")
            print("lambda_Linear Lagrangian (w):\n", np.round(self.lambda_LinearLagBackupHospital, 3))
            print("The coefficient of w after combining the Quadratic and Linear penalty::\n", np.round(self.LagrangianBackupHospital, 3))
            print("----------------------")

    def ComputeLagrangian(self, prevlag, independentvalue, implementablevalue):
        #if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- ComputeLagrangian")
        lambda_LinearLag = prevlag + (self.rho_PenaltyParameter * (independentvalue - implementablevalue))

        lagrangian = lambda_LinearLag - (2 * Constants.PHCoeeff_QuadraticPart * self.rho_PenaltyParameter * implementablevalue)

        return lambda_LinearLag, lagrangian
    
   #This function run the algorithm
    def Run(self):
        if Constants.Debug: print("\n We are in 'ProgressiveHedging' Class -- Run")
        self.PrintOnlyFirstStagePreviousValue = Constants.PrintOnlyFirstStageDecision
        if Constants.PrintOnlyFirstStageDecision:
            Constants.PrintOnlyFirstStageDecision = False
            # raise NameError("Progressive Hedging requires to print the full solution, set Constants.PrintOnlyFirstStageDecision to False")

        self.InitTrace()
        self.CurrentSolution = [None for w in self.ScenarioNrSet]

        while not self.CheckStopingCriterion():
            print("######################## PH Iteration: ", self.CurrentIteration)
            # Solve each scenario independentely
            self.SolveScenariosIndependently()

            # Create an implementable solution on the scenario tree
            sol = self.CurrentImplementableSolution
            if sol is not None:
                self.PreviousImplementableSolution = SimpleNamespace(
                    ACFEstablishment_x_wi         = sol.ACFEstablishment_x_wi.copy(),
                    LandRescueVehicle_thetaVar_wim= sol.LandRescueVehicle_thetaVar_wim.copy(),
                    BackupHospital_W_whhPrime     = sol.BackupHospital_W_whhPrime.copy(),)            
                #self.PreviousImplementableSolution = copy.deepcopy(self.CurrentImplementableSolution)

            self.CurrentImplementableSolution = self.CreateImplementableSolution()

            self.CurrentIteration += 1

            if self.CurrentIteration == 1:
                    self.rho_PenaltyParameter = Constants.Rho_PH_PenaltyParameter

            if (Constants.Dynamic_rho_PenaltyParameter and not Constants.Dynamic_Learning_rho_PenaltyParameter) and (self.CurrentIteration > 1) and (self.CurrentIteration % 10 == 0):
                self.rho_PenaltyParameter += Constants.Increase_rate_dynamic_rho
                if Constants.Debug: print(f"Updated rho_PenaltyParameter to {self.rho_PenaltyParameter} after {self.CurrentIteration} iterations.")
            
            if (Constants.Dynamic_Learning_rho_PenaltyParameter) and (self.CurrentIteration > 1):
                self.UpdatePenaltyParameter_rho()

            # Update the lagrangian multiplier
            self.UpdateLagragianMultipliers()

            #if Constants.Debug:
            #    self.PrintCurrentIteration()

        GivenACFEstablishment_Applicable_wi = [[round(value) for value in row] for row in self.CurrentImplementableSolution.ACFEstablishment_x_wi]
        GivenACFEstablishment_Applicable_i = self.Check_ACF_Establishment_Budget_Constraint(self.CurrentImplementableSolution.ACFEstablishment_x_wi[0], GivenACFEstablishment_Applicable_wi[0])
        
        GivenNrLandRescueVehicle_Applicable_wim = [[[round(value) for value in inner] for inner in outer] for outer in self.CurrentImplementableSolution.LandRescueVehicle_thetaVar_wim]
        GivenNrLandRescueVehicle_Applicable_im = self.Check_LandRescueVehicleAllocation_Constraints(GivenACFEstablishment_Applicable_i, GivenNrLandRescueVehicle_Applicable_wim[0])

        GivenBackupHospital_Applicable_whhprime = [[[round(value) for value in inner] for inner in outer] for outer in self.CurrentImplementableSolution.BackupHospital_W_whhPrime]
        GivenBackupHospital_Applicable_hhprime = self.Check_Hospitals_Compatibility_Constraint(GivenBackupHospital_Applicable_whhprime[0])

        # Now, replicate these scenario-0 values for all scenarios using one loop:
        self.GivenACFEstablishment_Applicable = []
        self.GivenNrLandRescueVehicle_Applicable = []
        self.GivenBackupHospital_Applicable = []

        for _ in self.ScenarioNrSet:
            # For the 1D list, make a shallow copy
            self.GivenACFEstablishment_Applicable.append(GivenACFEstablishment_Applicable_i[:])
            # For the 2D lists, make a deep copy of each row
            self.GivenNrLandRescueVehicle_Applicable.append([row[:] for row in GivenNrLandRescueVehicle_Applicable_im])
            self.GivenBackupHospital_Applicable.append([row[:] for row in GivenBackupHospital_Applicable_hhprime])
            
        self.Original_MIPSolver = MIPSolver(
                                        instance=self.Instance,
                                        model=Constants.Two_Stage,
                                        scenariotree=self.ScenarioTree,
                                        nrscenario=self.TreeStructure[1],
                                        givenACFEstablishment=self.GivenACFEstablishment_Applicable,
                                        givenNrLandRescueVehicle=self.GivenNrLandRescueVehicle_Applicable,
                                        givenBackupHospital=self.GivenBackupHospital_Applicable,
                                        evaluatesolution=True, # I set it as True, since, only when the evaluation mode is true, it sets the x values as their Given ones.
                                        logfile="NO")
        
        self.Original_MIPSolver.BuildModel()
        PHA_Final_solution = self.Original_MIPSolver.Solve(True)

        self.CurrentImplementableSolution.PHCost = PHA_Final_solution.TotalCost

        self.CurrentImplementableSolution.PHNrIteration = self.CurrentIteration
        self.WriteInTraceFile("End of PH algorithm ------- Cost: %r ------------ Time (sec): %.2f" % (self.CurrentImplementableSolution.PHCost, self.duration))
        Constants.PrintOnlyFirstStageDecision = self.PrintOnlyFirstStagePreviousValue

        # Round your three first‐stage variables to integers
        Tool.round_nested_list(self.CurrentImplementableSolution.ACFEstablishment_x_wi)
        Tool.round_nested_list(self.CurrentImplementableSolution.LandRescueVehicle_thetaVar_wim)
        Tool.round_nested_list(self.CurrentImplementableSolution.BackupHospital_W_whhPrime)

        return self.CurrentImplementableSolution

    def Check_ACF_Establishment_Budget_Constraint(self, x_var_i, Rounded_x_var_row, RandomRemoval=False):
        """
        Check the budget constraint for ACF establishment. If the sum exceeds the budget, 
        remove the ACFs with the lowest x_var value one by one until the budget is met.
        If RandomRemoval is True, ACFs will be removed randomly instead of based on their capacity.
        """
        # Calculate the initial sum of fixed costs
        total_cost = sum(self.Instance.Fixed_Cost_ACF_Constraint[i] * Rounded_x_var_row[i] for i in self.Instance.ACFSet)
        
        if total_cost <= self.Instance.Total_Budget_ACF_Establishment:
            return Rounded_x_var_row  # If the constraint is satisfied, return the rounded solution as is.
        
        # If RandomRemoval is False, use the existing approach (remove based on smallest x_var values)
        if not RandomRemoval:
            # Filter out the elements of x_var where the value is zero
            value_index_pairs = [(x_var_i[i], i) for i in self.Instance.ACFSet if x_var_i[i] > 0]
            
            # Sort by x_var value in ascending order (smallest values first)
            value_index_pairs.sort()  # Sort based on x_var value
            
            # Iterate and remove the ACFs with the lowest x_var values one by one
            for _, index in value_index_pairs:
                # Set the corresponding Rounded_x_var element to 0
                Rounded_x_var_row[index] = 0
                total_cost = sum(self.Instance.Fixed_Cost_ACF_Constraint[i] * Rounded_x_var_row[i] for i in self.Instance.ACFSet)
                
                # If the budget is satisfied, stop removing
                if total_cost <= self.Instance.Total_Budget_ACF_Establishment:
                    break
        
        # If RandomRemoval is True, remove ACFs randomly until the budget is satisfied
        else:
            acf_indices = [i for i in self.Instance.ACFSet if x_var_i[i] > 0]  # Indices of ACFs that are established
            random.shuffle(acf_indices)  # Shuffle the ACFs to remove them randomly
            
            for i in acf_indices:
                # Set the corresponding Rounded_x_var element to 0
                Rounded_x_var_row[i] = 0
                total_cost = sum(self.Instance.Fixed_Cost_ACF_Constraint[i] * Rounded_x_var_row[i] for i in self.Instance.ACFSet)
                
                # If the budget is satisfied, stop removing
                if total_cost <= self.Instance.Total_Budget_ACF_Establishment:
                    break
        
        # After modifying, if we still don't meet the budget, return the modified rounded x solution
        return Rounded_x_var_row   
    
    def check_connection_between_x_and_thetaVar(self, Rounded_ACFEstablishment_x_i, Rounded_thetaVar_im):
        """
        Ensure that if an ACF is not established (x[0][i] == 0), then no vehicles are assigned to it (thetaVar[0][m][i] == 0).
        """
        for i in self.Instance.ACFSet:
            if Rounded_ACFEstablishment_x_i[i] == 0:  # If ACF i is not established
                for m in self.Instance.RescueVehicleSet:  # For each rescue vehicle
                    Rounded_thetaVar_im[i][m] = 0  # Set vehicles to 0 if the ACF is not established

        return Rounded_thetaVar_im
    
    def check_limited_number_of_rescue_vehicles(self, Rounded_thetaVar_im):
        """
        Ensure that the number of vehicles assigned to each ACF does not exceed the available number of rescue vehicles.
        If exceeded, reduce the assignments starting with ACFs with lower capacities (smallest first).
        """
        for m in self.Instance.RescueVehicleSet:
            # Calculate the total number of vehicles assigned to ACF m in scenario 0
            total_assigned = sum(Rounded_thetaVar_im[i][m] for i in self.Instance.ACFSet)
            
            if total_assigned > self.Instance.Number_Rescue_Vehicle_ACF[m]:
                # If the total number of vehicles exceeds the available capacity, reduce the surplus
                surplus = total_assigned - self.Instance.Number_Rescue_Vehicle_ACF[m]
                
                # Sort the ACFs by their capacity (from lowest to highest)
                acf_with_capacity = [(self.Instance.ACF_Bed_Capacity[i], i) for i in self.Instance.ACFSet]
                acf_with_capacity.sort(key=lambda x: x[0])  # Sort by capacity in ascending order (smallest first)
                
                # Continue reducing surplus by 1 vehicle at a time from each ACF, starting from the lowest capacity
                while surplus > 0:  # Keep looping until the surplus is reduced to 0
                    for _, i in acf_with_capacity:
                        if surplus <= 0:
                            break  # Stop if surplus is reduced to 0
                        if Rounded_thetaVar_im[i][m] > 0:
                            reduction = min(1, Rounded_thetaVar_im[i][m])  # Reduce by 1 at a time
                            Rounded_thetaVar_im[i][m] -= reduction
                            surplus -= reduction

        return Rounded_thetaVar_im

    def Check_LandRescueVehicleAllocation_Constraints(self, Rounded_ACFEstablishment_x_i, Rounded_thetaVar_im):
        """
        Check the constraints for the land rescue vehicle allocation:
        1. Ensure that if an ACF is not established (x = 0), then no vehicles are assigned (thetaVar = 0).
        2. Ensure that the total number of vehicles assigned to each ACF does not exceed the available vehicles.
        """
        # Step 1: Check the connection between x and thetaVar
        Rounded_thetaVar_im = self.check_connection_between_x_and_thetaVar(Rounded_ACFEstablishment_x_i, Rounded_thetaVar_im)
        
        # Step 2: Check and adjust the total number of vehicles assigned to each ACF
        Rounded_thetaVar_im = self.check_limited_number_of_rescue_vehicles(Rounded_thetaVar_im)
        
        return Rounded_thetaVar_im

    def Check_Hospitals_Compatibility_Constraint(self, Rounded_w_hhprime):
        """
        Ensure that if w_[omega][h][h'] = 1, then hospital h' must be compatible with hospital h.
        If not, set w_[omega][h][h'] to 0.
        """
        for h in self.Instance.HospitalSet:
            for h_prime in self.Instance.HospitalSet:
                if Rounded_w_hhprime[h][h_prime] == 1:
                    # Check compatibility
                    if h_prime not in self.Instance.K_h.get(h, set()):
                        Rounded_w_hhprime[h][h_prime] = 0  # Set to 0 if not compatible
        
        return Rounded_w_hhprime