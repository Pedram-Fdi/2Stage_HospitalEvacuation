from __future__ import absolute_import, division, print_function
from Constants import Constants
from Solution import Solution
from EvaluatorIdentificator import EvaluatorIdentificator
from EvaluationSimulator import EvaluationSimulator
import subprocess
import pickle
import pandas as pd
import csv
import datetime

#This class contains the method to call the simulator and run the evauation
class Evaluator( object ):

    # Constructor
    def __init__(self, instance, testidentifier, evaluatoridentificator, solver):
        if Constants.Debug: print("\n We are in 'Evaluator' Class -- Constructor")        
        self.TestIdentifier = testidentifier
        self.AttentionModelInTestIdentifierChanged = False
        self.EvalutorIdentificator = evaluatoridentificator
        self.Solutions = self.GetPreviouslyFoundSolution()
        self.Instance = instance
        self.Solver = solver
        self.OutOfSampleTestResult = []
        self.InSampleTestResult =[]

    #Return the solution to evaluate
    def GetPreviouslyFoundSolution(self):
        if Constants.Debug: print("\n We are in 'Evaluator' Class -- GetPreviouslyFoundSolution")        

        result = []
        seeds = [self.TestIdentifier.ScenarioSeed]
        for s in seeds:
            try:
                self.TestIdentifier.ScenarioSeed = s
                if Constants.Debug: print("self.TestIdentifier.ScenarioSeed: ", self.TestIdentifier.ScenarioSeed)

                filedescription = self.TestIdentifier.GetAsString()
                if Constants.Debug: print("filedescription: ", filedescription)

                solution = Solution()
                solution.ReadFromFile(filedescription, self.TestIdentifier)
                result.append(solution)

            except IOError:
                if Constants.Debug:
                    print(IOError)
                    print("No solution found for seed %d" % s)

        return result

    #Get the temporary file containing the results of the simulation
    def GetEvaluationFileName(self):
        if Constants.Debug: print("\n We are in 'Evaluator' Class -- GetEvaluationFileName")

        result = Constants.GetEvaluationFolder() + self.TestIdentifier.GetAsString() + '_' + self.EvalutorIdentificator.GetAsString()
        
        return result
    
    #run the simulation
    def EvaluateSingleSol(self):
        if Constants.Debug: print("\n We are in 'Evaluator' Class -- EvaluateSingleSol")
        tmpmodel = self.TestIdentifier.Model
        filedescription = self.TestIdentifier.GetAsString()

        self.AttentionModelInTestIdentifierChanged = False
        MIPModel = self.TestIdentifier.Model
        if Constants.IsDeterministic(self.TestIdentifier.Model):
            MIPModel = Constants.Two_Stage
            if Constants.Debug: print("Changed to Deterministic Model: Two_Stage")

        solution = Solution()

        if Constants.RunEvaluationInSeparatedJob:
            if Constants.Debug: print("Reading solution from file")
            solution.ReadFromFile(filedescription, self.TestIdentifier)
        else:
            solution = self.GetPreviouslyFoundSolution()[0]

            if not solution.IsPartialSolution:
                if Constants.Debug: print("Solution is not partial, computing cost")
                solution.ComputeCost()

        evaluator = EvaluationSimulator(self.Instance, [solution],
                                        testidentificator=self.TestIdentifier,
                                        evaluatoridentificator=self.EvalutorIdentificator,
                                        treestructure=self.Solver.GetTreeStructure(),
                                        model=MIPModel)
        
        self.TestIdentifier.Model = tmpmodel

        self.OutOfSampleTestResult = evaluator.EvaluateXFixSolution(saveevaluatetab=True,
                                                                     filename=self.GetEvaluationFileName())


        self.GatherEvaluation()

    def GatherEvaluation(self):
        if Constants.Debug: print("\n We are in 'Evaluator' Class -- GatherEvaluation")
        
        evaluator = EvaluationSimulator(self.Instance, 
                                        solutions=[], 
                                        testidentificator=self.TestIdentifier,
                                        evaluatoridentificator=self.EvalutorIdentificator,
                                        treestructure=self.Solver.GetTreeStructure())
        EvaluationTab = []
        ProbabilitiesTab =[]
        KPIStats = []
        nrfile = 0
        #Creat the evaluation table
        currentseedvalue = self.TestIdentifier.ScenarioSeed
        for seed in [self.TestIdentifier.ScenarioSeed]:#SeedArray:
            filename = self.GetEvaluationFileName()
            if Constants.Debug: print("filename: ", filename)
            try:

                self.TestIdentifier.ScenarioSeed = seed
                #print "open file %rEvaluator.txt"%filename
                with open(filename + "_Evaluator.txt", 'rb') as f:
                    list = pickle.load(f)
                    EvaluationTab.append(list)
                    
                with open(filename + "_Probabilities.txt", 'rb') as f:
                    list = pickle.load(f)
                    ProbabilitiesTab.append(list)

                with open(filename + "_KPIStat.txt", "rb") as f:  # Pickling
                    list = pickle.load(f)
                    KPIStats.append(list)
                    nrfile =nrfile +1
            except IOError:
               # if Constants.Debug:
                    if Constants.Debug: print("No evaluation file found for seed %d %r" %(seed, filename))

        if nrfile >= 1:
            KPIStat = [sum(e) / len(e) for e in zip(*KPIStats)]

            self.OutOfSampleTestResult = evaluator.ComputeStatistic(EvaluationTab, ProbabilitiesTab, KPIStat, -1)
            
            self.InSampleTestResult = self.ComputeInSampleStatistis()
            
            self.PrintFinalResult()

        self.TestIdentifier.ScenarioSeed = currentseedvalue

    def PrintFinalResult(self):
        if Constants.Debug: print("\n We are in 'Evaluator' Class -- PrintFinalResult")

        # Data for the "Generic Information" sheet
        generic_data = self.TestIdentifier.GetAsStringList() + self.EvalutorIdentificator.GetAsStringList()
        generic_columns = ["Instance", "Model", "Solver", "ScenarioGeneration", "NrScenario", "ScenarioSeed", 
                           "PHAObj", "PHAPenalty", "ALNSRL", "ALNSRL_DeepQ", "RLSelectionMethod", "BBC_Accelerator", 
                           "ClusteringMethod", "All Scenario", "NrEvaluation", "Policy Generation", "Time Horizon"]
        generic_df = pd.DataFrame([generic_data], columns=generic_columns) 

        # Data for the "InSample" sheet
        # Ensure self.InSampleTestResult is structured correctly for DataFrame creation
        insample_columns = ["GRB Cost", "GRB Time", "GRB Gap(%)", "GRB Nr Constraints", "GRB Nr Variables", 
                            "PHA Cost", "PHA Nr Iteration", "Total Time", 
                            "ACF Establishment Cost", "Land Rescue Vehicle Assign. Cost", "Backup Hospital Cost", 
                            "Casualty Transfer Cost", "UnsatisfiedCasualties Cost", "DischargedPatients Cost",
                            "LandEvacuatedPatients Cost", "AerialEvacuatedPatients Cost", "UnevacuatedPatient Cost",
                            "AvailableCapFacility Cost",  
                            "% On-Time Transfer", "% On-Time Evacuation", "% Not Evacuated", 
                            "Nr ACF Established", "Nr Land Res. Vehicle Assigned", "Nr Backup Hospitals",
                            "Evaluation Duration"]
        insample_df = pd.DataFrame([self.InSampleTestResult], columns=insample_columns)

        # Data for the "OutOfSample" sheet
        # Ensure self.OutOfSampleTestResult is structured correctly for DataFrame creation
        outofsample_columns = ["Mean", "LB", "UB", "MinAverage", "MaxAverage", "nrerror", "GRB Cost", "GRB Time", "GRB Gap(%)", 
                                "GRB Nr Constraints", "GRB Nr Variables", "PHA Cost", "PHA Nr Iteration",
                                "Total Time", 
                                "ACF Establishment Cost", "Land Rescue Vehicle Assign. Cost", "Backup Hospital Cost", 
                                "Casualty Transfer Cost", "UnsatisfiedCasualties Cost", "DischargedPatients Cost",
                                "LandEvacuatedPatients Cost", "AerialEvacuatedPatients Cost", "UnevacuatedPatient Cost",
                                "AvailableCapFacility Cost",
                                "% On-Time Transfer", "% On-Time Evacuation", "% Not Evacuated", 
                                "Nr ACF Established", "Nr Land Res. Vehicle Assigned", "Nr Backup Hospitals", 
                                "Evaluation Duration"]
        outofsample_df = pd.DataFrame([self.OutOfSampleTestResult], columns=outofsample_columns)

        # Define file path
        file_path = f'./Test/TestResult_{self.TestIdentifier.GetAsString()}_{self.EvalutorIdentificator.GetAsString()}.xlsx'

        # Write data to different sheets within the same Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            generic_df.to_excel(writer, sheet_name='Generic Information', index=False)
            insample_df.to_excel(writer, sheet_name='InSample', index=False)
            outofsample_df.to_excel(writer, sheet_name='OutOfSample', index=False)


    # return a set of statistic associated with solving the problem
    def ComputeInSampleStatistis(self):
        if Constants.Debug: print("\n We are in 'Evaluator' Class -- ComputeInSampleStatistis") 
        InSampleKPIStat = []

        solutions = self.GetPreviouslyFoundSolution()
        lengthinsamplekpi = -1

        for solution in solutions:
            if not Constants.PrintOnlyFirstStageDecision:
                solution.ComputeStatistics()
            
            insamplekpisstate = solution.PrintStatistics(self.TestIdentifier, "InSample", -1, 0, -1, True, self.TestIdentifier.ScenarioSampling)

            lengthinsamplekpi = len(insamplekpisstate)
            InSampleKPIStat = [0] * lengthinsamplekpi
            
            for i in range(lengthinsamplekpi):
                InSampleKPIStat[i] = InSampleKPIStat[i] + insamplekpisstate[i]

        for i in range(lengthinsamplekpi):
            InSampleKPIStat[i] = InSampleKPIStat[i] / len(solutions)

        return InSampleKPIStat
    
    #This function runs the evaluation for the just completed test :
    def RunEvaluation(self):
        if Constants.Debug: print("\n We are in 'Evaluator' Class -- RunEvaluation")

        if self.EvalutorIdentificator.NrEvaluation > 0:
            policyset = ["Re-solve"]

            perfectsenarioset = [0]
            if self.Instance.Distribution == Constants.Binomial:
                perfectsenarioset = [0, 1]
            for policy in policyset:
                for perfectset in perfectsenarioset:
                    if Constants.RunEvaluationInSeparatedJob:
                        jobname = "./Jobs/job_evaluate_%s_%s_%s_%s_%s_%s_%s" % (
                            self.TestIdentifier.InstanceName,
                            self.TestIdentifier.Model,
                            self.TestIdentifier.NrScenario,
                            self.TestIdentifier.ScenarioSampling,
                            self.TestIdentifier.Method,
                            policy,
                            self.TestIdentifier.ScenarioSeed)
                        subprocess.call(["qsub", jobname])
                    else:
                        PolicyGeneration = policy
                        NearestNeighborStrategy = policy
                        AllScenario = perfectset
                        if AllScenario == 1:
                            NrEvaluation = 4096
                        else:
                            NrEvaluation = self.EvalutorIdentificator.NrEvaluation

                        self.EvalutorIdentificator = EvaluatorIdentificator(policy,
                                                                            NrEvaluation,
                                                                            self.EvalutorIdentificator.TimeHorizon,
                                                                            perfectset)

                        self.EvaluateSingleSol()
        else:
            self.InSampleTestResult = self.ComputeInSampleStatistis()
            self.PrintFinalResult()