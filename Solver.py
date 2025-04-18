import time
import csv
import datetime
import re
import numpy as np
import os
from Constants import Constants
import gurobipy as gp
from gurobipy import *
from ScenarioTree import ScenarioTree
from MIPSolver import MIPSolver
from ProgressiveHedging import ProgressiveHedging
from ALNS import ALNS
from BranchandBendersCut import BranchandBendersCut



class Solver(object):

    #Constructor
    def __init__(self, instance, testidentifier):
        if Constants.Debug: print("\n We are in 'Solver' Class -- Constructor")
        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.ScenarioGeneration = self.TestIdentifier.ScenarioSampling
        self.GivenACFEstablishment = []
        self.GivenNrLandRescueVehicle = []
        self.GivenBackupHospital = []

        self.TestDescription = self.TestIdentifier.GetAsString()

        self.TreeStructure = self.GetTreeStructure()

    #Define the tree  structur do be used
    def GetTreeStructure(self):
        if Constants.Debug: print("\nWe are in 'Solver' Class -- GetTreeStructure")
        
        if (self.TestIdentifier.Model == Constants.Two_Stage) or (self.TestIdentifier.Model == Constants.Average):
            # Extract the number of scenarios
            intPart_NrScenario = re.findall(r'\d+', self.TestIdentifier.NrScenario)
            intNrScenario = int(intPart_NrScenario[0]) if intPart_NrScenario else 10  # Default to 10 scenarios
            
            # Two-stage structure: [1, Number of Scenarios]
            treestructure = [1, intNrScenario]
            
            if Constants.Debug:
                print("Tree structure for Two-Stage Model:\n", treestructure)
            
            return treestructure


    #This method call the right method
    def Solve(self):
        if Constants.Debug: print("\n We are in 'Solver' Class -- Solve")
        solution = None

        if self.TestIdentifier.Model == Constants.Two_Stage:
            if self.TestIdentifier.Solver == Constants.MIP:
                solution = self.Solve_Use_Two_Stage()

            if self.TestIdentifier.Solver == Constants.ProgressiveHedging:
                Constants.We_Are_in_PHA = True
                self.TreeStructure = self.GetTreeStructure()
                self.ProgressiveHedging = ProgressiveHedging(self.Instance, 
                                                             self.TestIdentifier, 
                                                             self.TreeStructure)
                solution = self.ProgressiveHedging.Run()  
                Constants.We_Are_in_PHA = False   

            if self.TestIdentifier.Solver == Constants.ALNS:
                self.TreeStructure = self.GetTreeStructure()
                self.ALNS = ALNS(self.Instance, 
                                 self.TestIdentifier, 
                                 self.TreeStructure,
                                 use_rl = self.TestIdentifier.ALNSRL,
                                 use_deep_q = self.TestIdentifier.ALNSRL_DeepQ,
                                 selection_method = self.TestIdentifier.RLSelectionMethod)
                solution = self.ALNS.Run()  
        
            if self.TestIdentifier.Solver == Constants.BBC:
                self.TreeStructure = self.GetTreeStructure()
                self.BBC = BranchandBendersCut(self.Instance, 
                                                self.TestIdentifier, 
                                                self.TreeStructure)
                solution = self.BBC.Run()                  

        if self.TestIdentifier.Model == Constants.Average:
            solution = self.Solve_Use_Two_Stage()

        self.PrintSolutionToFile(solution)

        return solution

    def PrintSolutionToFile(self, solution):
        if Constants.Debug: print("\n We are in 'Solver' Class -- PrintSolutionToFile")

        if Constants.Debug: print("------------Moving from 'Solver' Class ('PrintSolutionToFile' Function) to 'TestIdentifier' class (GetAsString))---------------")
        testdescription = self.TestIdentifier.GetAsString()
        if Constants.Debug: print("------------Moving BACK from 'TestIdentifier' class (GetAsString) to 'Solver' Class ('PrintSolutionToFile' Function))---------------")
        
        if Constants.PrintSolutionFileToPickle:
            solution.PrintToPickle(testdescription)
        
        if Constants.PrintSolutionFileToExcel:
            solution.PrintToExcel(testdescription)

    #Solve the two-stage version of the problem    
    def Solve_Use_Two_Stage(self):
        if Constants.Debug: print("\n We are in 'Solver' Class -- Solve_Use_Two_Stage")

        tmpmodel = self.TestIdentifier.Model

        start = time.time()

        average = False

        if Constants.IsDeterministic(self.TestIdentifier.Model):
            average = True
            nrscenario = 1
            self.TestIdentifier.Model = Constants.Average

        # Use these demands to build the optimization model
        treestructure = self.TreeStructure
        solution, mipsolver = self.LocationAllocation(treestructure, average, recordsolveinfo=True)

        end = time.time()
        solution.TotalTime = round((end - start), 2)

        self.Model = tmpmodel

        return solution

    def save_scenario_to_file(self, scenario, instance_name):
        """
        Save the demand parameter from a Scenario object to a text file in the 'Instances' subfolder.

        Parameters:
        - scenario (ScenarioTree): The ScenarioTree object containing the demand data.
        - instance_name (str): Name of the instance, used for the file name.
        """
        # Extract the demand parameter from the Scenario object
        CasualtyDemand = scenario.CasualtyDemand  # (num_scenarios, T, L)
        HospitalDisruption = scenario.HospitalDisruption  # (num_scenarios, H)
        PatientDemand = scenario.PatientDemand  # (num_scenarios, J, H)
        PatientDischargedPercentage = scenario.PatientDischargedPercentage  # (num_scenarios, T, J, U)

        # Specify the directory to save the file
        output_directory = os.path.join(os.getcwd(), "Instances")  # "./Instances"

        # Ensure the directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Define the file name and path
        file_name = f"{instance_name}_Scenarios.txt"
        file_path = os.path.join(output_directory, file_name)

        with open(file_path, 'w') as file:
            num_scenarios = CasualtyDemand.shape[0]

            for s in range(num_scenarios):
                file.write(f"\n=====================================\n")
                file.write(f"Scenario {s + 1}:\n")
                file.write(f"=====================================\n\n")

                #Save CasualtyDemand (3D → 2D)
                file.write(f"CasualtyDemand:\n")
                np.savetxt(file, CasualtyDemand[s].reshape(-1, CasualtyDemand.shape[-1]), fmt='%d', delimiter=' ')
                file.write("\n")

                #Save HospitalDisruption (2D → 1D)
                file.write(f"HospitalDisruption:\n")
                np.savetxt(file, HospitalDisruption[s].reshape(1, -1), fmt='%d', delimiter=' ')
                file.write("\n")

                #Save PatientDemand (3D → 2D)
                file.write(f"PatientDemand:\n")
                np.savetxt(file, PatientDemand[s].reshape(-1, PatientDemand.shape[-1]), fmt='%d', delimiter=' ')
                file.write("\n")

                #Save PatientDischargedPercentage (4D → 2D)
                file.write(f"PatientDischargedPercentage:\n")
                reshaped_patient_discharged = PatientDischargedPercentage[s].reshape(-1, PatientDischargedPercentage.shape[-1])
                np.savetxt(file, reshaped_patient_discharged, fmt='%.4f', delimiter=' ')
                file.write("\n")

        print(f"Scenarios saved to {file_path}")

    def LocationAllocation(self, treestructur, averagescenario=False, recordsolveinfo=False):    
        if Constants.Debug: print("\n We are in 'Solver' Class -- CRP")
        scenariotreemodel = self.TestIdentifier.Model

        # Generate scenarios
        Scenario = ScenarioTree(instance=self.Instance,
                                tree_structure=self.TreeStructure,
                                scenario_seed=self.TestIdentifier.ScenarioSeed,
                                averagescenariotree=averagescenario,
                                scenariogenerationmethod=self.ScenarioGeneration)
        print("CasualtyDemand:\n", Scenario.CasualtyDemand)
        print("HospitalDisruption:\n", Scenario.HospitalDisruption)
        print("PatientDemand:\n", Scenario.PatientDemand)
        print("PatientDischargedPercentage:\n", Scenario.PatientDischargedPercentage)

        # Save scenarios to file
        if Constants.Debug:
            self.save_scenario_to_file(Scenario, self.TestIdentifier.InstanceName)

        MIPModel = self.TestIdentifier.Model

        if self.TestIdentifier.Model == Constants.Average:
            Scenario.TreeStructure[1] = 1
            Scenario.Probability = 1
            mipsolver = MIPSolver(instance = self.Instance, 
                                    model = MIPModel, 
                                    scenariotree = Scenario,
                                    nrscenario = 1,
                                    givenACFEstablishment = self.GivenACFEstablishment,
                                    givenNrLandRescueVehicle = self.GivenNrLandRescueVehicle,
                                    givenBackupHospital = self.GivenBackupHospital,
                                    logfile=self.TestDescription)
        else:
            mipsolver = MIPSolver(instance = self.Instance, 
                                model = MIPModel, 
                                scenariotree = Scenario,
                                nrscenario = Scenario.TreeStructure[1],
                                givenACFEstablishment = self.GivenACFEstablishment,
                                givenNrLandRescueVehicle = self.GivenNrLandRescueVehicle,
                                givenBackupHospital = self.GivenBackupHospital,
                                logfile = self.TestDescription) 

        if Constants.Debug: print("Start to model in Gurobi")  

        mipsolver.BuildModel()

        solution = mipsolver.Solve()

        return solution, mipsolver