from ScenarioTree import ScenarioTree
from Constants import Constants
from MIPSolver import MIPSolver
from Solution import Solution
import gurobipy as gp
from gurobipy import *
from Scenario import Scenario
from BendersCutCallback import BendersCutCallback
import numpy as np
import os
import time

class BranchandBendersCut:
    def __init__(self, 
                 instance, 
                 testidentifier, 
                 treestructure,
                 scenariotree=None,
                 givenfacilityestablishments=[]):
        """
        Initialize the ALNS class.
        :param instance: Instance of the problem.
        :param testidentifier: Test configuration for ALNS.
        :param treestructure: Tree structure for the scenarios.
        """
        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure
        self.CurrentIteration = 0


        # Generate scenarios
        self.GenerateScenarios(scenariotree)

        # The variable StartFixedTransVariables and ... gives the index at which each variable type start
        self.StartFacilityEstablishmentVariables = 0
        self.StartProductionVariables = 0
        self.StartShortageVariables = 0
        self.StartthetaVariables = 0

        if (Constants.UseLBF):
            self.StartProductionVariables_LBF = 0
            self.StartShortageVariables_LBF = 0

        # Required lists for Facility Capacity Constraint
        self.FacilityCapacityConstraint_Objects = [[None 
                                                    for i in self.Instance.FacilitySet] 
                                                    for t in self.Instance.TimeBucketSet] 
        self.PIFacilityCapacityConstraint_Names = []
        self.Concerned_t_FacilityCapacityConstraint = []
        self.Concerned_i_FacilityCapacityConstraint = []


        # Required lists for demand Flow Constraint
        self.DemandFlowConstraint_Objects = [[None 
                                                for J in self.Instance.DemandSet] 
                                                for t in self.Instance.TimeBucketSet] 
        self.PIDemandFlowConstraint_Names = []
        self.Concerned_t_DemandFlowConstraint = []
        self.Concerned_j_DemandFlowConstraint = []        

        self.ComputeIndices()

        # Build Master model
        self.BuildMaster()

        # Build Master model
        self.BuildSubProblem()

        self.TraceFileName = "./Temp/BBCtrace_%s_Evaluation_%s.txt" % (self.TestIdentifier.GetAsString(), Constants.Evaluation_Part)
        
        # For saving the final First-stage solution for all scenarios (its repeated in fact)
        self.solfacilityEstablishment = [[-1 for i in self.Instance.FacilitySet]
                                            for w in self.ScenarioSet]
        
    def InitTrace(self):
        if Constants.Debug: print("\n We are in 'BranchandBendersCut' Class -- InitTrace")
        if Constants.PrintPHATrace:
            self.TraceFile = open(self.TraceFileName, "w")
            self.TraceFile.write("Start the BBC algorithm \n")
            self.TraceFile.close()

    def WriteInTraceFile(self, string):
        if Constants.Debug: print("\n We are in 'BranchandBendersCut' Class -- WriteInTraceFile")

        if Constants.PrintPHATrace:
            self.TraceFile = open(self.TraceFileName, "a")
            self.TraceFile.write(string)
            self.TraceFile.close()

    def GenerateScenarios(self, scenariotree=None):

        if Constants.Debug:
            print("\n We are in 'ALNS' Class -- GenerateScenarios")

        if scenariotree is None:
            self.ScenarioTree = ScenarioTree(
                instance=self.Instance,
                tree_structure=self.TreeStructure,
                scenario_seed=self.TestIdentifier.ScenarioSeed,
                scenariogenerationmethod=self.TestIdentifier.ScenarioSampling
            )
        else:
            self.ScenarioTree = scenariotree
        
        self.NrScenario = self.ScenarioTree.TreeStructure[1]
        self.ScenarioProbability = (1/self.NrScenario)    
        self.ScenarioSet = range(self.NrScenario)    

    # Compute the start of index and the number of variables for the considered instance
    def ComputeIndices(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- ComputeIndices")

        self.NrFacilityLocationVariables = self.Instance.NrFacilities  
        self.NrProductionVariables = self.Instance.NrTimeBucket * self.Instance.NrFacilities *  self.Instance.NrDemandLocations                                                                                  #We do not have NrApheresisAssignmentVariables var. in t=0!
        self.NrShortageVariables = self.Instance.NrTimeBucket * self.Instance.NrDemandLocations                                                                                  #We do not have NrApheresisAssignmentVariables var. in t=0!
        if(Constants.UseMultiCut):
            self.NrthetaVariables = self.NrScenario
        else:
            self.NrthetaVariables = 1
        
        if(Constants.UseLBF):
            self.NrProductionVariables_LBF = self.Instance.NrTimeBucket * self.Instance.NrFacilities *  self.Instance.NrDemandLocations                                                                                  #We do not have NrApheresisAssignmentVariables var. in t=0!
            self.NrShortageVariables_LBF = self.Instance.NrTimeBucket * self.Instance.NrDemandLocations 


        self.StartFacilityEstablishmentVariables = 0
        self.StartProductionVariables = self.StartFacilityEstablishmentVariables + self.NrFacilityLocationVariables
        self.StartShortageVariables = self.StartProductionVariables + self.NrProductionVariables
        self.StartthetaVariables = self.StartShortageVariables + self.NrShortageVariables

        if(Constants.UseLBF):
            self.StartProductionVariables_LBF = self.StartthetaVariables + self.NrthetaVariables
            self.StartShortageVariables_LBF = self.StartProductionVariables_LBF + self.NrProductionVariables_LBF            


        if Constants.Debug:
            print("StartFacilityEstablishmentVariables: ", self.StartFacilityEstablishmentVariables)
            print("NrFacilityLocationVariables: ", self.NrFacilityLocationVariables)
            print("StartProductionVariables: ", self.StartProductionVariables)
            print("NrProductionVariables: ", self.NrProductionVariables)
            print("StartShortageVariables: ", self.StartShortageVariables)
            print("NrShortageVariables: ", self.NrShortageVariables)
            print("StartthetaVariables: ", self.StartthetaVariables)
            print("NrthetaVariables: ", self.NrthetaVariables)
            if(Constants.UseLBF):
                print("StartProductionVariables_LBF: ", self.StartProductionVariables_LBF)
                print("NrProductionVariables_LBF: ", self.NrProductionVariables_LBF)
                print("StartShortageVariables_LBF: ", self.StartShortageVariables_LBF)
                print("NrShortageVariables_LBF: ", self.NrShortageVariables_LBF)

    def GetStartFacilityEstablishmentVariables(self):
        return self.StartFacilityEstablishmentVariables
    
    def GetStartthetaVariables(self):
        return self.StartthetaVariables

    def GetStartProductionVariable(self):
        return self.StartProductionVariables
    
    def GetStartProductionVariable_LBF(self):
        return self.StartProductionVariables_LBF

    def GetStartShortageVariable(self):
        return self.StartShortageVariables
            
    def GetStartShortageVariable_LBF(self):
        return self.StartShortageVariables_LBF
            
    # the function GetIndexACFEstablishmentVariable returns the index of the variable x_{i}
    def GetIndexFacilityEstablishmentVariable(self, i):
        # if Constants.Debug: print("We are in 'MIPSolver' Class -- GetIndexFacilityEstablishmentVariable")
        return self.StartFacilityEstablishmentVariables + i
    
    def GetIndexthetaVariable(self, w):
        if(Constants.UseMultiCut):
            return self.StartthetaVariables + w
        else:
            return self.StartthetaVariables

    def GetIndexProductionVariable(self, t, i, j):

        return self.StartProductionVariables \
            + t * self.Instance.NrFacilities * self.Instance.NrDemandLocations \
            + i * self.Instance.NrDemandLocations \
            + j
    
    def GetIndexProductionVariable_LBF(self, t, i, j):

        return self.StartProductionVariables_LBF \
            + t * self.Instance.NrFacilities * self.Instance.NrDemandLocations \
            + i * self.Instance.NrDemandLocations \
            + j
    
    def GetIndexShortageVariable(self, t, j):

        return self.StartShortageVariables \
            + t * self.Instance.NrDemandLocations \
            + j
        
    def GetIndexShortageVariable_LBF(self, t, j):

        return self.StartShortageVariables_LBF \
            + t * self.Instance.NrDemandLocations \
            + j
        
    def GetfacilityestablishmentCoeff(self, i):
        return self.Instance.Fixed_Establishment_Cost[i]

    def GetProductionCoeff(self, t, i, j):
        return self.Instance.Transportation_Cost[i][j]

    def GetShortageCoeff(self, t, j):
        return self.Instance.Unserved_Penalty_Cost[j]
        
    def PrintAllVariables_Master(self):
        print("\n### Facility Establishment Variables ###")
        for index, var in self.Facility_Establishment_Var.items():
            print(f"Index: {index}, Variable: {var.VarName}, Obj Coeff: {var.Obj}, Type: {var.VType}")

        print("\n### theta Variables ###")
        for index, var in self.theta_Var.items():
            print(f"Index: {index}, Variable: {var.VarName}, Obj Coeff: {var.Obj}, Type: {var.VType}")

    def PrintAllVariables_Sub(self):
        print("\n### Production Variables ###")
        for index, var in self.Production_Var.items():
            print(f"Index: {index}, Variable: {var.VarName}, Obj Coeff: {var.Obj}, Type: {var.VType}")

        print("\n### Shortage Variables ###")
        for index, var in self.Shortage_Var.items():
            print(f"Index: {index}, Variable: {var.VarName}, Obj Coeff: {var.Obj}, Type: {var.VType}")

    def CreateVariable_and_Objective_Function_Master(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateVariable_and_Objective_Function_Master")

        ################################# Facility Establishment Variable #################################
        facilityestablishmentcost = {}
        self.Facility_Establishment_Var = {}
        for i in self.Instance.FacilitySet:                
            Index_Cost = self.GetIndexFacilityEstablishmentVariable(i) - self.GetStartFacilityEstablishmentVariables()               
            facilityestablishmentcost[Index_Cost] = self.GetfacilityestablishmentCoeff(i)
            Index_Var = self.GetIndexFacilityEstablishmentVariable(i)
            var_name = f"x_i_{i}_index_{Index_Var}"
            self.Facility_Establishment_Var[Index_Var] = self.MasterModel.addVar(vtype=GRB.BINARY, obj=facilityestablishmentcost[Index_Cost], lb=0, ub=1, name=var_name)
        self.MasterModel.update()


        ################################# theta Variable #################################
        self.theta_Var = {}
        if(Constants.UseMultiCut):
            for w in self.ScenarioSet:                
                Index_Var = self.GetIndexthetaVariable(w)
                var_name = f"theta_w_{w}_index_{Index_Var}"
                self.theta_Var[Index_Var] = self.MasterModel.addVar(vtype=GRB.CONTINUOUS, obj=self.ScenarioProbability, lb=0, name=var_name)
        else:
            Index_Var = self.GetIndexthetaVariable(0)
            var_name = f"theta_index_{Index_Var}"
            self.theta_Var[Index_Var] = self.MasterModel.addVar(vtype=GRB.CONTINUOUS, obj=1, lb=0, name=var_name)
        self.MasterModel.update()

        self.Production_Var_LBF = {}
        self.Shortage_Var_LBF = {}
        if (Constants.UseLBF):
            ############################################## Variables for LBF Accelerator
            ################################# Production Variable #################################
            for t in self.Instance.TimeBucketSet:  
                for i in self.Instance.FacilitySet:
                    for j in self.Instance.DemandSet:
                        Index_Cost = self.GetIndexProductionVariable_LBF(t, i, j) - self.GetStartProductionVariable_LBF()
                        Index_Var = self.GetIndexProductionVariable_LBF(t, i, j)
                        var_name = f"y_LBF_t_{t}_i_{i}_j_{j}_index_{Index_Var}"
                        self.Production_Var_LBF[Index_Var] = self.MasterModel.addVar(vtype=GRB.CONTINUOUS, obj=0, lb=0, ub=GRB.INFINITY, name=var_name)
            self.MasterModel.update()
            
            ################################# Shortage Variable #################################
            for t in self.Instance.TimeBucketSet:  
                for j in self.Instance.DemandSet:
                    Index_Cost = self.GetIndexShortageVariable_LBF(t, j) - self.GetStartShortageVariable_LBF()
                    Index_Var = self.GetIndexShortageVariable_LBF(t, j)
                    var_name = f"z_LBF_t_{t}_j_{j}_index_{Index_Var}"
                    self.Shortage_Var_LBF[Index_Var] = self.MasterModel.addVar(vtype=GRB.CONTINUOUS, obj=0, lb=0, ub=GRB.INFINITY, name=var_name)
            self.MasterModel.update()


        ############################################## Set minimization
        self.MasterModel.setObjective(self.MasterModel.getObjective(), GRB.MINIMIZE)

        if Constants.Debug:
            self.PrintAllVariables_Master()

    # Define the constraint of the model
    def CreateConstraints_Master(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateConstraints_Master")

        ## Add any constraints related to Master here
        #There is no constraint in the MP for this model at the beginning!
        
        if(Constants.UseLBF):
            ## Adding sub-problem constraints only for Average Scenario
            self.CreatethetaLBFConstraint()
            self.CreateACFTreatmentCapacityConstraint_LBF()
            self.CreateDemandFlowConstraint_LBF() 

    def CreatethetaLBFConstraint(self):
        if(Constants.UseMultiCut):
            for w in self.ScenarioSet:
                var_theta = [self.GetIndexthetaVariable(w)]

                coeff_theta = [1.0]
                
                ### The objective function of the sub problem for the average scenario
                vars_y_LBF = [self.GetIndexProductionVariable_LBF(t, i, j)
                                for j in self.Instance.DemandSet
                                for i in self.Instance.FacilitySet
                                for t in self.Instance.TimeBucketSet]
                
                coeff_y_LBF = [-1.0 * self.ScenarioProbability * self.Instance.Transportation_Cost[i][j]
                                for j in self.Instance.DemandSet
                                for i in self.Instance.FacilitySet
                                for t in self.Instance.TimeBucketSet]     

                vars_z_LBF = [self.GetIndexShortageVariable_LBF(t, j)
                                for j in self.Instance.DemandSet
                                for t in self.Instance.TimeBucketSet]
                
                coeff_z_LBF = [-1.0 * self.ScenarioProbability * self.Instance.Unserved_Penalty_Cost[j]
                                for j in self.Instance.DemandSet
                                for t in self.Instance.TimeBucketSet]    
                                                              
                
                ############ Create the left-hand side of the constraint       
                LeftHandSide_theta = gp.quicksum(coeff_theta[i] * self.theta_Var[var_theta[i]] for i in range(len(var_theta)))
                
                LeftHandSide_y = gp.quicksum(coeff_y_LBF[i] * self.Production_Var_LBF[vars_y_LBF[i]] for i in range(len(vars_y_LBF)))

                LeftHandSide_z = gp.quicksum(coeff_z_LBF[i] * self.Shortage_Var_LBF[vars_z_LBF[i]] for i in range(len(vars_z_LBF)))

                LeftHandSide = LeftHandSide_theta + LeftHandSide_y + LeftHandSide_z
                
                ############ Define the right-hand side (RHS) of the constraint
                RightHandSide =  0
                
                ############ Add the constraint to the model
                constraint_name = f"theta_LBF_w_{w}"
                
                self.MasterModel.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
        else:
            var_theta = [self.GetIndexthetaVariable(0)]

            coeff_theta = [1.0]

            ### The objective function of the sub problem for the average scenario
            vars_y_LBF = [self.GetIndexProductionVariable_LBF(t, i, j)
                            for j in self.Instance.DemandSet
                            for i in self.Instance.FacilitySet
                            for t in self.Instance.TimeBucketSet]
            
            coeff_y_LBF = [-1.0 * self.ScenarioProbability * self.Instance.Transportation_Cost[i][j]
                            for j in self.Instance.DemandSet
                            for i in self.Instance.FacilitySet
                            for t in self.Instance.TimeBucketSet]     

            vars_z_LBF = [self.GetIndexShortageVariable_LBF(t, j)
                            for j in self.Instance.DemandSet
                            for t in self.Instance.TimeBucketSet]
            
            coeff_z_LBF = [-1.0 * self.ScenarioProbability * self.Instance.Unserved_Penalty_Cost[j]
                            for j in self.Instance.DemandSet
                            for t in self.Instance.TimeBucketSet]    
                                                            
            
            ############ Create the left-hand side of the constraint       
            LeftHandSide_theta = gp.quicksum(coeff_theta[i] * self.theta_Var[var_theta[i]] for i in range(len(var_theta)))
            
            LeftHandSide_y = gp.quicksum(coeff_y_LBF[i] * self.Production_Var_LBF[vars_y_LBF[i]] for i in range(len(vars_y_LBF)))

            LeftHandSide_z = gp.quicksum(coeff_z_LBF[i] * self.Shortage_Var_LBF[vars_z_LBF[i]] for i in range(len(vars_z_LBF)))

            LeftHandSide = LeftHandSide_theta + LeftHandSide_y + LeftHandSide_z
            
            ############ Define the right-hand side (RHS) of the constraint
            RightHandSide =  0
            
            ############ Add the constraint to the model
            constraint_name = f"theta_LBF"
            
            self.MasterModel.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)

    def CreateACFTreatmentCapacityConstraint_LBF(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateACFTreatmentCapacityConstraint_LBF")

        for t in self.Instance.TimeBucketSet:
            for i in self.Instance.FacilitySet:

                vars_x = [self.GetIndexFacilityEstablishmentVariable(i)]
                
                coeff_x = [-1.0 * self.Instance.Facility_Capacity[i]]  

                vars_y = [self.GetIndexProductionVariable_LBF(t, i, j)
                            for j in self.Instance.DemandSet]
                
                coeff_y = [1.0 
                            for j in self.Instance.DemandSet]                                    
                
                ############ Create the left-hand side of the constraint       
                LeftHandSide_x = gp.quicksum(coeff_x[i] * self.Facility_Establishment_Var[vars_x[i]] for i in range(len(vars_x)))

                LeftHandSide_y = gp.quicksum(coeff_y[i] * self.Production_Var_LBF[vars_y[i]] for i in range(len(vars_y)))

                LeftHandSide = LeftHandSide_x + LeftHandSide_y
                
                ############ Define the right-hand side (RHS) of the constraint
                RightHandSide =  0
                
                ############ Add the constraint to the model
                constraint_name = f"FacilityCapacity_LBF_t_{t}_i_{i}"
                
                self.MasterModel.addConstr(LeftHandSide <= RightHandSide, name=constraint_name)

    def CreateDemandFlowConstraint_LBF(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateDemandFlowConstraint_LBF")

        for t in self.Instance.TimeBucketSet:
            for j in self.Instance.DemandSet:

                vars_y = [self.GetIndexProductionVariable_LBF(t, i, j) 
                          for i in self.Instance.FacilitySet]
                
                coeff_y = [1.0  for i in self.Instance.FacilitySet]

                vars_z = [self.GetIndexShortageVariable_LBF(t, j)]
                coeff_z = [1.0]

                ############ Create the left-hand side of the constraint
                LeftHandSide_y = gp.quicksum(coeff_y[i] * self.Production_Var_LBF[vars_y[i]] for i in range(len(vars_y)))
                LeftHandSide_z = gp.quicksum(coeff_z[i] * self.Shortage_Var_LBF[vars_z[i]] for i in range(len(vars_z)))
                
                LeftHandSide = LeftHandSide_y + LeftHandSide_z
                
                ############ Define the right-hand side (RHS) of the constraint
                RightHandSide = self.ScenarioTree.Demand_LBF[0][t][j] 

                ############ Add the constraint to the model
                constraint_name = f"DemandFlow_LBF_t_{t}_j_{j}"
                self.MasterModel.addConstr(LeftHandSide == RightHandSide, name=constraint_name)

    def CreateACFTreatmentCapacityConstraint(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateACFTreatmentCapacityConstraint")

        for t in self.Instance.TimeBucketSet:
            for i in self.Instance.FacilitySet:
                
                vars_y = [self.GetIndexProductionVariable(t, i, j)
                            for j in self.Instance.DemandSet]
                
                coeff_y = [-1.0 
                            for j in self.Instance.DemandSet]                                   
                
                ############ Create the left-hand side of the constraint       
                LeftHandSide_y = gp.quicksum(coeff_y[i] * self.Production_Var[vars_y[i]] for i in range(len(vars_y)))
                                                    
                LeftHandSide = LeftHandSide_y
                
                ############ Define the right-hand side (RHS) of the constraint
                RightHandSide = -1.0 * self.Instance.Facility_Capacity[i]
                
                ############ Add the constraint to the model
                constraint_name = f"FacilityCapacity_t_{t}_i_{i}"
                
                constraint = self.SubModel.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                
                self.FacilityCapacityConstraint_Objects[t][i] = constraint
                self.PIFacilityCapacityConstraint_Names.append(constraint_name)
                self.Concerned_t_FacilityCapacityConstraint.append(t)
                self.Concerned_i_FacilityCapacityConstraint.append(i)

    def CreateDemandFlowConstraint(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateDemandFlowConstraint")

        for t in self.Instance.TimeBucketSet:
            for j in self.Instance.DemandSet:

                vars_y = [self.GetIndexProductionVariable(t, i, j) for i in self.Instance.FacilitySet]
                coeff_y = [1.0  for i in self.Instance.FacilitySet]

                vars_z = [self.GetIndexShortageVariable(t, j)]
                coeff_z = [1.0]

                ############ Create the left-hand side of the constraint
                LeftHandSide_y = gp.quicksum(coeff_y[i] * self.Production_Var[vars_y[i]] for i in range(len(vars_y)))
                LeftHandSide_z = gp.quicksum(coeff_z[i] * self.Shortage_Var[vars_z[i]] for i in range(len(vars_z)))
                
                LeftHandSide = LeftHandSide_y + LeftHandSide_z
                
                ############ Define the right-hand side (RHS) of the constraint
                # For now just use scenario 0 data -- it will be reset during algorithm as it loops through scnearios
                RightHandSide = self.ScenarioTree.Demand[0][t][j] 

                ############ Add the constraint to the model
                constraint_name = f"DemandFlow_t_{t}_j_{j}"
                constraint = self.SubModel.addConstr(LeftHandSide == RightHandSide, name=constraint_name)

                self.DemandFlowConstraint_Objects[t][j] = constraint
                self.PIDemandFlowConstraint_Names.append(constraint_name)
                self.Concerned_t_DemandFlowConstraint.append(t)
                self.Concerned_j_DemandFlowConstraint.append(j)
    
    def Gathering_Info_ACFTreatmentCapacityConstraint(self):
        self.FacilityCapacityInfo = {
            "constraints": self.FacilityCapacityConstraint_Objects,
            "pi_names": self.PIFacilityCapacityConstraint_Names,
            "concerned_t": self.Concerned_t_FacilityCapacityConstraint,
            "concerned_i": self.Concerned_i_FacilityCapacityConstraint
            }

    def Gathering_Info_DemandFlowConstraint(self):
        self.DemandFlowInfo = {
            "constraints": self.DemandFlowConstraint_Objects,
            "pi_names": self.PIDemandFlowConstraint_Names,
            "concerned_t": self.Concerned_t_DemandFlowConstraint,
            "concerned_j": self.Concerned_j_DemandFlowConstraint
            }
        
    def CreateConstraints_Sub(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateConstraints")

        self.CreateACFTreatmentCapacityConstraint()
        self.Gathering_Info_ACFTreatmentCapacityConstraint()

        self.CreateDemandFlowConstraint()
        self.Gathering_Info_DemandFlowConstraint()

    def CreateVariable_and_Objective_Function_Sub(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateVariable_and_Objective_Function_Sub")

        ################################# Production Variable #################################
        productioncost = {}
        self.Production_Var = {}
        for t in self.Instance.TimeBucketSet:  
            for i in self.Instance.FacilitySet:
                for j in self.Instance.DemandSet:
                    Index_Cost = self.GetIndexProductionVariable(t, i, j) - self.GetStartProductionVariable()
                    productioncost[Index_Cost] = self.GetProductionCoeff(t, i, j)
                    Index_Var = self.GetIndexProductionVariable(t, i, j)
                    var_name = f"y_t_{t}_i_{i}_j_{j}_index_{Index_Var}"
                    self.Production_Var[Index_Var] = self.SubModel.addVar(vtype=GRB.CONTINUOUS, obj=productioncost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.SubModel.update()
        
        ################################# Shortage Variable #################################
        shortagecost = {}
        self.Shortage_Var = {}
        for t in self.Instance.TimeBucketSet:  
            for j in self.Instance.DemandSet:
                Index_Cost = self.GetIndexShortageVariable(t, j) - self.GetStartShortageVariable()
                shortagecost[Index_Cost] = self.GetShortageCoeff(t, j)
                Index_Var = self.GetIndexShortageVariable(t, j)
                var_name = f"z_t_{t}_j_{j}_index_{Index_Var}"
                self.Shortage_Var[Index_Var] = self.SubModel.addVar(vtype=GRB.CONTINUOUS, obj=shortagecost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.SubModel.update()

        ############################################## Set minimization
        self.SubModel.setObjective(self.SubModel.getObjective(), GRB.MINIMIZE)

        if Constants.Debug:
            self.PrintAllVariables_Sub()

    def SetParameter_Master(self):
        # Enable lazy constraints on the master:
        self.MasterModel.Params.lazyConstraints = 1
        self.MasterModel.setParam(GRB.Param.TimeLimit, Constants.AlgorithmTimeLimit)

    def BuildMaster(self):
        if Constants.Debug: print("\n We are in 'BranchandBendersCut' Class -- BuildMaster")

        # Initialize the Gurobi model
        self.MasterModel = gp.Model("Master_LocAlloc")

        # Create variables and objective function
        self.CreateVariable_and_Objective_Function_Master()

        # Create all constraints
        self.CreateConstraints_Master()

        self.SetParameter_Master()

    def SetParameter_Sub(self):
        self.SubModel.params.logtoconsole=0

    def BuildSubProblem(self):
        if Constants.Debug: print("\n We are in 'BranchandBendersCut' Class -- BuildSubProblem")

        # Initialize the Gurobi model
        self.SubModel = gp.Model("Sub_LocAlloc")

        # Create variables and objective function
        self.CreateVariable_and_Objective_Function_Sub()
        # Create all constraints
        self.CreateConstraints_Sub()

        # Set some parameters
        self.SetParameter_Sub()

    def write_lp_file(self, model, model_name):
        """
        Write the given model to an .lp file with a formatted name.

        :param model: The optimization model (e.g., self.MasterModel or self.SubModel).
        :param model_name: A string representing the model's name (e.g., "MasterModel", "SubModel").
        """
        if Constants.Debug_lp_Files:
            # Create the directory "MIP_Model_LP" in the current working directory
            lp_dir = os.path.join(os.getcwd(), "MIP_Model_LP")
            os.makedirs(lp_dir, exist_ok=True)  # Creates the directory if it does not exist

            # Variables to include in the filename
            T = self.Instance.NrTimeBucket
            I = self.Instance.NrFacilities
            J = self.Instance.NrDemandLocations
            scenario = len(self.ScenarioSet)

            # Format the filename with the variables
            lp_filename = f"{model_name}__T_{T}_I_{I}_J_{J}_Scenario_{scenario}.lp"
            
            # Full path for the .lp file
            lp_full_path = os.path.join(lp_dir, lp_filename)

            # Write the model to the .lp file
            model.write(lp_full_path)

    def Store_Optimal_FirstStage_Solutions(self):
        for w in self.ScenarioSet:
            for i in self.Instance.FacilitySet:
                index_var = self.GetIndexFacilityEstablishmentVariable(i)
                self.solfacilityEstablishment[w][i] = self.Facility_Establishment_Var[index_var].X

    def Run(self): 
        start_time = time.time()  # or time.perf_counter() for higher precision

        self.InitTrace()
        self.write_lp_file(self.MasterModel, "Master")

        # Instantiate the callback object:
        benders_callback = BendersCutCallback(  instance = self.Instance,
                                                testidentifier = self.TestIdentifier,
                                                scenarioTree = self.ScenarioTree,
                                                master=self.MasterModel,
                                                sub=self.SubModel,
                                                startFacilityEstablishmentVariables=self.StartFacilityEstablishmentVariables,
                                                startthetaVariables=self.StartthetaVariables,
                                                theta_Var=self.theta_Var,
                                                facility_Establishment_Var=self.Facility_Establishment_Var,
                                                facilityCapacityInfo=self.FacilityCapacityInfo,
                                                demandFlowInfo=self.DemandFlowInfo)
        
        if(Constants.UseWarmUp):
            benders_callback.WarmStart_Master_Model()  

        # Run the optimization with the object-oriented callback:
        self.MasterModel.optimize(benders_callback)

        print('current optimal solution:')
        print('objval = ', self.MasterModel.objVal)

        self.Store_Optimal_FirstStage_Solutions()

        # Record the end time and compute the total runtime.
        end_time = time.time()  # or time.perf_counter()
        total_time = end_time - start_time

        self.Fixed_MIPSolver = MIPSolver(instance=self.Instance,
                                            model=Constants.Two_Stage,
                                            scenariotree=self.ScenarioTree,
                                            nrscenario=self.TreeStructure[1],
                                            givenfacilitylocation=self.solfacilityEstablishment,
                                            logfile="NO")
        self.Fixed_MIPSolver.BuildModel()
        Final_solution = self.Fixed_MIPSolver.Solve(True)

        final_gap_LB = (Final_solution.GRBCost - self.MasterModel.ObjVal) / self.MasterModel.ObjVal
        final_gap_UB = (Final_solution.GRBCost - self.MasterModel.ObjVal) / Final_solution.GRBCost
        final_trace_str = "\nLB: {:.4f}, Final gap (Based on LB): {:.2%}, Final gap (Based on UB): {:.2%},Total time spent in Run: {:.2f} seconds\n".format(
            self.MasterModel.ObjVal, final_gap_LB, final_gap_UB, total_time)
        self.WriteInTraceFile(final_trace_str)


        return Final_solution