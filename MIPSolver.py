import pandas as pd
import time
import numpy as np
import math
from Constants import Constants
import itertools
import gurobipy as gp
from gurobipy import *
import os
from Tool import Tool
from Solution import Solution
from ScenarioTree import ScenarioTree


class MIPSolver(object):
    # constructor
    def __init__(self,
                 instance,
                 model,
                 scenariotree,
                 nrscenario,
                 givenACFEstablishment = [],
                 givenNrLandRescueVehicle = [],
                 givenBackupHospital = [],
                 evaluatesolution = False,
                 logfile=""):
        
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- Constructor")
        # Define some attributes and functions which help to set the index of the variable.

        ####### First-Stage Variables
        self.NrACFEstablishmentVariables = 0
        self.Nr_PHA_ZPlus_ACFEstablishmentVariables = 0
        self.Nr_PHA_ZMinus_ACFEstablishmentVariables = 0
        self.StartACFEstablishmentVariables = 0
        self.Start_PHA_ZPlus_ACFEstablishmentVariables = 0
        self.Start_PHA_ZMinus_ACFEstablishmentVariables = 0

        self.NrLandRescueVehicleVariables = 0
        self.Nr_PHA_ZPlus_LandRescueVehicleVariables = 0
        self.Nr_PHA_ZMinus_LandRescueVehicleVariables = 0
        self.StartLandRescueVehicleVariables = 0
        self.Start_PHA_ZPlus_LandRescueVehicleVariables = 0
        self.Start_PHA_ZMinus_LandRescueVehicleVariables = 0

        self.NrBackupHospitalVariables = 0
        self.Nr_PHA_ZPlus_BackupHospitalVariables = 0
        self.Nr_PHA_ZMinus_BackupHospitalVariables = 0
        self.StartBackupHospitalVariables = 0
        self.Start_PHA_ZPlus_BackupHospitalVariables = 0
        self.Start_PHA_ZMinus_BackupHospitalVariables = 0

        ####### Second-Stage Variables
        self.NrCasualtyTransferVariables = 0
        self.StartCasualtyTransferVariables = 0

        self.NrUnsatisfiedCasualtiesVariables = 0
        self.StartUnsatisfiedCasualtiesVariables = 0

        self.NrDischargedPatientsVariables = 0
        self.StartDischargedPatientsVariables = 0

        self.NrLandEvacuatedPatientsVariables = 0
        self.StartLandEvacuatedPatientsVariables = 0

        self.NrAerialEvacuatedPatientsVariables = 0
        self.StartAerialEvacuatedPatientsVariables = 0

        self.NrUnevacuatedPatientsVariables = 0
        self.StartUnevacuatedPatientsVariables = 0

        self.NrAvailableCapFacilityVariables = 0
        self.StartAvailableCapFacilityVariables = 0

        self.Instance = instance

        #Takes value in Average/Two_Stage/Multi_Stage
        self.Model = model        

        #The set of scenarios used to solve the instance
        self.DemandScenarioTree = scenariotree
        self.DemandScenarioTree.Owner = self
        self.NrScenario = nrscenario
        self.ScenarioProbability = (1/self.NrScenario)

        self.Scenarios = scenariotree.GetAllScenarioSet()

        self.ComputeIndices()

        self.GivenACFEstablishment = givenACFEstablishment
        self.GivenNrLandRescueVehicle = givenNrLandRescueVehicle
        self.GivenBackupHospital = givenBackupHospital

        self.ScenarioSet = range(self.NrScenario)

        self.logfilename= logfile
        #This list is filled after the resolution of the MIP
        self.SolveInfo = []

        #This list will contain the set of constraint number for each flow constraint
        self.MaxBackupConstraintNR = [[None for _ in self.Instance.HospitalSet]
                                            for _ in self.ScenarioSet]

        self.TotalBudgetConstraintNR = [None for _ in self.ScenarioSet]

        self.LandVehicleAssignmentConstraintNR = [[None for _ in self.Instance.RescueVehicleSet] 
                                                        for _ in self.ScenarioSet]
        
        self.VehicleACFConnectionConstraintNR = [[[None for _ in self.Instance.RescueVehicleSet] 
                                                        for _ in self.Instance.ACFSet] 
                                                        for _ in self.ScenarioSet]
        
        self.CasualtyAllocationConstraintNR = [[[[None for _ in self.Instance.DisasterAreaSet] 
                                                        for _ in self.Instance.InjuryLevelSet] 
                                                        for _ in self.Instance.TimeBucketSet] 
                                                        for _ in self.ScenarioSet]
        
        self.PatientAllocationConstraintNR = [[[[None for _ in self.Instance.HospitalSet] 
                                                    for _ in self.Instance.InjuryLevelSet] 
                                                    for _ in self.Instance.TimeBucketSet] 
                                                    for _ in self.ScenarioSet]
            
        self.HospitalCapConstraintNR = [[[None for _ in self.Instance.HospitalSet] 
                                              for _ in self.Instance.TimeBucketSet] 
                                              for _ in self.ScenarioSet]
        
        self.MaxHospitalCapConstraintNR = [[[None for _ in self.Instance.HospitalSet] 
                                                for _ in self.Instance.TimeBucketSet] 
                                                for _ in self.ScenarioSet]
        
        self.DischargedHospitalConstraintNR = [[[[None for _ in self.Instance.InjuryLevelSet] 
                                                    for _ in self.Instance.HospitalSet] 
                                                    for _ in self.Instance.TimeBucketSet] 
                                                    for _ in self.ScenarioSet]
        
        self.ACFCapConstraintNR = [[[None for _ in self.Instance.ACFSet] 
                                            for _ in self.Instance.TimeBucketSet] 
                                            for _ in self.ScenarioSet]
        
        self.MaxACFCapConstraintNR = [[[None for _ in self.Instance.ACFSet] 
                                              for _ in self.Instance.TimeBucketSet] 
                                              for _ in self.ScenarioSet]
        
        self.DischargedACFConstraintNR = [[[[None for _ in self.Instance.InjuryLevelSet] 
                                              for _ in self.Instance.ACFSet] 
                                              for _ in self.Instance.TimeBucketSet] 
                                              for _ in self.ScenarioSet]
            
        self.LandResVehicleCapHosConstraintNR = [[[[None for _ in self.Instance.RescueVehicleSet] 
                                                        for _ in self.Instance.HospitalSet] 
                                                        for _ in self.Instance.TimeBucketSet] 
                                                        for _ in self.ScenarioSet]
        
        self.LandResVehicleCapACFConstraintNR = [[[[None for _ in self.Instance.RescueVehicleSet] 
                                                        for _ in self.Instance.ACFSet] 
                                                        for _ in self.Instance.TimeBucketSet] 
                                                        for _ in self.ScenarioSet]
        
        self.AerialResVehicleCapConstraintNR = [[[None for _ in self.Instance.HospitalSet] 
                                                        for _ in self.Instance.TimeBucketSet] 
                                                        for _ in self.ScenarioSet]
        
        self.EvacuationBackupConnectionConstraintNR = [[[[None for _ in self.Instance.HospitalSet] 
                                                            for _ in self.Instance.HospitalSet] 
                                                            for _ in self.Instance.TimeBucketSet] 
                                                            for _ in self.ScenarioSet]
        
        self.FirstHospitalCapConstraintNR = [[None for _ in self.Instance.HospitalSet] 
                                                    for _ in self.ScenarioSet]
        
        self.FirstACFCapConstraintNR = [[None for _ in self.Instance.ACFSet] 
                                              for _ in self.ScenarioSet]
        

        ################ PHA Part        
        self.LinearPHA_ACFEstablishmentConstraintNR = [[None for _ in self.Instance.ACFSet] 
                                                             for _ in self.ScenarioSet]
        
        self.LinearPHA_LandRescueVehicleConstraintNR = [[[None for _ in self.Instance.RescueVehicleSet] 
                                                                for _ in self.Instance.ACFSet] 
                                                                for _ in self.ScenarioSet]
        
        self.LinearPHA_BackupHospitalConstraintNR = [[[None for _ in self.Instance.HospitalSet] 
                                                            for _ in self.Instance.HospitalSet] 
                                                            for _ in self.ScenarioSet]
        
        ############### Non-anticipativity
        self.NonAnticipativityACFConstraintNR = [[None for _ in self.Instance.ACFSet] 
                                                        for _ in self.ScenarioSet[:-1]]         #Because in pairwise non-anticipativity constraints we are not counting on the last scenario! (See the model)
        
        self.NonAnticipativityVehicleConstraintNR = [[[None for _ in self.Instance.RescueVehicleSet] 
                                                            for _ in self.Instance.ACFSet] 
                                                            for _ in self.ScenarioSet[:-1]]         #Because in pairwise non-anticipativity constraints we are not counting on the last scenario! (See the model)
        
        self.NonAnticipativityBackupConstraintNR = [[[None for _ in self.Instance.HospitalSet] 
                                                            for _ in self.Instance.HospitalSet]
                                                            for _ in self.ScenarioSet[:-1]]         #Because in pairwise non-anticipativity constraints we are not counting on the last scenario! (See the model)
        
        self.ACFEstablishmentVarConstraintNR = []
        self.LandRescueVehicleVarConstraintNR = []
        self.BackupHospitalVarConstraintNR = []

        self.EvaluateSolution = evaluatesolution

    # Compute the start of index and the number of variables for the considered instance
    def ComputeIndices(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- ComputeIndices")

        ################## First-Stage Variables
        self.NrACFEstablishmentVariables = self.NrScenario * self.Instance.NrACFs        # We create x_si, since we may develope PHA to solve the problem
        if Constants.We_Are_in_PHA and Constants.Quadratic_to_Linear_PHA:
            self.Nr_PHA_ZPlus_ACFEstablishmentVariables = self.NrScenario * self.Instance.NrACFs
            self.Nr_PHA_ZMinus_ACFEstablishmentVariables = self.NrScenario * self.Instance.ACFs 
        
        self.NrLandRescueVehicleVariables = self.NrScenario * self.Instance.NrACFs * self.Instance.NrRescueVehicles        
        if Constants.We_Are_in_PHA and Constants.Quadratic_to_Linear_PHA:
            self.Nr_PHA_ZPlus_LandRescueVehicleVariables = self.NrScenario * self.Instance.NrACFs * self.Instance.NrRescueVehicles
            self.Nr_PHA_ZMinus_LandRescueVehicleVariables = self.NrScenario * self.Instance.NrACFs * self.Instance.NrRescueVehicles 
        
        self.NrBackupHospitalVariables = self.NrScenario * self.Instance.NrHospitals  * self.Instance.NrHospitals      
        if Constants.We_Are_in_PHA and Constants.Quadratic_to_Linear_PHA:
            self.Nr_PHA_ZPlus_BackupHospitalVariables = self.NrScenario * self.Instance.NrHospitals * self.Instance.NrHospitals
            self.Nr_PHA_ZMinus_BackupHospitalVariables = self.NrScenario * self.Instance.NrHospitals * self.Instance.NrHospitals
        
        ################## Second-Stage Variables
        self.NrCasualtyTransferVariables = self.NrScenario * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrDisasterAreas * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles                
        self.NrUnsatisfiedCasualtiesVariables = self.NrScenario * self.Instance.NrTimeBucket *  self.Instance.NrInjuries * self.Instance.NrDisasterAreas         
        self.NrDischargedPatientsVariables = self.NrScenario * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrMedFacilities
        self.NrLandEvacuatedPatientsVariables = self.NrScenario * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrHospitals * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles 
        self.NrAerialEvacuatedPatientsVariables = self.NrScenario * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrHospitals * self.Instance.NrACFs * self.Instance.NrHospitals * self.Instance.NrRescueVehicles 
        self.NrUnevacuatedPatientsVariables = self.NrScenario * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrHospitals
        self.NrAvailableCapFacilityVariables = self.NrScenario * self.Instance.NrTimeBucket * self.Instance.NrMedFacilities 
        

        self.StartACFEstablishmentVariables = 0
        self.StartLandRescueVehicleVariables = self.StartACFEstablishmentVariables + self.NrACFEstablishmentVariables
        self.StartBackupHospitalVariables = self.StartLandRescueVehicleVariables + self.NrLandRescueVehicleVariables

        self.StartCasualtyTransferVariables = self.StartBackupHospitalVariables + self.NrBackupHospitalVariables
        self.StartUnsatisfiedCasualtiesVariables = self.StartCasualtyTransferVariables + self.NrCasualtyTransferVariables
        self.StartDischargedPatientsVariables = self.StartUnsatisfiedCasualtiesVariables + self.NrUnsatisfiedCasualtiesVariables
        self.StartLandEvacuatedPatientsVariables = self.StartDischargedPatientsVariables + self.NrDischargedPatientsVariables
        self.StartAerialEvacuatedPatientsVariables = self.StartLandEvacuatedPatientsVariables + self.NrLandEvacuatedPatientsVariables
        self.StartUnevacuatedPatientsVariables = self.StartAerialEvacuatedPatientsVariables + self.NrAerialEvacuatedPatientsVariables
        self.StartAvailableCapFacilityVariables = self.StartUnevacuatedPatientsVariables + self.NrUnevacuatedPatientsVariables

        #Linear PHA Related Constraints
        self.Start_PHA_ZPlus_ACFEstablishmentVariables = self.StartAvailableCapFacilityVariables + self.NrAvailableCapFacilityVariables
        self.Start_PHA_ZMinus_ACFEstablishmentVariables = self.Start_PHA_ZPlus_ACFEstablishmentVariables + self.Nr_PHA_ZPlus_ACFEstablishmentVariables
        
        self.Start_PHA_ZPlus_LandRescueVehicleVariables = self.Start_PHA_ZMinus_ACFEstablishmentVariables + self.Nr_PHA_ZMinus_ACFEstablishmentVariables
        self.Start_PHA_ZMinus_LandRescueVehicleVariables = self.Start_PHA_ZPlus_LandRescueVehicleVariables + self.Nr_PHA_ZPlus_LandRescueVehicleVariables
        
        self.Start_PHA_ZPlus_BackupHospitalVariables = self.Start_PHA_ZMinus_LandRescueVehicleVariables + self.Nr_PHA_ZMinus_LandRescueVehicleVariables
        self.Start_PHA_ZMinus_BackupHospitalVariables = self.Start_PHA_ZPlus_BackupHospitalVariables + self.Nr_PHA_ZPlus_BackupHospitalVariables

        if Constants.Debug:
            print("StartACFEstablishmentVariables: ", self.StartACFEstablishmentVariables)
            print("NrACFEstablishmentVariables: ", self.NrACFEstablishmentVariables)
            print("StartLandRescueVehicleVariables: ", self.StartLandRescueVehicleVariables)
            print("NrLandRescueVehicleVariables: ", self.NrLandRescueVehicleVariables)
            print("StartBackupHospitalVariables: ", self.StartBackupHospitalVariables)
            print("NrBackupHospitalVariables: ", self.NrBackupHospitalVariables)

            print("StartCasualtyTransferVariables: ", self.StartCasualtyTransferVariables)
            print("NrCasualtyTransferVariables: ", self.NrCasualtyTransferVariables)
            print("StartUnsatisfiedCasualtiesVariables: ", self.StartUnsatisfiedCasualtiesVariables)
            print("NrUnsatisfiedCasualtiesVariables: ", self.NrUnsatisfiedCasualtiesVariables)
            print("StartDischargedPatientsVariables: ", self.StartDischargedPatientsVariables)
            print("NrDischargedPatientsVariables: ", self.NrDischargedPatientsVariables)
            print("StartLandEvacuatedPatientsVariables: ", self.StartLandEvacuatedPatientsVariables)
            print("NrLandEvacuatedPatientsVariables: ", self.NrLandEvacuatedPatientsVariables)
            print("StartAerialEvacuatedPatientsVariables: ", self.StartAerialEvacuatedPatientsVariables)
            print("NrAerialEvacuatedPatientsVariables: ", self.NrAerialEvacuatedPatientsVariables)
            print("StartUnevacuatedPatientsVariables: ", self.StartUnevacuatedPatientsVariables)
            print("NrUnevacuatedPatientsVariables: ", self.NrUnevacuatedPatientsVariables)
            print("StartAvailableCapFacilityVariables: ", self.StartAvailableCapFacilityVariables)
            print("NrAvailableCapFacilityVariables: ", self.NrAvailableCapFacilityVariables)
            
            print("Start_PHA_ZPlus_ACFEstablishmentVariables: ", self.Start_PHA_ZPlus_ACFEstablishmentVariables)
            print("Nr_PHA_ZPlus_ACFEstablishmentVariables: ", self.Nr_PHA_ZPlus_ACFEstablishmentVariables)
            print("Start_PHA_ZMinus_ACFEstablishmentVariables: ", self.Start_PHA_ZMinus_ACFEstablishmentVariables)
            print("Nr_PHA_ZMinus_ACFEstablishmentVariables: ", self.Nr_PHA_ZMinus_ACFEstablishmentVariables)
            print("Start_PHA_ZPlus_LandRescueVehicleVariables: ", self.Start_PHA_ZPlus_LandRescueVehicleVariables)
            print("Nr_PHA_ZPlus_LandRescueVehicleVariables: ", self.Nr_PHA_ZPlus_LandRescueVehicleVariables)
            print("Start_PHA_ZMinus_LandRescueVehicleVariables: ", self.Start_PHA_ZMinus_LandRescueVehicleVariables)
            print("Nr_PHA_ZMinus_LandRescueVehicleVariables: ", self.Nr_PHA_ZMinus_LandRescueVehicleVariables)
            print("Start_PHA_ZPlus_BackupHospitalVariables: ", self.Start_PHA_ZPlus_BackupHospitalVariables)
            print("Nr_PHA_ZPlus_BackupHospitalVariables: ", self.Nr_PHA_ZPlus_BackupHospitalVariables)
            print("Start_PHA_ZMinus_BackupHospitalVariables: ", self.Start_PHA_ZMinus_BackupHospitalVariables)
            print("Nr_PHA_ZMinus_BackupHospitalVariables: ", self.Nr_PHA_ZMinus_BackupHospitalVariables)

    # the function GetIndexACFEstablishmentVariable returns the index of the variable x_{i}
    def GetIndexACFEstablishmentVariable(self, w, i):
        return self.StartACFEstablishmentVariables \
            + w * self.Instance.NrACFs \
            + i
    
    def GetIndexLandRescueVehicleVariable(self, w, i, m):
        return self.StartLandRescueVehicleVariables \
            + w * self.Instance.NrACFs * self.Instance.NrRescueVehicles \
            + i * self.Instance.NrRescueVehicles \
            + m
    
    def GetIndexBackupHospitalVariable(self, w, h, hprime):
        return self.StartBackupHospitalVariables \
            + w * self.Instance.NrHospitals * self.Instance.NrHospitals \
            + hprime * self.Instance.NrHospitals \
            + h
    
    def GetIndexFacilityEstablishmentVariable(self, w, i):
        return self.StartFacilityEstablishmentVariables \
            + w * self.Instance.NrFacilities \
            + i
    
    def GetIndex_PHA_ZPlus_LandRescueVehicleVariable(self, w, i, m):
        return self.Start_PHA_ZPlus_LandRescueVehicleVariables \
            + w * self.Instance.NrRescueVehicles * self.Instance.NrACFs \
            + m * self.Instance.NrACFs \
            + i
    
    def GetIndex_PHA_ZPlus_BackupHospitalVariable(self, w, h, hprime):
        return self.Start_PHA_ZPlus_BackupHospitalVariables \
            + w * self.Instance.NrHospitals * self.Instance.NrHospitals \
            + hprime * self.Instance.NrHospitals \
            + h
    
    def GetIndex_PHA_ZPlus_FacilEstablishmentVariable(self, w, i):
        return self.Start_PHA_ZPlus_FacilEstablishmentVariables \
            + w * self.Instance.NrFacilities \
            + i
    
    def GetIndex_PHA_ZPlus_ACFEstablishmentVariable(self, w, i):
        return self.Start_PHA_ZPlus_ACFEstablishmentVariables \
            + w * self.Instance.NrACFs \
            + i
    
    def GetIndex_PHA_ZMinus_LandRescueVehicleVariable(self, w, i, m):
        return self.Start_PHA_ZMinus_LandRescueVehicleVariables \
            + w * self.Instance.NrRescueVehicles * self.Instance.NrACFs \
            + m * self.Instance.NrACFs \
            + i
    
    def GetIndex_PHA_ZMinus_BackupHospitalVariable(self, w, h, hprime):
        return self.Start_PHA_ZMinus_BackupHospitalVariables \
            + w * self.Instance.NrHospitals * self.Instance.NrHospitals \
            + hprime * self.Instance.NrHospitals \
            + h
    
    def GetIndex_PHA_ZMinus_FacilEstablishmentVariable(self, w, i):
        return self.Start_PHA_ZMinus_FacilEstablishmentVariables \
            + w * self.Instance.NrFacilities \
            + i
    
    def GetIndex_PHA_ZMinus_ACFEstablishmentVariable(self, w, i):
        return self.Start_PHA_ZMinus_ACFEstablishmentVariables \
            + w * self.Instance.NrACFs \
            + i

    def GetIndexCasualtyTransferVariables(self, w, t, j, l, u, m):
        return self.StartCasualtyTransferVariables \
            + w * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrDisasterAreas * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles \
            + t * self.Instance.NrInjuries * self.Instance.NrDisasterAreas * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles \
            + j * self.Instance.NrDisasterAreas * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles \
            + l * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles \
            + u * self.Instance.NrRescueVehicles \
            + m

    def GetIndexUnsatisfiedCasualtiesVariables(self, w, t, j, l):
        return self.StartUnsatisfiedCasualtiesVariables \
            + w * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrDisasterAreas \
            + t * self.Instance.NrInjuries * self.Instance.NrDisasterAreas \
            + j * self.Instance.NrDisasterAreas \
            + l

    def GetIndexDischargedPatientsVariables(self, w, t, j, u):
        return self.StartDischargedPatientsVariables \
                + w * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrMedFacilities \
                + t * self.Instance.NrInjuries * self.Instance.NrMedFacilities \
                + j * self.Instance.NrMedFacilities \
                + u
    
    def GetIndexLandEvacuatedPatientsVariables(self, w, t, j, h, u, m):
        return self.StartLandEvacuatedPatientsVariables \
                + w * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrHospitals * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles \
                + t * self.Instance.NrInjuries * self.Instance.NrHospitals * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles \
                + j * self.Instance.NrHospitals * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles \
                + h * self.Instance.NrMedFacilities * self.Instance.NrRescueVehicles \
                + u * self.Instance.NrRescueVehicles \
                + m
        
    def GetIndexAerialEvacuatedPatientsVariables(self, w, t, j, h, i, hprime, m):
        return self.StartAerialEvacuatedPatientsVariables \
                + w * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrHospitals * self.Instance.NrACFs * self.Instance.NrHospitals * self.Instance.NrRescueVehicles \
                + t * self.Instance.NrInjuries * self.Instance.NrHospitals * self.Instance.NrACFs * self.Instance.NrHospitals * self.Instance.NrRescueVehicles \
                + j * self.Instance.NrHospitals * self.Instance.NrACFs * self.Instance.NrHospitals * self.Instance.NrRescueVehicles \
                + h * self.Instance.NrACFs * self.Instance.NrHospitals * self.Instance.NrRescueVehicles \
                + i * self.Instance.NrHospitals * self.Instance.NrRescueVehicles \
                + hprime * self.Instance.NrRescueVehicles \
                + m
    
    def GetIndexUnevacuatedPatientsVariables(self, w, t, j, h):
        return self.StartUnevacuatedPatientsVariables \
                + w * self.Instance.NrTimeBucket * self.Instance.NrInjuries * self.Instance.NrHospitals \
                + t * self.Instance.NrInjuries * self.Instance.NrHospitals \
                + j * self.Instance.NrHospitals \
                + h
    
    def GetIndexAvailableCapFacilityVariables(self, w, t, u):
        return self.StartAvailableCapFacilityVariables \
                + w * self.Instance.NrTimeBucket * self.Instance.NrMedFacilities \
                + t * self.Instance.NrMedFacilities \
                + u
    
    def GetIndexProductionVariable(self, w, t, i, j):
        return self.StartProductionVariables \
            + w * self.Instance.NrTimeBucket * self.Instance.NrFacilities * self.Instance.NrDemandLocations\
            + t * self.Instance.NrFacilities * self.Instance.NrDemandLocations \
            + i * self.Instance.NrDemandLocations \
            + j
    
    def GetIndexShortageVariable(self, w, t, j):

        return self.StartShortageVariables \
            + w * self.Instance.NrTimeBucket * self.Instance.NrDemandLocations\
            + t * self.Instance.NrDemandLocations \
            + j
        
    def GetStartACFEstablishmentVariables(self):
        return self.StartACFEstablishmentVariables
    
    def GetStartLandRescueVehicleVariables(self):
        return self.StartLandRescueVehicleVariables
    
    def GetStartBackupHospitalVariables(self):
        return self.StartBackupHospitalVariables
    
    def GetStartFacilityEstablishmentVariables(self):
        return self.StartFacilityEstablishmentVariables
    
    def GetStart_PHA_ZPlus_LandRescueVehicleVariables(self):
        return self.Start_PHA_ZPlus_LandRescueVehicleVariables
    
    def GetStart_PHA_ZPlus_BackupHospitalVariables(self):
        return self.Start_PHA_ZPlus_BackupHospitalVariables
    
    def GetStart_PHA_ZPlus_FacilEstablishmentVariables(self):
        return self.Start_PHA_ZPlus_FacilEstablishmentVariables
    
    def GetStart_PHA_ZPlus_ACFEstablishmentVariables(self):
        return self.Start_PHA_ZPlus_ACFEstablishmentVariables
    
    def GetStart_PHA_ZMinus_LandRescueVehicleVariables(self):
        return self.Start_PHA_ZMinus_LandRescueVehicleVariables
    
    def GetStart_PHA_ZMinus_BackupHospitalVariables(self):
        return self.Start_PHA_ZMinus_BackupHospitalVariables
    
    def GetStart_PHA_ZMinus_FacilEstablishmentVariables(self):
        return self.Start_PHA_ZMinus_FacilEstablishmentVariables
    
    def GetStart_PHA_ZMinus_ACFEstablishmentVariables(self):
        return self.Start_PHA_ZMinus_ACFEstablishmentVariables
    
    def GetStartCasualtyTransferVariables(self):
        return self.StartCasualtyTransferVariables
    
    def GetStartUnsatisfiedCasualtiesVariables(self):
        return self.StartUnsatisfiedCasualtiesVariables
    
    def GetStartDischargedPatientsVariables(self):
        return self.StartDischargedPatientsVariables
    
    def GetStartLandEvacuatedPatientsVariables(self):
        return self.StartLandEvacuatedPatientsVariables
        
    def GetStartAerialEvacuatedPatientsVariables(self):
        return self.StartAerialEvacuatedPatientsVariables

    def GetStartUnevacuatedPatientsVariables(self):
        return self.StartUnevacuatedPatientsVariables
    
    def GetStartAvailableCapFacilityVariables(self):
        return self.StartAvailableCapFacilityVariables
    
    def GetStartProductionVariable(self):
        return self.StartProductionVariables
    
    def GetStartShortageVariable(self):
        return self.StartShortageVariables
        
    def GetACFestablishmentCoeff_Obj(self, i):
        return self.Instance.Fixed_Cost_ACF_Objective[i] * self.ScenarioProbability
    
    def GetlandRescueVehicleCoeff(self, i, m):
        return self.Instance.VehicleAssignment_Cost[m] * self.ScenarioProbability
    
    def GetbackupHospitalCoeff(self, h, hprime):
        return self.Instance.CoordinationCost[h, hprime] * self.ScenarioProbability
    
    def GetfacilityestablishmentCoeff(self, i):
        return self.Instance.Fixed_Establishment_Cost[i] * self.ScenarioProbability
        
    def GetCasualtyTransferCoeff(self, w, l, u):
        if u < self.Instance.NrHospitals:
            return self.Instance.Time_D_H_Land[l][u] * self.ScenarioProbability
        else:
            return self.Instance.Time_D_A_Land[l][u - self.Instance.NrHospitals] * self.ScenarioProbability
    
    def GetUnsatisfiedCasualtiesCoeff(self, w, j):
        return self.Instance.Casualty_Shortage_Cost[j] * self.ScenarioProbability
    
    def GetLandEvacuatedPatientsCoeff(self, w, t, j, h, u, m):
        if Constants.Risk == "Constant":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Constant[t][j][h][u][m] * self.ScenarioProbability
        elif Constants.Risk == "Linear":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Linear[t][j][h][u][m] * self.ScenarioProbability
        elif Constants.Risk == "Exponential":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Exponential[t][j][h][u][m] * self.ScenarioProbability

    def GetAerialEvacuatedPatientsCoeff(self, w, t, j, h, i, hprime, m):
        if Constants.Risk == "Constant":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Constant[t][j][h][i][hprime][m] * self.ScenarioProbability
        elif Constants.Risk == "Linear":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Linear[t][j][h][i][hprime][m] * self.ScenarioProbability
        elif Constants.Risk == "Exponential":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Exponential[t][j][h][i][hprime][m] * self.ScenarioProbability

    def GetUnevacuatedPatientsCoeff(self, w, t, j, h):
        if Constants.Risk == "Constant":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.CumulativeThreatRiskConstant[t][j][h] * self.ScenarioProbability
        elif Constants.Risk == "Linear":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.CumulativeThreatRiskLinear[t][j][h] * self.ScenarioProbability
        elif Constants.Risk == "Exponential":
            return self.Instance.EvacuationRiskCost[j] * self.Instance.CumulativeThreatRiskExponential[t][j][h] * self.ScenarioProbability

    def GetProductionCoeff(self, w, t, i, j):
        return self.Instance.Transportation_Cost[i][j] * self.ScenarioProbability
            
    def GetShortageCoeff(self, w, t, j):
        return self.Instance.Unserved_Penalty_Cost[j] * self.ScenarioProbability

    def GetNrACFEstablishmentVariable(self):
        return self.NrACFEstablishmentVariables
    
    def GetNrLandRescueVehicleVariable(self):
        return self.NrLandRescueVehicleVariables
    
    def GetNrBackupHospitalVariable(self):
        return self.NrBackupHospitalVariables
    
    def GetNr_PHA_ZPlus_FacilEstablishmentVariable(self):
        return self.Nr_PHA_ZPlus_FacilLocationVariables
    
    def GetNr_PHA_ZMinus_FacilEstablishmentVariable(self):
        return self.Nr_PHA_ZMinus_FacilLocationVariables
    
    def PrintAllVariables(self):
        print("\n### Facility Establishment Variables ###")
        for index, var in self.Facility_Establishment_Var.items():
            print(f"Index: {index}, Variable: {var.VarName}, Obj Coeff: {var.Obj}, Type: {var.VType}")

        print("\n### Production Variables ###")
        for index, var in self.Production_Var.items():
            print(f"Index: {index}, Variable: {var.VarName}, Obj Coeff: {var.Obj}, Type: {var.VType}")

        print("\n### Shortage Variables ###")
        for index, var in self.Shortage_Var.items():
            print(f"Index: {index}, Variable: {var.VarName}, Obj Coeff: {var.Obj}, Type: {var.VType}")

    def CreateMaxBackupConstraint(self):
        for w in self.ScenarioSet:
            for h in self.Instance.HospitalSet:
                vars_W = [self.GetIndexBackupHospitalVariable(w, h, hprime)
                            for hprime in self.Instance.K_h.get(h, set())]  # Only take hprime in K_h
                coeff_W = [-1.0 
                            for hprime in self.Instance.K_h.get(h, set())]

                ############ Create the left-hand side of the constraint       
                LeftHandSide_W = gp.quicksum(coeff_W[i] * self.BackupHospital_Var[vars_W[i]] for i in range(len(vars_W)))        
                LeftHandSide = LeftHandSide_W
                
                ############ Define the right-hand side (RHS) of the constraint
                RightHandSide = -1.0 * self.Instance.Max_Backup_Hospital[h]
                
                ############ Add the constraint to the model
                constraint_name = f"MaxBackup_w_{w}_h_{h}"
                constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                self.MaxBackupConstraintNR[w][h] = constraint
    
    def CreateTotalBudgetConstraint(self):
        for w in self.ScenarioSet:
            vars_x = [self.GetIndexACFEstablishmentVariable(w, i)
                        for i in self.Instance.ACFSet]  
            coeff_x = [-1.0 * self.Instance.Fixed_Cost_ACF_Constraint[i]
                        for i in self.Instance.ACFSet]

            ############ Create the left-hand side of the constraint       
            LeftHandSide_x = gp.quicksum(coeff_x[i] * self.ACFEstablishment_Var[vars_x[i]] for i in range(len(vars_x)))        
            LeftHandSide = LeftHandSide_x
            
            ############ Define the right-hand side (RHS) of the constraint
            RightHandSide = -1.0 * self.Instance.Total_Budget_ACF_Establishment
            
            ############ Add the constraint to the model
            constraint_name = f"TotalBudget_w_{w}"
            constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
            self.TotalBudgetConstraintNR[w] = constraint

    def CreateLandVehicleAssignmentConstraint(self):
        for w in self.ScenarioSet:
            for m in self.Instance.RescueVehicleSet:
                vars_thetaVar = [self.GetIndexLandRescueVehicleVariable(w, i, m)
                                    for i in self.Instance.ACFSet] 
                coeff_thetaVar = [-1.0 
                                    for i in self.Instance.ACFSet]

                ############ Create the left-hand side of the constraint       
                LeftHandSide_thetaVar = gp.quicksum(coeff_thetaVar[i] * self.LandRescueVehicle_Var[vars_thetaVar[i]] for i in range(len(vars_thetaVar)))        
                LeftHandSide = LeftHandSide_thetaVar
                
                ############ Define the right-hand side (RHS) of the constraint
                RightHandSide = -1.0 * self.Instance.Number_Rescue_Vehicle_ACF[m]
                
                ############ Add the constraint to the model
                constraint_name = f"LandVehicleAssignment_w_{w}_m_{m}"
                constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                self.LandVehicleAssignmentConstraintNR[w][m] = constraint
    
    def CreateVehicleACFConnectionConstraint(self):
        for w in self.ScenarioSet:
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:

                    vars_thetaVar = [self.GetIndexLandRescueVehicleVariable(w, i, m)] 
                    coeff_thetaVar = [-1.0]

                    vars_x = [self.GetIndexACFEstablishmentVariable(w, i)] 
                    coeff_x = [1.0 * self.Instance.Number_Rescue_Vehicle_ACF[m]]

                    ############ Create the left-hand side of the constraint       
                    LeftHandSide_thetaVar = gp.quicksum(coeff_thetaVar[i] * self.LandRescueVehicle_Var[vars_thetaVar[i]] for i in range(len(vars_thetaVar)))        
                    LeftHandSide_x = gp.quicksum(coeff_x[i] * self.ACFEstablishment_Var[vars_x[i]] for i in range(len(vars_x)))        
                    LeftHandSide = LeftHandSide_thetaVar + LeftHandSide_x
                    
                    ############ Define the right-hand side (RHS) of the constraint
                    RightHandSide = 0
                    
                    ############ Add the constraint to the model
                    constraint_name = f"VehicleACFConnection_w_{w}_i_{i}_m_{m}"
                    constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                    self.LandVehicleAssignmentConstraintNR[w][m] = constraint

    def CreateACFTreatmentCapacityConstraint(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateACFTreatmentCapacityConstraint")

        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for i in self.Instance.FacilitySet:

                    vars_y = [self.GetIndexProductionVariable(w, t, i, j)
                                for j in self.Instance.DemandSet]
                    coeff_y = [-1.0 
                                for j in self.Instance.DemandSet]
                    
                    vars_x = [self.GetIndexFacilityEstablishmentVariable(w, i)]
                    coeff_x = [self.Instance.Facility_Capacity[i]]

                    ############ Create the left-hand side of the constraint       
                    LeftHandSide_y = gp.quicksum(coeff_y[i] * self.Production_Var[vars_y[i]] for i in range(len(vars_y)))        
                    LeftHandSide_x = gp.quicksum(coeff_x[i] * self.Facility_Establishment_Var[vars_x[i]] for i in range(len(vars_x)))
                    LeftHandSide = LeftHandSide_y + LeftHandSide_x
                    
                    ############ Define the right-hand side (RHS) of the constraint
                    RightHandSide = 0
                    
                    ############ Add the constraint to the model
                    constraint_name = f"FacilityCapacity_w_{w}_t_{t}_i_{i}"
                    constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                    self.FacilityCapacityConstraintNR[w][t][i] = constraint

    def CreateDemandFlowConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.DemandSet:

                    vars_y = [self.GetIndexProductionVariable(w, t, i, j) 
                                for i in self.Instance.FacilitySet]
                    coeff_y = [1.0  
                                for i in self.Instance.FacilitySet]

                    vars_z = [self.GetIndexShortageVariable(w, t, j)]
                    coeff_z = [1.0]

                    ############ Create the left-hand side of the constraint
                    LeftHandSide_y = gp.quicksum(coeff_y[i] * self.Production_Var[vars_y[i]] for i in range(len(vars_y)))
                    LeftHandSide_z = gp.quicksum(coeff_z[i] * self.Shortage_Var[vars_z[i]] for i in range(len(vars_z)))
                    LeftHandSide = LeftHandSide_y + LeftHandSide_z
                    
                    ############ Define the right-hand side (RHS) of the constraint
                    RightHandSide = self.DemandScenarioTree.Demand[w][t][j]  
                    ############ Add the constraint to the model
                    constraint_name = f"DemandFlow_w_{w}_t_{t}_j_{j}"
                    constraint = self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)
                    self.DemandFlowConstraintNR[w][t][j] = constraint

    def CreateCasualtyAllocationConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for l in self.Instance.DisasterAreaSet:

                        vars_qh = [self.GetIndexCasualtyTransferVariables(w, t, j, l, h, m)
                                    for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                    for h in self.Instance.HospitalSet if self.Instance.J_u[j][h] == 1 and self.DemandScenarioTree.HospitalDisruption[w][h] != 1]
                        coeff_qh = [1.0 
                                    for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                    for h in self.Instance.HospitalSet if self.Instance.J_u[j][h] == 1 and self.DemandScenarioTree.HospitalDisruption[w][h] != 1]

                        vars_qi = [self.GetIndexCasualtyTransferVariables(w, t, j, l, self.Instance.NrHospitals + i, m)
                                    for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                    for i in self.Instance.ACFSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                        coeff_qi = [1.0 
                                    for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                    for i in self.Instance.ACFSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                        
                        vars_mu = [self.GetIndexUnsatisfiedCasualtiesVariables(w, t, j, l)]
                        coeff_mu = [1.0]

                        vars_mu_prev = []
                        coeff_mu_prev = []
                        if t > 0:
                            vars_mu_prev = [self.GetIndexUnsatisfiedCasualtiesVariables(w, t-1, j, l)]
                            coeff_mu_prev = [-1.0]


                        ############ Create the left-hand side of the constraint
                        LeftHandSide_qh = gp.quicksum(coeff_qh[i] * self.CasualtyTransfer_Var[vars_qh[i]] for i in range(len(vars_qh)))
                        LeftHandSide_qi = gp.quicksum(coeff_qi[i] * self.CasualtyTransfer_Var[vars_qi[i]] for i in range(len(vars_qi)))
                        LeftHandSide_mu = gp.quicksum(coeff_mu[i] * self.UnsatisfiedCasualties_Var[vars_mu[i]] for i in range(len(vars_mu)))
                        LeftHandSide_mu_prev = gp.quicksum(coeff_mu_prev[i] * self.UnsatisfiedCasualties_Var[vars_mu_prev[i]] for i in range(len(vars_mu_prev)))
                        LeftHandSide = LeftHandSide_qh + LeftHandSide_qi + LeftHandSide_mu + LeftHandSide_mu_prev
                        
                        ############ Define the right-hand side (RHS) of the constraint
                        RightHandSide = self.DemandScenarioTree.CasualtyDemand[w][t][j][l]  
                        ############ Add the constraint to the model
                        constraint_name = f"CasualtyAllocation_w_{w}_t_{t}_j_{j}_l_{l}"
                        constraint = self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)
                        self.CasualtyAllocationConstraintNR[w][t][j][l] = constraint
    
    def CreatePatientAllocationConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        if self.DemandScenarioTree.HospitalDisruption[w][h] == 1:

                            vars_u_L_h = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, hprime, m)
                                            for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                            for hprime in self.Instance.HospitalSet
                                            if self.Instance.J_u[j][hprime] == 1 
                                            and self.DemandScenarioTree.HospitalDisruption[w][hprime] != 1
                                            and hprime in self.Instance.K_h.get(h, set())]
                            coeff_u_L_h = [1.0 
                                            for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                            for hprime in self.Instance.HospitalSet
                                            if self.Instance.J_u[j][hprime] == 1 
                                            and self.DemandScenarioTree.HospitalDisruption[w][hprime] != 1
                                            and hprime in self.Instance.K_h.get(h, set())]

                            vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                        for hprime in self.Instance.HospitalSet if self.Instance.J_u[j][hprime] == 1 and self.DemandScenarioTree.HospitalDisruption[w][hprime] != 1  and hprime in self.Instance.K_h.get(h, set())  # Ensure hprime is in K_h[h]
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set]
                            coeff_u_A = [1.0 
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                        for hprime in self.Instance.HospitalSet if self.Instance.J_u[j][hprime] == 1 and self.DemandScenarioTree.HospitalDisruption[w][hprime] != 1  and hprime in self.Instance.K_h.get(h, set())  # Ensure hprime is in K_h[h]
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set]
                                                        
                            vars_u_L_i = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, self.Instance.NrHospitals + i, m)
                                            for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                            for i in self.Instance.ACFSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                            coeff_u_L_i = [1.0 
                                            for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                            for i in self.Instance.ACFSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]


                            vars_Phi = [self.GetIndexUnevacuatedPatientsVariables(w, t, j, h)]
                            coeff_Phi = [1.0]

                            vars_Phi_prev = []
                            coeff_Phi_prev = []
                            if t > 0:
                                vars_Phi_prev = [self.GetIndexUnevacuatedPatientsVariables(w, t-1, j, h)]
                                coeff_Phi_prev = [-1.0]

                            ############ Create the left-hand side of the constraint
                            LeftHandSide_u_L_h = gp.quicksum(coeff_u_L_h[i] * self.LandEvacuatedPatients_Var[vars_u_L_h[i]] for i in range(len(vars_u_L_h)))
                            LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))
                            LeftHandSide_u_L_i = gp.quicksum(coeff_u_L_i[i] * self.LandEvacuatedPatients_Var[vars_u_L_i[i]] for i in range(len(vars_u_L_i)))
                            LeftHandSide_Phi = gp.quicksum(coeff_Phi[i] * self.UnevacuatedPatients_Var[vars_Phi[i]] for i in range(len(vars_Phi)))
                            LeftHandSide_Phi_prev = gp.quicksum(coeff_Phi_prev[i] * self.UnevacuatedPatients_Var[vars_Phi_prev[i]] for i in range(len(vars_Phi_prev)))
                            LeftHandSide = LeftHandSide_u_L_h + LeftHandSide_u_A + LeftHandSide_u_L_i + LeftHandSide_Phi + LeftHandSide_Phi_prev
                            
                            ############ Define the right-hand side (RHS) of the constraint
                            if t == 0:
                                RightHandSide = self.DemandScenarioTree.PatientDemand[w][j][h]  
                            else:
                                RightHandSide = 0
                            ############ Add the constraint to the model
                            constraint_name = f"PatientAllocation_w_{w}_t_{t}_j_{j}_h_{h}"
                            constraint = self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)
                            self.PatientAllocationConstraintNR[w][t][j][h] = constraint
    
    def CreateHospitalCapConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:

                    vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, tprime, j, hprime, i, h, m)
                                for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                for hprime in self.Instance.HospitalSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]
                    coeff_u_A = [-1.0 
                                for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                for hprime in self.Instance.HospitalSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]


                    vars_q = [self.GetIndexCasualtyTransferVariables(w, tprime, j, l, h, m)
                                for l in self.Instance.DisasterAreaSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]
                    coeff_q = [-1.0 
                                for l in self.Instance.DisasterAreaSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]
                                                
                    vars_u_L = [self.GetIndexLandEvacuatedPatientsVariables(w, tprime, j, hprime, h, m)
                                for hprime in self.Instance.HospitalSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]
                    coeff_u_L = [-1.0 
                                for hprime in self.Instance.HospitalSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]

                    vars_sigmaVar = [self.GetIndexDischargedPatientsVariables(w, tprime, j, h)
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1
                                        for tprime in range(t + 1)]
                    coeff_sigmaVar = [+1.0 
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1
                                        for tprime in range(t + 1)]

                    vars_zeta = [self.GetIndexAvailableCapFacilityVariables(w, t, h)]
                    coeff_zeta = [+1.0]
                                        
                    ############ Create the left-hand side of the constraint
                    LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))
                    LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))                    
                    LeftHandSide_u_L = gp.quicksum(coeff_u_L[i] * self.LandEvacuatedPatients_Var[vars_u_L[i]] for i in range(len(vars_u_L)))
                    LeftHandSide_sigmaVar = gp.quicksum(coeff_sigmaVar[i] * self.DischargedPatients_Var[vars_sigmaVar[i]] for i in range(len(vars_sigmaVar)))
                    LeftHandSide_zeta = gp.quicksum(coeff_zeta[i] * self.AvailableCapFacility_Var[vars_zeta[i]] for i in range(len(vars_zeta)))
                    LeftHandSide = LeftHandSide_u_A + LeftHandSide_q + LeftHandSide_u_L + LeftHandSide_sigmaVar + LeftHandSide_zeta
                    
                    ############ Define the right-hand side (RHS) of the constraint
                    RightHandSide = 0 

                    ############ Add the constraint to the model
                    constraint_name = f"HospitalCap_w_{w}_t_{t}_h_{h}"
                    constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                    self.HospitalCapConstraintNR[w][t][h] = constraint
    
    def CreateMaxHospitalCapConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:

                    vars_zeta = [self.GetIndexAvailableCapFacilityVariables(w, t, h)]
                    coeff_zeta = [-1.0]
                                        
                    ############ Create the left-hand side of the constraint
                    LeftHandSide_zeta = gp.quicksum(coeff_zeta[i] * self.AvailableCapFacility_Var[vars_zeta[i]] for i in range(len(vars_zeta)))
                    LeftHandSide = LeftHandSide_zeta
                    
                    ############ Define the right-hand side (RHS) of the constraint
                    RightHandSide = (1 - self.DemandScenarioTree.HospitalDisruption[w][h]) * -1.0 * self.Instance.Hospital_Bed_Capacity[h]

                    ############ Add the constraint to the model
                    constraint_name = f"MaxHospitalCap_w_{w}_t_{t}_h_{h}"
                    constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                    self.MaxHospitalCapConstraintNR[w][t][h] = constraint
    
    def CreateDischargedHospitalConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:
                    for j in self.Instance.InjuryLevelSet:
                        if self.Instance.J_u[j][h] == 1:

                            vars_sigmaVar = [self.GetIndexDischargedPatientsVariables(w, t, j, h)]
                            coeff_sigmaVar = [-1.0]

                            vars_q = [self.GetIndexCasualtyTransferVariables(w, t - k, j, l, h, m)
                                        for l in self.Instance.DisasterAreaSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_q = [1.0 * self.DemandScenarioTree.PatientDischargedPercentage[w][k][j][h] 
                                        for l in self.Instance.DisasterAreaSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]

                            vars_u_L = [self.GetIndexLandEvacuatedPatientsVariables(w, t - k, j, hprime, h, m)
                                        for hprime in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_u_L = [1.0 * self.DemandScenarioTree.PatientDischargedPercentage[w][k][j][h] 
                                        for hprime in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]

                            vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t - k, j, hprime, i, h, m)
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for hprime in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_u_A = [1.0 * self.DemandScenarioTree.PatientDischargedPercentage[w][k][j][h] 
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for hprime in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]             
                                                                                        
                            ############ Create the left-hand side of the constraint
                            LeftHandSide_sigmaVar = gp.quicksum(coeff_sigmaVar[i] * self.DischargedPatients_Var[vars_sigmaVar[i]] for i in range(len(vars_sigmaVar)))
                            LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))
                            LeftHandSide_u_L = gp.quicksum(coeff_u_L[i] * self.LandEvacuatedPatients_Var[vars_u_L[i]] for i in range(len(vars_u_L)))
                            LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))
                            LeftHandSide = LeftHandSide_sigmaVar + LeftHandSide_q + LeftHandSide_u_L + LeftHandSide_u_A
                            
                            ############ Define the right-hand side (RHS) of the constraint
                            RightHandSide = 0

                            ############ Add the constraint to the model
                            constraint_name = f"DischargedHospital_w_{w}_t_{t}_h_{h}_j_{j}"
                            constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                            self.DischargedHospitalConstraintNR[w][t][h][j] = constraint

    def CreateACFCapConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for i in self.Instance.ACFSet:

                    vars_q = [self.GetIndexCasualtyTransferVariables(w, tprime, j, l, self.Instance.NrHospitals + i, m)
                                for l in self.Instance.DisasterAreaSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]
                    coeff_q = [-1.0 
                                for l in self.Instance.DisasterAreaSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]
                                                
                    vars_u_L = [self.GetIndexLandEvacuatedPatientsVariables(w, tprime, j, h, self.Instance.NrHospitals + i, m)
                                for h in self.Instance.HospitalSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]
                    coeff_u_L = [-1.0 
                                for h in self.Instance.HospitalSet
                                for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1
                                for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                for tprime in range(t + 1)]

                    vars_sigmaVar = [self.GetIndexDischargedPatientsVariables(w, tprime, j, self.Instance.NrHospitals + i)
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1
                                        for tprime in range(t + 1)]
                    coeff_sigmaVar = [+1.0 
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1
                                        for tprime in range(t + 1)]

                    vars_zeta = [self.GetIndexAvailableCapFacilityVariables(w, t, self.Instance.NrHospitals + i)]
                    coeff_zeta = [+1.0]
                                        
                    ############ Create the left-hand side of the constraint
                    LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))                    
                    LeftHandSide_u_L = gp.quicksum(coeff_u_L[i] * self.LandEvacuatedPatients_Var[vars_u_L[i]] for i in range(len(vars_u_L)))
                    LeftHandSide_sigmaVar = gp.quicksum(coeff_sigmaVar[i] * self.DischargedPatients_Var[vars_sigmaVar[i]] for i in range(len(vars_sigmaVar)))
                    LeftHandSide_zeta = gp.quicksum(coeff_zeta[i] * self.AvailableCapFacility_Var[vars_zeta[i]] for i in range(len(vars_zeta)))
                    LeftHandSide = LeftHandSide_q + LeftHandSide_u_L + LeftHandSide_sigmaVar + LeftHandSide_zeta
                    
                    ############ Define the right-hand side (RHS) of the constraint
                    RightHandSide = 0 

                    ############ Add the constraint to the model
                    constraint_name = f"ACFCap_w_{w}_t_{t}_i_{i}"
                    constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                    self.ACFCapConstraintNR[w][t][i] = constraint

    def CreateMaxACFCapConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for i in self.Instance.ACFSet:

                    vars_zeta = [self.GetIndexAvailableCapFacilityVariables(w, t, self.Instance.NrHospitals + i)]
                    coeff_zeta = [-1.0]

                    vars_x = [self.GetIndexACFEstablishmentVariable(w, i)]
                    coeff_x = [+1.0 * self.Instance.ACF_Bed_Capacity[i]]
                                        
                    ############ Create the left-hand side of the constraint
                    LeftHandSide_zeta = gp.quicksum(coeff_zeta[i] * self.AvailableCapFacility_Var[vars_zeta[i]] for i in range(len(vars_zeta)))
                    LeftHandSide_x = gp.quicksum(coeff_x[i] * self.ACFEstablishment_Var[vars_x[i]] for i in range(len(vars_x)))
                    LeftHandSide = LeftHandSide_zeta + LeftHandSide_x
                    
                    ############ Define the right-hand side (RHS) of the constraint
                    RightHandSide = 0

                    ############ Add the constraint to the model
                    constraint_name = f"MaxACFCap_w_{w}_t_{t}_i_{i}"
                    constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                    self.MaxACFCapConstraintNR[w][t][i] = constraint

    def CreateDischargedACFConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for i in self.Instance.ACFSet:
                    for j in self.Instance.InjuryLevelSet:
                        if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1:

                            vars_sigmaVar = [self.GetIndexDischargedPatientsVariables(w, t, j, self.Instance.NrHospitals + i)]
                            coeff_sigmaVar = [-1.0]

                            vars_q = [self.GetIndexCasualtyTransferVariables(w, t - k, j, l, self.Instance.NrHospitals + i, m)
                                        for l in self.Instance.DisasterAreaSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_q = [1.0 * self.DemandScenarioTree.PatientDischargedPercentage[w][k][j][self.Instance.NrHospitals + i] 
                                        for l in self.Instance.DisasterAreaSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]

                            vars_u_L = [self.GetIndexLandEvacuatedPatientsVariables(w, t - k, j, h, self.Instance.NrHospitals + i, m)
                                        for h in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_u_L = [1.0 * self.DemandScenarioTree.PatientDischargedPercentage[w][k][j][self.Instance.NrHospitals + i] 
                                        for h in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]       
                                                                                        
                            ############ Create the left-hand side of the constraint
                            LeftHandSide_sigmaVar = gp.quicksum(coeff_sigmaVar[i] * self.DischargedPatients_Var[vars_sigmaVar[i]] for i in range(len(vars_sigmaVar)))
                            LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))
                            LeftHandSide_u_L = gp.quicksum(coeff_u_L[i] * self.LandEvacuatedPatients_Var[vars_u_L[i]] for i in range(len(vars_u_L)))
                            LeftHandSide = LeftHandSide_sigmaVar + LeftHandSide_q + LeftHandSide_u_L
                            
                            ############ Define the right-hand side (RHS) of the constraint
                            RightHandSide = 0

                            ############ Add the constraint to the model
                            constraint_name = f"DischargedACF_w_{w}_t_{t}_i_{i}_j_{j}"
                            constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                            self.DischargedACFConstraintNR[w][t][i][j] = constraint

    def CreateLandResVehicleCapHosConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:
                    for m in self.Instance.RescueVehicleSet:
                        

                        vars_q = []
                        coeff_q = []
                        if self.DemandScenarioTree.HospitalDisruption[w][h] != 1:
                            vars_q = [self.GetIndexCasualtyTransferVariables(w, t, j, l, h, m)
                                        for l in self.Instance.DisasterAreaSet
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                            coeff_q = [-1.0 * self.Instance.Time_D_H_Land[l][h]
                                        for l in self.Instance.DisasterAreaSet
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                        
                        vars_u_L_Hos = []
                        coeff_u_L_Hos = []
                        vars_u_A = []
                        coeff_u_A = []
                        vars_u_L_ACF = []
                        coeff_u_L_ACF = []
                        if self.DemandScenarioTree.HospitalDisruption[w][h] == 1:
                            vars_u_L_Hos = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, hprime, m)
                                            for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set())
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                            coeff_u_L_Hos = [-1.0 * self.Instance.Time_H_H_Land[h][hprime]
                                            for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set())
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                                                        
                            vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set())
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                            coeff_u_A = [-1.0 * self.Instance.Time_A_H_Land[i][h]
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set())
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                                                        
                            vars_u_L_ACF = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, self.Instance.NrHospitals + i, m)
                                            for i in self.Instance.ACFSet
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                            coeff_u_L_ACF = [-1.0 * self.Instance.Time_A_H_Land[i][h]
                                            for i in self.Instance.ACFSet
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                                                
                        ############ Create the left-hand side of the constraint
                        LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))                    
                        LeftHandSide_u_L_Hos = gp.quicksum(coeff_u_L_Hos[i] * self.LandEvacuatedPatients_Var[vars_u_L_Hos[i]] for i in range(len(vars_u_L_Hos)))                        
                        LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))
                        LeftHandSide_u_L_ACF = gp.quicksum(coeff_u_L_ACF[i] * self.LandEvacuatedPatients_Var[vars_u_L_ACF[i]] for i in range(len(vars_u_L_ACF)))
                        LeftHandSide = LeftHandSide_q + LeftHandSide_u_L_Hos + LeftHandSide_u_A + LeftHandSide_u_L_ACF
                        
                        ############ Define the right-hand side (RHS) of the constraint
                        RightHandSide = -1.0 * self.Instance.Land_Rescue_Vehicle_Capacity[m] * self.Instance.Number_Land_Rescue_Vehicle_Hospital[m][h] 

                        ############ Add the constraint to the model
                        constraint_name = f"LandResVehicleCapHos_w_{w}_t_{t}_h_{h}_m_{m}"
                        constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                        self.LandResVehicleCapHosConstraintNR[w][t][h][m] = constraint

    def CreateLandResVehicleCapACFConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for i in self.Instance.ACFSet:
                    for m in self.Instance.RescueVehicleSet:
                        
                        vars_q = [self.GetIndexCasualtyTransferVariables(w, t, j, l, self.Instance.NrHospitals + i, m)
                                    for l in self.Instance.DisasterAreaSet
                                    for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                        coeff_q = [-1.0 * self.Instance.Distance_D_A[l][i]
                                    for l in self.Instance.DisasterAreaSet
                                    for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                                                                    
                        vars_thetaVar = [self.GetIndexLandRescueVehicleVariable(w, i, m)]
                        coeff_thetaVar = [+1.0 * self.Instance.Land_Rescue_Vehicle_Capacity[m]]

                        ############ Create the left-hand side of the constraint
                        LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))                    
                        LeftHandSide_thetaVar = gp.quicksum(coeff_thetaVar[i] * self.LandRescueVehicle_Var[vars_thetaVar[i]] for i in range(len(vars_thetaVar)))                    
                        LeftHandSide = LeftHandSide_q + LeftHandSide_thetaVar
                        
                        ############ Define the right-hand side (RHS) of the constraint
                        RightHandSide = 0 

                        ############ Add the constraint to the model
                        constraint_name = f"LandResVehicleCapACF_w_{w}_t_{t}_i_{i}_m_{m}"
                        constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                        self.LandResVehicleCapACFConstraintNR[w][t][i][m] = constraint

    def CreateAerialResVehicleCapConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:
                    if self.DemandScenarioTree.HospitalDisruption[w][h] == 1:
                        
                        vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                    for m in self.Instance.RescueVehicleSet   
                                    for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                    for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set()) and self.DemandScenarioTree.HospitalDisruption[w][hprime] != 1
                                    for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                        coeff_u_A = [-1.0 * self.Instance.Time_A_H_Aerial[i][hprime]
                                    for m in self.Instance.RescueVehicleSet   
                                    for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                    for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set()) and self.DemandScenarioTree.HospitalDisruption[w][hprime] != 1
                                    for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                                                                              
                        ############ Create the left-hand side of the constraint
                        LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))                    
                        LeftHandSide = LeftHandSide_u_A
                        
                        ############ Define the right-hand side (RHS) of the constraint
                        RightHandSide = -1.0 * self.Instance.Aerial_Rescue_Vehicle_Capacity[0] * self.Instance.Available_Aerial_Vehicles_Hospital[h] 

                        ############ Add the constraint to the model
                        constraint_name = f"AerialResVehicleCap_w_{w}_t_{t}_h_{h}"
                        constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                        self.AerialResVehicleCapConstraintNR[w][t][h] = constraint

    def CreateEvacuationBackupConnectionConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:  
                    for hprime in self.Instance.HospitalSet:
                        if h != hprime:
                            vars_u_L_Hos = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, hprime, m)
                                            for m in self.Instance.RescueVehicleSet
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                            coeff_u_L_Hos = [-1.0
                                            for m in self.Instance.RescueVehicleSet
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                                           
                            vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                        for m in self.Instance.RescueVehicleSet                                        
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                            coeff_u_A = [-1.0
                                        for m in self.Instance.RescueVehicleSet                                        
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]

                            vars_W = [self.GetIndexBackupHospitalVariable(w, h, hprime)]
                            coeff_W = [+1.0 * self.Instance.Hospital_Bed_Capacity[hprime]]
                                                                                    
            
                            ############ Create the left-hand side of the constraint
                            LeftHandSide_u_L_Hos = gp.quicksum(coeff_u_L_Hos[i] * self.LandEvacuatedPatients_Var[vars_u_L_Hos[i]] for i in range(len(vars_u_L_Hos)))                        
                            LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))
                            LeftHandSide_W = gp.quicksum(coeff_W[i] * self.BackupHospital_Var[vars_W[i]] for i in range(len(vars_W)))
                            LeftHandSide = LeftHandSide_u_L_Hos + LeftHandSide_u_A + LeftHandSide_W
                            
                            ############ Define the right-hand side (RHS) of the constraint
                            RightHandSide = 0 

                            ############ Add the constraint to the model
                            constraint_name = f"EvacuationBackupConnection_w_{w}_t_{t}_h_{h}_h'_{hprime}"
                            constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                            self.EvacuationBackupConnectionConstraintNR[w][t][h][hprime] = constraint

    #The following constraint is for linearization of |x-x\{Bar}| in the PHA's objective function.
    def CreatLinearFacilPHAConstraint(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreatLinearPHAConstraint")

        for w in self.ScenarioSet:
            for i in self.Instance.FacilitySet:

                vars_x = [self.GetIndexFacilityEstablishmentVariable(w, i)]

                coeff_x = [-1.0]

                vars_zPlus = [self.GetIndex_PHA_ZPlus_FacilEstablishmentVariable(w, i)]
                coeff_zPlus = [1.0]
                
                vars_zMinus = [self.GetIndex_PHA_ZMinus_FacilEstablishmentVariable(w, i)]
                coeff_zMinus = [-1.0]

                ############ Create the left-hand side of the constraint
                LeftHandSide_x = gp.quicksum(coeff_x[i] * self.Facility_Establishment_Var[vars_x[i]] for i in range(len(vars_x)))
                LeftHandSide_zPlus = gp.quicksum(coeff_zPlus[i] * self.PHA_ZPlus_FacilEstablishment_Var[vars_zPlus[i]] for i in range(len(vars_zPlus)))
                LeftHandSide_zMinus = gp.quicksum(coeff_zMinus[i] * self.PHA_ZMinus_FacilEstablishment_Var[vars_zMinus[i]] for i in range(len(vars_zMinus)))
                
                LeftHandSide = LeftHandSide_x + LeftHandSide_zPlus + LeftHandSide_zMinus
                
                ############ Define the right-hand side (RHS) of the constraint
                RightHandSide = -1               #For starting, we set the RHS to 1, BUT we dynamically update it later in PHA class!
                ############ Add the constraint to the model
                constraint_name = f"LinearPHA_w_{w}_i_{i}"
                
                constraint = self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)
                
                self.LinearPHA_FacilConstraintNR[w][i] = constraint

    def CreateCopyGivenACFEstablishmentConstraints(self):
        self.ACFEstablishmentVarConstraintNR = [["" for _ in self.Instance.ACFSet] 
                                                    for _ in self.ScenarioSet]
        AlreadyAdded = [False for _ in range(self.GetNrACFEstablishmentVariable())]

        # FixedVars equal to the given ones
        for i in self.Instance.ACFSet:
            for w in self.ScenarioSet:
                indexvariable = self.GetIndexACFEstablishmentVariable(w, i)
                indexinarray = indexvariable - self.GetStartACFEstablishmentVariables()
                if not AlreadyAdded[indexinarray]:
                    vars_x = [self.GetIndexACFEstablishmentVariable(w, i)]
                    AlreadyAdded[indexinarray] = True
                    if isinstance(self.GivenACFEstablishment[0], list):  # 2D case
                        righthandside = min(round(self.GivenACFEstablishment[w][i]), 1)
                    else:
                        righthandside = min(round(self.GivenACFEstablishment[i]), 1)
                    # Add constraint
                    constraint_name = f"CopyGivenACFEstablishment_w_{w}_i_{i}"
                    if Constants.Debug: print(f"Adding constraint: {constraint_name} with RHS: {righthandside}")
                    self.LocAloc.addConstr(self.ACFEstablishment_Var[vars_x[0]] == righthandside, name=constraint_name)
                    self.ACFEstablishmentVarConstraintNR[w][i] = constraint_name
    
    def CreateCopyGivenLandRescueVehicleConstraints(self):
        self.LandRescueVehicleVarConstraintNR = [[[None for _ in self.Instance.RescueVehicleSet] 
                                                    for _ in self.Instance.ACFSet]
                                                    for _ in self.ScenarioSet]

        AlreadyAdded = [False for _ in range(self.GetNrLandRescueVehicleVariable())]

        # Iterate through scenarios, ACFs, and vehicle types
        for w in self.ScenarioSet:
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    indexvariable = self.GetIndexLandRescueVehicleVariable(w, i, m)
                    indexinarray = indexvariable - self.GetStartLandRescueVehicleVariables()

                    # Ensure index is valid before checking `AlreadyAdded`
                    if 0 <= indexinarray < len(AlreadyAdded) and not AlreadyAdded[indexinarray]:
                        AlreadyAdded[indexinarray] = True
                        CheckGivenNrLandRescueVehicle = np.array(self.GivenNrLandRescueVehicle)
                        # Determine the right-hand side (RHS) value for 2D or 3D input
                        if isinstance(CheckGivenNrLandRescueVehicle, np.ndarray):
                            if CheckGivenNrLandRescueVehicle.ndim == 3:  # 3D case
                                righthandside = round(self.GivenNrLandRescueVehicle[w][i][m])
                            elif CheckGivenNrLandRescueVehicle.ndim == 2:  # 2D case
                                righthandside = round(self.GivenNrLandRescueVehicle[i][m])
                            else:
                                raise ValueError("GivenNrLandRescueVehicle must be 2D or 3D.")
                        else:
                            raise TypeError("GivenNrLandRescueVehicle must be a NumPy array.")

                        # Add constraint to Gurobi
                        constraint_name = f"CopyGivenNrLandRescueVehicle_w_{w}_i_{i}_m_{m}"
                        if Constants.Debug: print(f"Adding constraint: {constraint_name} with RHS: {righthandside}")
                        self.LocAloc.addConstr(self.LandRescueVehicle_Var[indexvariable] == righthandside, name=constraint_name)
                        self.LandRescueVehicleVarConstraintNR[w][i][m] = constraint_name

    def CreateCopyGivenBackupHospitalConstraints(self):
        self.BackupHospitalVarConstraintNR = [[[None for _ in self.Instance.HospitalSet] 
                                                    for _ in self.Instance.HospitalSet]
                                                    for _ in self.ScenarioSet]

        AlreadyAdded = [False for _ in range(self.GetNrBackupHospitalVariable())]

        # Iterate through scenarios, ACFs, and vehicle types
        for w in self.ScenarioSet:
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:
                    indexvariable = self.GetIndexBackupHospitalVariable(w, h, hprime)
                    indexinarray = indexvariable - self.GetStartBackupHospitalVariables()

                    # Ensure index is valid before checking `AlreadyAdded`
                    if 0 <= indexinarray < len(AlreadyAdded) and not AlreadyAdded[indexinarray]:
                        AlreadyAdded[indexinarray] = True
                        CheckGivenBackupHospital = np.array(self.GivenBackupHospital)
                        # Determine the right-hand side (RHS) value for 2D or 3D input
                        if isinstance(CheckGivenBackupHospital, np.ndarray):
                            if CheckGivenBackupHospital.ndim == 3:  # 3D case
                                righthandside = round(self.GivenBackupHospital[w][h][hprime])
                            elif CheckGivenBackupHospital.ndim == 2:  # 2D case
                                righthandside = round(self.GivenBackupHospital[h][hprime])
                            else:
                                raise ValueError("GivenBackupHospital must be 2D or 3D.")
                        else:
                            raise TypeError("GivenBackupHospital must be a NumPy array.")

                        # Add constraint to Gurobi
                        constraint_name = f"CopyGivenBackupHospital_w_{w}_h_{h}_h'_{hprime}"
                        if Constants.Debug: print(f"Adding constraint: {constraint_name} with RHS: {righthandside}")
                        self.LocAloc.addConstr(self.BackupHospital_Var[indexvariable] == righthandside, name=constraint_name)
                        self.BackupHospitalVarConstraintNR[w][h][hprime] = constraint_name
    
    # This function creates the non anticipitativity constraint
    def CreateNonanticipativityConstraints_x(self): 
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateNonanticipativityConstraints_x")

        for w in self.ScenarioSet[:-1]:
            for i in self.Instance.ACFSet:                
                vars_x1 = self.GetIndexACFEstablishmentVariable(w, i)
                vars_x2 = self.GetIndexACFEstablishmentVariable(w+1, i)

                # Retrieving variable values
                var_value_x1 = self.ACFEstablishment_Var[vars_x1]
                var_value_x2 = self.ACFEstablishment_Var[vars_x2]
                
                coeff_x1 = 1.0
                coeff_x2 = -1.0

                # Calculating left hand side components
                LeftHandSide_x1 = coeff_x1 * var_value_x1
                LeftHandSide_x2 = coeff_x2 * var_value_x2
                LeftHandSide = LeftHandSide_x1 + LeftHandSide_x2
                
                RightHandSide = 0.0

                # Adding the constraint
                constraint_name = f"Nonanticipativity_x_w1_{w}_w2_{(w+1)}_i_{i}"
                self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)
    
    # This function creates the non anticipitativity constraint
    def CreateNonanticipativityConstraints_thetaVar(self): 
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateNonanticipativityConstraints_x")

        for w in self.ScenarioSet[:-1]:
            for i in self.Instance.ACFSet:                
                for m in self.Instance.RescueVehicleSet:                
                    vars_thetaVar1 = self.GetIndexLandRescueVehicleVariable(w, i, m)
                    vars_thetaVar2 = self.GetIndexLandRescueVehicleVariable(w+1, i, m)

                    # Retrieving variable values
                    var_value_thetaVar1 = self.LandRescueVehicle_Var[vars_thetaVar1]
                    var_value_thetaVar2 = self.LandRescueVehicle_Var[vars_thetaVar2]
                    
                    coeff_thetaVar1 = 1.0
                    coeff_thetaVar2 = -1.0

                    # Calculating left hand side components
                    LeftHandSide_thetaVar1 = coeff_thetaVar1 * var_value_thetaVar1
                    LeftHandSide_thetaVar2 = coeff_thetaVar2 * var_value_thetaVar2
                    LeftHandSide = LeftHandSide_thetaVar1 + LeftHandSide_thetaVar2
                    
                    RightHandSide = 0.0

                    # Adding the constraint
                    constraint_name = f"Nonanticipativity_thetaVar_w1_{w}_w2_{(w+1)}_i_{i}_m_{m}"
                    self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)

    # This function creates the non anticipitativity constraint
    def CreateNonanticipativityConstraints_W(self): 
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateNonanticipativityConstraints_x")

        for w in self.ScenarioSet[:-1]:
            for h in self.Instance.HospitalSet:                
                for hprime in self.Instance.HospitalSet:                
                    vars_W1 = self.GetIndexBackupHospitalVariable(w, h, hprime)
                    vars_W2 = self.GetIndexBackupHospitalVariable(w+1, h, hprime)

                    # Retrieving variable values
                    var_value_W1 = self.BackupHospital_Var[vars_W1]
                    var_value_W2 = self.BackupHospital_Var[vars_W2]
                    
                    coeff_W1 = 1.0
                    coeff_W2 = -1.0

                    # Calculating left hand side components
                    LeftHandSide_W1 = coeff_W1 * var_value_W1
                    LeftHandSide_W2 = coeff_W2 * var_value_W2
                    LeftHandSide = LeftHandSide_W1 + LeftHandSide_W2
                    
                    RightHandSide = 0.0

                    # Adding the constraint
                    constraint_name = f"Nonanticipativity_W_w1_{w}_w2_{(w+1)}_h_{h}_h'_{hprime}"
                    self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)

    # This function define the variables and related objective functions
    def CreateVariable_and_Objective_Function(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateVariable_and_Objective_Function")

        ################################# ACF Establishment Variable #################################
        acfestablishmentcost = {}
        self.ACFEstablishment_Var = {}
        for w in self.ScenarioSet:
            for i in self.Instance.ACFSet:                   
                Index_Cost = self.GetIndexACFEstablishmentVariable(w, i) - self.GetStartACFEstablishmentVariables()               
                acfestablishmentcost[Index_Cost] = self.GetACFestablishmentCoeff_Obj(i)
                Index_Var = self.GetIndexACFEstablishmentVariable(w, i)
                var_name = f"x_w_{w}_i_{i}_index_{Index_Var}"
                if self.EvaluateSolution:
                    self.ACFEstablishment_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj = acfestablishmentcost[Index_Cost], lb=0, ub=1, name=var_name)
                else:
                    self.ACFEstablishment_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.BINARY, obj = acfestablishmentcost[Index_Cost], lb=0, ub=1, name=var_name)
        self.LocAloc.update()
        if Constants.We_Are_in_PHA and Constants.Quadratic_to_Linear_PHA:
            ################################# PHA_ZPlus ACF Variable #################################
            self.PHA_ZPlus_ACFEstablishment_Var = {}
            for w in self.ScenarioSet:
                for i in self.Instance.ACFSet:
                    Index_Cost = self.GetIndex_PHA_ZPlus_ACFEstablishmentVariable(w, i) - self.GetStart_PHA_ZPlus_ACFEstablishmentVariables()               
                    Index_Var = self.GetIndex_PHA_ZPlus_ACFEstablishmentVariable(w, i)
                    var_name = f"z_x_Plus_w_{w}_i_{i}_index_{Index_Var}"
                    self.PHA_ZPlus_ACFEstablishment_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=0, lb=0, ub=1, name=var_name)
            self.LocAloc.update()

            ################################# PHA_ZMinus ACF Variable #################################
            self.PHA_ZMinus_ACFEstablishment_Var = {}
            for w in self.ScenarioSet:
                for i in self.Instance.ACFSet:
                    Index_Cost = self.GetIndex_PHA_ZMinus_ACFEstablishmentVariable(w, i) - self.GetStart_PHA_ZMinus_ACFEstablishmentVariables()               
                    Index_Var = self.GetIndex_PHA_ZMinus_ACFEstablishmentVariable(w, i)
                    var_name = f"z_x_Minus_w_{w}_i_{i}_index_{Index_Var}"
                    self.PHA_ZMinus_ACFEstablishment_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=0, lb=0, ub=1, name=var_name)
            self.LocAloc.update()

        ################################# Land Rescue Vehicle Variable #################################
        landRescueVehiclecost = {}
        self.LandRescueVehicle_Var = {}
        for w in self.ScenarioSet:
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:               
                    Index_Cost = self.GetIndexLandRescueVehicleVariable(w, i, m) - self.GetStartLandRescueVehicleVariables()               
                    landRescueVehiclecost[Index_Cost] = self.GetlandRescueVehicleCoeff(i, m)
                    Index_Var = self.GetIndexLandRescueVehicleVariable(w, i, m)
                    var_name = f"thetaVar_w_{w}_i_{i}_m_{m}_index_{Index_Var}"
                    if self.EvaluateSolution:
                        self.LandRescueVehicle_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj = landRescueVehiclecost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
                    else:
                        self.LandRescueVehicle_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.INTEGER, obj = landRescueVehiclecost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.LocAloc.update()
        if Constants.We_Are_in_PHA and Constants.Quadratic_to_Linear_PHA:
            ################################# PHA_ZPlus Land Rescue Vehicle Variable #################################
            self.PHA_ZPlus_LandRescueVehicle_Var = {}
            for w in self.ScenarioSet:
                for m in self.Instance.RescueVehicleSet:
                    for i in self.Instance.ACFSet:
                        Index_Cost = self.GetIndex_PHA_ZPlus_LandRescueVehicleVariable(w, i, m) - self.GetStart_PHA_ZPlus_LandRescueVehicleVariables()               
                        Index_Var = self.GetIndex_PHA_ZPlus_LandRescueVehicleVariable(w, i, m)
                        var_name = f"z_thetaVar_Plus_w_{w}_i_{i}_m_{m}_index_{Index_Var}"
                        self.PHA_ZPlus_LandRescueVehicle_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=0, lb=0, ub=GRB.INFINITY, name=var_name)
            self.LocAloc.update()

            ################################# PHA_ZMinus Land Rescue Vehicle Variable #################################
            self.PHA_ZMinus_LandRescueVehicle_Var = {}
            for w in self.ScenarioSet:
                for m in self.Instance.RescueVehicleSet:
                    for i in self.Instance.ACFSet:
                        Index_Cost = self.GetIndex_PHA_ZMinus_LandRescueVehicleVariable(w, i, m) - self.GetStart_PHA_ZMinus_LandRescueVehicleVariables()               
                        Index_Var = self.GetIndex_PHA_ZMinus_LandRescueVehicleVariable(w, i, m)
                        var_name = f"z_thetaVar_Minus_w_{w}_i_{i}_m_{m}_index_{Index_Var}"
                        self.PHA_ZMinus_LandRescueVehicle_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=0, lb=0, ub=GRB.INFINITY, name=var_name)
            self.LocAloc.update()

        ################################# Backup Hospital Variable #################################
        backupHospitalcost = {}
        self.BackupHospital_Var = {}
        for w in self.ScenarioSet:
            for h in self.Instance.HospitalSet:                   
                for hprime in self.Instance.HospitalSet:                   
                    Index_Cost = self.GetIndexBackupHospitalVariable(w, h, hprime) - self.GetStartBackupHospitalVariables()               
                    backupHospitalcost[Index_Cost] = self.GetbackupHospitalCoeff(h, hprime)
                    Index_Var = self.GetIndexBackupHospitalVariable(w, h, hprime)
                    var_name = f"W_w_{w}_h_{h}_h'_{hprime}_index_{Index_Var}"

                    if self.EvaluateSolution:
                        self.BackupHospital_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=backupHospitalcost[Index_Cost], lb=0, ub=1, name=var_name)
                    else:
                        self.BackupHospital_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.BINARY, obj=backupHospitalcost[Index_Cost], lb=0, ub=1, name=var_name)
        self.LocAloc.update()
        if Constants.We_Are_in_PHA and Constants.Quadratic_to_Linear_PHA:
            ################################# PHA_ZPlus Backup Hospital Variable #################################
            self.PHA_ZPlus_BackupHospital_Var = {}
            for w in self.ScenarioSet:
                for h in self.Instance.HospitalSet:
                    for hprime in self.Instance.HospitalSet:
                        Index_Cost = self.GetIndex_PHA_ZPlus_BackupHospitalVariable(w, h, hprime) - self.GetStart_PHA_ZPlus_BackupHospitalVariables()               
                        Index_Var = self.GetIndex_PHA_ZPlus_BackupHospitalVariable(w, h, hprime)
                        var_name = f"z_W_Plus_w_{w}_i_{i}_index_{Index_Var}"
                        self.PHA_ZPlus_BackupHospital_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=0, lb=0, ub=1, name=var_name)
            self.LocAloc.update()

            ################################# PHA_ZMinus Backup Hospital Variable #################################
            self.PHA_ZMinus_BackupHospital_Var = {}
            for w in self.ScenarioSet:
                for h in self.Instance.HospitalSet:
                    for hprime in self.Instance.HospitalSet:
                        Index_Cost = self.GetIndex_PHA_ZMinus_BackupHospitalVariable(w, h, hprime) - self.GetStart_PHA_ZMinus_BackupHospitalVariables()               
                        Index_Var = self.GetIndex_PHA_ZMinus_BackupHospitalVariable(w, h, hprime)
                        var_name = f"z_W_Minus_w_{w}_i_{i}_index_{Index_Var}"
                        self.PHA_ZMinus_BackupHospital_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=0, lb=0, ub=1, name=var_name)
            self.LocAloc.update()

        ################################# Casualty Transfer Variable #################################
        casualtyTransferCost = {}
        self.CasualtyTransfer_Var = {}
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:  
                for j in self.Instance.InjuryLevelSet:
                    for l in self.Instance.DisasterAreaSet:
                        for u in self.Instance.MedFacilitySet:
                            for m in self.Instance.RescueVehicleSet:
                                Index_Cost = self.GetIndexCasualtyTransferVariables(w, t, j, l, u, m) - self.GetStartCasualtyTransferVariables()
                                casualtyTransferCost[Index_Cost] = self.GetCasualtyTransferCoeff(w, l, u)
                                Index_Var = self.GetIndexCasualtyTransferVariables(w, t, j, l, u, m)
                                var_name = f"q_w_{w}_t_{t}_j_{j}_l_{l}_u_{u}_m_{m}_index_{Index_Var}"
                                self.CasualtyTransfer_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=casualtyTransferCost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.LocAloc.update()
        
        ################################# Unsatisfied Casualties Variable #################################
        unsatisfiedCasualtiesCost = {}
        self.UnsatisfiedCasualties_Var = {}
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:  
                for j in self.Instance.InjuryLevelSet:
                    for l in self.Instance.DisasterAreaSet:
                        Index_Cost = self.GetIndexUnsatisfiedCasualtiesVariables(w, t, j, l) - self.GetStartUnsatisfiedCasualtiesVariables()
                        unsatisfiedCasualtiesCost[Index_Cost] = self.GetUnsatisfiedCasualtiesCoeff(w, j)
                        Index_Var = self.GetIndexUnsatisfiedCasualtiesVariables(w, t, j, l)
                        var_name = f"mu_w_{w}_t_{t}_j_{j}_l_{l}_index_{Index_Var}"
                        self.UnsatisfiedCasualties_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=unsatisfiedCasualtiesCost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.LocAloc.update()
        
        ################################# Discharged Patients Variable #################################
        dischargedPatientsCost = {}
        self.DischargedPatients_Var = {}
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:  
                for j in self.Instance.InjuryLevelSet:
                    for u in self.Instance.MedFacilitySet:
                        Index_Cost = self.GetIndexDischargedPatientsVariables(w, t, j, u) - self.GetStartDischargedPatientsVariables()
                        dischargedPatientsCost[Index_Cost] = 0
                        Index_Var = self.GetIndexDischargedPatientsVariables(w, t, j, u)
                        var_name = f"sigmaVar_w_{w}_t_{t}_j_{j}_u_{u}_index_{Index_Var}"
                        self.DischargedPatients_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=dischargedPatientsCost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.LocAloc.update()
        
        ################################# Land Evacuated Patients Variable #################################
        landEvacuatedPatientsCost = {}
        self.LandEvacuatedPatients_Var = {}
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:  
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        for u in self.Instance.MedFacilitySet:
                            for m in self.Instance.RescueVehicleSet:
                                Index_Cost = self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, u, m) - self.GetStartLandEvacuatedPatientsVariables()
                                landEvacuatedPatientsCost[Index_Cost] = self.GetLandEvacuatedPatientsCoeff(w, t, j, h, u, m)
                                Index_Var = self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, u, m)
                                var_name = f"u_L_w_{w}_t_{t}_j_{j}_h_{h}_u_{u}_m_{m}_index_{Index_Var}"
                                self.LandEvacuatedPatients_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=landEvacuatedPatientsCost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.LocAloc.update()
                
        ################################# Aerial Evacuated Patients Variable #################################
        aerialEvacuatedPatientsCost = {}
        self.AerialEvacuatedPatients_Var = {}
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:  
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        for i in self.Instance.ACFSet:
                            for hprime in self.Instance.HospitalSet:
                                for m in self.Instance.RescueVehicleSet:
                                    Index_Cost = self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m) - self.GetStartAerialEvacuatedPatientsVariables()
                                    aerialEvacuatedPatientsCost[Index_Cost] = self.GetAerialEvacuatedPatientsCoeff(w, t, j, h, i, hprime, m)
                                    Index_Var = self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                    var_name = f"u_A_w_{w}_t_{t}_j_{j}_h_{h}_i_{i}_h'_{hprime}_m_{m}_index_{Index_Var}"
                                    self.AerialEvacuatedPatients_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=aerialEvacuatedPatientsCost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.LocAloc.update()
        
        ################################# Unevacuated Patients Variable #################################
        unevacuatedPatientsCost = {}
        self.UnevacuatedPatients_Var = {}
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:  
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        Index_Cost = self.GetIndexUnevacuatedPatientsVariables(w, t, j, h) - self.GetStartUnevacuatedPatientsVariables()
                        # Check if t is the last time bucket (i.e., T)
                        if t == self.Instance.TimeBucketSet[-1]:  # Assuming the last time bucket is T
                            unevacuatedPatientsCost[Index_Cost] = self.GetUnevacuatedPatientsCoeff(w, t, j, h)
                        else:
                            unevacuatedPatientsCost[Index_Cost] = 0     
                        Index_Var = self.GetIndexUnevacuatedPatientsVariables(w, t, j, h)
                        var_name = f"Phi_w_{w}_t_{t}_j_{j}_h_{h}_index_{Index_Var}"
                        self.UnevacuatedPatients_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=unevacuatedPatientsCost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.LocAloc.update()
        
        ################################# Available Cap Facility Variable #################################
        availableCapFacilityCost = {}
        self.AvailableCapFacility_Var = {}
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:  
                for u in self.Instance.MedFacilitySet:
                    Index_Cost = self.GetIndexAvailableCapFacilityVariables(w, t, u) - self.GetStartAvailableCapFacilityVariables()
                    availableCapFacilityCost[Index_Cost] = 0
                    Index_Var = self.GetIndexAvailableCapFacilityVariables(w, t, u)
                    var_name = f"zeta_Hos_w_{w}_t_{t}_u_{u}_index_{Index_Var}"
                    self.AvailableCapFacility_Var[Index_Var] = self.LocAloc.addVar(vtype=GRB.CONTINUOUS, obj=availableCapFacilityCost[Index_Cost], lb=0, ub=GRB.INFINITY, name=var_name)
        self.LocAloc.update()
        
        ############################################## Set minimization
        self.LocAloc.setObjective(self.LocAloc.getObjective(), GRB.MINIMIZE)

    # Define the constraint of the model
    def CreateConstraints(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateConstraints")

        #self.CreateMaxBackupConstraint()
        self.CreateTotalBudgetConstraint()
        self.CreateLandVehicleAssignmentConstraint()
        self.CreateVehicleACFConnectionConstraint()
        self.CreateCasualtyAllocationConstraint()
        self.CreatePatientAllocationConstraint()
        self.CreateHospitalCapConstraint()
        self.CreateMaxHospitalCapConstraint()
        self.CreateDischargedHospitalConstraint()
        self.CreateACFCapConstraint()
        self.CreateMaxACFCapConstraint()
        self.CreateDischargedACFConstraint()
        self.CreateLandResVehicleCapHosConstraint()
        self.CreateLandResVehicleCapACFConstraint()
        self.CreateAerialResVehicleCapConstraint()
        self.CreateEvacuationBackupConnectionConstraint()

        if Constants.We_Are_in_PHA == False:
            self.CreateNonanticipativityConstraints_x()
            self.CreateNonanticipativityConstraints_thetaVar()
            self.CreateNonanticipativityConstraints_W()

        if Constants.We_Are_in_PHA and Constants.Quadratic_to_Linear_PHA:
            self.CreatLinearFacilPHAConstraint()

        if self.EvaluateSolution:
            self.CreateCopyGivenACFEstablishmentConstraints()
            self.CreateCopyGivenLandRescueVehicleConstraints()
            self.CreateCopyGivenBackupHospitalConstraints()

    def BuildModel(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- BuildModel")
        
        # Initialize the Gurobi model
        self.LocAloc = gp.Model("LocationAllocation")

        # Create variables and objective function
        self.CreateVariable_and_Objective_Function()

        # Create all constraints
        self.CreateConstraints()
        
        self.Parameter_Tunning()

    def Parameter_Tunning(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- Parameter_Tunning")

        self.LocAloc.setParam('TimeLimit', Constants.MIPTimeLimit)
        self.LocAloc.setParam('MIPGap', self.Instance.My_EpGap)
        
        self.LocAloc.setParam('Threads', 1)
        self.LocAloc.setParam('OutputFlag', Constants.ModelOutputFlag)  # Prevents Gurobi from showing the optimization process!

    def ReadNrVariableConstraint(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- ReadNrVariableConstraint")

        num_vars = self.LocAloc.numVars
        num_constrs = self.LocAloc.numConstrs

        return num_vars, num_constrs
    
    def Check_Optimality_and_Print_Solutions(self):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- Check_Optimality_and_Print_Solutions")

        solution = {}
        if self.LocAloc.status == GRB.OPTIMAL:
            print("Optimal solution found.")
            objective_value = self.LocAloc.objVal
            solution['objective_value'] = objective_value
            solution['variables'] = {}
            if(Constants.Debug): print(f"$$$$$$$$ Objective Value: {objective_value}")
            if(Constants.Debug): print("------------------------")
            ######### First-Stage variables
            for index, var_obj in self.ACFEstablishment_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X

            for index, var_obj in self.LandRescueVehicle_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X

            for index, var_obj in self.BackupHospital_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X

            ######### Second-Stage variables
            for index, var_obj in self.CasualtyTransfer_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X       
            for index, var_obj in self.UnsatisfiedCasualties_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X       
            for index, var_obj in self.DischargedPatients_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X       
            for index, var_obj in self.LandEvacuatedPatients_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X       
            for index, var_obj in self.AerialEvacuatedPatients_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X       
            for index, var_obj in self.UnevacuatedPatients_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X       
            for index, var_obj in self.AvailableCapFacility_Var.items():
                if var_obj.X > 1e-6:  
                    if Constants.Debug: print(f"{var_obj.VarName} = {var_obj.X}")
                    solution['variables'][var_obj.VarName] = var_obj.X       
        else:
            print(f"No optimal solution found. Status code: {self.LocAloc.status}")
        return solution
        
    def Solve(self, createsolution = True ):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- Solve")

        start_time = time.time()

        self.Parameter_Tunning()

        end_modeling = time.time()

        if Constants.Debug_lp_Files:
            # Create the directory "MIP_Model_LP" in the current working directory
            if self.EvaluateSolution:
                lp_dir = os.path.join(os.getcwd(), "MIP_Model_LP_Evaluation")
            else:
                lp_dir = os.path.join(os.getcwd(), "MIP_Model_LP")

            if not os.path.exists(lp_dir):
                os.makedirs(lp_dir)

            # Variables to include in the filename
            T = self.Instance.NrTimeBucket
            I = self.Instance.NrACFs
            H = self.Instance.NrHospitals
            L = self.Instance.NrDisasterAreas
            M = self.Instance.NrRescueVehicles
            scenario = len(self.ScenarioSet)

            # Format the filename with the variables
            lp_filename = f"MIP_Model_T_{T}_I_{I}_H_{H}_L_{L}_M_{M}_Scenario_{scenario}.lp"

            # Write the model to a file before optimization
            lp_full_path = os.path.join(lp_dir, lp_filename)
            self.LocAloc.write(lp_full_path)

        self.LocAloc.optimize()

        nrvariable, nrconstraints = self.ReadNrVariableConstraint()

        buildtime = round(end_modeling - start_time, 2)
        solvetime = round(time.time() - end_modeling, 2)

        #sol includes objective function and variables' values
        sol = self.Check_Optimality_and_Print_Solutions()

        if self.LocAloc.status == GRB.OPTIMAL:
            if Constants.Debug:
                print("GRB Solve Time(s): %r   GRB build time(s): %s   cost: %s" % (solvetime, buildtime, sol['objective_value']))
            if createsolution:
                Solution = self.CreateCRPSolution(sol, solvetime, nrvariable, nrconstraints)
            else:
                Solution = None
            return Solution
        elif self.LocAloc.status == GRB.INF_OR_UNBD:
            print("Solution status: INFEASIBLE OR UNBOUNDED")
            # Write the model to an LP file
            self.LocAloc.write("infeasible_model.lp")
        elif self.LocAloc.status == GRB.INFEASIBLE:
            print("Solution status: INFEASIBLE")
            # Write the model to an LP file
            self.LocAloc.write("infeasible_model.lp")
            print("Model is infeasible. Identifying conflicts...")
            # Compute the IIS (Irreducible Inconsistent Subsystem)
            self.LocAloc.computeIIS()
            for c in self.LocAloc.getConstrs():
                if c.IISConstr:
                    print(f"Infeasible constraint!!!!!!!!!!!!!!: {c.constrName}")
        elif self.LocAloc.status == GRB.UNBOUNDED:
            print("Solution status: UNBOUNDED")
            # Write the model to an LP file
            self.LocAloc.write("unbounded_model.lp")

    def CreateCRPSolution(self, sol, solvetime, nrvariable, nrconstraints):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- CreateCRPSolution")

        scenarios = self.Scenarios
        scenarioset = self.ScenarioSet

        objvalue = sol['objective_value']
        
        ################################# x ------- Create an empty list to store the solution values based on the index
        # Step 1: Get values for scenario 0 (first scenario)
        sol_x_values_by_index = []
        Final_ACFEstablishmentCost = 0
        for i in self.Instance.ACFSet:
            index = self.GetIndexACFEstablishmentVariable(0, i)  # For w=0
            if index in self.ACFEstablishment_Var:
                Final_ACFEstablishmentCost += (self.Instance.Fixed_Cost_ACF_Objective[i] * self.ScenarioProbability * self.ACFEstablishment_Var[index].X)
                sol_x_values_by_index.append(self.ACFEstablishment_Var[index].X)

        # Step 2: Replicate values for all scenarios
        sol_x_values_by_index *= len(self.ScenarioSet)  # Repeat values for all scenarios

        # Step 3: Transform into 2D array (one row per scenario)
        solACFEstablishment_x_wi = Tool.Transform2d(sol_x_values_by_index, len(self.ScenarioSet), len(self.Instance.ACFSet))
        if Constants.Debug: print("solACFEstablishment_x_wi:\n ", solACFEstablishment_x_wi)

        ################################# thetaVar ------- Create an empty list to store the solution values based on the index
        sol_thetaVar_values_by_index = []
        Final_LandRescueVehicleCost = 0

        # Step 1: Get values for scenario 0 (first scenario)
        for i in self.Instance.ACFSet:
            for m in self.Instance.RescueVehicleSet:
                index = self.GetIndexLandRescueVehicleVariable(0, i, m)  # For w=0
                if index in self.LandRescueVehicle_Var:
                    Final_LandRescueVehicleCost += (self.Instance.VehicleAssignment_Cost[m] * self.ScenarioProbability * self.LandRescueVehicle_Var[index].X)
                    sol_thetaVar_values_by_index.append(self.LandRescueVehicle_Var[index].X)

        # Step 2: Replicate values for all scenarios
        sol_thetaVar_values_by_index *= len(self.ScenarioSet)  # Repeat values for all scenarios

        # Step 3: Transform into 3D array (one row per scenario)
        solLandRescueVehicle_thetaVar_wim = Tool.Transform3d(sol_thetaVar_values_by_index, len(self.ScenarioSet), len(self.Instance.ACFSet), len(self.Instance.RescueVehicleSet))
        if Constants.Debug: print("solLandRescueVehicle_thetaVar_wim:\n ", solLandRescueVehicle_thetaVar_wim)


        ################################# W ------- Create an empty list to store the solution values based on the index
        sol_W_values_by_index = []
        Final_BackupHospitalCost = 0

        # Step 1: Get values for scenario 0 (first scenario)
        for h in self.Instance.HospitalSet:
            for hprime in self.Instance.HospitalSet:
                index = self.GetIndexBackupHospitalVariable(0, h, hprime)  # For w=0
                if index in self.BackupHospital_Var:
                    Final_BackupHospitalCost += (self.Instance.CoordinationCost[h, hprime] * self.ScenarioProbability * self.BackupHospital_Var[index].X)
                    sol_W_values_by_index.append(self.BackupHospital_Var[index].X)

        # Step 2: Replicate values for all scenarios
        sol_W_values_by_index *= len(self.ScenarioSet)  # Repeat values for all scenarios

        # Step 3: Transform into 3D array (one row per scenario)
        solBackupHospital_W_whhPrime = Tool.Transform3d(sol_W_values_by_index, len(self.ScenarioSet), len(self.Instance.HospitalSet), len(self.Instance.HospitalSet))
        if Constants.Debug: print("solBackupHospital_W_whhPrime:\n ", solBackupHospital_W_whhPrime)

        Final_CasualtyTransferCost = 0
        solCasualtyTransfer_q_wtjlum = []
        Final_UnsatisfiedCasualtiesCost = 0
        solUnsatisfiedCasualties_mu_wtjl = []
        Final_DischargedPatientsCost = 0
        solDischargedPatients_sigmaVar_wtju = []
        Final_LandEvacuatedPatientsCost = 0
        solLandEvacuatedPatients_u_L_wtjhum = []
        Final_AerialEvacuatedPatientsCost = 0
        solAerialEvacuatedPatients_u_A_wtjhihPrimem = []
        Final_UnevacuatedPatientsCost = 0
        solUnevacuatedPatients_Phi_wtjh = []
        Final_AvailableCapFacilityCost = 0
        solAvailableCapFacility_zeta_wtu = []
        if Constants.Obtain_SecondStage_Solution == True:
            ########################################## q ------- Create an empty list to store the solution values based on the index
            sol_q_values_by_index = []
            for w in self.ScenarioSet:
                for t in self.Instance.TimeBucketSet:
                    for j in self.Instance.InjuryLevelSet:
                        for l in self.Instance.DisasterAreaSet:
                            for u in self.Instance.MedFacilitySet:
                                for m in self.Instance.RescueVehicleSet:
                                    index = self.GetIndexCasualtyTransferVariables(w, t, j, l, u, m)
                                    if index in self.CasualtyTransfer_Var:
                                        if u < self.Instance.NrHospitals:
                                            Final_CasualtyTransferCost += self.Instance.Time_D_H_Land[l][u] * self.ScenarioProbability * self.CasualtyTransfer_Var[index].X
                                        else:
                                            Final_CasualtyTransferCost += self.Instance.Time_D_A_Land[l][u - self.Instance.NrHospitals] * self.ScenarioProbability * self.CasualtyTransfer_Var[index].X     
                                        sol_q_values_by_index.append(self.CasualtyTransfer_Var[index].X)
            if Constants.Debug: print("Final_CasualtyTransferCost: ", Final_CasualtyTransferCost)
            solCasualtyTransfer_q_wtjlum = Tool.Transform6d(sol_q_values_by_index, len(self.ScenarioSet), len(self.Instance.TimeBucketSet), len(self.Instance.InjuryLevelSet), len(self.Instance.DisasterAreaSet), len(self.Instance.MedFacilitySet), len(self.Instance.RescueVehicleSet))
            if Constants.Debug: print("solCasualtyTransfer_q_wtjlum:\n ", solCasualtyTransfer_q_wtjlum)

            ########################################## mu ------- Create an empty list to store the solution values based on the index
            sol_mu_values_by_index = []
            for w in self.ScenarioSet:
                for t in self.Instance.TimeBucketSet:
                    for j in self.Instance.InjuryLevelSet:
                        for l in self.Instance.DisasterAreaSet:
                            index = self.GetIndexUnsatisfiedCasualtiesVariables(w, t, j, l)
                            if index in self.UnsatisfiedCasualties_Var:
                                Final_UnsatisfiedCasualtiesCost += (self.Instance.Casualty_Shortage_Cost[j] * self.ScenarioProbability * self.UnsatisfiedCasualties_Var[index].X)
                                sol_mu_values_by_index.append(self.UnsatisfiedCasualties_Var[index].X)
            if Constants.Debug: print("Final_UnsatisfiedCasualtiesCost: ", Final_UnsatisfiedCasualtiesCost)
            solUnsatisfiedCasualties_mu_wtjl = Tool.Transform4d(sol_mu_values_by_index, len(self.ScenarioSet), len(self.Instance.TimeBucketSet), len(self.Instance.InjuryLevelSet), len(self.Instance.DisasterAreaSet))
            if Constants.Debug: print("solUnsatisfiedCasualties_mu_wtjl:\n ", solUnsatisfiedCasualties_mu_wtjl)

            ########################################## sigmaVar ------- Create an empty list to store the solution values based on the index
            sol_sigmaVar_values_by_index = []
            for w in self.ScenarioSet:
                for t in self.Instance.TimeBucketSet:
                    for j in self.Instance.InjuryLevelSet:
                        for u in self.Instance.MedFacilitySet:
                            index = self.GetIndexDischargedPatientsVariables(w, t, j, u)
                            if index in self.DischargedPatients_Var:
                                Final_DischargedPatientsCost += 0
                                sol_sigmaVar_values_by_index.append(self.DischargedPatients_Var[index].X)
            if Constants.Debug: print("Final_DischargedPatientsCost: ", Final_DischargedPatientsCost)
            solDischargedPatients_sigmaVar_wtju = Tool.Transform4d(sol_sigmaVar_values_by_index, len(self.ScenarioSet), len(self.Instance.TimeBucketSet), len(self.Instance.InjuryLevelSet), len(self.Instance.MedFacilitySet))
            if Constants.Debug: print("solDischargedPatients_sigmaVar_wtju:\n ", solDischargedPatients_sigmaVar_wtju)

            ########################################## u_L ------- Create an empty list to store the solution values based on the index
            sol_u_L_values_by_index = []
            for w in self.ScenarioSet:
                for t in self.Instance.TimeBucketSet:
                    for j in self.Instance.InjuryLevelSet:
                        for h in self.Instance.HospitalSet:
                            for u in self.Instance.MedFacilitySet:
                                for m in self.Instance.RescueVehicleSet:
                                    index = self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, u, m)
                                    if index in self.LandEvacuatedPatients_Var:
                                        if Constants.Risk == "Constant":
                                            Final_LandEvacuatedPatientsCost += self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Constant[t][j][h][u][m] * self.ScenarioProbability * self.LandEvacuatedPatients_Var[index].X
                                        elif Constants.Risk == "Linear":
                                            Final_LandEvacuatedPatientsCost += self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Linear[t][j][h][u][m] * self.ScenarioProbability * self.LandEvacuatedPatients_Var[index].X
                                        elif Constants.Risk == "Exponential":
                                            Final_LandEvacuatedPatientsCost += self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Exponential[t][j][h][u][m] * self.ScenarioProbability * self.LandEvacuatedPatients_Var[index].X
                                        sol_u_L_values_by_index.append(self.LandEvacuatedPatients_Var[index].X)
            if Constants.Debug: print("Final_LandEvacuatedPatientsCost: ", Final_LandEvacuatedPatientsCost)
            solLandEvacuatedPatients_u_L_wtjhum = Tool.Transform6d(sol_u_L_values_by_index, len(self.ScenarioSet), len(self.Instance.TimeBucketSet), len(self.Instance.InjuryLevelSet), len(self.Instance.HospitalSet), len(self.Instance.MedFacilitySet), len(self.Instance.RescueVehicleSet))
            if Constants.Debug: print("solLandEvacuatedPatients_u_L_wtjhum:\n ", solLandEvacuatedPatients_u_L_wtjhum)

            ########################################## u_A ------- Create an empty list to store the solution values based on the index
            sol_u_A_values_by_index = []
            for w in self.ScenarioSet:
                for t in self.Instance.TimeBucketSet:
                    for j in self.Instance.InjuryLevelSet:
                        for h in self.Instance.HospitalSet:
                            for i in self.Instance.ACFSet:
                                for hprime in self.Instance.HospitalSet:
                                    for m in self.Instance.RescueVehicleSet:
                                        index = self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                        if index in self.AerialEvacuatedPatients_Var:
                                            if Constants.Risk == "Constant":
                                                Final_AerialEvacuatedPatientsCost += self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Constant[t][j][h][i][hprime][m] * self.ScenarioProbability * self.AerialEvacuatedPatients_Var[index].X
                                            elif Constants.Risk == "Linear":
                                                Final_AerialEvacuatedPatientsCost += self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Linear[t][j][h][i][hprime][m] * self.ScenarioProbability * self.AerialEvacuatedPatients_Var[index].X
                                            elif Constants.Risk == "Exponential":
                                                Final_AerialEvacuatedPatientsCost += self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Exponential[t][j][h][i][hprime][m] * self.ScenarioProbability * self.AerialEvacuatedPatients_Var[index].X
                                            sol_u_A_values_by_index.append(self.AerialEvacuatedPatients_Var[index].X)
            if Constants.Debug: print("Final_AerialEvacuatedPatientsCost: ", Final_AerialEvacuatedPatientsCost)
            solAerialEvacuatedPatients_u_A_wtjhihPrimem = Tool.Transform7d(sol_u_A_values_by_index, len(self.ScenarioSet), len(self.Instance.TimeBucketSet), len(self.Instance.InjuryLevelSet), len(self.Instance.HospitalSet), len(self.Instance.ACFSet), len(self.Instance.HospitalSet), len(self.Instance.RescueVehicleSet))
            if Constants.Debug: print("solAerialEvacuatedPatients_u_A_wtjhihPrimem:\n ", solAerialEvacuatedPatients_u_A_wtjhihPrimem)

            ########################################## Phi ------- Create an empty list to store the solution values based on the index
            sol_Phi_values_by_index = []
            for w in self.ScenarioSet:
                for t in self.Instance.TimeBucketSet:
                    for j in self.Instance.InjuryLevelSet:
                        for h in self.Instance.HospitalSet:
                            index = self.GetIndexUnevacuatedPatientsVariables(w, t, j, h)
                            if index in self.UnevacuatedPatients_Var:
                                if t == self.Instance.TimeBucketSet[-1]:  # Assuming the last time bucket is T
                                    Final_UnevacuatedPatientsCost += self.GetUnevacuatedPatientsCoeff(w, t, j, h) * self.UnevacuatedPatients_Var[index].X
                                else:
                                    Final_UnevacuatedPatientsCost += 0 * self.ScenarioProbability * self.UnevacuatedPatients_Var[index].X
                                sol_Phi_values_by_index.append(self.UnevacuatedPatients_Var[index].X)
            if Constants.Debug: print("Final_UnevacuatedPatientsCost: ", Final_UnevacuatedPatientsCost)
            solUnevacuatedPatients_Phi_wtjh = Tool.Transform4d(sol_Phi_values_by_index, len(self.ScenarioSet), len(self.Instance.TimeBucketSet), len(self.Instance.InjuryLevelSet), len(self.Instance.HospitalSet))
            if Constants.Debug: print("solUnevacuatedPatients_Phi_wtjh:\n ", solUnevacuatedPatients_Phi_wtjh)

            ########################################## zigma ------- Create an empty list to store the solution values based on the index
            sol_zeta_values_by_index = []
            for w in self.ScenarioSet:
                for t in self.Instance.TimeBucketSet:
                    for u in self.Instance.MedFacilitySet:
                        index = self.GetIndexAvailableCapFacilityVariables(w, t, u)
                        if index in self.AvailableCapFacility_Var:
                            Final_AvailableCapFacilityCost += (0 * self.ScenarioProbability * self.AvailableCapFacility_Var[index].X)
                            sol_zeta_values_by_index.append(self.AvailableCapFacility_Var[index].X)
            if Constants.Debug: print("Final_AvailableCapFacilityCost: ", Final_AvailableCapFacilityCost)
            solAvailableCapFacility_zeta_wtu = Tool.Transform3d(sol_zeta_values_by_index, len(self.ScenarioSet), len(self.Instance.TimeBucketSet), len(self.Instance.MedFacilitySet))
            if Constants.Debug: print("solAvailableCapFacility_zeta_wtu:\n ", solAvailableCapFacility_zeta_wtu)

        ##########################################

        solution = Solution(instance=self.Instance, 
                            solACFEstablishment_x_wi = solACFEstablishment_x_wi, 
                            solLandRescueVehicle_thetaVar_wim = solLandRescueVehicle_thetaVar_wim, 
                            solBackupHospital_W_whhPrime = solBackupHospital_W_whhPrime, 
                            solCasualtyTransfer_q_wtjlum = solCasualtyTransfer_q_wtjlum, 
                            solUnsatisfiedCasualties_mu_wtjl = solUnsatisfiedCasualties_mu_wtjl, 
                            solDischargedPatients_sigmaVar_wtju = solDischargedPatients_sigmaVar_wtju, 
                            solLandEvacuatedPatients_u_L_wtjhum = solLandEvacuatedPatients_u_L_wtjhum, 
                            solAerialEvacuatedPatients_u_A_wtjhihPrimem = solAerialEvacuatedPatients_u_A_wtjhihPrimem, 
                            solUnevacuatedPatients_Phi_wtjh = solUnevacuatedPatients_Phi_wtjh, 
                            solAvailableCapFacility_zeta_wtu = solAvailableCapFacility_zeta_wtu, 
                            Final_ACFEstablishmentCost = Final_ACFEstablishmentCost, 
                            Final_LandRescueVehicleCost = Final_LandRescueVehicleCost, 
                            Final_BackupHospitalCost = Final_BackupHospitalCost, 
                            Final_CasualtyTransferCost = Final_CasualtyTransferCost, 
                            Final_UnsatisfiedCasualtiesCost = Final_UnsatisfiedCasualtiesCost, 
                            Final_DischargedPatientsCost = Final_DischargedPatientsCost, 
                            Final_LandEvacuatedPatientsCost = Final_LandEvacuatedPatientsCost, 
                            Final_AerialEvacuatedPatientsCost = Final_AerialEvacuatedPatientsCost, 
                            Final_UnevacuatedPatientsCost = Final_UnevacuatedPatientsCost, 
                            Final_AvailableCapFacilityCost = Final_AvailableCapFacilityCost, 
                            scenarioset=scenarios, 
                            scenariotree=self.DemandScenarioTree, 
                            partialsolution=False)
        if Constants.Debug: print("------------Moving BACK from 'Solution' class (Constructor) to 'MIPSolver' Class ('CreateCRPSolution' Function))---------------")
        
        solution.GRBCost = objvalue
        solution.GRBGap = 0
        solution.GRBNrVariables = nrvariable
        solution.GRBNrConstraints = nrconstraints
        solution.GRBTime = solvetime

        return solution

    def UpdateCasualtyAllocationConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for l in self.Instance.DisasterAreaSet:
                        # Remove the old constraint from the model
                        if self.CasualtyAllocationConstraintNR[w][t][j][l] is not None:
                            self.LocAloc.remove(self.CasualtyAllocationConstraintNR[w][t][j][l])

                        vars_qh = [self.GetIndexCasualtyTransferVariables(w, t, j, l, h, m)
                                    for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                    for h in self.Instance.HospitalSet if self.Instance.J_u[j][h] == 1 and self.Scenarios[w].HospitalDisruption[h] != 1]
                        coeff_qh = [1.0 
                                    for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                    for h in self.Instance.HospitalSet if self.Instance.J_u[j][h] == 1 and self.Scenarios[w].HospitalDisruption[h] != 1]

                        vars_qi = [self.GetIndexCasualtyTransferVariables(w, t, j, l, self.Instance.NrHospitals + i, m)
                                    for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                    for i in self.Instance.ACFSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                        coeff_qi = [1.0 
                                    for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                    for i in self.Instance.ACFSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                        
                        vars_mu = [self.GetIndexUnsatisfiedCasualtiesVariables(w, t, j, l)]
                        coeff_mu = [1.0]

                        vars_mu_prev = []
                        coeff_mu_prev = []
                        if t > 0:
                            vars_mu_prev = [self.GetIndexUnsatisfiedCasualtiesVariables(w, t-1, j, l)]
                            coeff_mu_prev = [-1.0]


                        ############ Create the left-hand side of the constraint
                        LeftHandSide_qh = gp.quicksum(coeff_qh[i] * self.CasualtyTransfer_Var[vars_qh[i]] for i in range(len(vars_qh)))
                        LeftHandSide_qi = gp.quicksum(coeff_qi[i] * self.CasualtyTransfer_Var[vars_qi[i]] for i in range(len(vars_qi)))
                        LeftHandSide_mu = gp.quicksum(coeff_mu[i] * self.UnsatisfiedCasualties_Var[vars_mu[i]] for i in range(len(vars_mu)))
                        LeftHandSide_mu_prev = gp.quicksum(coeff_mu_prev[i] * self.UnsatisfiedCasualties_Var[vars_mu_prev[i]] for i in range(len(vars_mu_prev)))
                        LeftHandSide = LeftHandSide_qh + LeftHandSide_qi + LeftHandSide_mu + LeftHandSide_mu_prev
                        
                        ############ Define the right-hand side (RHS) of the constraint
                        RightHandSide = self.Scenarios[w].CasualtyDemand[t][j][l]  
                        ############ Add the constraint to the model
                        constraint_name = f"CasualtyAllocation_w_{w}_t_{t}_j_{j}_l_{l}"
                        constraint = self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)
                        self.CasualtyAllocationConstraintNR[w][t][j][l] = constraint

    def UpdatePatientAllocationConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        if self.Scenarios[w].HospitalDisruption[h] == 1:
                            # Remove the old constraint from the model
                            if self.PatientAllocationConstraintNR[w][t][j][h] is not None:
                                self.LocAloc.remove(self.PatientAllocationConstraintNR[w][t][j][h])

                            vars_u_L_h = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, hprime, m)
                                            for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                            for hprime in self.Instance.HospitalSet
                                            if self.Instance.J_u[j][hprime] == 1 
                                            and self.Scenarios[w].HospitalDisruption[hprime] != 1
                                            and hprime in self.Instance.K_h.get(h, set())]
                            coeff_u_L_h = [1.0 
                                            for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                            for hprime in self.Instance.HospitalSet
                                            if self.Instance.J_u[j][hprime] == 1 
                                            and self.Scenarios[w].HospitalDisruption[hprime] != 1
                                            and hprime in self.Instance.K_h.get(h, set())]

                            vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                        for hprime in self.Instance.HospitalSet if self.Instance.J_u[j][hprime] == 1 and self.Scenarios[w].HospitalDisruption[hprime] != 1  and hprime in self.Instance.K_h.get(h, set())  # Ensure hprime is in K_h[h]
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set]
                            coeff_u_A = [1.0 
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                        for hprime in self.Instance.HospitalSet if self.Instance.J_u[j][hprime] == 1 and self.Scenarios[w].HospitalDisruption[hprime] != 1  and hprime in self.Instance.K_h.get(h, set())  # Ensure hprime is in K_h[h]
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set]
                                                        
                            vars_u_L_i = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, self.Instance.NrHospitals + i, m)
                                            for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                            for i in self.Instance.ACFSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                            coeff_u_L_i = [1.0 
                                            for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1
                                            for i in self.Instance.ACFSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]


                            vars_Phi = [self.GetIndexUnevacuatedPatientsVariables(w, t, j, h)]
                            coeff_Phi = [1.0]

                            vars_Phi_prev = []
                            coeff_Phi_prev = []
                            if t > 0:
                                vars_Phi_prev = [self.GetIndexUnevacuatedPatientsVariables(w, t-1, j, h)]
                                coeff_Phi_prev = [-1.0]

                            ############ Create the left-hand side of the constraint
                            LeftHandSide_u_L_h = gp.quicksum(coeff_u_L_h[i] * self.LandEvacuatedPatients_Var[vars_u_L_h[i]] for i in range(len(vars_u_L_h)))
                            LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))
                            LeftHandSide_u_L_i = gp.quicksum(coeff_u_L_i[i] * self.LandEvacuatedPatients_Var[vars_u_L_i[i]] for i in range(len(vars_u_L_i)))
                            LeftHandSide_Phi = gp.quicksum(coeff_Phi[i] * self.UnevacuatedPatients_Var[vars_Phi[i]] for i in range(len(vars_Phi)))
                            LeftHandSide_Phi_prev = gp.quicksum(coeff_Phi_prev[i] * self.UnevacuatedPatients_Var[vars_Phi_prev[i]] for i in range(len(vars_Phi_prev)))
                            LeftHandSide = LeftHandSide_u_L_h + LeftHandSide_u_A + LeftHandSide_u_L_i + LeftHandSide_Phi + LeftHandSide_Phi_prev
                            
                            ############ Define the right-hand side (RHS) of the constraint
                            if t == 0:
                                RightHandSide = self.Scenarios[w].PatientDemand[j][h]  
                            else:
                                RightHandSide = 0
                            ############ Add the constraint to the model
                            constraint_name = f"PatientAllocation_w_{w}_t_{t}_j_{j}_h_{h}"
                            constraint = self.LocAloc.addConstr(LeftHandSide == RightHandSide, name=constraint_name)
                            self.PatientAllocationConstraintNR[w][t][j][h] = constraint

    def UpdateMaxHospitalCapConstraint(self):

        for w_idx, w in enumerate(self.ScenarioSet):
            for t_idx, t in enumerate(self.Instance.TimeBucketSet):
                for h_idx, h in enumerate(self.Instance.HospitalSet):
                    # Calculate the new right-hand side for the constraint
                    righthandside = (1 - self.Scenarios[w].HospitalDisruption[h]) * -1.0 * self.Instance.Hospital_Bed_Capacity[h]
                    # Retrieve the constraint reference
                    constr_ref = self.MaxHospitalCapConstraintNR[w_idx][t_idx][h_idx]
                    # Update the RHS of the constraint
                    constr_ref.RHS = righthandside
    
    def UpdateDischargedHospitalConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:
                    for j in self.Instance.InjuryLevelSet:
                        if self.Instance.J_u[j][h] == 1:
                            # Remove the old constraint from the model
                            if self.DischargedHospitalConstraintNR[w][t][h][j] is not None:
                                self.LocAloc.remove(self.DischargedHospitalConstraintNR[w][t][h][j])                            

                            vars_sigmaVar = [self.GetIndexDischargedPatientsVariables(w, t, j, h)]
                            coeff_sigmaVar = [-1.0]

                            vars_q = [self.GetIndexCasualtyTransferVariables(w, t - k, j, l, h, m)
                                        for l in self.Instance.DisasterAreaSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_q = [1.0 * self.Scenarios[w].PatientDischargedPercentage[k][j][h] 
                                        for l in self.Instance.DisasterAreaSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]

                            vars_u_L = [self.GetIndexLandEvacuatedPatientsVariables(w, t - k, j, hprime, h, m)
                                        for hprime in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_u_L = [1.0 * self.Scenarios[w].PatientDischargedPercentage[k][j][h] 
                                        for hprime in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]

                            vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t - k, j, hprime, i, h, m)
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for hprime in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_u_A = [1.0 * self.Scenarios[w].PatientDischargedPercentage[k][j][h] 
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for hprime in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]             
                                                                                        
                            ############ Create the left-hand side of the constraint
                            LeftHandSide_sigmaVar = gp.quicksum(coeff_sigmaVar[i] * self.DischargedPatients_Var[vars_sigmaVar[i]] for i in range(len(vars_sigmaVar)))
                            LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))
                            LeftHandSide_u_L = gp.quicksum(coeff_u_L[i] * self.LandEvacuatedPatients_Var[vars_u_L[i]] for i in range(len(vars_u_L)))
                            LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))
                            LeftHandSide = LeftHandSide_sigmaVar + LeftHandSide_q + LeftHandSide_u_L + LeftHandSide_u_A
                            
                            ############ Define the right-hand side (RHS) of the constraint
                            RightHandSide = 0

                            ############ Add the constraint to the model
                            constraint_name = f"DischargedHospital_w_{w}_t_{t}_h_{h}_j_{j}"
                            constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                            self.DischargedHospitalConstraintNR[w][t][h][j] = constraint

    def UpdateDischargedACFConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for i in self.Instance.ACFSet:
                    for j in self.Instance.InjuryLevelSet:
                        if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1:
                            # Remove the old constraint from the model
                            if self.DischargedACFConstraintNR[w][t][i][j] is not None:
                                self.LocAloc.remove(self.DischargedACFConstraintNR[w][t][i][j]) 

                            vars_sigmaVar = [self.GetIndexDischargedPatientsVariables(w, t, j, self.Instance.NrHospitals + i)]
                            coeff_sigmaVar = [-1.0]

                            vars_q = [self.GetIndexCasualtyTransferVariables(w, t - k, j, l, self.Instance.NrHospitals + i, m)
                                        for l in self.Instance.DisasterAreaSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_q = [1.0 * self.Scenarios[w].PatientDischargedPercentage[k][j][self.Instance.NrHospitals + i] 
                                        for l in self.Instance.DisasterAreaSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]

                            vars_u_L = [self.GetIndexLandEvacuatedPatientsVariables(w, t - k, j, h, self.Instance.NrHospitals + i, m)
                                        for h in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]
                            coeff_u_L = [1.0 * self.Scenarios[w].PatientDischargedPercentage[k][j][self.Instance.NrHospitals + i] 
                                        for h in self.Instance.HospitalSet
                                        for m in self.Instance.RescueVehicleSet if self.Instance.J_m[j][m] == 1  # Ensure m is valid for j
                                        for k in range(t + 1)]       
                                                                                        
                            ############ Create the left-hand side of the constraint
                            LeftHandSide_sigmaVar = gp.quicksum(coeff_sigmaVar[i] * self.DischargedPatients_Var[vars_sigmaVar[i]] for i in range(len(vars_sigmaVar)))
                            LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))
                            LeftHandSide_u_L = gp.quicksum(coeff_u_L[i] * self.LandEvacuatedPatients_Var[vars_u_L[i]] for i in range(len(vars_u_L)))
                            LeftHandSide = LeftHandSide_sigmaVar + LeftHandSide_q + LeftHandSide_u_L
                            
                            ############ Define the right-hand side (RHS) of the constraint
                            RightHandSide = 0

                            ############ Add the constraint to the model
                            constraint_name = f"DischargedACF_w_{w}_t_{t}_i_{i}_j_{j}"
                            constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                            self.DischargedACFConstraintNR[w][t][i][j] = constraint

    def UpdateLandResVehicleCapHosConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:
                    for m in self.Instance.RescueVehicleSet:
                        # Remove the old constraint from the model
                        if self.LandResVehicleCapHosConstraintNR[w][t][h][m] is not None:
                            self.LocAloc.remove(self.LandResVehicleCapHosConstraintNR[w][t][h][m])                         

                        vars_q = []
                        coeff_q = []
                        if self.Scenarios[w].HospitalDisruption[h] != 1:
                            vars_q = [self.GetIndexCasualtyTransferVariables(w, t, j, l, h, m)
                                        for l in self.Instance.DisasterAreaSet
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                            coeff_q = [-1.0 * self.Instance.Time_D_H_Land[l][h]
                                        for l in self.Instance.DisasterAreaSet
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                        
                        vars_u_L_Hos = []
                        coeff_u_L_Hos = []
                        vars_u_A = []
                        coeff_u_A = []
                        vars_u_L_ACF = []
                        coeff_u_L_ACF = []
                        if self.Scenarios[w].HospitalDisruption[h] == 1:
                            vars_u_L_Hos = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, hprime, m)
                                            for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set())
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                            coeff_u_L_Hos = [-1.0 * self.Instance.Time_H_H_Land[h][hprime]
                                            for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set())
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                                                        
                            vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set())
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                            coeff_u_A = [-1.0 * self.Instance.Time_A_H_Land[i][h]
                                        for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                        for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set())
                                        for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                                                        
                            vars_u_L_ACF = [self.GetIndexLandEvacuatedPatientsVariables(w, t, j, h, self.Instance.NrHospitals + i, m)
                                            for i in self.Instance.ACFSet
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                            coeff_u_L_ACF = [-1.0 * self.Instance.Time_A_H_Land[i][h]
                                            for i in self.Instance.ACFSet
                                            for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][self.Instance.NrHospitals + i] == 1]
                                                
                        ############ Create the left-hand side of the constraint
                        LeftHandSide_q = gp.quicksum(coeff_q[i] * self.CasualtyTransfer_Var[vars_q[i]] for i in range(len(vars_q)))                    
                        LeftHandSide_u_L_Hos = gp.quicksum(coeff_u_L_Hos[i] * self.LandEvacuatedPatients_Var[vars_u_L_Hos[i]] for i in range(len(vars_u_L_Hos)))                        
                        LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))
                        LeftHandSide_u_L_ACF = gp.quicksum(coeff_u_L_ACF[i] * self.LandEvacuatedPatients_Var[vars_u_L_ACF[i]] for i in range(len(vars_u_L_ACF)))
                        LeftHandSide = LeftHandSide_q + LeftHandSide_u_L_Hos + LeftHandSide_u_A + LeftHandSide_u_L_ACF
                        
                        ############ Define the right-hand side (RHS) of the constraint
                        RightHandSide = -1.0 * self.Instance.Land_Rescue_Vehicle_Capacity[m] * self.Instance.Number_Land_Rescue_Vehicle_Hospital[m][h] 

                        ############ Add the constraint to the model
                        constraint_name = f"LandResVehicleCapHos_w_{w}_t_{t}_h_{h}_m_{m}"
                        constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                        self.LandResVehicleCapHosConstraintNR[w][t][h][m] = constraint

    def UpdateAerialResVehicleCapConstraint(self):
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for h in self.Instance.HospitalSet:
                    if self.Scenarios[w].HospitalDisruption[h] == 1:
                        # Remove the old constraint from the model
                        if self.AerialResVehicleCapConstraintNR[w][t][h] is not None:
                            self.LocAloc.remove(self.AerialResVehicleCapConstraintNR[w][t][h])    

                        vars_u_A = [self.GetIndexAerialEvacuatedPatientsVariables(w, t, j, h, i, hprime, m)
                                    for m in self.Instance.RescueVehicleSet   
                                    for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                    for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set()) and self.DemandScenarioTree.HospitalDisruption[w][hprime] != 1
                                    for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                        coeff_u_A = [-1.0 * self.Instance.Time_A_H_Aerial[i][hprime]
                                    for m in self.Instance.RescueVehicleSet   
                                    for i in self.Instance.ACFSet if i in self.Instance.I_A_Set   
                                    for hprime in self.Instance.HospitalSet if hprime in self.Instance.K_h.get(h, set()) and self.DemandScenarioTree.HospitalDisruption[w][hprime] != 1
                                    for j in self.Instance.InjuryLevelSet if self.Instance.J_u[j][h] == 1]
                                                                              
                        ############ Create the left-hand side of the constraint
                        LeftHandSide_u_A = gp.quicksum(coeff_u_A[i] * self.AerialEvacuatedPatients_Var[vars_u_A[i]] for i in range(len(vars_u_A)))                    
                        LeftHandSide = LeftHandSide_u_A
                        
                        ############ Define the right-hand side (RHS) of the constraint
                        RightHandSide = -1.0 * self.Instance.Aerial_Rescue_Vehicle_Capacity[0] * self.Instance.Available_Aerial_Vehicles_Hospital[h] 

                        ############ Add the constraint to the model
                        constraint_name = f"AerialResVehicleCap_w_{w}_t_{t}_h_{h}"
                        constraint = self.LocAloc.addConstr(LeftHandSide >= RightHandSide, name=constraint_name)
                        self.AerialResVehicleCapConstraintNR[w][t][h] = constraint

    def ModifyMipForScenarioTree(self, scenariotree):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- ModifyMipForScenarioTree")

        self.DemandScenarioTree = scenariotree
        self.DemandScenarioTree.Owner = self
        self.NrScenario = scenariotree.TreeStructure[1]
        if Constants.Debug: print(f"Number of scenarios: {self.NrScenario}")
        self.ComputeIndices()
        self.Scenarios = scenariotree.GetAllScenarioSet()
        if Constants.Debug: print(f"Retrieved {len(self.Scenarios)} scenarios.")
        self.ScenarioSet = range(self.NrScenario)
        if Constants.Debug: print(f"ScenarioSet range set to 0 to {self.NrScenario - 1}.")

        # Update Constraints for new scenarios
        self.UpdateCasualtyAllocationConstraint()
        self.UpdatePatientAllocationConstraint()
        self.UpdateMaxHospitalCapConstraint()
        self.UpdateDischargedHospitalConstraint()
        self.UpdateDischargedACFConstraint()
        self.UpdateLandResVehicleCapHosConstraint()
        self.UpdateAerialResVehicleCapConstraint()

        self.LocAloc.update()   
    
    def ModifyMipForFacil_LinearPHA(self, Implementable_FacilityEstablishment_x_wi):
        if Constants.Debug: print("\n We are in 'MIPSolver' Class -- ModifyMipForFacil_LinearPHA")
        print("Implementable_FacilityEstablishment_x_wi:\n", Implementable_FacilityEstablishment_x_wi)

        # Update the right-hand side (RHS) of the Linear PHA for Facil constraints
        for w_idx, w in enumerate(self.ScenarioSet):
            for i_idx, i in enumerate(self.Instance.FacilitySet):
                # Calculate the new right-hand side for the constraint

                righthandside = -1.0 * Implementable_FacilityEstablishment_x_wi[w][i]

                # Retrieve the constraint reference
                constr_ref = self.LinearPHA_FacilConstraintNR[w_idx][i_idx]
                
                # Update the RHS of the constraint
                constr_ref.RHS = righthandside

        self.LocAloc.update()  

    def ChangeACFEstablishmentVarToContinuous(self):
        if Constants.Debug: print("\n We are in 'SDDPStage' Class -- (ChangeACFEstablishmentVarToBinary)")

        for w in self.ScenarioSet:
            for i in self.Instance.ACFSet:

                Index_Var = self.GetIndexACFEstablishmentVariable(w, i)
                variable_to_update = self.ACFEstablishment_Var[Index_Var]

                variable_to_update.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)
                variable_to_update.setAttr(GRB.Attr.LB, 0.0)
                variable_to_update.setAttr(GRB.Attr.UB, 1.0)
                self.LocAloc.update()          
    
    def ChangeLandRescueVehicleVarToContinuous(self):
        if Constants.Debug: print("\n We are in 'SDDPStage' Class -- (ChangeLandRescueVehicleVarToContinuous)")

        for w in self.ScenarioSet:
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:

                    Index_Var = self.GetIndexLandRescueVehicleVariable(w, i, m)
                    variable_to_update = self.LandRescueVehicle_Var[Index_Var]

                    variable_to_update.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)
                    variable_to_update.setAttr(GRB.Attr.LB, 0.0)
                    variable_to_update.setAttr(GRB.Attr.UB, GRB.INFINITY)
                    self.LocAloc.update()          
    
    def ChangeBackupHospitalVarToContinuous(self):
        if Constants.Debug: print("\n We are in 'SDDPStage' Class -- (ChangeBackupHospitalVarToContinuous)")

        for w in self.ScenarioSet:
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:

                    Index_Var = self.GetIndexBackupHospitalVariable(w, h, hprime)
                    variable_to_update = self.BackupHospital_Var[Index_Var]

                    variable_to_update.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)
                    variable_to_update.setAttr(GRB.Attr.LB, 0.0)
                    variable_to_update.setAttr(GRB.Attr.UB, 1.0)
                    self.LocAloc.update()          