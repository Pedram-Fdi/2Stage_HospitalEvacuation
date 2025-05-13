from __future__ import absolute_import, division, print_function
from datetime import datetime
import math
import csv
from ScenarioTree import ScenarioTree
from Constants import Constants
from Tool import Tool
from Instance import Instance
import openpyxl as opxl
from ast import literal_eval
import numpy as np
import pandas as pd
from Scenario import Scenario
from TestIdentificator import TestIdentificator

#This class define a solution to MRP
class Solution(object):

    #constructor
    def __init__(self, 
                 instance=None, 
                 solACFEstablishment_x_wi = None, 
                 solLandRescueVehicle_thetaVar_wim = None, 
                 solBackupHospital_W_whhPrime = None, 
                 solCasualtyTransfer_q_wtjlum = None, 
                 solUnsatisfiedCasualties_mu_wtjl = None, 
                 solDischargedPatients_sigmaVar_wtju = None, 
                 solLandEvacuatedPatients_u_L_wtjhum = None, 
                 solAerialEvacuatedPatients_u_A_wtjhihPrimem = None, 
                 solUnevacuatedPatients_Phi_wtjh = None, 
                 solAvailableCapFacility_zeta_wtu = None, 
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
                 scenarioset=None, 
                 scenariotree=None, 
                 partialsolution=False):
        if Constants.Debug: print("\n We are in 'Solution' Class -- Constructor")
        
        self.acfestablishmentcost = -1
        self.landRescueVehiclecost = -1
        self.backupHospitalcost = -1

        self.casualtyTransferCost = -1
        self.unsatisfiedCasualtiesCost = -1
        self.dischargedPatientsCost = -1
        self.landEvacuatedPatientsCost = -1
        self.aerialEvacuatedPatientsCost = -1
        self.unevacuatedPatientsCost = -1
        self.availableCapFacilityCost = -1

        self.InSamplePercentOnTimeTransfer = -1
        self.InSamplePercentOnTimeEvacuation = -1
        self.InSamplePercentNotEvacuatedAtAll = -1

        self.InSampleAverageOnTimeTransfer = []
        self.InSampleAverageOnTimeEvacuation = []
        self.InSampleAverageNotEvacuatedAtAll = []


        self.InSampleAverageACFEstablishment = []
        self.InSampleAverageLandRescueVehicle = []
        self.InSampleAverageBackupHospital = []


        self.TotalCost =-1

        self.Instance = instance

        
        self.InSampleAverageCasualtyTransfer = []
        self.InSampleAverageUnsatisfiedCasualties = []
        self.InSampleAverageDischargedPatients = []
        self.InSampleAverageLandEvacuatedPatients = []
        self.InSampleAverageAerialEvacuatedPatients = []
        self.InSampleAverageUnevacuatedPatients = []
        self.InSampleAverageAvailableCapFacility = []

        

        # The objecie value as outputed by GRB,
        self.GRBCost = -1
        self.GRBGap = -1
        self.GRBTime = -1
        self.TotalTime = 0
        self.GRBNrConstraints = -1
        self.GRBNrVariables = -1

        self.PHCost = -1
        self.PHNrIteration = -1

        #The set of scenario on which the solution is found
        self.Scenarioset = scenarioset

        # Wrap the ScenarioTree input in a list if it's not already a list
        if scenariotree is not None:
            self.ScenarioTree = [scenariotree]  # Initialize with the input tree
        else:
            self.ScenarioTree = []  # Empty list if no tree is provided


        if not self.Scenarioset is None:
            self.SenarioNrset = range(len(self.Scenarioset))
        if Constants.Debug: print("\n We are in 'Solution' Class -- Constructor")

        self.ACFEstablishment_x_wi = solACFEstablishment_x_wi
        self.LandRescueVehicle_thetaVar_wim = solLandRescueVehicle_thetaVar_wim
        self.BackupHospital_W_whhPrime = solBackupHospital_W_whhPrime

        self.CasualtyTransfer_q_wtjlum = solCasualtyTransfer_q_wtjlum
        self.UnsatisfiedCasualties_mu_wtjl = solUnsatisfiedCasualties_mu_wtjl
        self.DischargedPatients_sigmaVar_wtju = solDischargedPatients_sigmaVar_wtju
        self.LandEvacuatedPatients_u_L_wtjhum = solLandEvacuatedPatients_u_L_wtjhum
        self.AerialEvacuatedPatients_u_A_wtjhihPrimem = solAerialEvacuatedPatients_u_A_wtjhihPrimem
        self.UnevacuatedPatients_Phi_wtjh = solUnevacuatedPatients_Phi_wtjh
        self.AvailableCapFacility_zeta_wtu = solAvailableCapFacility_zeta_wtu
        
        self.FinalACFEstablishmentCost = Final_ACFEstablishmentCost
        self.FinalLandRescueVehicleCost = Final_LandRescueVehicleCost
        self.FinalBackupHospitalCost = Final_BackupHospitalCost

        self.FinalCasualtyTransferCost = Final_CasualtyTransferCost
        self.FinalUnsatisfiedCasualtiesCost = Final_UnsatisfiedCasualtiesCost
        self.FinalDischargedPatientsCost = Final_DischargedPatientsCost
        self.FinalLandEvacuatedPatientsCost = Final_LandEvacuatedPatientsCost
        self.FinalAerialEvacuatedPatientsCost = Final_AerialEvacuatedPatientsCost
        self.FinalUnevacuatedPatientsCost = Final_UnevacuatedPatientsCost
        self.FinalAvailableCapFacilityCost = Final_AvailableCapFacilityCost

        # Total cost calculation
        self.TotalCost = (  self.FinalACFEstablishmentCost
                          + self.FinalLandRescueVehicleCost 
                          + self.FinalBackupHospitalCost 
                          + self.FinalCasualtyTransferCost 
                          + self.FinalUnsatisfiedCasualtiesCost 
                          + self.FinalDischargedPatientsCost 
                          + self.FinalLandEvacuatedPatientsCost 
                          + self.FinalAerialEvacuatedPatientsCost 
                          + self.FinalUnevacuatedPatientsCost 
                          + self.FinalAvailableCapFacilityCost 
                          )
        if Constants.Debug: print("TotalCost: ", self.TotalCost)
            
        self.IsPartialSolution = partialsolution

    #This function return the number of time bucket covered by the solution
    def GetConsideredTimeBucket(self):
        result = self.Instance.TimeBucketSet
        return result
    
    #This function return a set of dataframes descirbing the solution
    def DataFrameFromList(self):
        if Constants.Debug: print("\n We are in 'Solution' Class -- DataFrameFromList")
        

        scenarioset = range(len(self.Scenarioset))
        
        timebucketset = self.GetConsideredTimeBucket()              

        ######################################## ACF Establishment Data Frame
        solACFEstablishment = [[self.ACFEstablishment_x_wi[w][i] 
                                    for w in scenarioset] 
                                    for i in self.Instance.ACFSet]
        if Constants.Debug: print("solACFEstablishment: ", solACFEstablishment)
        ScenarioLables = list(range(len(self.Scenarioset))) 
        aCFEstablishment_df = pd.DataFrame(solACFEstablishment, index=self.Instance.ACFSet, columns=ScenarioLables)
        aCFEstablishment_df.index.name = "ACFLoc" 
        if Constants.Debug: print("aCFEstablishment_df:\n ", aCFEstablishment_df)

        ######################################################## Land Rescue Vehicle Data Frame
        solLandRescueVehicle = []
        for i in self.Instance.ACFSet:
            LandRescueVehicle_Values = []
            for w in scenarioset:
                for m in self.Instance.RescueVehicleSet:
                    LandRescueVehicleExtraction = self.LandRescueVehicle_thetaVar_wim[w][i][m]
                    LandRescueVehicle_Values.append(LandRescueVehicleExtraction)
            solLandRescueVehicle.append(LandRescueVehicle_Values)
        if Constants.Debug: print("solLandRescueVehicle:\n ", solLandRescueVehicle)
        iterables_2d = [range(len(self.Scenarioset)), range(len(self.Instance.RescueVehicleSet))]
        multiindex_2d = pd.MultiIndex.from_product(iterables_2d, names=['Scenario', 'Vehicle'])
        landRescueVehicle_df = pd.DataFrame(solLandRescueVehicle, index=self.Instance.ACFSet, columns=multiindex_2d)
        landRescueVehicle_df.index.name = "ACFLoc"
        if Constants.Debug: print("landRescueVehicle_df:\n ", landRescueVehicle_df)

        ######################################## Backup Hospital Data Frame
        solBackupHospital = []
        for h in self.Instance.HospitalSet:
            BackupHospital_Values = []
            for w in scenarioset:
                for hprime in self.Instance.HospitalSet:
                    BackupHospitalExtraction = self.BackupHospital_W_whhPrime[w][h][hprime]
                    BackupHospital_Values.append(BackupHospitalExtraction)
            solBackupHospital.append(BackupHospital_Values)
        if Constants.Debug: print("solBackupHospital:\n ", solBackupHospital)
        iterables_2d = [range(len(self.Scenarioset)), range(len(self.Instance.HospitalSet))]
        multiindex_2d = pd.MultiIndex.from_product(iterables_2d, names=['Scenario', 'HospitalLocP'])
        backupHospital_df = pd.DataFrame(solBackupHospital, index=self.Instance.HospitalSet, columns=multiindex_2d)
        backupHospital_df.index.name = "HospitalLoc"
        if Constants.Debug: print("backupHospital_df:\n ", backupHospital_df)

        ######################################################## Casualty Transfer Data Frame
        solCasualtyTransfer = []
        for j in self.Instance.InjuryLevelSet:
            CasualtyTransfer_values = []
            for w in scenarioset:
                for t in timebucketset:
                    for l in self.Instance.DisasterAreaSet:
                        for u in self.Instance.MedFacilitySet:
                            for m in self.Instance.RescueVehicleSet: 
                                Extraction = self.CasualtyTransfer_q_wtjlum[w][t][j][l][u][m]
                                CasualtyTransfer_values.append(Extraction)
            solCasualtyTransfer.append(CasualtyTransfer_values)
        iterables_6d = [range(len(self.Scenarioset)), timebucketset, range(len(self.Instance.DisasterAreaSet)), range(len(self.Instance.MedFacilitySet)), range(len(self.Instance.RescueVehicleSet))]
        multiindex_6d = pd.MultiIndex.from_product(iterables_6d, names=['Scenario', 'Time', 'DisasterLoc', 'MedFacilityLoc', 'Vehicle'])
        casualtyTransfer_df = pd.DataFrame(solCasualtyTransfer, index=self.Instance.InjuryLevelSet, columns=multiindex_6d)
        casualtyTransfer_df.index.name = "Injury"
        if Constants.Debug: print("casualtyTransfer_df:\n ", casualtyTransfer_df)

        ######################################################## Unsatisfied Casualties Data Frame
        solUnsatisfiedCasualties = []
        for j in self.Instance.InjuryLevelSet:
            UnsatisfiedCasualties_Values = []
            for w in scenarioset:
                for t in timebucketset:
                    for l in self.Instance.DisasterAreaSet:
                        Extraction = self.UnsatisfiedCasualties_mu_wtjl[w][t][j][l]
                        UnsatisfiedCasualties_Values.append(Extraction)
            solUnsatisfiedCasualties.append(UnsatisfiedCasualties_Values)
        iterables_3d = [range(len(self.Scenarioset)), timebucketset, range(len(self.Instance.DisasterAreaSet))]
        multiindex_3d = pd.MultiIndex.from_product(iterables_3d, names=['Scenario', 'Time', 'DisasterLoc'])
        unsatisfiedCasualties_df = pd.DataFrame(solUnsatisfiedCasualties, index=self.Instance.InjuryLevelSet, columns=multiindex_3d)
        unsatisfiedCasualties_df.index.name = "Injury"
        if Constants.Debug: print("unsatisfiedCasualties_df:\n ", unsatisfiedCasualties_df)

        ######################################################## Discharged Patients Data Frame
        solDischargedPatients = []
        for j in self.Instance.InjuryLevelSet:
            DischargedPatients_Values = []
            for w in scenarioset:
                for t in timebucketset:
                    for u in self.Instance.MedFacilitySet:
                        Extraction = self.DischargedPatients_sigmaVar_wtju[w][t][j][u]
                        DischargedPatients_Values.append(Extraction)
            solDischargedPatients.append(DischargedPatients_Values)
        iterables_3d = [range(len(self.Scenarioset)), timebucketset, range(len(self.Instance.MedFacilitySet))]
        multiindex_3d = pd.MultiIndex.from_product(iterables_3d, names=['Scenario', 'Time', 'MedFacilityLoc'])
        dischargedPatients_df = pd.DataFrame(solDischargedPatients, index=self.Instance.InjuryLevelSet, columns=multiindex_3d)
        dischargedPatients_df.index.name = "Injury"
        if Constants.Debug: print("dischargedPatients_df:\n ", dischargedPatients_df)

        ######################################################## Land Evacuated Patients Data Frame
        solLandEvacuatedPatients = []
        for j in self.Instance.InjuryLevelSet:
            LandEvacuatedPatients_values = []
            for w in scenarioset:
                for t in timebucketset:
                    for h in self.Instance.HospitalSet:
                        for u in self.Instance.MedFacilitySet:
                            for m in self.Instance.RescueVehicleSet: 
                                Extraction = self.LandEvacuatedPatients_u_L_wtjhum[w][t][j][h][u][m]
                                LandEvacuatedPatients_values.append(Extraction)
            solLandEvacuatedPatients.append(LandEvacuatedPatients_values)
        iterables_6d = [range(len(self.Scenarioset)), timebucketset, range(len(self.Instance.HospitalSet)), range(len(self.Instance.MedFacilitySet)), range(len(self.Instance.RescueVehicleSet))]
        multiindex_6d = pd.MultiIndex.from_product(iterables_6d, names=['Scenario', 'Time', 'HospitalLoc', 'MedFacilityLoc', 'Vehicle'])
        landEvacuatedPatients_df = pd.DataFrame(solLandEvacuatedPatients, index=self.Instance.InjuryLevelSet, columns=multiindex_6d)
        landEvacuatedPatients_df.index.name = "Injury"
        if Constants.Debug: print("landEvacuatedPatients_df:\n ", landEvacuatedPatients_df)

        ######################################################## Aerial Evacuated Patients Data Frame
        solAerialEvacuatedPatients = []
        for j in self.Instance.InjuryLevelSet:
            AerialEvacuatedPatients_values = []
            for w in scenarioset:
                for t in timebucketset:
                    for h in self.Instance.HospitalSet:
                        for i in self.Instance.ACFSet:
                            for hprime in self.Instance.HospitalSet:
                                for m in self.Instance.RescueVehicleSet:  # New 7th dimension
                                    Extraction = self.AerialEvacuatedPatients_u_A_wtjhihPrimem[w][t][j][h][i][hprime][m]
                                    AerialEvacuatedPatients_values.append(Extraction)
            solAerialEvacuatedPatients.append(AerialEvacuatedPatients_values)
        iterables_7d = [range(len(self.Scenarioset)), timebucketset, range(len(self.Instance.HospitalSet)), range(len(self.Instance.ACFSet)), range(len(self.Instance.HospitalSet)), range(len(self.Instance.RescueVehicleSet))]
        multiindex_7d = pd.MultiIndex.from_product(iterables_7d, names=['Scenario', 'Time', 'HospitalLoc', 'ACFLoc', 'HospitalLocP', 'Vehicle'])
        aerialEvacuatedPatients_df = pd.DataFrame(solAerialEvacuatedPatients, index=self.Instance.InjuryLevelSet, columns=multiindex_7d)
        aerialEvacuatedPatients_df.index.name = "Injury"
        if Constants.Debug: print("aerialEvacuatedPatients_df:\n ", aerialEvacuatedPatients_df)

        ######################################################## Unevacuated Patients Data Frame
        solUnevacuatedPatients = []
        for j in self.Instance.InjuryLevelSet:
            UnevacuatedPatients_Values = []
            for w in scenarioset:
                for t in timebucketset:
                    for h in self.Instance.HospitalSet:
                        Extraction = self.UnevacuatedPatients_Phi_wtjh[w][t][j][h]
                        UnevacuatedPatients_Values.append(Extraction)
            solUnevacuatedPatients.append(UnevacuatedPatients_Values)
        iterables_3d = [range(len(self.Scenarioset)), timebucketset, range(len(self.Instance.HospitalSet))]
        multiindex_3d = pd.MultiIndex.from_product(iterables_3d, names=['Scenario', 'Time', 'HospitalLoc'])
        unevacuatedPatients_df = pd.DataFrame(solUnevacuatedPatients, index=self.Instance.InjuryLevelSet, columns=multiindex_3d)
        unevacuatedPatients_df.index.name = "Injury"
        if Constants.Debug: print("unevacuatedPatients_df:\n ", unevacuatedPatients_df)

        ######################################################## Available Cap Facility Data Frame
        solAvailableCapFacility = []
        for u in self.Instance.MedFacilitySet:
            AvailableCapFacility_values = []
            for w in scenarioset:
                for t in timebucketset:
                    Extraction = self.AvailableCapFacility_zeta_wtu[w][t][u]
                    AvailableCapFacility_values.append(Extraction)
            solAvailableCapFacility.append(AvailableCapFacility_values)
        iterables_3d = [range(len(self.Scenarioset)), timebucketset]
        multiindex_3d = pd.MultiIndex.from_product(iterables_3d, names=['Scenario', 'Time'])
        availableCapFacility_df = pd.DataFrame(solAvailableCapFacility, index=self.Instance.MedFacilitySet, columns=multiindex_3d)
        availableCapFacility_df.index.name = "MedFacilityLoc"
        if Constants.Debug: print("availableCapFacility_df:\n", availableCapFacility_df)

        return aCFEstablishment_df, \
                landRescueVehicle_df, \
                backupHospital_df, \
                casualtyTransfer_df, \
                unsatisfiedCasualties_df, \
                dischargedPatients_df, \
                landEvacuatedPatients_df, \
                aerialEvacuatedPatients_df, \
                unevacuatedPatients_df, \
                availableCapFacility_df

    # return the path to binary file of the solution is saved
    def GetSolutionPickleFileNameStart(self, description, dataframename):
        
        result ="./Solutions/"+  description + "_" + dataframename
            
        return result
    
    def GetGeneralInfoDf(self):
        if Constants.Debug: print("\n We are in 'Solution' Class -- GetGeneralInfoDf")

        model = "Rule"

        # Create a dictionary for general information with appropriate keys and values
        general_info = {
            "Name": self.Instance.InstanceName,
            "Distribution": self.Instance.Distribution,
            "Model": model,
            "GRBCost": round(self.GRBCost, 2),
            "GRBTime": self.GRBTime,
            "PHCost": self.PHCost,
            "PHNrIteration": self.PHNrIteration,
            "TotalTime": self.TotalTime,
            "GRBGap": self.GRBGap,
            "GRBNrConstraints": self.GRBNrConstraints,
            "GRBNrVariables": self.GRBNrVariables,
            "IsPartialSolution": self.IsPartialSolution
        }

        # Convert the dictionary into a DataFrame
        # Since general_info represents a single row, we turn it into a DataFrame by passing
        # the values in a list (to form a row) and specify the column names using the keys from the dictionary
        generaldf = pd.DataFrame([general_info], columns=general_info.keys())

        # Optionally, if you want to set one of the columns as the index of the DataFrame, you can do so.
        # For example, if you want "Name" as the index:
        # generaldf.set_index("Name", inplace=True)
        
        return generaldf

    def GetSolutionFileName(self, description):
        if Constants.Debug: print("\n We are in 'Solution' Class -- GetSolutionFileName")

        if Constants.PrintDetailsExcelFiles:
            result = "./Solutions/" + description + "_Solution.xlsx"
        else:
            result = "./Solutions/" + description + "_Solution.txt"  # Changed .xlsx to .txt
        
        return result
    
    #This function print the solution in an Excel file in the folder "Solutions"
    def PrintToExcel(self, description):
        if Constants.Debug: print("\n We are in 'Solution' Class -- PrintToExcel")

        aCFEstablishment_df, \
        landRescueVehicle_df, \
        backupHospital_df, \
        casualtyTransfer_df, \
        unsatisfiedCasualties_df, \
        dischargedPatients_df, \
        landEvacuatedPatients_df, \
        aerialEvacuatedPatients_df, \
        unevacuatedPatients_df, \
        availableCapFacility_df = self.DataFrameFromList()

        file_name = self.GetSolutionFileName(description)
        if file_name.endswith(".txt"):
            file_name = file_name.replace(".txt", ".xlsx")

        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            
            aCFEstablishment_df.to_excel(writer, sheet_name='ACF Establishment')
            landRescueVehicle_df.to_excel(writer, sheet_name='Land Rescue Vehicle')
            backupHospital_df.to_excel(writer, sheet_name='Backup Hospital')


            #casualtyTransfer_df.to_excel(writer, sheet_name='Casualty Transfer')
            #unsatisfiedCasualties_df.to_excel(writer, sheet_name='Unsatisfied Casualties')
            #dischargedPatients_df.to_excel(writer, sheet_name='Discharged Patients')
            #landEvacuatedPatients_df.to_excel(writer, sheet_name='Land Evacuated Patients')
            #aerialEvacuatedPatients_df.to_excel(writer, sheet_name='Aerial Evacuated Patients')
            #unevacuatedPatients_df.to_excel(writer, sheet_name='Unevacuated Patients')
            #availableCapFacility_df.to_excel(writer, sheet_name='Available Cap Facility')

            generaldf = self.GetGeneralInfoDf()
            generaldf.to_excel(writer, sheet_name="Generic")

            # Constructing ScenarioTree info DataFrame correctly
            # The original approach seems to be incorrect as it would raise an error due to shape mismatch.
            # Assuming 'scenariotreeinfo' is intended to be a single row in the DataFrame:
            if len(self.ScenarioTree) == 1:  # Single ScenarioTree
                scenariotreeinfo = [[self.Instance.InstanceName, 
                                    self.ScenarioTree[0].ScenarioSeed, 
                                    self.ScenarioTree[0].TreeStructure, 
                                    self.ScenarioTree[0].AverageScenarioTree, 
                                    self.ScenarioTree[0].ScenarioGenerationMethod]]
            else:  # Multiple ScenarioTrees
                scenariotreeinfo = []
                for tree in self.ScenarioTree:
                    scenariotreeinfo.append([self.Instance.InstanceName, 
                                            tree.ScenarioSeed, 
                                            tree.TreeStructure, 
                                            tree.AverageScenarioTree, 
                                            tree.ScenarioGenerationMethod])
            columnstab = ["Name", "Seed", "TreeStructure", "AverageScenarioTree", "ScenarioGenerationMethod"]
            scenariotreeinfo_df = pd.DataFrame(scenariotreeinfo, columns=columnstab)
            scenariotreeinfo_df.to_excel(writer, sheet_name="ScenarioTree")

    # This function print the solution different pickle files
    def PrintToPickle(self, description):
        if Constants.Debug: print("\n We are in 'Solution' Class -- PrintToPickle")

        aCFEstablishment_df, \
        landRescueVehicle_df, \
        backupHospital_df, \
        casualtyTransfer_df, \
        unsatisfiedCasualties_df, \
        dischargedPatients_df, \
        landEvacuatedPatients_df, \
        aerialEvacuatedPatients_df, \
        unevacuatedPatients_df, \
        availableCapFacility_df = self.DataFrameFromList()

        aCFEstablishment_df.to_pickle(self.GetSolutionPickleFileNameStart(description, 'ACFEstablishment'))
        landRescueVehicle_df.to_pickle(self.GetSolutionPickleFileNameStart(description, 'LandRescueVehicle'))
        backupHospital_df.to_pickle(self.GetSolutionPickleFileNameStart(description, 'BackupHospital'))

        casualtyTransfer_df.to_pickle(self.GetSolutionPickleFileNameStart(description,  'CasualtyTransfer'))
        unsatisfiedCasualties_df.to_pickle(self.GetSolutionPickleFileNameStart(description,  'UnsatisfiedCasualties'))
        dischargedPatients_df.to_pickle(self.GetSolutionPickleFileNameStart(description,  'DischargedPatients'))
        landEvacuatedPatients_df.to_pickle(self.GetSolutionPickleFileNameStart(description,  'LandEvacuatedPatients'))
        aerialEvacuatedPatients_df.to_pickle(self.GetSolutionPickleFileNameStart(description,  'AerialEvacuatedPatients'))
        unevacuatedPatients_df.to_pickle(self.GetSolutionPickleFileNameStart(description,  'UnevacuatedPatients'))
        availableCapFacility_df.to_pickle(self.GetSolutionPickleFileNameStart(description,  'AvailableCapFacility'))

        generaldf = self.GetGeneralInfoDf()
        generaldf.to_pickle(self.GetSolutionPickleFileNameStart(description, "Generic"))

        if len(self.ScenarioTree) == 1:  # Single ScenarioTree case
            scenariotreeinfo = [self.Instance.InstanceName, 
                                self.ScenarioTree[0].ScenarioSeed, 
                                self.ScenarioTree[0].TreeStructure, 
                                self.ScenarioTree[0].AverageScenarioTree, 
                                self.ScenarioTree[0].ScenarioGenerationMethod]
        else:  # Multiple ScenarioTrees
            scenariotreeinfo = []
            for tree in self.ScenarioTree:
                scenariotreeinfo.append([self.Instance.InstanceName, 
                                        tree.ScenarioSeed, 
                                        tree.TreeStructure, 
                                        tree.AverageScenarioTree, 
                                        tree.ScenarioGenerationMethod])
        columnstab = ["Name", "Seed", "TreeStructure", "AverageScenarioTree", "ScenarioGenerationMethod"]
        scenariotreeinfo = pd.DataFrame(scenariotreeinfo, index=columnstab)
        scenariotreeinfo.to_pickle( self.GetSolutionPickleFileNameStart(description,  "ScenarioTree") )

    #This function set the attributes of the solution from the excel/binary file
    def ReadFromFile(self, description, TestIdentifier):
        if Constants.Debug: print("\n We are in 'Solution' Class -- ReadFromFile")
        
        aCFEstablishment_df, \
        landRescueVehicle_df, \
        backupHospital_df, \
        casualtyTransfer_df, \
        unsatisfiedCasualties_df, \
        dischargedPatients_df, \
        landEvacuatedPatients_df, \
        aerialEvacuatedPatients_df, \
        unevacuatedPatients_df, \
        availableCapFacility_df, \
        instanceinfo, \
        scenariotreeinfo \
        = self.ReadPickleFiles(description)

        self.Instance = Instance(instanceinfo['Name'].iloc[0])
        
        self.Instance.LoadInstanceFromPickle(instanceinfo['Name'].iloc[0])

        scenariogenerationm = scenariotreeinfo.at['ScenarioGenerationMethod', 0]
        avgscenariotree = scenariotreeinfo.at['AverageScenarioTree', 0]
        scenariotreeseed = int(scenariotreeinfo.at['Seed', 0])
        branchingstructure  = literal_eval(str(scenariotreeinfo.at['TreeStructure', 0]))

        #######3 WHYYYYYYYYYYYYY YOU ARE HERE?
        model = instanceinfo['Model'].iloc[0]
        Constants.ClusteringMethod = TestIdentifier.Clustering         ## We disabled clustering for doing Evaluation (to  have the same evaluations instances for all scenarios, and now, again we being it back to what it was at the first!)
        
        self.ScenarioTree = ScenarioTree(instance=self.Instance,
                                        tree_structure=branchingstructure,
                                        scenario_seed=scenariotreeseed,
                                        averagescenariotree=avgscenariotree,
                                        scenariogenerationmethod=scenariogenerationm)
        

        self.IsPartialSolution = instanceinfo['IsPartialSolution'].iloc[0]
        self.GRBCost = instanceinfo['GRBCost'].iloc[0] 
        self.GRBTime = instanceinfo['GRBTime'].iloc[0]  
        self.TotalTime = instanceinfo['TotalTime'].iloc[0] 
        self.GRBGap = instanceinfo['GRBGap'].iloc[0]  
        self.GRBNrConstraints = instanceinfo['GRBNrConstraints'].iloc[0] 
        self.GRBNrVariables = instanceinfo['GRBNrVariables'].iloc[0] 
        self.PHCost = instanceinfo['PHCost'].iloc[0] 
        self.PHNrIteration = instanceinfo['PHNrIteration'].iloc[0] 
        
        if Constants.Debug:
            print(f"GRBCost: {self.GRBCost}")
            print(f"GRBTime: {self.GRBTime}")
            print(f"TotalTime: {self.TotalTime}")
            print(f"GRBGap: {self.GRBGap}")
            print(f"GRBNrConstraints: {self.GRBNrConstraints}")
            print(f"GRBNrVariables: {self.GRBNrVariables}")
            print(f"PHCost: {self.PHCost}")
            print(f"PHNrIteration: {self.PHNrIteration}")
        
        self.Scenarioset = self.ScenarioTree.GetAllScenarioSet()

        if self.IsPartialSolution:
            self.Scenarioset = [self.Scenarioset[0]]        

        self.ListFromDataFrame(aCFEstablishment_df, 
                               landRescueVehicle_df,
                               backupHospital_df,
                               casualtyTransfer_df, 
                               unsatisfiedCasualties_df, 
                               dischargedPatients_df, 
                               landEvacuatedPatients_df, 
                               aerialEvacuatedPatients_df, 
                               unevacuatedPatients_df, 
                               availableCapFacility_df)

        if not self.IsPartialSolution:
            self.ComputeCost()

    #This function return a set of dataframes describing the content of the binary file
    def ReadPickleFiles(self, description):
        if Constants.Debug: print("\n We are in 'Solution' Class -- ReadPickleFiles")
        # The supplychain is defined in the sheet named "01_LL" and the data are in the sheet "01_SD"

        aCFEstablishment_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'ACFEstablishment'))
        landRescueVehicle_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'LandRescueVehicle'))
        backupHospital_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'BackupHospital'))

        casualtyTransfer_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description,  'CasualtyTransfer'))
        unsatisfiedCasualties_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description,  'UnsatisfiedCasualties'))
        dischargedPatients_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description,  'DischargedPatients'))
        landEvacuatedPatients_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description,  'LandEvacuatedPatients'))
        aerialEvacuatedPatients_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description,  'AerialEvacuatedPatients'))
        unevacuatedPatients_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description,  'UnevacuatedPatients'))
        availableCapFacility_df = pd.read_pickle(self.GetSolutionPickleFileNameStart(description,  'AvailableCapFacility'))

        instanceinfo = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, "Generic"))
        scenariotreeinfo = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, "ScenarioTree"))

        return aCFEstablishment_df, \
                landRescueVehicle_df, \
                backupHospital_df, \
                casualtyTransfer_df, \
                unsatisfiedCasualties_df, \
                dischargedPatients_df, \
                landEvacuatedPatients_df, \
                aerialEvacuatedPatients_df, \
                unevacuatedPatients_df, \
                availableCapFacility_df, \
                instanceinfo, \
                scenariotreeinfo
    
        #This function creates a solution from the set of dataframe given in paramter  
    
    def ListFromDataFrame(self, 
                          aCFEstablishment_df, 
                          landRescueVehicle_df, 
                          backupHospital_df, 
                          casualtyTransfer_df, 
                          unsatisfiedCasualties_df, 
                          dischargedPatients_df, 
                          landEvacuatedPatients_df, 
                          aerialEvacuatedPatients_df, 
                          unevacuatedPatients_df, 
                          availableCapFacility_df):
        if Constants.Debug: print("\n We are in 'Solution' Class -- ListFromDataFrame")

        scenarioset = range(len(self.Scenarioset))

        timebucketset = self.GetConsideredTimeBucket()

        ###################################### ACF Establishment
        self.ACFEstablishment_x_wi = [[aCFEstablishment_df.loc[i, (w)] 
                                        for i in self.Instance.ACFSet] 
                                        for w in scenarioset]        
        if Constants.Debug: print("self.ACFEstablishment_x_wi: ", self.ACFEstablishment_x_wi)
        
        ###################################### Land Rescue Vehicle
        self.LandRescueVehicle_thetaVar_wim = [[[landRescueVehicle_df.loc[i, (w, m)] 
                                                    for m in self.Instance.RescueVehicleSet] 
                                                    for i in self.Instance.ACFSet] 
                                                    for w in scenarioset]
        if Constants.Debug: print("self.LandRescueVehicle_thetaVar_wim: ", self.LandRescueVehicle_thetaVar_wim)

        ###################################### Backup Hospital
        self.BackupHospital_W_whhPrime = [[[backupHospital_df.loc[h, (w, hprime)] 
                                                    for hprime in self.Instance.HospitalSet] 
                                                    for h in self.Instance.HospitalSet] 
                                                    for w in scenarioset]
        if Constants.Debug: print("self.BackupHospital_W_whhPrime: ", self.BackupHospital_W_whhPrime)

        ###################################### Casualty Transfer
        self.CasualtyTransfer_q_wtjlum = [[[[[[casualtyTransfer_df.loc[j, (w, t, l, u, m)] 
                                                for m in self.Instance.RescueVehicleSet] 
                                                for u in self.Instance.MedFacilitySet] 
                                                for l in self.Instance.DisasterAreaSet] 
                                                for j in self.Instance.InjuryLevelSet] 
                                                for t in timebucketset] 
                                                for w in scenarioset]
        if Constants.Debug: print("self.CasualtyTransfer_q_wtjlum: ", self.CasualtyTransfer_q_wtjlum)

        ###################################### Unsatisfied Casualties
        self.UnsatisfiedCasualties_mu_wtjl = [[[[unsatisfiedCasualties_df.loc[j, (w, t, l)] 
                                                    for l in self.Instance.DisasterAreaSet] 
                                                    for j in self.Instance.InjuryLevelSet] 
                                                    for t in timebucketset] 
                                                    for w in scenarioset]
        if Constants.Debug: print("self.UnsatisfiedCasualties_mu_wtjl: ", self.UnsatisfiedCasualties_mu_wtjl)

        ###################################### Discharged Patients
        self.DischargedPatients_sigmaVar_wtju = [[[[dischargedPatients_df.loc[j, (w, t, u)] 
                                                    for u in self.Instance.MedFacilitySet] 
                                                    for j in self.Instance.InjuryLevelSet] 
                                                    for t in timebucketset] 
                                                    for w in scenarioset]
        if Constants.Debug: print("self.DischargedPatients_sigmaVar_wtju: ", self.DischargedPatients_sigmaVar_wtju)

        ###################################### Land Evacuated Patients
        self.LandEvacuatedPatients_u_L_wtjhum = [[[[[[landEvacuatedPatients_df.loc[j, (w, t, h, u, m)] 
                                                        for m in self.Instance.RescueVehicleSet] 
                                                        for u in self.Instance.MedFacilitySet] 
                                                        for h in self.Instance.HospitalSet] 
                                                        for j in self.Instance.InjuryLevelSet] 
                                                        for t in timebucketset] 
                                                        for w in scenarioset]
        if Constants.Debug: print("self.LandEvacuatedPatients_u_L_wtjhum: ", self.LandEvacuatedPatients_u_L_wtjhum)

        ###################################### Aerial Evacuated Patients
        self.AerialEvacuatedPatients_u_A_wtjhihPrimem = [[[[[[[aerialEvacuatedPatients_df.loc[j, (w, t, h, i, hprime, m)] 
                                                            for m in self.Instance.RescueVehicleSet] 
                                                            for hprime in self.Instance.HospitalSet] 
                                                            for i in self.Instance.ACFSet] 
                                                            for h in self.Instance.HospitalSet] 
                                                            for j in self.Instance.InjuryLevelSet] 
                                                            for t in timebucketset] 
                                                            for w in scenarioset]
        if Constants.Debug: print("self.AerialEvacuatedPatients_u_A_wtjhihPrimem: ", self.AerialEvacuatedPatients_u_A_wtjhihPrimem)

        ###################################### Unevacuated Patients
        self.UnevacuatedPatients_Phi_wtjh = [[[[unevacuatedPatients_df.loc[j, (w, t, h)] 
                                                for h in self.Instance.HospitalSet] 
                                                for j in self.Instance.InjuryLevelSet] 
                                                for t in timebucketset] 
                                                for w in scenarioset]
        if Constants.Debug: print("self.UnevacuatedPatients_Phi_wtjh: ", self.UnevacuatedPatients_Phi_wtjh)

        ###################################### Available CapFacility
        self.AvailableCapFacility_zeta_wtu = [[[availableCapFacility_df.loc[u, (w, t)] 
                                                for u in self.Instance.MedFacilitySet] 
                                                for t in timebucketset] 
                                                for w in scenarioset]
        if Constants.Debug: print("self.AvailableCapFacility_zeta_wtu: ", self.AvailableCapFacility_zeta_wtu)

    def ComputeCost(self):
        if Constants.Debug: print("\n We are in 'Solution' Class -- ComputeCost")
        
        self.TotalCost, \
            self.ACFestablishmentcost, \
            self.LandRescueVehiclecost, \
            self.BackupHospitalcost, \
                self.CasualtyTransferCost, \
                self.UnsatisfiedCasualtiesCost, \
                self.DischargedPatientsCost, \
                self.LandEvacuatedPatientsCost, \
                self.AerialEvacuatedPatientsCost, \
                self.UnevacuatedPatientsCost, \
                self.AvailableCapFacilityCost \
                    = self.GetCostInInterval(self.Instance.TimeBucketSet)

    #This function return the costs encountered in a specific time interval
    def GetCostInInterval(self, timerange):
        if Constants.Debug: print("\n We are in 'Solution' Class -- GetCostInInterval")
        
        ScenarioProbability = (1/len(self.Scenarioset))

        acfestablishmentcost = 0
        landRescueVehiclecost = 0
        backupHospitalcost = 0

        casualtyTransferCost = 0
        unsatisfiedCasualtiesCost = 0
        dischargedPatientsCost = 0
        landEvacuatedPatientsCost = 0
        aerialEvacuatedPatientsCost = 0
        unevacuatedPatientsCost = 0
        availableCapFacilityCost = 0

        ########################## ACF Establishment
        for w in range(len(self.Scenarioset)):
            for i in self.Instance.ACFSet:
                acfestablishmentcost += self.ACFEstablishment_x_wi[w][i] \
                                            * self.Instance.Fixed_Cost_ACF_Objective[i] \
                                            * ScenarioProbability
        ########################## Land Rescue Vehicle
        for w in range(len(self.Scenarioset)):
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    landRescueVehiclecost += self.LandRescueVehicle_thetaVar_wim[w][i][m] \
                                            * self.Instance.VehicleAssignment_Cost[m] \
                                            * ScenarioProbability
                    
        ########################## Backup Hospital
        for w in range(len(self.Scenarioset)):
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:
                    backupHospitalcost += self.BackupHospital_W_whhPrime[w][h][hprime] \
                                            * self.Instance.CoordinationCost[h, hprime] \
                                            * ScenarioProbability

        ########################## Casualty Transfer
        for w in range(len(self.Scenarioset)):
            for t in timerange:
                for j in self.Instance.InjuryLevelSet:
                    for l in self.Instance.DisasterAreaSet:
                        for u in self.Instance.MedFacilitySet:
                            for m in self.Instance.RescueVehicleSet:
                                if u < self.Instance.NrHospitals:
                                    casualtyTransferCost += self.CasualtyTransfer_q_wtjlum[w][t][j][l][u][m] \
                                                            * self.Instance.Time_D_H_Land[l][u] \
                                                            * ScenarioProbability
                                else:
                                    casualtyTransferCost += self.CasualtyTransfer_q_wtjlum[w][t][j][l][u][m] \
                                                            * self.Instance.Time_D_A_Land[l][u - self.Instance.NrHospitals] \
                                                            * ScenarioProbability
                                                                     
        ########################## Unsatisfied Casualties
        for w in range(len(self.Scenarioset)):
            for t in timerange:
                for j in self.Instance.InjuryLevelSet:
                    for l in self.Instance.DisasterAreaSet:
                            unsatisfiedCasualtiesCost += self.UnsatisfiedCasualties_mu_wtjl[w][t][j][l] \
                                                        * self.Instance.Casualty_Shortage_Cost[j] \
                                                        * ScenarioProbability

        ########################## Discharged Patients
        for w in range(len(self.Scenarioset)):
            for t in timerange:
                for j in self.Instance.InjuryLevelSet:
                    for u in self.Instance.MedFacilitySet:
                            dischargedPatientsCost += self.DischargedPatients_sigmaVar_wtju[w][t][j][u] \
                                                        * 0 \
                                                        * ScenarioProbability

        ########################## Land Evacuated Patients
        for w in range(len(self.Scenarioset)):
            for t in timerange:
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        for u in self.Instance.MedFacilitySet:
                            for m in self.Instance.RescueVehicleSet:
                                if Constants.Risk == "Constant":
                                    landEvacuatedPatientsCost += self.LandEvacuatedPatients_u_L_wtjhum[w][t][j][h][u][m] \
                                                            * self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Constant[t][j][h][u][m] \
                                                            * ScenarioProbability
                                elif Constants.Risk == "Linear":
                                    landEvacuatedPatientsCost += self.LandEvacuatedPatients_u_L_wtjhum[w][t][j][h][u][m] \
                                                            * self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Linear[t][j][h][u][m] \
                                                            * ScenarioProbability
                                elif Constants.Risk == "Exponential":
                                    landEvacuatedPatientsCost += self.LandEvacuatedPatients_u_L_wtjhum[w][t][j][h][u][m] \
                                                            * self.Instance.EvacuationRiskCost[j] * self.Instance.LandEvacuationRisk_Exponential[t][j][h][u][m] \
                                                            * ScenarioProbability                                    

        ########################## Aerial Evacuated Patients
        for w in range(len(self.Scenarioset)):
            for t in timerange:
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                        for i in self.Instance.ACFSet:
                            for hprime in self.Instance.HospitalSet:
                                for m in self.Instance.RescueVehicleSet:
                                    if Constants.Risk == "Constant":
                                        aerialEvacuatedPatientsCost += self.AerialEvacuatedPatients_u_A_wtjhihPrimem[w][t][j][h][i][hprime][m] \
                                                                    * self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Constant[t][j][h][i][hprime][m] \
                                                                    * ScenarioProbability
                                    elif Constants.Risk == "Linear":
                                        aerialEvacuatedPatientsCost += self.AerialEvacuatedPatients_u_A_wtjhihPrimem[w][t][j][h][i][hprime][m] \
                                                                    * self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Linear[t][j][h][i][hprime][m] \
                                                                    * ScenarioProbability
                                    elif Constants.Risk == "Exponential":
                                        aerialEvacuatedPatientsCost += self.AerialEvacuatedPatients_u_A_wtjhihPrimem[w][t][j][h][i][hprime][m] \
                                                                    * self.Instance.EvacuationRiskCost[j] * self.Instance.AerialEvacuationRisk_Exponential[t][j][h][i][hprime][m] \
                                                                    * ScenarioProbability

        ########################## Unevacuated Patients
        for w in range(len(self.Scenarioset)):
            for t in timerange:
                for j in self.Instance.InjuryLevelSet:
                    for h in self.Instance.HospitalSet:
                            if t == self.Instance.TimeBucketSet[-1]:
                                if Constants.Risk == "Constant":
                                    unevacuatedPatientsCost += self.UnevacuatedPatients_Phi_wtjh[w][t][j][h] \
                                                                * self.Instance.EvacuationRiskCost[j] * self.Instance.CumulativeThreatRiskConstant[t][j][h] \
                                                                * ScenarioProbability
                                if Constants.Risk == "Linear":
                                    unevacuatedPatientsCost += self.UnevacuatedPatients_Phi_wtjh[w][t][j][h] \
                                                                * self.Instance.EvacuationRiskCost[j] * self.Instance.CumulativeThreatRiskLinear[t][j][h] \
                                                                * ScenarioProbability
                                if Constants.Risk == "Exponential":
                                    unevacuatedPatientsCost += self.UnevacuatedPatients_Phi_wtjh[w][t][j][h] \
                                                                * self.Instance.EvacuationRiskCost[j] * self.Instance.CumulativeThreatRiskExponential[t][j][h] \
                                                                * ScenarioProbability
                            else:
                                unevacuatedPatientsCost += 0

        ########################## Available Cap Facility
        for w in range(len(self.Scenarioset)):
            for t in timerange:
                for u in self.Instance.MedFacilitySet:
                    availableCapFacilityCost += self.AvailableCapFacility_zeta_wtu[w][t][u] \
                                                * 0 \
                                                * ScenarioProbability

        totalcost = (  acfestablishmentcost 
                     + landRescueVehiclecost 
                     + backupHospitalcost 
                     + casualtyTransferCost 
                     + unsatisfiedCasualtiesCost 
                     + dischargedPatientsCost 
                     + landEvacuatedPatientsCost 
                     + aerialEvacuatedPatientsCost 
                     + unevacuatedPatientsCost 
                     + availableCapFacilityCost)
        
        if Constants.Debug:
            print("ScenarioProbability: ", ScenarioProbability)
            print(f"ACFestablishment Cost: {acfestablishmentcost}")
            print(f"LandRescueVehicle Cost: {landEvacuatedPatientsCost}")
            print(f"BackupHospital Cost: {backupHospitalcost}")
            print(f"CasualtyTransfer Cost: {casualtyTransferCost}")
            print(f"UnsatisfiedCasualties Cost: {unsatisfiedCasualtiesCost}")
            print(f"DischargedPatients Cost: {dischargedPatientsCost}")
            print(f"LandEvacuatedPatients Cost: {landEvacuatedPatientsCost}")
            print(f"AerialEvacuatedPatients Cost: {aerialEvacuatedPatientsCost}")
            print(f"UnevacuatedPatients Cost: {unevacuatedPatientsCost}")
            print(f"AvailableCapFacility Cost: {availableCapFacilityCost}")
            print(f"Total Cost: {totalcost}")  

        return totalcost, \
                acfestablishmentcost, \
                landEvacuatedPatientsCost, \
                backupHospitalcost, \
                    casualtyTransferCost, \
                    unsatisfiedCasualtiesCost, \
                    dischargedPatientsCost, \
                    landEvacuatedPatientsCost, \
                    aerialEvacuatedPatientsCost, \
                    unevacuatedPatientsCost, \
                    availableCapFacilityCost

    #This function compute some statistic on the current solution
    def ComputeStatistics(self):

        if Constants.Debug: print("\n We are in 'Solution' Class -- ComputeStatistics")

        ################ ACF Establishment
        if Constants.Debug: print("\ACFEstablishment_x_wi:\n", self.ACFEstablishment_x_wi)
        self.InSampleAverageACFEstablishment = [sum(self.ACFEstablishment_x_wi[w][i]  for w in self.SenarioNrset) /len(self.SenarioNrset)
                                                for i in self.Instance.ACFSet]
        if Constants.Debug: print("InSampleAverageACFEstablishment:\n", self.InSampleAverageACFEstablishment)
        
        ################ Land Rescue Vehicle
        if Constants.Debug: print("\n LandRescueVehicle_thetaVar_wim:\n", self.LandRescueVehicle_thetaVar_wim)
        self.InSampleAverageLandRescueVehicle = [[round(sum(self.LandRescueVehicle_thetaVar_wim[w][i][m] for w in self.SenarioNrset) / len(self.SenarioNrset),2)
                                                    for m in self.Instance.RescueVehicleSet]
                                                    for i in self.Instance.ACFSet]
        if Constants.Debug: print("\InSampleAverageLandRescueVehicle:\n", self.InSampleAverageLandRescueVehicle)
        
        ################ Backup Hospital
        if Constants.Debug: print("\n BackupHospital_W_whhPrime:\n", self.BackupHospital_W_whhPrime)
        self.InSampleAverageBackupHospital = [[round(sum(self.BackupHospital_W_whhPrime[w][h][hprime] for w in self.SenarioNrset) / len(self.SenarioNrset),2)
                                                    for hprime in self.Instance.HospitalSet]
                                                    for h in self.Instance.HospitalSet]
        if Constants.Debug: print("\InSampleAverageBackupHospital:\n", self.InSampleAverageLandRescueVehicle)
        
        ################ Casualty Transfer
        if Constants.Debug: print("\n CasualtyTransfer_q_wtjlum:\n", self.CasualtyTransfer_q_wtjlum)
        self.InSampleAverageCasualtyTransfer = [[[[[round(sum(self.CasualtyTransfer_q_wtjlum[w][t][j][l][u][m] for w in self.SenarioNrset) / len(self.SenarioNrset), 2) 
                                            for m in self.Instance.RescueVehicleSet]
                                            for u in self.Instance.MedFacilitySet]
                                            for l in self.Instance.DisasterAreaSet]
                                            for j in self.Instance.InjuryLevelSet]
                                            for t in self.Instance.TimeBucketSet]
        if Constants.Debug: print("\InSampleAverageCasualtyTransfer:\n", self.InSampleAverageCasualtyTransfer)
        
        ################ Unsatisfied Casualties
        if Constants.Debug: print("\n UnsatisfiedCasualties_mu_wtjl:\n", self.UnsatisfiedCasualties_mu_wtjl)
        self.InSampleAverageUnsatisfiedCasualties = [[[round(sum(self.UnsatisfiedCasualties_mu_wtjl[w][t][j][l] for w in self.SenarioNrset) / len(self.SenarioNrset), 2) 
                                            for l in self.Instance.DisasterAreaSet]
                                            for j in self.Instance.InjuryLevelSet]
                                            for t in self.Instance.TimeBucketSet]
        if Constants.Debug: print("\InSampleAverageUnsatisfiedCasualties:\n", self.InSampleAverageUnsatisfiedCasualties)
        
        ################ Discharged Patients
        if Constants.Debug: print("\n DischargedPatients_sigmaVar_wtju:\n", self.DischargedPatients_sigmaVar_wtju)
        self.InSampleAverageDischargedPatients = [[[round(sum(self.DischargedPatients_sigmaVar_wtju[w][t][j][u] for w in self.SenarioNrset) / len(self.SenarioNrset), 2) 
                                            for u in self.Instance.MedFacilitySet]
                                            for j in self.Instance.InjuryLevelSet]
                                            for t in self.Instance.TimeBucketSet]
        if Constants.Debug: print("\InSampleAverageDischargedPatients:\n", self.InSampleAverageDischargedPatients)
        
        ################ Land Evacuated Patients
        if Constants.Debug: print("\n LandEvacuatedPatients_u_L_wtjhum:\n", self.LandEvacuatedPatients_u_L_wtjhum)
        self.InSampleAverageLandEvacuatedPatients = [[[[[round(sum(self.LandEvacuatedPatients_u_L_wtjhum[w][t][j][h][u][m] for w in self.SenarioNrset) / len(self.SenarioNrset), 2) 
                                            for m in self.Instance.RescueVehicleSet]
                                            for u in self.Instance.MedFacilitySet]
                                            for h in self.Instance.HospitalSet]
                                            for j in self.Instance.InjuryLevelSet]
                                            for t in self.Instance.TimeBucketSet]
        if Constants.Debug: print("\InSampleAverageLandEvacuatedPatients:\n", self.InSampleAverageLandEvacuatedPatients)
        
        ################ Aerial Evacuated Patients
        if Constants.Debug: print("\n AerialEvacuatedPatients_u_A_wtjhihPrimem:\n", self.AerialEvacuatedPatients_u_A_wtjhihPrimem)
        self.InSampleAverageAerialEvacuatedPatients = [[[[[[round(sum(self.AerialEvacuatedPatients_u_A_wtjhihPrimem[w][t][j][h][i][hprime][m] for w in self.SenarioNrset) / len(self.SenarioNrset), 2) 
                                            for m in self.Instance.RescueVehicleSet]
                                            for hprime in self.Instance.HospitalSet]
                                            for i in self.Instance.ACFSet]
                                            for h in self.Instance.HospitalSet]
                                            for j in self.Instance.InjuryLevelSet]
                                            for t in self.Instance.TimeBucketSet]
        if Constants.Debug: print("\InSampleAverageAerialEvacuatedPatients:\n", self.InSampleAverageAerialEvacuatedPatients)
        
        ################ Unevacuated Patients 
        if Constants.Debug: print("\n UnevacuatedPatients_Phi_wtjh:\n", self.UnevacuatedPatients_Phi_wtjh)
        self.InSampleAverageUnevacuatedPatients = [[[round(sum(self.UnevacuatedPatients_Phi_wtjh[w][t][j][h] for w in self.SenarioNrset) / len(self.SenarioNrset), 2) 
                                            for h in self.Instance.HospitalSet]
                                            for j in self.Instance.InjuryLevelSet]
                                            for t in self.Instance.TimeBucketSet]
        if Constants.Debug: print("\InSampleAverageUnevacuatedPatients:\n", self.InSampleAverageUnevacuatedPatients)
        
        ################ Available Cap Facility
        if Constants.Debug: print("\n AvailableCapFacility_zeta_wtu:\n", self.AvailableCapFacility_zeta_wtu)
        self.InSampleAverageAvailableCapFacility = [[round(sum(self.AvailableCapFacility_zeta_wtu[w][t][u] for w in self.SenarioNrset) / len(self.SenarioNrset), 2) 
                                            for u in self.Instance.MedFacilitySet]
                                            for t in self.Instance.TimeBucketSet]
        if Constants.Debug: print("\InSampleAverageAvailableCapFacility:\n", self.InSampleAverageAvailableCapFacility)
        

        ############### Calculates the average amount of "Casualty Demand" met on-time!
        self.InSampleAverageOnTimeTransfer = [[[(sum(max([self.ScenarioTree[w].CasualtyDemand[0][t][j][l] - self.UnsatisfiedCasualties_mu_wtjl[w][t][j][l],0])  for w in self.SenarioNrset) / len(self.SenarioNrset))
                                                for l in self.Instance.DisasterAreaSet]
                                                for j in self.Instance.InjuryLevelSet]
                                                for t in self.Instance.TimeBucketSet]
        if Constants.Debug: print("\n InSampleAverageOnTimeTransfer: ", self.InSampleAverageOnTimeTransfer)
        
        ############### Calculates the average amount of "Patient Demand" evacuated on-time! (any patient evacuated after the first period, is assumed delayed)
        self.InSampleAverageOnTimeEvacuation = [[(sum((self.ScenarioTree[w].PatientDemand[0][j][h] - self.UnevacuatedPatients_Phi_wtjh[w][0][j][h])
                        for w in self.SenarioNrset
                        if self.ScenarioTree[w].HospitalDisruption[0][h] == 1  # Only consider disrupted hospitals
                    ) / 
                    sum(1 for w in self.SenarioNrset if self.ScenarioTree[w].HospitalDisruption[0][h] == 1)  # Count disrupted hospital scenarios
                    if any(self.ScenarioTree[w].HospitalDisruption[0][h] == 1 for w in self.SenarioNrset)  # Avoid division by zero
                    else 0)
                    for h in self.Instance.HospitalSet]
                    for j in self.Instance.InjuryLevelSet]

        if Constants.Debug: print("\n InSampleAverageOnTimeEvacuation: ", self.InSampleAverageOnTimeEvacuation)

        ############### Calculates the average amount of "Patient Demand" NOT EVACUATED AT ALL!
        self.InSampleAverageNotEvacuatedAtAll = [[(sum(
                        self.UnevacuatedPatients_Phi_wtjh[w][-1][j][h]  # Last period t = T
                        for w in self.SenarioNrset
                        if self.ScenarioTree[w].HospitalDisruption[0][h] == 1) /
                    sum(1 for w in self.SenarioNrset if self.ScenarioTree[w].HospitalDisruption[0][h] == 1)  # Count disrupted hospital scenarios
                    if any(self.ScenarioTree[w].HospitalDisruption[0][h] == 1 for w in self.SenarioNrset)  # Avoid division by zero
                    else 0)
                for h in self.Instance.HospitalSet]
            for j in self.Instance.InjuryLevelSet]

        if Constants.Debug: print("\n InSampleAverageNotEvacuatedAtAll: ", self.InSampleAverageNotEvacuatedAtAll)

        ###############################################
        self.InSampleTotalCasualtyDemandPerScenario = [sum(sum(sum(self.ScenarioTree[w].CasualtyDemand[0][t][j][l]
                                                        for l in self.Instance.DisasterAreaSet)
                                                        for j in self.Instance.InjuryLevelSet)
                                                        for t in self.Instance.TimeBucketSet)
                                                        for w in self.SenarioNrset]
        if Constants.Debug: print("\n InSampleTotalCasualtyDemandPerScenario: ", self.InSampleTotalCasualtyDemandPerScenario)
        totalcasualtydemand = sum(self.InSampleTotalCasualtyDemandPerScenario)
        if Constants.Debug: print("totalcasualtydemand: ", totalcasualtydemand)
        
        ###############################################
        # Compute total patient demand per scenario (only considering disrupted hospitals)
        self.InSampleTotalPatientDemandPerScenario = [sum(sum(self.ScenarioTree[w].PatientDemand[0][j][h]
                    for h in self.Instance.HospitalSet if self.ScenarioTree[w].HospitalDisruption[0][h] == 1)
                for j in self.Instance.InjuryLevelSet)
            for w in self.SenarioNrset]
        if Constants.Debug: print("\n InSampleTotalPatientDemandPerScenario: ", self.InSampleTotalPatientDemandPerScenario)

        # Compute total patient demand across all scenarios
        totalpatientdemand = sum(self.InSampleTotalPatientDemandPerScenario)
        if Constants.Debug:print("totalpatientdemand: ", totalpatientdemand)

        ###############################################
        self.InSampleTotalTransferPerScenario = [sum(sum(sum(sum(sum(round(self.CasualtyTransfer_q_wtjlum[w][t][j][l][u][m],2)
                                                    for m in self.Instance.RescueVehicleSet)
                                                    for u in self.Instance.MedFacilitySet)
                                                    for l in self.Instance.DisasterAreaSet)
                                                    for j in self.Instance.InjuryLevelSet)
                                                    for t in self.Instance.TimeBucketSet)
                                                    for w in self.SenarioNrset]
        if Constants.Debug: print("\n InSampleTotalTransferPerScenario: ", self.InSampleTotalTransferPerScenario)

        totaltransfer = sum(self.InSampleTotalTransferPerScenario)
        if Constants.Debug: print("totaltransfer: ", totaltransfer)

        ###############################################
        self.InSampleTotalLandEvacuationPerScenario = [sum(sum(sum(sum(sum(round(self.LandEvacuatedPatients_u_L_wtjhum[w][t][j][h][u][m],2)
                                                    for m in self.Instance.RescueVehicleSet)
                                                    for u in self.Instance.MedFacilitySet)
                                                    for h in self.Instance.HospitalSet)
                                                    for j in self.Instance.InjuryLevelSet)
                                                    for t in self.Instance.TimeBucketSet)
                                                    for w in self.SenarioNrset]
        if Constants.Debug: print("\n InSampleTotalLandEvacuationPerScenario: ", self.InSampleTotalLandEvacuationPerScenario)

        self.InSampleTotalAerialEvacuationPerScenario = [sum(sum(sum(sum(sum(sum(round(self.AerialEvacuatedPatients_u_A_wtjhihPrimem[w][t][j][h][i][hprime][m],2)
                                                    for m in self.Instance.RescueVehicleSet)
                                                    for hprime in self.Instance.HospitalSet)
                                                    for i in self.Instance.ACFSet)
                                                    for h in self.Instance.HospitalSet)
                                                    for j in self.Instance.InjuryLevelSet)
                                                    for t in self.Instance.TimeBucketSet)
                                                    for w in self.SenarioNrset]
        if Constants.Debug: print("\n InSampleTotalAerialEvacuationPerScenario: ", self.InSampleTotalAerialEvacuationPerScenario)

        self.InSampleTotalEvacuationPerScenario = [
            self.InSampleTotalLandEvacuationPerScenario[w] + self.InSampleTotalAerialEvacuationPerScenario[w]
            for w in self.SenarioNrset]

        if Constants.Debug:print("\n InSampleTotalEvacuationPerScenario: ", self.InSampleTotalEvacuationPerScenario)

        totalevacuation = sum(self.InSampleTotalEvacuationPerScenario)
        if Constants.Debug: print("totalevacuation: ", totalevacuation)

        ###############################################
        self.InSampleTotalNotEvacuationAtAllPerScenario = [sum(sum(self.UnevacuatedPatients_Phi_wtjh[w][-1][j][h]
                    for j in self.Instance.InjuryLevelSet)
                for h in self.Instance.HospitalSet if self.ScenarioTree[w].HospitalDisruption[0][h] == 1)
            if any(self.ScenarioTree[w].HospitalDisruption[0][h] == 1 for h in self.Instance.HospitalSet)  # Check if any hospital is disrupted
            else 0
            for w in self.SenarioNrset]
        if Constants.Debug:print("\n InSampleTotalNotEvacuationAtAllPerScenario: ", self.InSampleTotalNotEvacuationAtAllPerScenario)

        ###############################################
        self.InSampleTotalOnTimeTransferPerScenario = [sum(sum(sum(
                                                                    max(self.ScenarioTree[w].CasualtyDemand[0][t][j][l] - self.UnsatisfiedCasualties_mu_wtjl[w][t][j][l], 0)
                                                                    for l in self.Instance.DisasterAreaSet)
                                                                    for j in self.Instance.InjuryLevelSet)
                                                                    for t in self.Instance.TimeBucketSet)
                                                                    for w in self.SenarioNrset]
        if Constants.Debug:print("\n InSampleTotalOnTimeTransferPerScenario: ", self.InSampleTotalOnTimeTransferPerScenario)

        ###############################################
        self.InSampleTotalOnTimeEvacuationPerScenario = [sum(sum(self.ScenarioTree[w].PatientDemand[0][j][h] - self.UnevacuatedPatients_Phi_wtjh[w][0][j][h]
                    for j in self.Instance.InjuryLevelSet)
                for h in self.Instance.HospitalSet if self.ScenarioTree[w].HospitalDisruption[0][h] == 1)
            if any(self.ScenarioTree[w].HospitalDisruption[0][h] == 1 for h in self.Instance.HospitalSet)  # Check if any hospital is disrupted
            else 0
            for w in self.SenarioNrset]
        if Constants.Debug:print("\n InSampleTotalOnTimeEvacuationPerScenario: ", self.InSampleTotalOnTimeEvacuationPerScenario)

        ###############################################
        nrscenario = len(self.ScenarioTree)
        self.InSampleAverageCasualtyDemand = round(sum(self.InSampleTotalCasualtyDemandPerScenario[w] for w in self.SenarioNrset) / nrscenario,2)
        if Constants.Debug: print("\n InSampleAverageCasualtyDemand: ", self.InSampleAverageCasualtyDemand)
        
        ###############################################
        num_disrupted_scenarios = sum(1 for w in self.SenarioNrset if any(self.ScenarioTree[w].HospitalDisruption[0][h] == 1 for h in self.Instance.HospitalSet))

        # Compute the average patient demand only considering disrupted scenarios
        if num_disrupted_scenarios > 0:
            self.InSampleAveragePatientDemand = round(sum(self.InSampleTotalPatientDemandPerScenario[w] for w in self.SenarioNrset) / num_disrupted_scenarios, 2)
        else:
            self.InSampleAveragePatientDemand = 0  # Avoid division by zero
        if Constants.Debug:
            print("\n Number of disrupted scenarios: ", num_disrupted_scenarios)
            print("\n InSampleAveragePatientDemand: ", self.InSampleAveragePatientDemand)
            
        ###############################################
        self.InSampleAverageTransfer = round(sum(self.InSampleTotalTransferPerScenario[w] for w in self.SenarioNrset) / nrscenario, 2)
        if Constants.Debug: print("\n InSampleAverageTransfer: ", self.InSampleAverageTransfer)
        
        ###############################################
        self.InSampleAverageEvacuation = round(sum(self.InSampleTotalEvacuationPerScenario[w] for w in self.SenarioNrset) / num_disrupted_scenarios,2)
        if Constants.Debug: print("\n InSampleAverageEvacuation: ", self.InSampleAverageEvacuation)
        
        ###############################################
        self.InSampleAverageNotEvacuationAtAll = round(sum(self.InSampleTotalNotEvacuationAtAllPerScenario[w] for w in self.SenarioNrset) / num_disrupted_scenarios,2)
        if Constants.Debug: print("\n InSampleAverageNotEvacuationAtAll: ", self.InSampleAverageNotEvacuationAtAll)

        ###############################################
        self.InSamplePercentOnTimeTransfer = round(100 * (sum(self.InSampleTotalOnTimeTransferPerScenario[s] for s in self.SenarioNrset)) / totalcasualtydemand, 2)
        if Constants.Debug: print("\n InSamplePercentOnTimeTransfer (%): ", self.InSamplePercentOnTimeTransfer)
        
        ###############################################
        self.InSamplePercentOnTimeEvacuation = round(100 * (sum(self.InSampleTotalOnTimeEvacuationPerScenario[s] for s in self.SenarioNrset)) / totalpatientdemand, 2)
        if Constants.Debug: print("\n InSamplePercentOnTimeEvacuation: ", self.InSamplePercentOnTimeEvacuation)
        
        ###############################################
        self.InSamplePercentNotEvacuatedAtAll = round(100 * (sum(self.InSampleTotalNotEvacuationAtAllPerScenario[s] for s in self.SenarioNrset)) / totalpatientdemand, 2)
        if Constants.Debug: print("\n InSamplePercentNotEvacuatedAtAll: ", self.InSamplePercentNotEvacuatedAtAll)

    def GetNrACFEstablishment(self):
        if not isinstance(self.ScenarioTree, list):     # Always Convert ScenarioTree to a list.
            self.ScenarioTree = [self.ScenarioTree]

        if(Constants.Debug): print("self.ScenarioTree: ", self.ScenarioTree)
        result = sum(self.ACFEstablishment_x_wi[w][i] 
                     for i in self.Instance.ACFSet 
                     for w in range(len(self.ScenarioTree)))
        result = result / len(self.ScenarioTree)

        return result
    
    def GetNrLandVehicleAssignment(self):
        if not isinstance(self.ScenarioTree, list):     # Always Convert ScenarioTree to a list.
            self.ScenarioTree = [self.ScenarioTree]

        result = sum(self.LandRescueVehicle_thetaVar_wim[w][i][m] 
                     for m in self.Instance.RescueVehicleSet 
                     for i in self.Instance.ACFSet 
                     for w in range(len(self.ScenarioTree)))
        result = result / len(self.ScenarioTree)

        return result
    
    def GetNrBackupHospital(self):
        if not isinstance(self.ScenarioTree, list):     # Always Convert ScenarioTree to a list.
            self.ScenarioTree = [self.ScenarioTree]

        result = sum(self.BackupHospital_W_whhPrime[w][h][hprime] 
                     for hprime in self.Instance.HospitalSet 
                     for h in self.Instance.HospitalSet 
                     for w in range(len(self.ScenarioTree)))
        result = result / len(self.ScenarioTree)

        return result
    
    #This function print the statistic in an Excel file
    def PrintStatistics(self, testidentifier, filepostscript, offsetseed, nrevaluation,  evaluationduration, insample, evaluationmethod):
        if Constants.Debug: print("\n We are in 'Solution' Class -- PrintStatistics")

        #To compute every statistic Constants.PrintOnlyFirstStageDecision should be False
        if (not Constants.PrintOnlyFirstStageDecision) or (not insample):
            
            if Constants.PrintDetailsExcelFiles:
                self.PrintDetailExcelStatistic( filepostscript, offsetseed, nrevaluation,  testidentifier, evaluationmethod )


            self.ComputeCost()            
        
        nracfs = self.GetNrACFEstablishment()
        nrlandvehiclesassigned = self.GetNrLandVehicleAssignment()
        nrbackuphospital = self.GetNrBackupHospital()

        kpistat = [ self.GRBCost,
                    self.GRBTime,
                    self.GRBGap,
                    self.GRBNrConstraints,
                    self.GRBNrVariables,
                    self.PHCost,
                    self.PHNrIteration,
                    self.TotalTime,
                    self.ACFestablishmentcost,
                    self.LandRescueVehiclecost,
                    self.BackupHospitalcost,
                    self.CasualtyTransferCost,
                    self.UnsatisfiedCasualtiesCost,
                    self.DischargedPatientsCost,
                    self.LandEvacuatedPatientsCost,
                    self.AerialEvacuatedPatientsCost,
                    self.UnevacuatedPatientsCost,
                    self.AvailableCapFacilityCost,
                    self.InSamplePercentOnTimeTransfer,
                    self.InSamplePercentOnTimeEvacuation,
                    self.InSamplePercentNotEvacuatedAtAll,
                    nracfs,
                    nrlandvehiclesassigned,
                    nrbackuphospital,
                    evaluationduration
                    ]

        # Assuming 'column_names' is a list of strings representing the column names
        column_names = ["GRB Cost", "GRB Time", "GRB Gap (%)", "GRB Nr Constraints", "GRB Nr Variables",   
                        "PHA Cost", "PHA Nr Iteration", "Total Time", 
                        "ACF Establishment Cost", 
                        "Land Rescue Vehicle Assign. Cost", 
                        "Backup Hospital Cost", 
                        "Casualty Transfer Cost",
                        "UnsatisfiedCasualties Cost",
                        "DischargedPatients Cost",
                        "LandEvacuatedPatients Cost",
                        "AerialEvacuatedPatients Cost",
                        "UnevacuatedPatient Cost",
                        "AvailableCapFacility Cost",
                        "% On-Time Transfer", 
                        "% On-Time Evacuation", 
                        "% Not Evacuated", 
                        "Nr ACF Established", 
                        "Nr Land Res. Vehicle Assigned", 
                        "Nr Backup Hospitals", 
                        "Evaluation Duration"]
        # Adding additional headers for data from testidentifier and other values
        additional_headers = ["Instance Name", "Model", "Solver", "Scenario Sampling", "Number of Scenarios", "Scenario Seed"]
        
        full_column_names = additional_headers + ["Out/In", "NrEvaluation"] + column_names

        data = testidentifier.GetAsStringList() + [filepostscript, len(self.ScenarioTree)] + kpistat


        if Constants.PrintDetailsExcelFiles:
            d = datetime.now()
            date = d.strftime('%m_%d_%Y_%H_%M_%S')
            myfile = open(r'./Test/Statistic/TestResult_%s_%r.csv' % (testidentifier.GetAsString(), filepostscript), 'w')
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(full_column_names)
            wr.writerow(data)
            myfile.close()


        return kpistat
    
    #This function print detailed statistics about the obtained solution (avoid using it as it consume memory)
    def PrintDetailExcelStatistic(self, filepostscript, offsetseed, nrevaluation,  testidentifier, evaluationmethod):
        if Constants.Debug: print("\n We are in 'Solution' Class -- PrintDetailExcelStatistic")

        scenarioset = range(len(self.ScenarioTree))

        d = datetime.now()
        date = d.strftime('%m_%d_%Y_%H_%M_%S')
        file_path = "./Solutions/" + testidentifier.GetAsString() + "_Statistics_" + filepostscript + ".xlsx"

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            avgACFEstablishment_df = pd.DataFrame(self.InSampleAverageACFEstablishment)
            avgACFEstablishment_df.to_excel(writer, sheet_name="Avg.ACFEstablishment")

            avgLandRescueVehicle_df = pd.DataFrame(self.InSampleAverageLandRescueVehicle)
            avgLandRescueVehicle_df.to_excel(writer, sheet_name="Avg.LandRescueVehicle")

            avgBackupHospital_df = pd.DataFrame(self.InSampleAverageBackupHospital)
            avgBackupHospital_df.to_excel(writer, sheet_name="Avg.BackupHospital")

            avgcasualtyTransfer_df = pd.DataFrame(self.InSampleAverageCasualtyTransfer)
            avgcasualtyTransfer_df.to_excel(writer, sheet_name="Avg.CasualtyTransfer")
            
            avgunsatisfiedCasualties_df = pd.DataFrame(self.InSampleAverageUnsatisfiedCasualties)
            avgunsatisfiedCasualties_df.to_excel(writer, sheet_name="Avg.UnsatisfiedCasualties")
            
            avgdischargedPatients_df = pd.DataFrame(self.InSampleAverageDischargedPatients)
            avgdischargedPatients_df.to_excel(writer, sheet_name="Avg.DischargedPatients")
            
            avglandEvacuatedPatients_df = pd.DataFrame(self.InSampleAverageLandEvacuatedPatients)
            avglandEvacuatedPatients_df.to_excel(writer, sheet_name="Avg.LandEvacuatedPatients")
            
            avgaerialEvacuatedPatients_df = pd.DataFrame(self.InSampleAverageAerialEvacuatedPatients)
            avgaerialEvacuatedPatients_df.to_excel(writer, sheet_name="Avg.AerialEvacuatedPatients")
            
            avgunevacuatedPatients_df = pd.DataFrame(self.InSampleAverageUnevacuatedPatients)
            avgunevacuatedPatients_df.to_excel(writer, sheet_name="Avg.UnevacuatedPatients")
            
            avgavailableCapFacility_df = pd.DataFrame(self.InSampleAverageAvailableCapFacility)
            avgavailableCapFacility_df.to_excel(writer, sheet_name="Avg.AvailableCapFacility")
            

            ######################### INFOR PER SCENARIO
            start_row = 0  # Keep track of where to write each DataFrame

            # Write Total Casualty Demand per scenario
            perscenario_casualtydemand_df = pd.DataFrame(
                [self.InSampleTotalCasualtyDemandPerScenario],
                index=["Total Casualty Demand"],
                columns=scenarioset)
            perscenario_casualtydemand_df.to_excel(writer, sheet_name="Info Per Scenario", startrow=start_row)
            start_row += len(perscenario_casualtydemand_df.index) + 2  # Move down after writing

            # Write Total Patient Demand per scenario
            perscenario_patientdemand_df = pd.DataFrame(
                [self.InSampleTotalPatientDemandPerScenario],
                index=["Total Patient Demand"],
                columns=scenarioset)
            perscenario_patientdemand_df.to_excel(writer, sheet_name="Info Per Scenario", startrow=start_row)
            start_row += len(perscenario_patientdemand_df.index) + 2

            # Write Total Casualty Transfer per scenario
            perscenario_casualtytransfer_df = pd.DataFrame(
                [self.InSampleTotalTransferPerScenario],
                index=["Total Casualty Transfer"],
                columns=scenarioset)
            perscenario_casualtytransfer_df.to_excel(writer, sheet_name="Info Per Scenario", startrow=start_row)
            start_row += len(perscenario_casualtytransfer_df.index) + 2

            # Write Total Patient Transfer per scenario
            perscenario_patienttransfer_df = pd.DataFrame(
                [self.InSampleTotalEvacuationPerScenario],
                index=["Total Patient Transfer"],
                columns=scenarioset)
            perscenario_patienttransfer_df.to_excel(writer, sheet_name="Info Per Scenario", startrow=start_row)

            ######################### END OF INFOR PER SCENARIO

            general = testidentifier.GetAsStringList() + [
                                                            self.InSampleAverageCasualtyDemand,
                                                            self.InSampleAveragePatientDemand,
                                                            self.InSampleAverageTransfer,
                                                            self.InSampleAverageEvacuation,
                                                            self.InSampleAverageNotEvacuationAtAll,
                                                            offsetseed,
                                                            nrevaluation,
                                                            testidentifier.ScenarioSeed,
                                                            evaluationmethod]

            # Ensure the general data is in a 2D list format for DataFrame conversion
            general = [general]  # Wrapping general list in another list to make it 2D

            columnstab = ["Instance", 
                          "Model", 
                          "Solver", 
                          "ScenarioGeneration", 
                          "NrScenario", 
                          "ScenarioSeed", 
                          "PHAObj", 
                          "PHAPenalty", 
                          "ALNSRL", 
                          "ALNSRL_DeepQ", 
                          "RLSelectionMethod", 
                          "BBC_Accelerator", 
                          "Clustering_Method",
                          "Average Casualty Dem.", 
                          "Average Patient Dem.", 
                          "Average Transfer", 
                          "Average Evacuation", 
                          "Average Not Evac. at All", 
                          "offsetseed", 
                          "nrevaluation", 
                          "solutionseed", 
                          "evaluationmethod"]

            generaldf = pd.DataFrame(general, columns=columnstab)
            generaldf.to_excel(writer, sheet_name="General")

    #This function merge solution2 into self. Assume that solution2 has a single scenario
    def Merge(self, solution2):
        if Constants.Debug:
            print("\n We are in 'Solution' Class -- Merge")

        if(Constants.Debug): print("Before---------- solution2.ScenarioTree:\n", solution2.ScenarioTree)
        if(Constants.Debug): print("Before---------- self.ScenarioTree:\n", self.ScenarioTree)

        # Flatten solution2.ScenarioTree if it's a list
        if isinstance(solution2.ScenarioTree, list):
            self.ScenarioTree.extend(solution2.ScenarioTree)  # Extend with elements from solution2
        else:
            self.ScenarioTree.append(solution2.ScenarioTree)  # Append single ScenarioTree object

        # Update scenario index range
        self.SenarioNrset = range(len(self.ScenarioTree))

        if(Constants.Debug): print("After---------- solution2.ScenarioTree:\n", solution2.ScenarioTree)
        if(Constants.Debug): print("After---------- self.ScenarioTree:\n", self.ScenarioTree)
        if(Constants.Debug): print("self.SenarioNrset: ", self.SenarioNrset)

        self.ACFEstablishment_x_wi = self.ACFEstablishment_x_wi + solution2.ACFEstablishment_x_wi
        if Constants.Debug: print("self.ACFEstablishment_x_wi:\n ", self.ACFEstablishment_x_wi)
        
        self.LandRescueVehicle_thetaVar_wim = self.LandRescueVehicle_thetaVar_wim + solution2.LandRescueVehicle_thetaVar_wim
        if Constants.Debug: print("self.LandRescueVehicle_thetaVar_wim:\n ", self.LandRescueVehicle_thetaVar_wim)
        
        self.BackupHospital_W_whhPrime = self.BackupHospital_W_whhPrime + solution2.BackupHospital_W_whhPrime
        if Constants.Debug: print("self.BackupHospital_W_whhPrime:\n ", self.BackupHospital_W_whhPrime)

        self.CasualtyTransfer_q_wtjlum = self.CasualtyTransfer_q_wtjlum + solution2.CasualtyTransfer_q_wtjlum
        if Constants.Debug: print("self.CasualtyTransfer_q_wtjlum:\n ", self.CasualtyTransfer_q_wtjlum)

        self.UnsatisfiedCasualties_mu_wtjl = self.UnsatisfiedCasualties_mu_wtjl + solution2.UnsatisfiedCasualties_mu_wtjl
        if Constants.Debug: print("self.UnsatisfiedCasualties_mu_wtjl:\n ", self.UnsatisfiedCasualties_mu_wtjl)

        self.DischargedPatients_sigmaVar_wtju = self.DischargedPatients_sigmaVar_wtju + solution2.DischargedPatients_sigmaVar_wtju
        if Constants.Debug: print("self.DischargedPatients_sigmaVar_wtju:\n ", self.DischargedPatients_sigmaVar_wtju)

        self.LandEvacuatedPatients_u_L_wtjhum = self.LandEvacuatedPatients_u_L_wtjhum + solution2.LandEvacuatedPatients_u_L_wtjhum
        if Constants.Debug: print("self.LandEvacuatedPatients_u_L_wtjhum:\n ", self.LandEvacuatedPatients_u_L_wtjhum)

        self.AerialEvacuatedPatients_u_A_wtjhihPrimem = self.AerialEvacuatedPatients_u_A_wtjhihPrimem + solution2.AerialEvacuatedPatients_u_A_wtjhihPrimem
        if Constants.Debug: print("self.AerialEvacuatedPatients_u_A_wtjhihPrimem:\n ", self.AerialEvacuatedPatients_u_A_wtjhihPrimem)

        self.UnevacuatedPatients_Phi_wtjh = self.UnevacuatedPatients_Phi_wtjh + solution2.UnevacuatedPatients_Phi_wtjh
        if Constants.Debug: print("self.UnevacuatedPatients_Phi_wtjh:\n ", self.UnevacuatedPatients_Phi_wtjh)

        self.AvailableCapFacility_zeta_wtu = self.AvailableCapFacility_zeta_wtu + solution2.AvailableCapFacility_zeta_wtu
        if Constants.Debug: print("self.AvailableCapFacility_zeta_wtu:\n ", self.AvailableCapFacility_zeta_wtu)