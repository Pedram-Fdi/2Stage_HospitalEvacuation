import os
import platform
import pandas as pd
from Constants import Constants
import random
import math
import numpy as np
import pickle
import googlemaps
import openpyxl as opxl
from scipy.stats import norm, lognorm, gamma


#uncomment to create a plot of the supply chain (might create some interference with CPLEX)
#import networkx as nx
#import matplotlib.pyplot as plt

class Instance(object):

    # Constructor
    def __init__(self, instanceName):

        self.InstanceName = instanceName

        self.NrTimeBucket = -1
        self.NrACFs = -1
        self.NrHospitals = -1
        self.NrMedFacilities = -1
        self.NrDisasterAreas = -1
        self.NrRescueVehicles = -1
        self.NrInjuries = -1
        self.NrScenarios = -1
        self.Distribution = "Uniform"

        self.Speed_Land = 50;                           # (km/h) Ref: (Fairness in ambulance routing for post disaster management)
        self.Speed_Land_AmBus = 40;                       
        self.Speed_Aerial = 115;                        # (km/h) Ref: (Optimal decision-making of post-disaster emergency material scheduling based on helicopter–truck–drone collaboration)                       
        self.Square_Dimension = 50;                     # In KM
        self.Working_Hours_per_Day = 12;                # working Hours for vehicles 
        self.Number_of_Planning_Days = 0.5;               # Each period is 12h (It should be always a multiplier of a day since I will use it in risk parameters too)


        #Domain of Parameters
        self.Min_ACF_Bed_Capacity = 250
        self.Max_ACF_Bed_Capacity = 500

        self.m2_Required_for_Each_Patient = 1.3
        self.Cost_of_Each_m2 = 25        

        self.Min_VehicleAssignment_Cost = 0.001
        self.Max_VehicleAssignment_Cost = 0.001

        self.Min_Demand_in_Each_Location = 0 * self.Number_of_Planning_Days
        self.Max_Demand_in_Each_Location = 100 * self.Number_of_Planning_Days

        # Injury level percentages
        self.Priority_Patient_Percent = {
            0: 0.40,  # High priority
            1: 0.60   # Low priority
        }

        self.Do_you_need_point_plot = 0

        self.Min_Hospital_Bed_Capacity = 400
        self.Max_Hospital_Bed_Capacity = 600        


        self.Min_Casualty_Shortage_Cost = 150000
        self.Max_Casualty_Shortage_Cost = 150000

        self.HighPriority_EvacuationRiskCost = 100000   

        self.Safety_Factor_Rescue_Vehicle_ACF = 3           # (Default for non-case: 5) For having 0 Shortage (For demand between [50,200], its defaul is on 3)
        self.Safety_Factor_Rescue_Vehicle_Hospital = 4      #For having 0 Shortage (For demand between [50,200], its defaul is on 1.5)

        self.MinHospitalOccupationRate = 0.50                 # (Accurate for Turkey: 55.3% Ref:https://www.statista.com/statistics/1116612/oecd-hospital-acute-care-occupancy-rates-select-countries-worldwide/) or Accurate: 69.8% Ref: https://www.oecd.org/en/publications/health-at-a-glance-2023_7a7afb35-en/full-report/hospital-beds-and-occupancy_10add5df.html
        self.MaxHospitalOccupationRate = 0.60                 # (Accurate for Turkey: 55.3% Ref:https://www.statista.com/statistics/1116612/oecd-hospital-acute-care-occupancy-rates-select-countries-worldwide/) or (Accurate: 69.8% Ref: https://www.oecd.org/en/publications/health-at-a-glance-2023_7a7afb35-en/full-report/hospital-beds-and-occupancy_10add5df.html
        
        self.MinAerial_ToBeAssigned = 5
        self.MaxAerial_ToBeAssigned = 10
        
        self.MinCoordinationCost = 200                        # if you set min to 2 then the Ref is: (An integrated blood supply chain network design during a pandemic)
        self.MaxCoordinationCost = 900                        # if you set max to 9 then the Ref is:  (An integrated blood supply chain network design during a pandemic)


        # Only Parameters used in the Mathematical Model
        self.ACF_Bed_Capacity = []
        self.Hospital_Bed_Capacity = []
        self.Fixed_Cost_ACF_Objective = []
        self.Fixed_Cost_ACF_Constraint = []
        self.Total_Budget_ACF_Establishment = []
        self.VehicleAssignment_Cost = []
        self.ForecastedAvgCasualtyDemand = []
        self.ForecastedSTDCasualtyDemand = []
        self.Land_Rescue_Vehicle_Capacity = []
        self.Aerial_Rescue_Vehicle_Capacity = []
        self.Distance_D_A = []
        self.Distance_A_H = []
        self.Distance_D_H = []
        self.Distance_A_A = []
        self.Distance_H_H = []
        self.Distance_U_U = []
        self.Time_D_A_Land = []
        self.Time_A_H_Land = []
        self.Time_D_H_Land = []
        self.Time_A_A_Land = []
        self.Time_H_H_Land = []
        self.Time_U_U_Land = []
        self.Time_D_A_Aerial = []
        self.Time_A_H_Aerial = []
        self.Time_D_H_Aerial = []
        self.Time_A_A_Aerial = []
        self.Time_H_H_Aerial = []
        self.Casualty_Shortage_Cost = []
        self.Number_Rescue_Vehicle_ACF = []
        self.Number_Land_Rescue_Vehicle_Hospital = []
        self.ForecastedAvgHospitalDisruption = []
        self.ForecastedSTDHospitalDisruption = []
        self.ForecastedAvgPatientDemand = []
        self.ForecastedSTDPatientDemand = []
        self.ForecastedAvgPercentagePatientDischarged = []
        self.ForecastedSTDPercentagePatientDischarged = []
        self.Max_Backup_Hospital = []
        self.Available_Aerial_Vehicles_Hospital = []
        self.CoordinationCost = []
        self.EvacuationRiskCost = []
        self.CumulativeThreatRiskConstant = []
        self.CumulativeThreatRiskLinear = []
        self.CumulativeThreatRiskExponential = []
        self.CumulativeLandTransportation_Risk = []
        self.CumulativeAerialTransportRisk = []
        self.LandEvacuationRisk_Constant = []
        self.LandEvacuationRisk_Linear = []
        self.LandEvacuationRisk_Exponential = []
        self.AerialEvacuationRisk_Constant = []
        self.AerialEvacuationRisk_Linear = []
        self.AerialEvacuationRisk_Exponential = []

        self.My_EpGap= 0.0001
        self.Do_you_want_Dependent_Hospital_Capacities_based_on_Demands = 0

    def ComputeIndices(self):
        if Constants.Debug: print("\n We are in 'Instance' Class -- ComputeIndices")

        self.TimeBucketSet = range(self.NrTimeBucket)
        self.ACFSet = range(self.NrACFs)
        self.HospitalSet = range(self.NrHospitals)
        self.MedFacilitySet = range(self.NrMedFacilities)
        self.DisasterAreaSet = range(self.NrDisasterAreas)
        self.RescueVehicleSet = range(self.NrRescueVehicles)
        self.InjuryLevelSet = range(self.NrInjuries)
        self.ScenariosSet = range(self.NrScenarios)

    def build_J_u(self):
        """Creates a binary matrix J_u of size [NrInjuries x NrMedFacilities]."""
        self.J_u = [[1] * self.NrHospitals + [0] * self.NrACFs,   # First row: Injury type 1
                    [1] * self.NrMedFacilities]                   # Second: Injury type 2
    
    def build_J_m(self):
        """Creates a binary matrix J_u of size [NrInjuries x NrRescueVehicles]."""
        self.J_m = [[1] * 1 + [1] * 1 + [0] * 1,   # First row: Injury type 1
                    [1] * self.NrRescueVehicles]    # Second: Injury type 2

    def assign_backup_hospitals(self, BackupPercentage, seed = None):
        """
        Assigns backup hospitals for each hospital.
        
        :param BackupPercentage: float, percentage of hospitals that should serve as backup for each hospital h.
        :param seed: int, random seed for reproducibility
        """

        if seed is not None:
            random.seed(seed)        

        self.K_h = {}  # Dictionary to store backup hospitals for each h

        for h in self.HospitalSet:
            # Number of backup hospitals for each hospital h
            num_backups = max(1, int(np.ceil((self.NrHospitals - 1) * BackupPercentage)))
            
            # Ensure the hospital does not select itself as backup
            possible_backups = list(set(range(self.NrHospitals)) - {h})

            # Randomly select backup hospitals
            self.K_h[h] = set(random.sample(possible_backups, min(num_backups, len(possible_backups))))

    def assign_aerial_acfs(self, percentage, seed=None):
        """
        Selects a percentage of ACFs randomly to be equipped with aerial evacuation area.
        
        :param percentage: float, percentage of ACFs to have aerial equipment (e.g., 0.3 for 30%)
        :param seed: int, random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            
        num_aerial_acfs = int(self.NrACFs * percentage)  # Number of ACFs to equip
        self.ACFs_Aerial_Set = set(random.sample(range(self.NrACFs), num_aerial_acfs))  # Randomly select ACFs

    def Generate_Data(self, seed=None):
        if Constants.Debug: print("\n We are in 'Instance' Class -- Generate_Data")

        if seed is not None:
            random.seed(seed)

        ################################## Generate ACF Bed Capacities
        for i in self.ACFSet:
            New_ACF_Bed_Capacity = random.randint(self.Min_ACF_Bed_Capacity, self.Max_ACF_Bed_Capacity )
            self.ACF_Bed_Capacity.append(New_ACF_Bed_Capacity)

        ################################## ACFs with Staging Area for Helicopter
        """Randomly assigns 50% of ACFs to the set I_A, which represents staging areas for helicopters."""
        num_selected = max(1, self.NrACFs // 2)  # Ensure at least one ACF is selected
        self.I_A_Set = set(random.sample(range(self.NrACFs), num_selected))  # Select random ACFs
        self.I_A_List = list(self.I_A_Set)  

        ################################## Generate Fixed Costs ACF based on ACF Bed Capacities for the Objective Function
        for i in self.ACFSet:
            New_Fixed_Cost_ACF_Objective = self.ACF_Bed_Capacity[i] * self.m2_Required_for_Each_Patient * self.Cost_of_Each_m2 * 0.00001
            New_Fixed_Cost_ACF_Objective = math.floor(1000 * New_Fixed_Cost_ACF_Objective) / 1000
            self.Fixed_Cost_ACF_Objective.append(New_Fixed_Cost_ACF_Objective)        

        ################################## Generate Fixed Costs ACF based on ACF Bed Capacities for the Constraint
        for i in self.ACFSet:
            New_Fixed_Cost_ACF_Constraint = self.ACF_Bed_Capacity[i] * self.m2_Required_for_Each_Patient * self.Cost_of_Each_m2
            New_Fixed_Cost_ACF_Constraint = math.floor(1000 * New_Fixed_Cost_ACF_Constraint) / 1000
            self.Fixed_Cost_ACF_Constraint.append(New_Fixed_Cost_ACF_Constraint)

        ################################## Generate Available Budget for ACFs
        Total_Required_Budget_for_ACF_Establishment = 0
        for i in self.ACFSet:
            Total_Required_Budget_for_ACF_Establishment += self.Fixed_Cost_ACF_Constraint[i]
        Total_Required_Budget_for_ACF_Establishment = ((Total_Required_Budget_for_ACF_Establishment * 17) / 20)
        Max_Required_Budget_for_ACF_Establishment = max(self.Fixed_Cost_ACF_Constraint)
        print("Total_Required_Budget_for_ACF_Establishment: ", Total_Required_Budget_for_ACF_Establishment)
        print("Max_Required_Budget_for_ACF_Establishment: ", Max_Required_Budget_for_ACF_Establishment)
        
        self.Total_Budget_ACF_Establishment = max(Total_Required_Budget_for_ACF_Establishment, Max_Required_Budget_for_ACF_Establishment)   # The total available budget to establish ACFs*/
        
        ################################## Generate VehicleAssignment_Cost       
        for m in self.RescueVehicleSet:  
            New_VehicleAssignment_Cost = random.uniform(self.Min_VehicleAssignment_Cost, self.Max_VehicleAssignment_Cost)
            New_VehicleAssignment_Cost = math.floor(1000 * New_VehicleAssignment_Cost) / 1000
            self.VehicleAssignment_Cost.append(New_VehicleAssignment_Cost)

        ##################################  Calculating Forecasted Average Demand
        self.ForecastedAvgCasualtyDemand = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.DisasterAreaSet)))
        for t in self.TimeBucketSet:
            for l in self.DisasterAreaSet:
                # Generate total demand for each location and time period
                total_demand = (self.Min_Demand_in_Each_Location + self.Max_Demand_in_Each_Location) / 2               #For uniform Distribution 
                for j in self.InjuryLevelSet:
                    # Allocate a portion of total demand to each injury level
                    demand_for_injury_level = round(total_demand * self.Priority_Patient_Percent[j])
                        
                    self.ForecastedAvgCasualtyDemand[t][j][l] = demand_for_injury_level

        ##################################  Calculating Forecasted Average Standard Deviation Demand
        self.ForecastedSTDCasualtyDemand = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.DisasterAreaSet)))
        
        for t in self.TimeBucketSet:
            for l in self.DisasterAreaSet:
                # Generate total demand for each location and time period
                total_STD_demand = (self.Max_Demand_in_Each_Location - self.Min_Demand_in_Each_Location) / 2
                
                for j in self.InjuryLevelSet:
                    # Allocate a portion of total demand to each injury level
                    STD_demand_for_injury_level = round(total_STD_demand * self.Priority_Patient_Percent[j])


                    self.ForecastedSTDCasualtyDemand[t][j][l] = STD_demand_for_injury_level

        ##################################  Calculating Average Hospital Bed Capacity
        self.Hospital_Bed_Capacity = np.zeros((len(self.HospitalSet)))  

        for h in self.HospitalSet:
        # Generate total bed for each location and time period
            if self.Do_you_want_Dependent_Hospital_Capacities_based_on_Demands == 1:
                total_bed = random.randint(self.Min_Hospital_Bed_Capacity, self.Max_Hospital_Bed_Capacity * (self.NrDisasterAreas/10))
                self.Hospital_Bed_Capacity[h] = total_bed
            else:
                #total_bed = random.randint(self.Min_Hospital_Bed_Capacity, self.Max_Hospital_Bed_Capacity)
                total_bed = random.randint(self.Min_Hospital_Bed_Capacity, self.Max_Hospital_Bed_Capacity)
                self.Hospital_Bed_Capacity[h] = total_bed   


        ##################################  Calculating Land Rescue Vehicle Capacity

        a = (self.Square_Dimension / 2)
        nominal_Land_Rescue_Vehicle_Capacity = math.floor(0.5 * (self.Speed_Land / a) * self.Working_Hours_per_Day * self.Number_of_Planning_Days)    # Approximate number of patients transferred in the planning Horizon (Ex: 1 week)
        nominal_Land_Rescue_Vehicle_Capacity_Ambus = math.floor(0.5 * (self.Speed_Land_AmBus / a) * self.Working_Hours_per_Day * self.Number_of_Planning_Days)    # Approximate number of patients transferred in the planning Horizon (Ex: 1 week)
        
        for m in self.RescueVehicleSet:  # Assuming Number_Vehicle_Mode means there are four modes (0 to 3)
            new_Land_Rescue_Vehicle_Capacity = 0
            if m == 0:
                new_Land_Rescue_Vehicle_Capacity = 1 * nominal_Land_Rescue_Vehicle_Capacity
            elif m == 1:
                new_Land_Rescue_Vehicle_Capacity = 2 * nominal_Land_Rescue_Vehicle_Capacity
            elif m == 2:
                new_Land_Rescue_Vehicle_Capacity = 20 * nominal_Land_Rescue_Vehicle_Capacity_Ambus      ## REf: (Decision support for hospital evacuation and emergency response) and (https://txemtf.org/avada_portfolio/ambus/)
            elif m >= 3:
                print("\nThe number of Vehicles (index m) cannot be more than 3!!!!!!!!!\n")
                input("Press Enter to continue...")  # system("Pause") equivalent in Python
            self.Land_Rescue_Vehicle_Capacity.append(new_Land_Rescue_Vehicle_Capacity)
        
        ##################################  Calculating Aerial Rescue Vehicle Capacity

        a = (self.Square_Dimension / 2)
        nominal_Aerial_Rescue_Vehicle_Capacity = math.floor(0.5 * (self.Speed_Aerial / a) * self.Working_Hours_per_Day * self.Number_of_Planning_Days)    # Approximate number of patients transferred in the planning Horizon (Ex: 1 week)
        new_Aerial_Rescue_Vehicle_Capacity = 10 * nominal_Aerial_Rescue_Vehicle_Capacity        ## Ref: (A bi-objective robust optimization model for disaster response planning under uncertainties)
        self.Aerial_Rescue_Vehicle_Capacity.append(new_Aerial_Rescue_Vehicle_Capacity)

        ##################################  Calculating Distances

        self.Hospital_Position = self.Generate_Positions(self.NrHospitals)
        print("Hospital_Positions: \n", self.Hospital_Position)

        self.ACF_Position = self.Generate_Positions(self.NrACFs)
        print("ACF_Position: \n", self.ACF_Position)

        self.DisasterArea_Position = self.Generate_Positions(self.NrDisasterAreas)
        print("DisasterArea_Position: \n", self.DisasterArea_Position)

        # Generate plotting data if required
        if self.Do_you_need_point_plot == 1:
            self.Plot_Positions()

        self.Distance_D_A = self.Calculate_Distances(self.DisasterArea_Position, self.ACF_Position)
        self.Distance_A_H = self.Calculate_Distances(self.ACF_Position, self.Hospital_Position)
        self.Distance_D_H = self.Calculate_Distances(self.DisasterArea_Position, self.Hospital_Position)
        self.Distance_A_A = self.Calculate_Distances_Within_Same(self.ACF_Position)
        self.Distance_H_H = self.Calculate_Distances_Within_Same(self.Hospital_Position)
        # Combine hospital and ACF positions into a single list
        MedFacility_Position = self.Hospital_Position + self.ACF_Position        
        self.Distance_U_U = self.Calculate_Distances_Within_Same(MedFacility_Position)

        # Calculate travel times based on land speed
        self.Time_D_A_Land = self.Calculate_Travel_Time_Land(self.Distance_D_A, self.Speed_Land)
        self.Time_A_H_Land = self.Calculate_Travel_Time_Land(self.Distance_A_H, self.Speed_Land)
        self.Time_D_H_Land = self.Calculate_Travel_Time_Land(self.Distance_D_H, self.Speed_Land)
        self.Time_A_A_Land = self.Calculate_Travel_Time_Land(self.Distance_A_A, self.Speed_Land)
        self.Time_H_H_Land = self.Calculate_Travel_Time_Land(self.Distance_H_H, self.Speed_Land)
        self.Time_U_U_Land = self.Calculate_Travel_Time_Land(self.Distance_U_U, self.Speed_Land)

        # Calculate travel times based on aerial speed
        self.Time_D_A_Aerial = self.Calculate_Travel_Time_Land(self.Distance_D_A, self.Speed_Aerial)
        self.Time_A_H_Aerial = self.Calculate_Travel_Time_Land(self.Distance_A_H, self.Speed_Aerial)
        self.Time_D_H_Aerial = self.Calculate_Travel_Time_Land(self.Distance_D_H, self.Speed_Aerial)
        self.Time_A_A_Aerial = self.Calculate_Travel_Time_Land(self.Distance_A_A, self.Speed_Aerial)
        self.Time_H_H_Aerial = self.Calculate_Travel_Time_Land(self.Distance_H_H, self.Speed_Aerial)

        ##################################  Calculating Casualty_Shortage_Cost
        self.Casualty_Shortage_Cost = np.zeros((len(self.InjuryLevelSet)))
        
        for j in self.InjuryLevelSet:
            New_Casualty_Shortage_Cost = random.randint(self.Min_Casualty_Shortage_Cost, self.Max_Casualty_Shortage_Cost)
            if j == 0:
                self.Casualty_Shortage_Cost[j] = New_Casualty_Shortage_Cost
            if j == 1:
                self.Casualty_Shortage_Cost[j] = New_Casualty_Shortage_Cost / 2

        ##################################  Calculating Number of Land Rescue Vehicles that can be assigned to each ACF
        self.Number_Rescue_Vehicle_ACF = self.allocate_rescue_vehicles()

        ##################################  Calculating Number_Land_Rescue_Vehicle_Hospital
        self.Number_Land_Rescue_Vehicle_Hospital = self.allocate_hospital_rescue_vehicles()

        ################################## Average Hospital Disruption
        for h in self.HospitalSet:
            self.ForecastedAvgHospitalDisruption.append(0.45)       ## REF: (Using Collapse Risk Assessments to Inform Seismic Safety Policy for Older Concrete Buildings)     
        self.ForecastedAvgHospitalDisruption = np.array(self.ForecastedAvgHospitalDisruption)

        ################################## STD Hospital Disruption
        for h in self.HospitalSet:
            self.ForecastedSTDHospitalDisruption.append(0.07)      ## REF: (Using Collapse Risk Assessments to Inform Seismic Safety Policy for Older Concrete Buildings)            
        self.ForecastedSTDHospitalDisruption = np.array(self.ForecastedSTDHospitalDisruption)

        ##################################  Calculating Average Hospital Patient
        self.ForecastedAvgPatientDemand = np.zeros((len(self.InjuryLevelSet), len(self.HospitalSet)))
        for j in self.InjuryLevelSet:
            for h in self.HospitalSet:
                Min_ForecastedAvgPatientDemand = round(self.MinHospitalOccupationRate * self.Priority_Patient_Percent[j] * self.Hospital_Bed_Capacity[h])
                Max_ForecastedAvgPatientDemand = round(self.MaxHospitalOccupationRate * self.Priority_Patient_Percent[j] * self.Hospital_Bed_Capacity[h])                
                total_patient = round((Max_ForecastedAvgPatientDemand + Min_ForecastedAvgPatientDemand) / 2)
                self.ForecastedAvgPatientDemand[j][h] = total_patient   

        ##################################  Calculating STD Hospital Patient
        self.ForecastedSTDPatientDemand = np.zeros((len(self.InjuryLevelSet), len(self.HospitalSet)))
        for j in self.InjuryLevelSet:
            for h in self.HospitalSet:
                Min_ForecastedSTDPatientDemand = round(self.MinHospitalOccupationRate * self.Priority_Patient_Percent[j] * self.Hospital_Bed_Capacity[h])
                Max_ForecastedSTDPatientDemand = round(self.MaxHospitalOccupationRate * self.Priority_Patient_Percent[j] * self.Hospital_Bed_Capacity[h])                
                total_patient = round((Max_ForecastedSTDPatientDemand - Min_ForecastedSTDPatientDemand) / 2)
                self.ForecastedSTDPatientDemand[j][h] = total_patient

        ##################################  Calculating Average and STD of Patient Discharged Percentage
        ForecastedAvgPercentagePatientDischarged_ujt, ForecastedSTDPercentagePatientDischarged_ujt = self.generate_patient_discharge_probabilities(distribution_type = "lognorm")
        # Convert lists to numpy arrays
        ForecastedAvgPercentagePatientDischarged_ujt = np.array(ForecastedAvgPercentagePatientDischarged_ujt)
        ForecastedSTDPercentagePatientDischarged_ujt = np.array(ForecastedSTDPercentagePatientDischarged_ujt)
        # Reshape from [u][j][t] to [t][j][u]
        self.ForecastedAvgPercentagePatientDischarged = np.transpose(ForecastedAvgPercentagePatientDischarged_ujt, (2, 1, 0))
        self.ForecastedSTDPercentagePatientDischarged = np.transpose(ForecastedSTDPercentagePatientDischarged_ujt, (2, 1, 0))

        ################################## Maximum Backup Hospital
        for h in self.HospitalSet:
            self.Max_Backup_Hospital.append((self.NrHospitals - 1)) 

        ##################################  Calculating Available Aerial Vehicles
        self.Available_Aerial_Vehicles_Hospital = np.zeros((len(self.HospitalSet)))
        
        for h in self.HospitalSet:
            New_Available_Aerial_Vehicles_Hospital = random.randint(self.MinAerial_ToBeAssigned, self.MaxAerial_ToBeAssigned)
            self.Available_Aerial_Vehicles_Hospital[h] = New_Available_Aerial_Vehicles_Hospital
        self.Available_Aerial_Vehicles_Hospital = np.array(self.Available_Aerial_Vehicles_Hospital)
        ##################################  Calculating Average Hospital Patient
        self.CoordinationCost = np.zeros((len(self.HospitalSet), len(self.HospitalSet)))
        for h in self.HospitalSet:
            for hPrime in self.HospitalSet:
                if h == hPrime:
                    New_CoordinationCost = 1000
                else:
                    New_CoordinationCost = random.randint(self.MinCoordinationCost, self.MaxCoordinationCost)
                self.CoordinationCost[h][hPrime] = New_CoordinationCost   
        
        
        ################################## From This point, I will generate RISK Parameters !!!!!!!!!!!!!!!!!!!!!
        ## REF: (Decision support for hospital evacuation and emergency response) 
        ## In this paper MS-II is our High-Priority Patients and MS-I is Low-Priority ones.

        # Risk Cost
        self.EvacuationRiskCost = np.zeros((len(self.InjuryLevelSet)))
        for j in self.InjuryLevelSet:
            if j==0: #High-Priority Evacuation risk cost
                self.EvacuationRiskCost[j] = self.HighPriority_EvacuationRiskCost
            elif j==1:
                self.EvacuationRiskCost[j] = self.HighPriority_EvacuationRiskCost / 2

        # Constant risk model
        ThreatRiskConstant = self.generate_threat_risk("constant")
        # Linear risk model
        ThreatRiskLinear = self.generate_threat_risk("linear")
        # Exponential risk model
        ThreatRiskExponential = self.generate_threat_risk("exponential")
        # Compute cumulative threat risk for each threat risk model
        self.CumulativeThreatRiskConstant = self.compute_cumulative_threat_risk(ThreatRiskConstant)
        self.CumulativeThreatRiskLinear = self.compute_cumulative_threat_risk(ThreatRiskLinear)
        self.CumulativeThreatRiskExponential = self.compute_cumulative_threat_risk(ThreatRiskExponential)

        # Initialize Land_Loading_Time as a list with the same length as the number of rescue vehicle modes
        self.Land_Loading_Time = [0] * len(self.RescueVehicleSet)

        # Assign loading times (in minutes) for each vehicle type
        self.Land_Loading_Time[0] = 10  # Advanced life support ambulances
        self.Land_Loading_Time[1] = 10  # Basic life support ambulances
        self.Land_Loading_Time[2] = 20  # Ambus (large ambulance/bus hybrid)

        # Define transportation risk parameters (β_jm)
        self.Land_Transportation_Risk = np.zeros((len(self.InjuryLevelSet), len(self.RescueVehicleSet)))

        # Assign risks for land vehicles
        self.Land_Transportation_Risk[0][0] = 0.0001  # High-priority, Advanced Ambulance
        self.Land_Transportation_Risk[0][1] = 0.0002  # High-priority, Basic Ambulance
        self.Land_Transportation_Risk[0][2] = 0.0005  # High-priority, Ambus
        self.Land_Transportation_Risk[1][0] = 0.00005  # Low-priority, Advanced Ambulance
        self.Land_Transportation_Risk[1][1] = 0.00005  # Low-priority, Basic Ambulance
        self.Land_Transportation_Risk[1][2] = 0.0001  # Low-priority, Ambus

        self.CumulativeLandTransportation_Risk = self.compute_cumulative_land_transportation_risk(self.Land_Transportation_Risk)

        # Assign risks for aerial transport (Helicopters)
        self.Aerial_Transportation_Risk = np.zeros(len(self.InjuryLevelSet))
        self.Aerial_Transportation_Risk[0] = 0.0002  # High-priority helicopter transport risk
        self.Aerial_Transportation_Risk[1] = 0.00008  # Low-priority helicopter transport risk

        self.Aerial_Loading_Time = 20

        self.CumulativeAerialTransportRisk = self.compute_cumulative_aerial_transportation_risk()
        
        self.LandEvacuationRisk_Constant = self.compute_land_evacuation_risk(self.CumulativeThreatRiskConstant)
        self.LandEvacuationRisk_Linear = self.compute_land_evacuation_risk(self.CumulativeThreatRiskLinear)
        self.LandEvacuationRisk_Exponential = self.compute_land_evacuation_risk(self.CumulativeThreatRiskExponential)

        self.AerialEvacuationRisk_Constant = self.compute_aerial_evacuation_risk(self.CumulativeThreatRiskConstant)
        self.AerialEvacuationRisk_Linear = self.compute_aerial_evacuation_risk(self.CumulativeThreatRiskLinear)
        self.AerialEvacuationRisk_Exponential = self.compute_aerial_evacuation_risk(self.CumulativeThreatRiskExponential)

    def Generate_Data_CaseStudy(self, seed=None):
        if Constants.Debug: print("\n We are in 'Instance' Class -- Generate_Data_CaseStudy")
        excel_file_path = r'C:\PhD\Thesis\Papers\3rd\Code\Case Data\Van\Van_Data.xlsx'  # The path to your Excel file

        if seed is not None:
            random.seed(seed)

        ################################## Generate ACF Bed Capacities
        sheet_name = 'TMCs_31'  # The sheet containing the ACF data
        
        # Read the Excel file
        df_acfs = pd.read_excel(excel_file_path, sheet_name=sheet_name)

        # Read the Excel file
        df_acfs = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)  # Set header=None if the file has no headers

        # If your file doesn't have column headers, reference columns by index:
        acf_numbers = df_acfs.iloc[1:32, 0].values  # ACF numbers (from column A, index 0)
        acf_capacities = df_acfs.iloc[1:32, 3].values  # ACF capacities (from column D, index 3)

        # Assign capacities based on extracted data
        for i, capacity in enumerate(acf_capacities):
            self.ACF_Bed_Capacity.append(capacity)
        print("ACF_Bed_Capacity: ", self.ACF_Bed_Capacity)

        ################################## ACFs with Staging Area for Helicopter
        """Randomly assigns 50% of ACFs to the set I_A, which represents staging areas for helicopters."""
        num_selected = max(1, self.NrACFs // 2)  # Ensure at least one ACF is selected
        self.I_A_Set = set(random.sample(range(self.NrACFs), num_selected))  # Select random ACFs
        self.I_A_List = list(self.I_A_Set)  

        ################################## Generate Fixed Costs ACF based on ACF Bed Capacities for the Objective Function
        for i in self.ACFSet:
            New_Fixed_Cost_ACF_Objective = self.ACF_Bed_Capacity[i] * self.m2_Required_for_Each_Patient * self.Cost_of_Each_m2 * 0.00001
            New_Fixed_Cost_ACF_Objective = math.floor(1000 * New_Fixed_Cost_ACF_Objective) / 1000
            self.Fixed_Cost_ACF_Objective.append(New_Fixed_Cost_ACF_Objective)        

        ################################## Generate Fixed Costs ACF based on ACF Bed Capacities for the Constraint
        for i in self.ACFSet:
            New_Fixed_Cost_ACF_Constraint = self.ACF_Bed_Capacity[i] * self.m2_Required_for_Each_Patient * self.Cost_of_Each_m2
            New_Fixed_Cost_ACF_Constraint = math.floor(1000 * New_Fixed_Cost_ACF_Constraint) / 1000
            self.Fixed_Cost_ACF_Constraint.append(New_Fixed_Cost_ACF_Constraint)

        ################################## Generate Available Budget for ACFs
        Total_Required_Budget_for_ACF_Establishment = 0
        for i in self.ACFSet:
            Total_Required_Budget_for_ACF_Establishment += self.Fixed_Cost_ACF_Constraint[i]
        Total_Required_Budget_for_ACF_Establishment = ((Total_Required_Budget_for_ACF_Establishment * 17) / 20)
        Max_Required_Budget_for_ACF_Establishment = max(self.Fixed_Cost_ACF_Constraint)
        print("Total_Required_Budget_for_ACF_Establishment: ", Total_Required_Budget_for_ACF_Establishment)
        print("Max_Required_Budget_for_ACF_Establishment: ", Max_Required_Budget_for_ACF_Establishment)
        
        self.Total_Budget_ACF_Establishment = max(Total_Required_Budget_for_ACF_Establishment, Max_Required_Budget_for_ACF_Establishment)   # The total available budget to establish ACFs*/
        
        ################################## Generate VehicleAssignment_Cost       
        for m in self.RescueVehicleSet:  
            New_VehicleAssignment_Cost = random.uniform(self.Min_VehicleAssignment_Cost, self.Max_VehicleAssignment_Cost)
            New_VehicleAssignment_Cost = math.floor(1000 * New_VehicleAssignment_Cost) / 1000
            self.VehicleAssignment_Cost.append(New_VehicleAssignment_Cost)

        ##################################  Calculating Forecasted Average Casualty Demand
        if self.NrDisasterAreas == 94:
            sheet_name = 'Demands_94'

            # Read the total average demands from the Excel file (assuming demands are in column D, cells D2:D95)
            df_demand = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            total_average_demands = df_demand.iloc[1:95, 3].values  # Reading demands from D2 to D95 (0-indexed)

            # Initialize ForecastedAvgCasualtyDemand array
            self.ForecastedAvgCasualtyDemand = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.DisasterAreaSet)))

            # Define a demand distribution factor to assign more demands to the earlier periods
            # Example: 50% of total demand is allocated to the first half of the periods, and 50% to the second half
            period_distribution_factors = np.linspace(0.7, 0.3, len(self.TimeBucketSet))  # This linearly decreases demand across periods
            period_distribution_factors /= period_distribution_factors.sum()  # Normalize to ensure the sum is 1.0

            # Iterate over each demand location
            for l, total_location_demand in enumerate(total_average_demands):
                
                # Normalize the total demand per location to be distributed across periods, injury types, and blood groups
                total_demand_allocated = 0
                
                # Distribute demand across time periods (more demand in earlier periods)
                for t, period_factor in enumerate(period_distribution_factors):
                    
                    # Calculate demand for the current time period
                    period_demand = total_location_demand * period_factor
                    
                    # Distribute this period's demand across injury levels
                    for j in self.InjuryLevelSet:
                        demand_for_injury_level = period_demand * self.Priority_Patient_Percent[j]
                                                    
                        # Assign the calculated demand to the forecasted demand matrix and ensure it's rounded up
                        self.ForecastedAvgCasualtyDemand[t][j][l] = round(demand_for_injury_level)
                        total_demand_allocated += self.ForecastedAvgCasualtyDemand[t][j][l]

                # Ensure the total demand allocated is equal to the total location demand from the Excel file
                scaling_factor = total_location_demand / total_demand_allocated if total_demand_allocated > 0 else 0
                
                # Scale the demands to match the total demand from the Excel file and ensure integer values
                for t in self.TimeBucketSet:
                    for j in self.InjuryLevelSet:
                        self.ForecastedAvgCasualtyDemand[t][j][l] = math.ceil(self.ForecastedAvgCasualtyDemand[t][j][l] * scaling_factor)
        
        elif self.NrDisasterAreas == 60:
            sheet_name = 'Demands_60'

            # Read the total average demands from the Excel file (assuming demands are in column D, cells D2:D95)
            df_demand = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            total_average_demands = df_demand.iloc[1:61, 4].values  # Reading demands from D2 to D95 (0-indexed)

            # Initialize ForecastedAvgCasualtyDemand array
            self.ForecastedAvgCasualtyDemand = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.DisasterAreaSet)))

            # Define a demand distribution factor to assign more demands to the earlier periods
            # Example: 50% of total demand is allocated to the first half of the periods, and 50% to the second half
            period_distribution_factors = np.linspace(0.7, 0.3, len(self.TimeBucketSet))  # This linearly decreases demand across periods
            period_distribution_factors /= period_distribution_factors.sum()  # Normalize to ensure the sum is 1.0

            # Iterate over each demand location
            for l, total_location_demand in enumerate(total_average_demands):
                
                # Normalize the total demand per location to be distributed across periods, injury types, and blood groups
                total_demand_allocated = 0
                
                # Distribute demand across time periods (more demand in earlier periods)
                for t, period_factor in enumerate(period_distribution_factors):
                    
                    # Calculate demand for the current time period
                    period_demand = total_location_demand * period_factor
                    
                    # Distribute this period's demand across injury levels
                    for j in self.InjuryLevelSet:
                        demand_for_injury_level = period_demand * self.Priority_Patient_Percent[j]
                                                    
                        # Assign the calculated demand to the forecasted demand matrix and ensure it's rounded up
                        self.ForecastedAvgCasualtyDemand[t][j][l] = round(demand_for_injury_level)
                        total_demand_allocated += self.ForecastedAvgCasualtyDemand[t][j][l]

                # Ensure the total demand allocated is equal to the total location demand from the Excel file
                scaling_factor = total_location_demand / total_demand_allocated if total_demand_allocated > 0 else 0
                
                # Scale the demands to match the total demand from the Excel file and ensure integer values
                for t in self.TimeBucketSet:
                    for j in self.InjuryLevelSet:
                        self.ForecastedAvgCasualtyDemand[t][j][l] = math.ceil(self.ForecastedAvgCasualtyDemand[t][j][l] * scaling_factor)
                
        elif self.NrDisasterAreas == 30:
            sheet_name = 'Demands_30'

            # Read the total average demands from the Excel file (assuming demands are in column D, cells D2:D95)
            df_demand = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            total_average_demands = df_demand.iloc[1:31, 3].values  # Reading demands from D2 to D95 (0-indexed)

            # Initialize ForecastedAvgCasualtyDemand array
            self.ForecastedAvgCasualtyDemand = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.DisasterAreaSet)))

            # Define a demand distribution factor to assign more demands to the earlier periods
            # Example: 50% of total demand is allocated to the first half of the periods, and 50% to the second half
            period_distribution_factors = np.linspace(0.7, 0.3, len(self.TimeBucketSet))  # This linearly decreases demand across periods
            period_distribution_factors /= period_distribution_factors.sum()  # Normalize to ensure the sum is 1.0

            # Iterate over each demand location
            for l, total_location_demand in enumerate(total_average_demands):
                
                # Normalize the total demand per location to be distributed across periods, injury types, and blood groups
                total_demand_allocated = 0
                
                # Distribute demand across time periods (more demand in earlier periods)
                for t, period_factor in enumerate(period_distribution_factors):
                    
                    # Calculate demand for the current time period
                    period_demand = total_location_demand * period_factor
                    
                    # Distribute this period's demand across injury levels
                    for j in self.InjuryLevelSet:
                        demand_for_injury_level = period_demand * self.Priority_Patient_Percent[j]
                                                    
                        # Assign the calculated demand to the forecasted demand matrix and ensure it's rounded up
                        self.ForecastedAvgCasualtyDemand[t][j][l] = round(demand_for_injury_level)
                        total_demand_allocated += self.ForecastedAvgCasualtyDemand[t][j][l]

                # Ensure the total demand allocated is equal to the total location demand from the Excel file
                scaling_factor = total_location_demand / total_demand_allocated if total_demand_allocated > 0 else 0
                
                # Scale the demands to match the total demand from the Excel file and ensure integer values
                for t in self.TimeBucketSet:
                    for j in self.InjuryLevelSet:
                        self.ForecastedAvgCasualtyDemand[t][j][l] = math.ceil(self.ForecastedAvgCasualtyDemand[t][j][l] * scaling_factor)
        
        # Sum all the elements in the ForecastedAverageDemand array to get the total demand
        total_demand = np.sum(self.ForecastedAvgCasualtyDemand) 
        # Print the total demand and the demand array

        ##################################  Calculating Forecasted Average Standard Deviation Casualty Demand
        if self.NrDisasterAreas == 94:
            sheet_name = 'Demands_94'

            # Read the Max and Min demands from the Excel file (assuming Max in F2:F95 and Min in G2:G95)
            df_demand = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            max_demands = df_demand.iloc[1:95, 6].values  # Reading max demands from F2 to F95 (0-indexed)
            min_demands = df_demand.iloc[1:95, 5].values  # Reading min demands from G2 to G95 (0-indexed)

            # Initialize ForecastedSTDCasualtyDemand array
            self.ForecastedSTDCasualtyDemand = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.DisasterAreaSet)))

            # Define a demand distribution factor for standard deviation, same as we did for average demand
            period_distribution_factors = np.linspace(0.7, 0.3, len(self.TimeBucketSet))  # This linearly decreases demand across periods
            period_distribution_factors /= period_distribution_factors.sum()  # Normalize to ensure the sum is 1.0

            # Iterate over each demand location
            for l, (max_demand, min_demand) in enumerate(zip(max_demands, min_demands)):
                
                # Calculate the total standard deviation demand for each location
                total_STD_demand = (max_demand - min_demand) / 2  # This gives the STD for each location

                total_std_demand_allocated = 0  # To track the sum of allocated demand
                
                # Distribute standard deviation across time periods (more STD in earlier periods)
                for t, period_factor in enumerate(period_distribution_factors):
                    
                    # Calculate the STD for the current time period
                    period_std_demand = total_STD_demand * period_factor
                    
                    # Distribute this period's STD across injury levels
                    for j in self.InjuryLevelSet:
                        std_demand_for_injury_level = period_std_demand * self.Priority_Patient_Percent[j]
                                                    
                        # Assign the calculated STD demand to the forecasted STD demand matrix and round
                        self.ForecastedSTDCasualtyDemand[t][j][l] = round(std_demand_for_injury_level)
                        total_std_demand_allocated += self.ForecastedSTDCasualtyDemand[t][j][l]

                # Ensure the total STD demand allocated is equal to the calculated total_STD_demand
                scaling_factor = total_STD_demand / total_std_demand_allocated if total_std_demand_allocated > 0 else 0
                
                # Scale the STD demands to match the total demand and ensure integer values
                for t in self.TimeBucketSet:
                    for j in self.InjuryLevelSet:
                        self.ForecastedSTDCasualtyDemand[t][j][l] = math.ceil(self.ForecastedSTDCasualtyDemand[t][j][l] * scaling_factor)
        
        if self.NrDisasterAreas == 60:
            sheet_name = 'Demands_60'

            # Read the Max and Min demands from the Excel file (assuming Max in F2:F95 and Min in G2:G95)
            df_demand = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            max_demands = df_demand.iloc[1:61, 7].values  # Reading max demands from F2 to F95 (0-indexed)
            min_demands = df_demand.iloc[1:61, 6].values  # Reading min demands from G2 to G95 (0-indexed)

            # Initialize ForecastedSTDCasualtyDemand array
            self.ForecastedSTDCasualtyDemand = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.DisasterAreaSet)))

            # Define a demand distribution factor for standard deviation, same as we did for average demand
            period_distribution_factors = np.linspace(0.7, 0.3, len(self.TimeBucketSet))  # This linearly decreases demand across periods
            period_distribution_factors /= period_distribution_factors.sum()  # Normalize to ensure the sum is 1.0

            # Iterate over each demand location
            for l, (max_demand, min_demand) in enumerate(zip(max_demands, min_demands)):
                
                # Calculate the total standard deviation demand for each location
                total_STD_demand = (max_demand - min_demand) / 2  # This gives the STD for each location

                total_std_demand_allocated = 0  # To track the sum of allocated demand
                
                # Distribute standard deviation across time periods (more STD in earlier periods)
                for t, period_factor in enumerate(period_distribution_factors):
                    
                    # Calculate the STD for the current time period
                    period_std_demand = total_STD_demand * period_factor
                    
                    # Distribute this period's STD across injury levels
                    for j in self.InjuryLevelSet:
                        std_demand_for_injury_level = period_std_demand * self.Priority_Patient_Percent[j]
                                                    
                        # Assign the calculated STD demand to the forecasted STD demand matrix and round
                        self.ForecastedSTDCasualtyDemand[t][j][l] = round(std_demand_for_injury_level)
                        total_std_demand_allocated += self.ForecastedSTDCasualtyDemand[t][j][l]

                # Ensure the total STD demand allocated is equal to the calculated total_STD_demand
                scaling_factor = total_STD_demand / total_std_demand_allocated if total_std_demand_allocated > 0 else 0
                
                # Scale the STD demands to match the total demand and ensure integer values
                for t in self.TimeBucketSet:
                    for j in self.InjuryLevelSet:
                        self.ForecastedSTDCasualtyDemand[t][j][l] = math.ceil(self.ForecastedSTDCasualtyDemand[t][j][l] * scaling_factor)
        
        if self.NrDisasterAreas == 30:
            sheet_name = 'Demands_30'

            # Read the Max and Min demands from the Excel file (assuming Max in F2:F95 and Min in G2:G95)
            df_demand = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
            max_demands = df_demand.iloc[1:31, 6].values  # Reading max demands from F2 to F95 (0-indexed)
            min_demands = df_demand.iloc[1:31, 5].values  # Reading min demands from G2 to G95 (0-indexed)

            # Initialize ForecastedSTDCasualtyDemand array
            self.ForecastedSTDCasualtyDemand = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.DisasterAreaSet)))

            # Define a demand distribution factor for standard deviation, same as we did for average demand
            period_distribution_factors = np.linspace(0.7, 0.3, len(self.TimeBucketSet))  # This linearly decreases demand across periods
            period_distribution_factors /= period_distribution_factors.sum()  # Normalize to ensure the sum is 1.0

            # Iterate over each demand location
            for l, (max_demand, min_demand) in enumerate(zip(max_demands, min_demands)):
                
                # Calculate the total standard deviation demand for each location
                total_STD_demand = (max_demand - min_demand) / 2  # This gives the STD for each location

                total_std_demand_allocated = 0  # To track the sum of allocated demand
                
                # Distribute standard deviation across time periods (more STD in earlier periods)
                for t, period_factor in enumerate(period_distribution_factors):
                    
                    # Calculate the STD for the current time period
                    period_std_demand = total_STD_demand * period_factor
                    
                    # Distribute this period's STD across injury levels
                    for j in self.InjuryLevelSet:
                        std_demand_for_injury_level = period_std_demand * self.Priority_Patient_Percent[j]
                                                    
                        # Assign the calculated STD demand to the forecasted STD demand matrix and round
                        self.ForecastedSTDCasualtyDemand[t][j][l] = round(std_demand_for_injury_level)
                        total_std_demand_allocated += self.ForecastedSTDCasualtyDemand[t][j][l]

                # Ensure the total STD demand allocated is equal to the calculated total_STD_demand
                scaling_factor = total_STD_demand / total_std_demand_allocated if total_std_demand_allocated > 0 else 0
                
                # Scale the STD demands to match the total demand and ensure integer values
                for t in self.TimeBucketSet:
                    for j in self.InjuryLevelSet:
                        self.ForecastedSTDCasualtyDemand[t][j][l] = math.ceil(self.ForecastedSTDCasualtyDemand[t][j][l] * scaling_factor)
        
        # Sum all the elements in the ForecastedStandardDeviationDemand array to get the total STD demand
        total_std_demand = np.sum(self.ForecastedSTDCasualtyDemand)

        # Ensure STD demand is not greater than average demand
        for t in self.TimeBucketSet:
            for j in self.InjuryLevelSet:
                for l in self.DisasterAreaSet:
                    if self.ForecastedSTDCasualtyDemand[t][j][l] > self.ForecastedAvgCasualtyDemand[t][j][l]:
                        self.ForecastedSTDCasualtyDemand[t][j][l] = self.ForecastedAvgCasualtyDemand[t][j][l]

        ##################################  Calculating Hospital Bed Capacity
        sheet_name = 'Hospitals'  # The sheet containing hospital capacities
        # Read hospital capacities from the Excel file (cells D5 to D8)
        df_hospitals = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
        hospital_capacities = df_hospitals.iloc[4:8, 3].values  # Reading capacities from D5 to D8 (0-indexed)
        
        # Initialize Hospital_Bed_Capacity array
        self.Hospital_Bed_Capacity = np.zeros(len(self.HospitalSet))
        
        for h in self.HospitalSet:
            # Assign hospital capacities from the Excel sheet
            self.Hospital_Bed_Capacity[h] = hospital_capacities[h]

        ##################################  Calculating Land Rescue Vehicle Capacity

        a = (self.Square_Dimension / 2)
        nominal_Land_Rescue_Vehicle_Capacity = math.floor(0.5 * (self.Speed_Land / a) * self.Working_Hours_per_Day * self.Number_of_Planning_Days)    # Approximate number of patients transferred in the planning Horizon (Ex: 1 week)
        nominal_Land_Rescue_Vehicle_Capacity_Ambus = math.floor(0.5 * (self.Speed_Land_AmBus / a) * self.Working_Hours_per_Day * self.Number_of_Planning_Days)    # Approximate number of patients transferred in the planning Horizon (Ex: 1 week)
        
        for m in self.RescueVehicleSet:  # Assuming Number_Vehicle_Mode means there are four modes (0 to 3)
            new_Land_Rescue_Vehicle_Capacity = 0
            if m == 0:
                new_Land_Rescue_Vehicle_Capacity = 1 * nominal_Land_Rescue_Vehicle_Capacity
            elif m == 1:
                new_Land_Rescue_Vehicle_Capacity = 2 * nominal_Land_Rescue_Vehicle_Capacity
            elif m == 2:
                new_Land_Rescue_Vehicle_Capacity = 20 * nominal_Land_Rescue_Vehicle_Capacity_Ambus      ## REf: (Decision support for hospital evacuation and emergency response) and (https://txemtf.org/avada_portfolio/ambus/)
            elif m >= 3:
                print("\nThe number of Vehicles (index m) cannot be more than 3!!!!!!!!!\n")
                input("Press Enter to continue...")  # system("Pause") equivalent in Python
            self.Land_Rescue_Vehicle_Capacity.append(new_Land_Rescue_Vehicle_Capacity)
        
        ##################################  Calculating Aerial Rescue Vehicle Capacity

        a = (self.Square_Dimension / 2)
        nominal_Aerial_Rescue_Vehicle_Capacity = math.floor(0.5 * (self.Speed_Aerial / a) * self.Working_Hours_per_Day * self.Number_of_Planning_Days)    # Approximate number of patients transferred in the planning Horizon (Ex: 1 week)
        new_Aerial_Rescue_Vehicle_Capacity = 10 * nominal_Aerial_Rescue_Vehicle_Capacity        ## Ref: (A bi-objective robust optimization model for disaster response planning under uncertainties)
        self.Aerial_Rescue_Vehicle_Capacity.append(new_Aerial_Rescue_Vehicle_Capacity)

        ##################################  Calculating Distances
        if self.NrDisasterAreas == 94:
            df_demands = pd.read_excel(excel_file_path, sheet_name='Demands_94', header=None)
            demand_latitudes = df_demands.iloc[1:95, 7].values 
            demand_longitudes = df_demands.iloc[1:95, 8].values
            self.DisasterArea_Position = list(zip(demand_latitudes, demand_longitudes))
        elif self.NrDisasterAreas == 60:
            df_demands = pd.read_excel(excel_file_path, sheet_name='Demands_60', header=None)
            demand_latitudes = df_demands.iloc[1:61, 8].values  
            demand_longitudes = df_demands.iloc[1:61, 9].values  
            self.DisasterArea_Position = list(zip(demand_latitudes, demand_longitudes))
        elif self.NrDisasterAreas == 30:
            df_demands = pd.read_excel(excel_file_path, sheet_name='Demands_30', header=None)
            demand_latitudes = df_demands.iloc[1:31, 7].values  
            demand_longitudes = df_demands.iloc[1:31, 8].values  
            self.DisasterArea_Position = list(zip(demand_latitudes, demand_longitudes))  

        # Read latitude and longitude for hospitals
        df_hospitals = pd.read_excel(excel_file_path, sheet_name='Hospitals', header=None)
        hospital_latitudes = df_hospitals.iloc[4:8, 4].values  # Latitude: E5 to E8
        hospital_longitudes = df_hospitals.iloc[4:8, 5].values  # Longitude: F5 to F8
        self.Hospital_Position = list(zip(hospital_latitudes, hospital_longitudes))

        # Read latitude and longitude for ACFs
        df_acfs = pd.read_excel(excel_file_path, sheet_name='TMCs_31', header=None)
        acf_latitudes = df_acfs.iloc[1:32, 4].values  # Latitude: E2 to E32
        acf_longitudes = df_acfs.iloc[1:32, 5].values  # Longitude: F2 to F32
        self.ACF_Position = list(zip(acf_latitudes, acf_longitudes))

        # Generate plotting data if required
        if self.Do_you_need_point_plot == 1:
            self.Plot_Positions()

        self.Distance_D_A = self.calculate_distances_haversine(self.DisasterArea_Position, self.ACF_Position)
        self.Distance_A_H = self.calculate_distances_haversine(self.ACF_Position, self.Hospital_Position)
        self.Distance_D_H = self.calculate_distances_haversine(self.DisasterArea_Position, self.Hospital_Position)
        self.Distance_A_A = self.calculate_distances_within_same_haversine(self.ACF_Position)
        self.Distance_H_H = self.calculate_distances_within_same_haversine(self.Hospital_Position)
        # Combine hospital and ACF positions into a single list
        MedFacility_Position = self.Hospital_Position + self.ACF_Position        
        self.Distance_U_U = self.calculate_distances_within_same_haversine(MedFacility_Position)

        # Calculate travel times based on land speed
        self.Time_D_A_Land = self.Calculate_Travel_Time_Land(self.Distance_D_A, self.Speed_Land)
        self.Time_A_H_Land = self.Calculate_Travel_Time_Land(self.Distance_A_H, self.Speed_Land)
        self.Time_D_H_Land = self.Calculate_Travel_Time_Land(self.Distance_D_H, self.Speed_Land)
        self.Time_A_A_Land = self.Calculate_Travel_Time_Land(self.Distance_A_A, self.Speed_Land)
        self.Time_H_H_Land = self.Calculate_Travel_Time_Land(self.Distance_H_H, self.Speed_Land)
        self.Time_U_U_Land = self.Calculate_Travel_Time_Land(self.Distance_U_U, self.Speed_Land)

        # Calculate travel times based on aerial speed
        self.Time_D_A_Aerial = self.Calculate_Travel_Time_Land(self.Distance_D_A, self.Speed_Aerial)
        self.Time_A_H_Aerial = self.Calculate_Travel_Time_Land(self.Distance_A_H, self.Speed_Aerial)
        self.Time_D_H_Aerial = self.Calculate_Travel_Time_Land(self.Distance_D_H, self.Speed_Aerial)
        self.Time_A_A_Aerial = self.Calculate_Travel_Time_Land(self.Distance_A_A, self.Speed_Aerial)
        self.Time_H_H_Aerial = self.Calculate_Travel_Time_Land(self.Distance_H_H, self.Speed_Aerial)

        ##################################  Calculating Casualty_Shortage_Cost
        self.Casualty_Shortage_Cost = np.zeros((len(self.InjuryLevelSet)))
        
        for j in self.InjuryLevelSet:
            New_Casualty_Shortage_Cost = random.randint(self.Min_Casualty_Shortage_Cost, self.Max_Casualty_Shortage_Cost)
            if j == 0:
                self.Casualty_Shortage_Cost[j] = New_Casualty_Shortage_Cost
            if j == 1:
                self.Casualty_Shortage_Cost[j] = New_Casualty_Shortage_Cost / 2

        ##################################  Calculating Number of Land Rescue Vehicles that can be assigned to each ACF
        self.Number_Rescue_Vehicle_ACF = self.allocate_rescue_vehicles()

        ##################################  Calculating Number_Land_Rescue_Vehicle_Hospital
        self.Number_Land_Rescue_Vehicle_Hospital = self.allocate_hospital_rescue_vehicles()

        ################################## Average Hospital Disruption
        for h in self.HospitalSet:
            self.ForecastedAvgHospitalDisruption.append(0.45)       ## REF: (Using Collapse Risk Assessments to Inform Seismic Safety Policy for Older Concrete Buildings)     
        self.ForecastedAvgHospitalDisruption = np.array(self.ForecastedAvgHospitalDisruption)

        ################################## STD Hospital Disruption
        for h in self.HospitalSet:
            self.ForecastedSTDHospitalDisruption.append(0.07)      ## REF: (Using Collapse Risk Assessments to Inform Seismic Safety Policy for Older Concrete Buildings)            
        self.ForecastedSTDHospitalDisruption = np.array(self.ForecastedSTDHospitalDisruption)

        ##################################  Calculating Average Hospital Patient
        self.ForecastedAvgPatientDemand = np.zeros((len(self.InjuryLevelSet), len(self.HospitalSet)))
        for j in self.InjuryLevelSet:
            for h in self.HospitalSet:
                MinHospitalOccupationRate_Turkey = 0.50
                MaxHospitalOccupationRate_Turkey = 0.55
                Min_ForecastedAvgPatientDemand = round(MinHospitalOccupationRate_Turkey * self.Priority_Patient_Percent[j] * self.Hospital_Bed_Capacity[h])
                Max_ForecastedAvgPatientDemand = round(MaxHospitalOccupationRate_Turkey * self.Priority_Patient_Percent[j] * self.Hospital_Bed_Capacity[h])                
                total_patient = round((Max_ForecastedAvgPatientDemand + Min_ForecastedAvgPatientDemand) / 2)
                self.ForecastedAvgPatientDemand[j][h] = total_patient   

        ##################################  Calculating STD Hospital Patient
        self.ForecastedSTDPatientDemand = np.zeros((len(self.InjuryLevelSet), len(self.HospitalSet)))
        for j in self.InjuryLevelSet:
            for h in self.HospitalSet:
                MinHospitalOccupationRate_Turkey = 0.55
                MaxHospitalOccupationRate_Turkey = 0.75
                Min_ForecastedSTDPatientDemand = round(MinHospitalOccupationRate_Turkey * self.Priority_Patient_Percent[j] * self.Hospital_Bed_Capacity[h])
                Max_ForecastedSTDPatientDemand = round(MaxHospitalOccupationRate_Turkey * self.Priority_Patient_Percent[j] * self.Hospital_Bed_Capacity[h])                
                total_patient = round((Max_ForecastedSTDPatientDemand - Min_ForecastedSTDPatientDemand) / 2)
                self.ForecastedSTDPatientDemand[j][h] = total_patient

        ##################################  Calculating Average and STD of Patient Discharged Percentage
        ForecastedAvgPercentagePatientDischarged_ujt, ForecastedSTDPercentagePatientDischarged_ujt = self.generate_patient_discharge_probabilities(distribution_type = "lognorm")
        # Convert lists to numpy arrays
        ForecastedAvgPercentagePatientDischarged_ujt = np.array(ForecastedAvgPercentagePatientDischarged_ujt)
        ForecastedSTDPercentagePatientDischarged_ujt = np.array(ForecastedSTDPercentagePatientDischarged_ujt)
        # Reshape from [u][j][t] to [t][j][u]
        self.ForecastedAvgPercentagePatientDischarged = np.transpose(ForecastedAvgPercentagePatientDischarged_ujt, (2, 1, 0))
        self.ForecastedSTDPercentagePatientDischarged = np.transpose(ForecastedSTDPercentagePatientDischarged_ujt, (2, 1, 0))

        ################################## Maximum Backup Hospital
        for h in self.HospitalSet:
            self.Max_Backup_Hospital.append((self.NrHospitals - 1)) 

        ##################################  Calculating Available Aerial Vehicles
        self.Available_Aerial_Vehicles_Hospital = np.zeros((len(self.HospitalSet)))
        
        for h in self.HospitalSet:
            New_Available_Aerial_Vehicles_Hospital = random.randint(self.MinAerial_ToBeAssigned, self.MaxAerial_ToBeAssigned)
            self.Available_Aerial_Vehicles_Hospital[h] = New_Available_Aerial_Vehicles_Hospital
        self.Available_Aerial_Vehicles_Hospital = np.array(self.Available_Aerial_Vehicles_Hospital)
        ##################################  Calculating Average Hospital Patient
        self.CoordinationCost = np.zeros((len(self.HospitalSet), len(self.HospitalSet)))
        for h in self.HospitalSet:
            for hPrime in self.HospitalSet:
                if h == hPrime:
                    New_CoordinationCost = 1000
                else:
                    New_CoordinationCost = random.randint(self.MinCoordinationCost, self.MaxCoordinationCost)
                self.CoordinationCost[h][hPrime] = New_CoordinationCost   
        
        
        ################################## From This point, I will generate RISK Parameters !!!!!!!!!!!!!!!!!!!!!
        ## REF: (Decision support for hospital evacuation and emergency response) 
        ## In this paper MS-II is our High-Priority Patients and MS-I is Low-Priority ones.

        # Risk Cost
        self.EvacuationRiskCost = np.zeros((len(self.InjuryLevelSet)))
        for j in self.InjuryLevelSet:
            if j==0: #High-Priority Evacuation risk cost
                self.EvacuationRiskCost[j] = self.HighPriority_EvacuationRiskCost
            elif j==1:
                self.EvacuationRiskCost[j] = self.HighPriority_EvacuationRiskCost / 2

        # Constant risk model
        ThreatRiskConstant = self.generate_threat_risk("constant")
        # Linear risk model
        ThreatRiskLinear = self.generate_threat_risk("linear")
        # Exponential risk model
        ThreatRiskExponential = self.generate_threat_risk("exponential")
        # Compute cumulative threat risk for each threat risk model
        self.CumulativeThreatRiskConstant = self.compute_cumulative_threat_risk(ThreatRiskConstant)
        self.CumulativeThreatRiskLinear = self.compute_cumulative_threat_risk(ThreatRiskLinear)
        self.CumulativeThreatRiskExponential = self.compute_cumulative_threat_risk(ThreatRiskExponential)

        # Initialize Land_Loading_Time as a list with the same length as the number of rescue vehicle modes
        self.Land_Loading_Time = [0] * len(self.RescueVehicleSet)

        # Assign loading times (in minutes) for each vehicle type
        self.Land_Loading_Time[0] = 10  # Advanced life support ambulances
        self.Land_Loading_Time[1] = 10  # Basic life support ambulances
        self.Land_Loading_Time[2] = 20  # Ambus (large ambulance/bus hybrid)

        # Define transportation risk parameters (β_jm)
        self.Land_Transportation_Risk = np.zeros((len(self.InjuryLevelSet), len(self.RescueVehicleSet)))

        # Assign risks for land vehicles
        self.Land_Transportation_Risk[0][0] = 0.0001  # High-priority, Advanced Ambulance
        self.Land_Transportation_Risk[0][1] = 0.0002  # High-priority, Basic Ambulance
        self.Land_Transportation_Risk[0][2] = 0.0005  # High-priority, Ambus
        self.Land_Transportation_Risk[1][0] = 0.00005  # Low-priority, Advanced Ambulance
        self.Land_Transportation_Risk[1][1] = 0.00005  # Low-priority, Basic Ambulance
        self.Land_Transportation_Risk[1][2] = 0.0001  # Low-priority, Ambus

        self.CumulativeLandTransportation_Risk = self.compute_cumulative_land_transportation_risk(self.Land_Transportation_Risk)

        # Assign risks for aerial transport (Helicopters)
        self.Aerial_Transportation_Risk = np.zeros(len(self.InjuryLevelSet))
        self.Aerial_Transportation_Risk[0] = 0.0002  # High-priority helicopter transport risk
        self.Aerial_Transportation_Risk[1] = 0.00008  # Low-priority helicopter transport risk

        self.Aerial_Loading_Time = 20

        self.CumulativeAerialTransportRisk = self.compute_cumulative_aerial_transportation_risk()
        
        self.LandEvacuationRisk_Constant = self.compute_land_evacuation_risk(self.CumulativeThreatRiskConstant)
        self.LandEvacuationRisk_Linear = self.compute_land_evacuation_risk(self.CumulativeThreatRiskLinear)
        self.LandEvacuationRisk_Exponential = self.compute_land_evacuation_risk(self.CumulativeThreatRiskExponential)

        self.AerialEvacuationRisk_Constant = self.compute_aerial_evacuation_risk(self.CumulativeThreatRiskConstant)
        self.AerialEvacuationRisk_Linear = self.compute_aerial_evacuation_risk(self.CumulativeThreatRiskLinear)
        self.AerialEvacuationRisk_Exponential = self.compute_aerial_evacuation_risk(self.CumulativeThreatRiskExponential)

    def allocate_rescue_vehicles(self):
        """
        Improved version of estimating and allocating rescue vehicles to ACFs.
        Ensures that:
            Number_Rescue_Vehicle_ACF[0] >= Number_Rescue_Vehicle_ACF[1] >= Number_Rescue_Vehicle_ACF[2]
        """

        # Initialize array
        self.Number_Rescue_Vehicle_ACF = np.zeros(len(self.RescueVehicleSet))

        # Step 1: Compute total estimated demand per period
        Total_Demand_Per_Period = self.ForecastedAvgCasualtyDemand.sum(axis=(1, 2))
        print("Total_Demand_Per_Period: ", Total_Demand_Per_Period)

        # Step 2: Find the max demand across periods
        max_demand = Total_Demand_Per_Period.max()

        # Step 3: Compute total number of vehicles needed
        total_vehicles_needed = 0
        vehicle_requirements = []
        
        for m in self.RescueVehicleSet:
            required_for_m = math.ceil(max_demand / self.Land_Rescue_Vehicle_Capacity[m])
            vehicle_requirements.append((m, required_for_m))
            total_vehicles_needed += required_for_m

        # Adjust based on safety factor
        total_vehicles_needed = math.ceil(total_vehicles_needed * self.Safety_Factor_Rescue_Vehicle_ACF)

        # Step 4: Allocate vehicles ensuring `m=0` gets the most
        vehicle_requirements.sort(key=lambda x: x[1], reverse=True)  # Sort in descending order
        remaining_vehicles = total_vehicles_needed

        for m, _ in vehicle_requirements:
            if remaining_vehicles > 0:
                if m != self.NrRescueVehicles - 1:  # If not the last type
                    assigned = math.ceil(remaining_vehicles / (self.NrRescueVehicles - m))  # More to lower indices
                    self.Number_Rescue_Vehicle_ACF[m] = assigned
                    remaining_vehicles -= assigned
                else:  # Last type gets the remaining
                    self.Number_Rescue_Vehicle_ACF[m] = remaining_vehicles

        # Step 5: Ensure descending order explicitly
        self.Number_Rescue_Vehicle_ACF = np.sort(self.Number_Rescue_Vehicle_ACF)[::-1]  # Sort in descending order
        
        return self.Number_Rescue_Vehicle_ACF        

    def allocate_hospital_rescue_vehicles(self):
        """
        Improved version of estimating and allocating rescue vehicles to hospitals.
        Ensures that:
            - Number_Land_Rescue_Vehicle_Hospital[0][h] >= Number_Land_Rescue_Vehicle_Hospital[1][h] >= Number_Land_Rescue_Vehicle_Hospital[2][h]
            - Vehicles are fairly distributed across hospitals
        """

        # Initialize array
        self.Number_Land_Rescue_Vehicle_Hospital = np.zeros((len(self.RescueVehicleSet), len(self.HospitalSet)))

        # Step 1: Compute total required vehicles per hospital
        total_vehicles_per_hospital = []
        total_global_demand = 0

        for h in self.HospitalSet:
            total_required = 0
            for m in self.RescueVehicleSet:
                required_for_m = math.ceil(self.Hospital_Bed_Capacity[h] / self.Land_Rescue_Vehicle_Capacity[m])
                total_required += required_for_m

            # Apply safety factor
            total_required = math.ceil(total_required * self.Safety_Factor_Rescue_Vehicle_Hospital)
            total_vehicles_per_hospital.append(total_required)
            total_global_demand += total_required

        # Step 2: Distribute rescue vehicles proportionally across hospitals
        remaining_vehicles = total_global_demand
        for h in self.HospitalSet:
            if remaining_vehicles <= 0:
                break

            # Allocate proportionally based on each hospital’s demand
            allocated = min(total_vehicles_per_hospital[h], remaining_vehicles)
            remaining_vehicles -= allocated

            # Step 3: Distribute across vehicle types
            for m in self.RescueVehicleSet:
                portion = allocated * ((self.NrRescueVehicles - m) / sum(range(1, self.NrRescueVehicles + 1)))  # Higher priority for lower index
                self.Number_Land_Rescue_Vehicle_Hospital[m][h] = math.ceil(portion)

        # Step 4: Ensure descending order
        for h in self.HospitalSet:
            self.Number_Land_Rescue_Vehicle_Hospital[:, h] = np.sort(self.Number_Land_Rescue_Vehicle_Hospital[:, h])[::-1]

            return self.Number_Land_Rescue_Vehicle_Hospital
                
    def compute_aerial_evacuation_risk(self, CumulativeThreatRisk):
        """
        Computes R_(tjhih'm)^A: The aerial evacuation risk for a patient with injury type j
        from a non-functional hospital h to a backup functional hospital h'
        using ACF i as a connection hub and a land rescue vehicle type m in period t.

        :param CumulativeThreatRisk: A 3D numpy array containing Λ_tjh values (cumulative threat risk).
                                    Can be Constant, Linear, or Exponential.
        :return: A 6D numpy array indexed by [t][j][h][i][h'][m], containing R_(tjhih'm)^A values.
        """
        # Initialize the aerial evacuation risk matrix
        AerialEvacuationRisk = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.HospitalSet), len(self.ACFSet), len(self.HospitalSet), len(self.RescueVehicleSet)))

        # Iterate over all indices
        for t in self.TimeBucketSet:  # Time period
            for j in self.InjuryLevelSet:  # Injury type
                for h in self.HospitalSet:  # Non-functional hospital
                    for i in self.ACFSet:  # Intermediate ACF
                        for h_prime in self.HospitalSet:  # Backup functional hospital
                            for m in self.RescueVehicleSet:  # Land vehicle type
                                # Fetch values for calculations
                                Lambda_tjh = CumulativeThreatRisk[t][j][h]  # Chosen threat risk model
                                Theta_jhihm_A = self.CumulativeAerialTransportRisk[j][h][i][h_prime][m]  # Aerial transport risk

                                # Compute aerial evacuation risk
                                AerialEvacuationRisk[t][j][h][i][h_prime][m] = 1 - ((1 - Lambda_tjh) * (1 - Theta_jhihm_A))

        return AerialEvacuationRisk

    def compute_land_evacuation_risk(self, CumulativeThreatRisk):
        """
        Computes R_tjhum^L: The land evacuation risk associated with evacuating a patient with injury type j
        from a non-functional hospital h to a backup medical facility u via a land rescue vehicle type m in period t.

        :param CumulativeThreatRisk: A 3D numpy array containing Λ_tjh values (cumulative threat risk).
                                    Can be Constant, Linear, or Exponential.
        :return: A 5D numpy array indexed by [t][j][h][u][m], containing R_tjhum^L values.
        """
        # Initialize the land evacuation risk matrix
        LandEvacuationRisk = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.HospitalSet), len(self.MedFacilitySet), len(self.RescueVehicleSet)))

        # Iterate over all indices
        for t in self.TimeBucketSet:  # Time period
            for j in self.InjuryLevelSet:  # Injury type
                for h in self.HospitalSet:  # Non-functional hospital
                    for u in self.MedFacilitySet:  # Backup medical facility (h or ACF)
                        for m in self.RescueVehicleSet:  # Land vehicle type
                            # Fetch values for calculations
                            Lambda_tjh = CumulativeThreatRisk[t][j][h]  # Threat risk (chosen model)
                            Theta_jhum_L = self.CumulativeLandTransportation_Risk[j][h][u][m]  # Land transport risk

                            # Compute land evacuation risk
                            LandEvacuationRisk[t][j][h][u][m] = 1 - ((1 - Lambda_tjh) * (1 - Theta_jhum_L))

        return LandEvacuationRisk

    def compute_cumulative_aerial_transportation_risk(self):
        """
        Computes Θ_(jhih'm)^A: The cumulative aerial transportation risk for a patient with injury type j
        evacuated from the non-functional hospital h to the backup functional hospital h'
        through ACF i using land rescue vehicle type m and an aerial rescue vehicle.

        :return: A 5D numpy array indexed by [j][h][i][h'][m], containing Θ_(jhih'm)^A values.
        """
        # Initialize the cumulative aerial transport risk matrix
        CumulativeAerialTransportRisk = np.zeros((len(self.InjuryLevelSet), len(self.HospitalSet), len(self.ACFSet), len(self.HospitalSet), len(self.RescueVehicleSet)))

        # Iterate over all indices
        for j in self.InjuryLevelSet:  # Injury type
            for h in self.HospitalSet:  # Non-functional hospital
                for i in self.ACFSet:  # Intermediate ACF
                    for h_prime in self.HospitalSet:  # Backup functional hospital
                        for m in self.RescueVehicleSet:  # Land vehicle type
                            # Fetch values for calculations
                            beta_jm_L = self.Land_Transportation_Risk[j][m]  # Land transport risk
                            travel_time_L = self.Time_A_H_Land[i][h]  # Land travel time (h → i)
                            loading_time_L = self.Land_Loading_Time[m]  # Land loading time

                            beta_j_A = self.Aerial_Transportation_Risk[j]  # Aerial transport risk
                            travel_time_A = self.Time_A_H_Aerial[i][h_prime]  # Aerial travel time (i → h')
                            loading_time_A = self.Aerial_Loading_Time  # Aerial loading time

                            # Compute exponent terms
                            exponent_L = travel_time_L + 2 * loading_time_L
                            exponent_A = travel_time_A + 2 * loading_time_A

                            # Compute cumulative aerial transportation risk
                            CumulativeAerialTransportRisk[j][h][i][h_prime][m] = 1 - (((1 - beta_jm_L) ** exponent_L) * ((1 - beta_j_A) ** exponent_A))

        return CumulativeAerialTransportRisk

    def compute_cumulative_land_transportation_risk(self, Land_Transportation_Risk):
        """
        Computes \Theta_jhum^L: The cumulative land transportation risk for patients
        evacuated from a non-functional hospital h to a backup medical facility u.

        :param Land_Transportation_Risk: A 2D numpy array containing β_jm^L values (land transportation risk per injury type j and vehicle m).
        :return: A 4D numpy array indexed by [j][h][u][m], containing Θ_jhum^L values.
        """
        # Initialize the cumulative risk matrix
        CumulativeLandTransportRisk = np.zeros((len(self.InjuryLevelSet), len(self.HospitalSet), len(self.MedFacilitySet), len(self.RescueVehicleSet)))

        # Iterate over injury levels, hospitals, medical facilities, and vehicle types
        for j in self.InjuryLevelSet:  # Injury type
            for h in self.HospitalSet:  # Non-functional hospital
                for u in self.MedFacilitySet:  # Backup medical facility
                    for m in self.RescueVehicleSet:  # Vehicle type
                        # Fetch values for calculation
                        beta_jm = Land_Transportation_Risk[j][m]  # Transportation risk for injury j, vehicle m
                        travel_time = self.Time_U_U_Land[h][u]  # Time from hospital h to facility u
                        loading_time = self.Land_Loading_Time[m]  # Loading time for vehicle m

                        # Compute exponent term (travel + 2 * loading time)
                        exponent = travel_time + 2 * loading_time
                        
                        # Compute cumulative risk Θ_jhum^L
                        CumulativeLandTransportRisk[j][h][u][m] = 1 - ((1 - beta_jm) ** exponent)

        return CumulativeLandTransportRisk

    def compute_cumulative_threat_risk(self, ThreatRisk):
        """
        Computes the cumulative threat risk Λ_tjh for each t, j, h.

        :param ThreatRisk: A 3D numpy array containing α_tjh values (threat risk at each time t).
        :return: A 3D numpy array containing Λ_tjh values (cumulative threat risk at each time t).
        """
        # Initialize the cumulative threat risk array
        CumulativeThreatRisk = np.zeros_like(ThreatRisk)

        # Iterate over injury levels and hospitals
        for j in range(len(self.InjuryLevelSet)):
            for h in range(len(self.HospitalSet)):
                # Compute cumulative threat risk over time t
                product_term = 1.0  # Start with 1.0 for the product in the formula
                for t in range(len(self.TimeBucketSet)):
                    product_term *= (1 - min(ThreatRisk[t][j][h], 1.0))  # Cap per-period risk at 1.0
                    CumulativeThreatRisk[t][j][h] = min(1 - product_term, 1.0)  # Ensure cumulative risk doesn't exceed 1.0

        return CumulativeThreatRisk

    def generate_threat_risk(self, distribution_type="constant"):
        """
        Generates alpha_tjh: The threat risk parameter for each injury type j, hospital h, and time period t.

        :param distribution_type: str, choose from ["constant", "linear", "exponential"]
        :return: A 3D list of threat risk values indexed by [t][j][h]
        """
        # Compute scaling factor: Converts the base 10-min slots to the new time slot size
        scaling_factor = (self.Number_of_Planning_Days * 24 * 60) / 10  # Convert to 10-min units

        # Adjusted risk parameters based on new time scale
        risk_params = {
            0: {  # High-priority patients
                "constant": 0.0020,  # No change for constant risk
                "linear": 0.000025 * scaling_factor,  # Scale UP for larger time slots
                "a_exp": 0.0000625,  # Base value unchanged
                "b_exp": 32 / scaling_factor  # Scale DOWN to accelerate risk accumulation
            },
            1: {  # Low-priority patients
                "constant": 0.0012,  # No change for constant risk
                "linear": 0.000015 * scaling_factor,  # Scale UP for larger time slots
                "a_exp": 0.0000375,  # Base value unchanged
                "b_exp": 34 / scaling_factor  # Scale DOWN to accelerate risk accumulation
            }
        }

        # Initialize risk matrix
        ThreatRisk = np.zeros((len(self.TimeBucketSet), len(self.InjuryLevelSet), len(self.HospitalSet)))

        # Iterate over time, injury levels, and hospitals
        for t in self.TimeBucketSet:
            for j in self.InjuryLevelSet:
                for h in self.HospitalSet:
                    if distribution_type == "constant":
                        ThreatRisk[t][j][h] = min(risk_params[j]["constant"], 1.0)  # Ensure max probability is 1.0
                    elif distribution_type == "linear":
                        ThreatRisk[t][j][h] = min(risk_params[j]["linear"] * (t + 1), 1.0)  # Ensure max probability is 1.0
                    elif distribution_type == "exponential":
                        ThreatRisk[t][j][h] = min(risk_params[j]["a_exp"] * np.exp((t + 1) / risk_params[j]["b_exp"]), 1.0)  # Cap at 1.0
                    else:
                        raise ValueError("Invalid distribution type. Choose from ['constant', 'linear', 'exponential'].")

        return ThreatRisk

    def generate_patient_discharge_probabilities(self, mean_hours=108.89,  # Mean stay for high-priority patients
                                                distribution_type="custom"):
        """
        Generates the percentage of patients discharged at each period after admission
        for each injury type j and medical facility u, based on a fixed discharge pattern.

        :param mean_hours: Mean duration (in hours) a high-priority patient stays at the hospital.
        :param distribution_type: str, currently only supports "custom" distribution.
        :return: Two lists containing mean and std dev discharge percentages (rounded to 2 decimals).
        """

        # Define maximum number of periods (always generate up to 20 periods)
        max_periods = 20

        # Convert hours to time periods (t)
        mean_periods_high = round(mean_hours / (self.Number_of_Planning_Days * 24))
        mean_periods_high = mean_periods_high - 1 # Because in python, indexes start from 0!
        mean_periods_low = round(mean_periods_high / 2)  # Low-priority patients are discharged faster
        mean_periods_low = mean_periods_low - 1 # Because in python, indexes start from 0!

        print(f"Mean stay in periods (high-priority): {mean_periods_high}")
        print(f"Mean stay in periods (low-priority): {mean_periods_low}")

        # Initialize lists
        ForecastedAvgPercentagePatientDischarged_ujt = []
        ForecastedSTDPercentagePatientDischarged_ujt = []

        # Define different parameters for high-priority (j=0) and low-priority (j=1) patients
        discharge_params = {
            0: {"mean": mean_periods_high},  # High-priority
            1: {"mean": mean_periods_low}   # Low-priority
        }

        # Iterate over medical facilities
        for u in self.MedFacilitySet:
            facility_avg = []
            facility_std = []

            # Iterate over injury levels
            for j in self.InjuryLevelSet:
                mean_stay = discharge_params[j]["mean"]

                # Define a discharge pattern: 50% at mean_stay, 50% spread around 4 periods before/after
                discharge_probs = np.zeros(max_periods)  # Always create up to 20 periods

                # Define discharge distribution range
                start_period = max(1, mean_stay - 6)  # Ensure no negative index
                end_period = min(mean_stay + 6, max_periods - 1)  # Ensure within max limits
                
                discharge_probs[mean_stay] = 0.45  # 50% discharged at the mean period
                
                if end_period > start_period:  # Only distribute if there's space
                    remaining_prob = 1 - discharge_probs[mean_stay]  # Remaining 50% to distribute
                    num_periods = end_period - start_period
                    distributed_probs = np.full(num_periods, remaining_prob / num_periods)  # Equal distribution

                    discharge_probs[start_period:end_period] += distributed_probs
                    discharge_probs[mean_stay] -= distributed_probs[0] ## Because, here we addd percentage two times!
                # Normalize to ensure sum ≤ 1
                discharge_probs = np.round(discharge_probs, 2)  # Keep two decimals
                
                # Ensure sum doesn't exceed 1
                discharge_probs = discharge_probs / discharge_probs.sum()

                # Standard deviation is 30% of probability for high-priority, 15% for low-priority
                discharge_std = np.zeros(max_periods)
                discharge_std[start_period:end_period] = (0.30 if j == 0 else 0.15) * discharge_probs[start_period:end_period]

                # Convert to lists and store only the required portion based on `self.TimeBucketSet`
                facility_avg.append(np.round(discharge_probs[:len(self.TimeBucketSet)], 3).tolist())
                facility_std.append(np.round(discharge_std[:len(self.TimeBucketSet)], 3).tolist())

            ForecastedAvgPercentagePatientDischarged_ujt.append(facility_avg)
            ForecastedSTDPercentagePatientDischarged_ujt.append(facility_std)

        return ForecastedAvgPercentagePatientDischarged_ujt, ForecastedSTDPercentagePatientDischarged_ujt

    def Calculate_Travel_Time_Land(self, distance_matrix, speed):
        """
        Calculates travel time between points in minutes based on distance and average speed.

        :param distance_matrix: List of lists, each sublist represents distances from one point to others.
        :param speed: float, the average land speed in km/h.
        :return: List of lists representing travel time in minutes.
        """
        travel_time = []
        for row in distance_matrix:
            time_row = [round((distance / speed) * 60, 2) if speed > 0 else float('inf') for distance in row]
            travel_time.append(time_row)
        return travel_time

    def Generate_Positions(self, number):
        if Constants.Debug: print("\n We are in 'Instance' Class -- Generate_Positions")

        positions = []
        for i in range(number):
            pos = []
            for j in range(2):  
                vv = random.random() * self.Square_Dimension
                vv = int(vv * 100) / 100.0  
                pos.append(vv)
            positions.append(pos)
        return positions

    def Calculate_Distances(self, set1, set2):
        distances = []
        for pos1 in set1:
            row = []
            for pos2 in set2:
                distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                row.append(round(distance, 2))
            distances.append(row)
        return distances

    # Haversine formula to calculate distances between two lat/lon points in kilometers
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0  # Radius of the Earth in km
        Average_Circuity_Factor_Turkey = 1.36  # Circuity factor for Turkey

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = (R * c) * Average_Circuity_Factor_Turkey  # Apply the circuity factor to the straight-line distance
        return round(distance, 2)

    # Calculate distances between two sets of positions using latitude and longitude
    def calculate_distances_haversine(self, set1, set2):
        distances = []
        for pos1 in set1:
            row = []
            for pos2 in set2:
                distance = self.haversine(pos1[0], pos1[1], pos2[0], pos2[1])  # Use self.haversine to call the method
                row.append(distance)
            distances.append(row)
        return distances

    # Calculate distances within the same set
    def calculate_distances_within_same_haversine(self, set_positions):
        distances = []
        for i, pos1 in enumerate(set_positions):
            row = []
            for j, pos2 in enumerate(set_positions):
                if i != j:
                    distance = self.haversine(pos1[0], pos1[1], pos2[0], pos2[1])  # Use self.haversine here as well
                else:
                    distance = 0  # Distance to itself is 0
                row.append(distance)
            distances.append(row)
        return distances
        
    def Calculate_Distances_Within_Same(self, positions):
        distances = []
        for i, pos1 in enumerate(positions):
            row = []
            for j, pos2 in enumerate(positions):
                if i == j:
                    distance = 1000  # Special case for distance to itself
                else:
                    distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                row.append(round(distance, 2))
            distances.append(row)
        return distances

    def Plot_Positions(self):
        if Constants.Debug: print("\nWe are in 'Instance' Class -- Plot_Positions")

        # Correct the indices in list comprehensions for plotting positions
        hospital_plot = [[pos[0], pos[1]] for pos in self.Hospital_Position]
        acf_plot = [[pos[0], pos[1]] for pos in self.ACF_Position]
        demand_location_plot = [[pos[0], pos[1]] for pos in self.DisasterArea_Position]

        # Print positions for verification
        print("Hospital Plotting Positions:", hospital_plot)
        print("ACF Plotting Positions:", acf_plot)
        print("Disaster Areas Plotting Positions:", demand_location_plot)

        # Call to actual plotting method, assuming it uses matplotlib to plot
        self.Plot_Points_matplotlib(hospital_plot, acf_plot, demand_location_plot)

    def Plot_Points_matplotlib(self, hospital, acf, demand_location):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(*zip(*hospital), c='blue', label='Hospitals')
        plt.scatter(*zip(*acf), c='green', label='ACFs')
        plt.scatter(*zip(*demand_location), c='red', label='Disaster Areas')
        plt.legend()
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Facility Positions')
        plt.grid(True)
        plt.show()

    # This function print the instance on the screen
    def Print_Attributes(self):
        """
        Prints important instance attributes and writes them to a text file.
        """
        if Constants.Debug: 
            print("\n We are in 'Instance' Class -- Print_Attributes")

        filename = f"./Instances/{self.InstanceName}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists

        with open(filename, 'w') as file:
            file.write("\n============= DEBUG INFORMATION =============\n")

            # Define variables to print
            debug_data = {
                'NrTimeBucket': self.NrTimeBucket,
                'NrACFs': self.NrACFs,
                'NrHospitals': self.NrHospitals,
                'NrMedFacilities': self.NrMedFacilities,
                'NrDisasterAreas': self.NrDisasterAreas,
                'NrRescueVehicles': self.NrRescueVehicles,
                'NrInjuries': self.NrInjuries,
                'K_h': self.K_h,
                'J_m': self.J_m,
                'J_u': self.J_u,
                'I_A_Set': self.I_A_Set,
                'I_A_List': self.I_A_List,
                'ACF_Bed_Capacity': self.ACF_Bed_Capacity,
                'Hospital_Bed_Capacity': self.Hospital_Bed_Capacity,
                'Fixed_Cost_ACF_Objective': self.Fixed_Cost_ACF_Objective,
                'Fixed_Cost_ACF_Constraint': self.Fixed_Cost_ACF_Constraint,
                'Total_Budget_ACF_Establishment': self.Total_Budget_ACF_Establishment,
                'VehicleAssignment_Cost': self.VehicleAssignment_Cost,
                'ForecastedAvgCasualtyDemand': self.ForecastedAvgCasualtyDemand,
                'ForecastedSTDCasualtyDemand': self.ForecastedSTDCasualtyDemand,
                'Land_Rescue_Vehicle_Capacity': self.Land_Rescue_Vehicle_Capacity,
                'Aerial_Rescue_Vehicle_Capacity': self.Aerial_Rescue_Vehicle_Capacity,
                'Distance_D_A': self.Distance_D_A,
                'Distance_A_H': self.Distance_A_H,
                'Distance_D_H': self.Distance_D_H,
                'Distance_A_A': self.Distance_A_A,
                'Distance_H_H': self.Distance_H_H,
                'Distance_U_U': self.Distance_U_U,
                'Time_D_A_Land': self.Time_D_A_Land,
                'Time_A_H_Land': self.Time_A_H_Land,
                'Time_D_H_Land': self.Time_D_H_Land,
                'Time_A_A_Land': self.Time_A_A_Land,
                'Time_H_H_Land': self.Time_H_H_Land,
                'Time_U_U_Land': self.Time_U_U_Land,
                'Time_D_A_Aerial': self.Time_D_A_Aerial,
                'Time_A_H_Aerial': self.Time_A_H_Aerial,
                'Time_D_H_Aerial': self.Time_D_H_Aerial,
                'Time_A_A_Aerial': self.Time_A_A_Aerial,
                'Time_H_H_Aerial': self.Time_H_H_Aerial,
                'Casualty_Shortage_Cost': self.Casualty_Shortage_Cost,
                'Number_Rescue_Vehicle_ACF': self.Number_Rescue_Vehicle_ACF,
                'Number_Land_Rescue_Vehicle_Hospital': self.Number_Land_Rescue_Vehicle_Hospital,
                'ForecastedAvgHospitalDisruption': self.ForecastedAvgHospitalDisruption,
                'ForecastedSTDHospitalDisruption': self.ForecastedSTDHospitalDisruption,
                'ForecastedAvgPatientDemand': self.ForecastedAvgPatientDemand,
                'ForecastedSTDPatientDemand': self.ForecastedSTDPatientDemand,
                'ForecastedAvgPercentagePatientDischarged': self.ForecastedAvgPercentagePatientDischarged,
                'ForecastedSTDPercentagePatientDischarged': self.ForecastedSTDPercentagePatientDischarged,
                'Max_Backup_Hospital': self.Max_Backup_Hospital,
                'Available_Aerial_Vehicles_Hospital': self.Available_Aerial_Vehicles_Hospital,
                'CoordinationCost': self.CoordinationCost,
                'EvacuationRiskCost': self.EvacuationRiskCost,
                'CumulativeThreatRiskConstant': self.CumulativeThreatRiskConstant,
                'CumulativeThreatRiskLinear': self.CumulativeThreatRiskLinear,
                'CumulativeThreatRiskExponential': self.CumulativeThreatRiskExponential,
                'CumulativeLandTransportation_Risk': self.CumulativeLandTransportation_Risk,
                'CumulativeAerialTransportRisk': self.CumulativeAerialTransportRisk,
                'LandEvacuationRisk_Constant': self.LandEvacuationRisk_Constant,
                'LandEvacuationRisk_Linear': self.LandEvacuationRisk_Linear,
                'LandEvacuationRisk_Exponential': self.LandEvacuationRisk_Exponential,
                'AerialEvacuationRisk_Constant': self.AerialEvacuationRisk_Constant,
                'AerialEvacuationRisk_Linear': self.AerialEvacuationRisk_Linear,
                'AerialEvacuationRisk_Exponential': self.AerialEvacuationRisk_Exponential
            }

            # Print & Write each variable in a structured way
            for key, value in debug_data.items():
                # Convert NumPy arrays to lists for better printing
                if isinstance(value, np.ndarray):
                    value = value.tolist()

                # Print variable title
                print(f"\n{key}:")
                file.write(f"\n{key}:\n")

                # Print and write values with structured formatting
                if isinstance(value, list):
                    # For lists/matrices, print line by line
                    for row in value:
                        print(row)
                        file.write(f"{row}\n")
                else:
                    # For single values, print directly
                    print(value)
                    file.write(f"{value}\n")

            print("\n============= END OF DEBUG INFO =============")
            file.write("\n============= END OF DEBUG INFO =============\n")

    def SaveInstanceToPickle(self):
        # Define the filename using the instance name with a .pkl extension
        filename = f"./Instances/{self.InstanceName}.pkl"
        
        # Use 'wb' to write in binary mode
        with open(filename, 'wb') as file:
            data_to_save = {
                'NrTimeBucket': self.NrTimeBucket,
                'NrACFs': self.NrACFs,
                'NrHospitals': self.NrHospitals,
                'NrMedFacilities': self.NrMedFacilities,
                'NrDisasterAreas': self.NrDisasterAreas,
                'NrRescueVehicles': self.NrRescueVehicles,
                'NrInjuries': self.NrInjuries,
                'K_h': self.K_h,
                'J_m': self.J_m,
                'J_u': self.J_u,
                'I_A_Set': self.I_A_Set,
                'I_A_List': self.I_A_List,
                'ACF_Bed_Capacity': self.ACF_Bed_Capacity,
                'Hospital_Bed_Capacity': self.Hospital_Bed_Capacity,
                'Fixed_Cost_ACF_Objective': self.Fixed_Cost_ACF_Objective,
                'Fixed_Cost_ACF_Constraint': self.Fixed_Cost_ACF_Constraint,
                'Total_Budget_ACF_Establishment': self.Total_Budget_ACF_Establishment,
                'VehicleAssignment_Cost': self.VehicleAssignment_Cost,
                'ForecastedAvgCasualtyDemand': self.ForecastedAvgCasualtyDemand,
                'ForecastedSTDCasualtyDemand': self.ForecastedSTDCasualtyDemand,
                'Land_Rescue_Vehicle_Capacity': self.Land_Rescue_Vehicle_Capacity,
                'Aerial_Rescue_Vehicle_Capacity': self.Aerial_Rescue_Vehicle_Capacity,
                'Distance_D_A': self.Distance_D_A,
                'Distance_A_H': self.Distance_A_H,
                'Distance_D_H': self.Distance_D_H,
                'Distance_A_A': self.Distance_A_A,
                'Distance_H_H': self.Distance_H_H,
                'Distance_U_U': self.Distance_U_U,
                'Time_D_A_Land': self.Time_D_A_Land,
                'Time_A_H_Land': self.Time_A_H_Land,
                'Time_D_H_Land': self.Time_D_H_Land,
                'Time_A_A_Land': self.Time_A_A_Land,
                'Time_H_H_Land': self.Time_H_H_Land,
                'Time_U_U_Land': self.Time_U_U_Land,
                'Time_D_A_Aerial': self.Time_D_A_Aerial,
                'Time_A_H_Aerial': self.Time_A_H_Aerial,
                'Time_D_H_Aerial': self.Time_D_H_Aerial,
                'Time_A_A_Aerial': self.Time_A_A_Aerial,
                'Time_H_H_Aerial': self.Time_H_H_Aerial,
                'Casualty_Shortage_Cost': self.Casualty_Shortage_Cost,
                'Number_Rescue_Vehicle_ACF': self.Number_Rescue_Vehicle_ACF,
                'Number_Land_Rescue_Vehicle_Hospital': self.Number_Land_Rescue_Vehicle_Hospital,
                'ForecastedAvgHospitalDisruption': self.ForecastedAvgHospitalDisruption,
                'ForecastedSTDHospitalDisruption': self.ForecastedSTDHospitalDisruption,
                'ForecastedAvgPatientDemand': self.ForecastedAvgPatientDemand,
                'ForecastedSTDPatientDemand': self.ForecastedSTDPatientDemand,
                'ForecastedAvgPercentagePatientDischarged': self.ForecastedAvgPercentagePatientDischarged,
                'ForecastedSTDPercentagePatientDischarged': self.ForecastedSTDPercentagePatientDischarged,
                'Max_Backup_Hospital': self.Max_Backup_Hospital,
                'Available_Aerial_Vehicles_Hospital': self.Available_Aerial_Vehicles_Hospital,
                'CoordinationCost': self.CoordinationCost,
                'EvacuationRiskCost': self.EvacuationRiskCost,
                'CumulativeThreatRiskConstant': self.CumulativeThreatRiskConstant,
                'CumulativeThreatRiskLinear': self.CumulativeThreatRiskLinear,
                'CumulativeThreatRiskExponential': self.CumulativeThreatRiskExponential,
                'CumulativeLandTransportation_Risk': self.CumulativeLandTransportation_Risk,
                'CumulativeAerialTransportRisk': self.CumulativeAerialTransportRisk,
                'LandEvacuationRisk_Constant': self.LandEvacuationRisk_Constant,
                'LandEvacuationRisk_Linear': self.LandEvacuationRisk_Linear,
                'LandEvacuationRisk_Exponential': self.LandEvacuationRisk_Exponential,
                'AerialEvacuationRisk_Constant': self.AerialEvacuationRisk_Constant,
                'AerialEvacuationRisk_Linear': self.AerialEvacuationRisk_Linear,
                'AerialEvacuationRisk_Exponential': self.AerialEvacuationRisk_Exponential
            }
            
            # Save the data using pickle
            pickle.dump(data_to_save, file)

        print(f"Data saved to {filename}")

    def LoadInstanceFromPickle(self, instancename):
        # Define the filename using the instance name with a .pkl extension
        if platform.system() == "Linux":
            # Use the absolute path for the Linux system
            instances_dir = "/home/pfarghad/Myschedulingmodel_3/RL/Instances"  # Replace with your Linux directory
        else:
            # Use the desired path for your local system (Windows in this case)
            instances_dir = r"C:\PhD\Thesis\Papers\3rd\Code\RL\Instances"
        
        # Define the filename using the instance name with a .pkl extension
        filename = os.path.join(instances_dir, f"{instancename}.pkl")
        print(f"Loading instance from {filename}")  # Debugging line

        # Use 'rb' to read in binary mode
        with open(filename, 'rb') as file:
            # Load the data using pickle
            data_loaded = pickle.load(file)
            
            self.NrTimeBucket = data_loaded['NrTimeBucket']
            self.NrACFs = data_loaded['NrACFs']
            self.NrHospitals = data_loaded['NrHospitals']
            self.NrMedFacilities = data_loaded['NrMedFacilities']
            self.NrDisasterAreas = data_loaded['NrDisasterAreas']
            self.NrRescueVehicles = data_loaded['NrRescueVehicles']
            self.NrInjuries = data_loaded['NrInjuries']
            self.K_h = data_loaded['K_h']
            self.J_m = data_loaded['J_m']
            self.J_u = data_loaded['J_u']
            self.I_A_Set = data_loaded['I_A_Set']
            self.I_A_List = data_loaded['I_A_List']
            self.ACF_Bed_Capacity = data_loaded['ACF_Bed_Capacity']
            self.Hospital_Bed_Capacity = data_loaded['Hospital_Bed_Capacity']
            self.Fixed_Cost_ACF_Objective = data_loaded['Fixed_Cost_ACF_Objective']
            self.Fixed_Cost_ACF_Constraint = data_loaded['Fixed_Cost_ACF_Constraint']
            self.Total_Budget_ACF_Establishment = data_loaded['Total_Budget_ACF_Establishment']
            self.VehicleAssignment_Cost = data_loaded['VehicleAssignment_Cost']
            self.ForecastedAvgCasualtyDemand = data_loaded['ForecastedAvgCasualtyDemand']
            self.ForecastedSTDCasualtyDemand = data_loaded['ForecastedSTDCasualtyDemand']
            self.Land_Rescue_Vehicle_Capacity = data_loaded['Land_Rescue_Vehicle_Capacity']
            self.Aerial_Rescue_Vehicle_Capacity = data_loaded['Aerial_Rescue_Vehicle_Capacity']
            self.Distance_D_A = data_loaded['Distance_D_A']
            self.Distance_A_H = data_loaded['Distance_A_H']
            self.Distance_D_H = data_loaded['Distance_D_H']
            self.Distance_A_A = data_loaded['Distance_A_A']
            self.Distance_H_H = data_loaded['Distance_H_H']
            self.Distance_U_U = data_loaded['Distance_U_U']
            self.Time_D_A_Land = data_loaded['Time_D_A_Land']
            self.Time_A_H_Land = data_loaded['Time_A_H_Land']
            self.Time_D_H_Land = data_loaded['Time_D_H_Land']
            self.Time_A_A_Land = data_loaded['Time_A_A_Land']
            self.Time_H_H_Land = data_loaded['Time_H_H_Land']
            self.Time_U_U_Land = data_loaded['Time_U_U_Land']
            self.Time_D_A_Aerial = data_loaded['Time_D_A_Aerial']
            self.Time_A_H_Aerial = data_loaded['Time_A_H_Aerial']
            self.Time_D_H_Aerial = data_loaded['Time_D_H_Aerial']
            self.Time_A_A_Aerial = data_loaded['Time_A_A_Aerial']
            self.Time_H_H_Aerial = data_loaded['Time_H_H_Aerial']
            self.Casualty_Shortage_Cost = data_loaded['Casualty_Shortage_Cost']
            self.Number_Rescue_Vehicle_ACF = data_loaded['Number_Rescue_Vehicle_ACF']
            self.Number_Land_Rescue_Vehicle_Hospital = data_loaded['Number_Land_Rescue_Vehicle_Hospital']
            self.ForecastedAvgHospitalDisruption = data_loaded['ForecastedAvgHospitalDisruption']
            self.ForecastedSTDHospitalDisruption = data_loaded['ForecastedSTDHospitalDisruption']
            self.ForecastedAvgPatientDemand = data_loaded['ForecastedAvgPatientDemand']
            self.ForecastedSTDPatientDemand = data_loaded['ForecastedSTDPatientDemand']
            self.ForecastedAvgPercentagePatientDischarged = data_loaded['ForecastedAvgPercentagePatientDischarged']
            self.ForecastedSTDPercentagePatientDischarged = data_loaded['ForecastedSTDPercentagePatientDischarged']
            self.Max_Backup_Hospital = data_loaded['Max_Backup_Hospital']
            self.Available_Aerial_Vehicles_Hospital = data_loaded['Available_Aerial_Vehicles_Hospital']
            self.CoordinationCost = data_loaded['CoordinationCost']
            self.EvacuationRiskCost = data_loaded['EvacuationRiskCost']
            self.CumulativeThreatRiskConstant = data_loaded['CumulativeThreatRiskConstant']
            self.CumulativeThreatRiskLinear = data_loaded['CumulativeThreatRiskLinear']
            self.CumulativeThreatRiskExponential = data_loaded['CumulativeThreatRiskExponential']
            self.CumulativeLandTransportation_Risk = data_loaded['CumulativeLandTransportation_Risk']
            self.CumulativeAerialTransportRisk = data_loaded['CumulativeAerialTransportRisk']
            self.LandEvacuationRisk_Constant = data_loaded['LandEvacuationRisk_Constant']
            self.LandEvacuationRisk_Linear = data_loaded['LandEvacuationRisk_Linear']
            self.LandEvacuationRisk_Exponential = data_loaded['LandEvacuationRisk_Exponential']
            self.AerialEvacuationRisk_Constant = data_loaded['AerialEvacuationRisk_Constant']
            self.AerialEvacuationRisk_Linear = data_loaded['AerialEvacuationRisk_Linear']
            self.AerialEvacuationRisk_Exponential = data_loaded['AerialEvacuationRisk_Exponential']

        self.ComputeIndices()
        if Constants.Debug: self.Print_Attributes()
        print(f"Data loaded from {filename}")
