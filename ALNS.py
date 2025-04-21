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
import random
import os
import numpy as np
from collections import deque
from DQLAgent import DQLAgent
from RLAgent import RLAgent
import itertools


class ALNS:
    def __init__(self, 
                 instance, 
                 testidentifier, 
                 treestructure,
                 scenariotree=None, 
                 max_iterations=Constants.Max_ALNS_Iterations, 
                 use_rl=False, 
                 use_deep_q='QL', 
                 selection_method='e-greedy', 
                 state_size_x=None,
                 state_size_thetaVar=None,
                 state_size_w=None,
                 givenACFestablishments=[],
                 givenlandRescueVehicles=[],
                 givenBackupHospitals=[]
                 ):
        """
        Initialize the ALNS class.
        :param instance: Instance of the problem.
        :param testidentifier: Test configuration for ALNS.
        :param treestructure: Tree structure for the scenarios.
        :param scenariotree: Scenario tree (optional).
        :param max_iterations: Maximum number of ALNS iterations.
        :param use_rl: Whether to use RL for operator selection
        :param use_deep_q: Whether to use Deep Q-Learning instead of Q-Learning        
        :param givenACFestablishments: Initial ACF establishments.
        :param givenlandRescueVehicles: Initial Land rescue Vehicle.
        :param givenBackupHospitals: Initial Backup hospital.
        """
        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.TreeStructure = treestructure
        self.MaxIterations = max_iterations
        self.CurrentIteration = 0

        self.use_RL = use_rl  # Whether to use RL for operator selection
        self.use_Deep_Q = use_deep_q  # Whether to use Deep Q-Learning
        self.state_size_x = state_size_x if state_size_x is not None else self.Instance.NrACFs
        self.state_size_thetaVar = state_size_thetaVar if state_size_thetaVar is not None else (self.Instance.NrACFs * self.Instance.NrRescueVehicles)
        self.state_size_w = state_size_w if state_size_w is not None else (self.Instance.NrHospitals * self.Instance.NrHospitals)
        
        self.state_size = self.state_size_x + self.state_size_thetaVar + self.state_size_w

        # Define your operator sets:
        binary_destroy_actions = ["NoDestroy", "Remove"]
        binary_repair_actions = ["Add", "Swap"]
        integer_destroy_actions = ["NoChange", "Decrease"]
        integer_repair_actions = ["NoChange", "Increase"]

        # Build the combined action space using Cartesian product:
        self.combined_actions = list(itertools.product(binary_destroy_actions, 
                                                    binary_repair_actions, 
                                                    integer_destroy_actions, 
                                                    integer_repair_actions))
        # The total number of actions is the length of the combined actions list.
        self.num_actions = len(self.combined_actions)

        if self.use_RL:
            if self.use_Deep_Q:
                # Initialize DQLAgent
                self.RL_Agent = DQLAgent(num_actions = self.num_actions,
                                         state_size = self.state_size,
                                         selection_method = selection_method,
                                         alpha = 0.001,  # Learning rate
                                         gamma = 0.9,
                                         epsilon = 0.1,
                                         buffer_size = 10000,
                                         batch_size = 64,
                                         target_update_freq = 1000)
            else:
                # Initialize QLearningAgent
                self.RL_Agent = RLAgent(num_actions = self.num_actions,
                                        selection_method = selection_method,
                                        alpha = 0.2,
                                        gamma = 0.9,
                                        epsilon = 0.1)
        else:
            self.RL_Agent = None  # No RL agent
            # Initialize operator weights and scores for roulette wheel selection.
            self.operator_weights = [1.0] * self.num_actions
            self.operator_scores = [0.0] * self.num_actions
            # Define the segment length (e.g., 20 iterations per segment)
            self.segment_length = 10

        # Initial best solution and cost
        self.current_solution_x = None
        self.current_solution_thetaVar = None
        self.current_solution_w = None
        self.best_solution = None
        self.best_cost = float('inf')

        # Handle given ACF establishments
        if givenACFestablishments:
            givenACFestablishments = [min(round(x), 1) for x in givenACFestablishments]
            if Constants.Debug: print("Given ACF establishments: ", givenACFestablishments)

        if givenlandRescueVehicles:
            givenlandRescueVehicles = [[round(x) for x in row] for row in givenlandRescueVehicles]
            if Constants.Debug: print("Given land Rescue Vehicles:", givenlandRescueVehicles)

        if givenBackupHospitals:
            givenBackupHospitals = [[round(x) for x in row] for row in givenBackupHospitals]
            if Constants.Debug: print("Given lHospital Backup:", givenBackupHospitals)

        self.GivenACFEstablishment = givenACFestablishments
        self.GivenNrLandRescueVehicle = givenlandRescueVehicles
        self.GivenBackupHospital = givenBackupHospitals

        self.SolveWithfixedACFEstablishment = bool(self.GivenACFEstablishment)

        # Generate scenarios
        self.GenerateScenarios(scenariotree)

        # Start timing
        self.StartTime_ALNS = time.time()

        # Build MIP model
        self.BuildMIPs2()

        self.TraceFileName = "./Temp/ALNStrace_%s_Evaluation_%s.txt" % (self.TestIdentifier.GetAsString(), Constants.Evaluation_Part)

        ## For Being more time efficient
        self.scenario_set = self.MIPSolver.ScenarioSet
        self.sorted_acfs = sorted(self.Instance.ACFSet, key=lambda i: self.Instance.ACF_Bed_Capacity[i], reverse=True)
        # Initialize a list to store log entries
        self.log_buffer = []
        # Set how often to flush the buffer to file (every 10 iterations)
        self.flush_interval = 10

    def InitTrace(self):
        if Constants.Debug: print("\n We are in 'ALNS' Class -- InitTrace")
        if Constants.PrintPHATrace:
            self.TraceFile = open(self.TraceFileName, "w")
            self.TraceFile.write("Start the ALNS algorithm \n")
            self.TraceFile.close()

    def WriteInTraceFile(self, string):
        # Append log message to the buffer
        self.log_buffer.append(string)

        # Check if it's time to flush the buffer to file
        if len(self.log_buffer) >= self.flush_interval:
            # Write all buffered messages to the file
            with open(self.TraceFileName, "a") as file:
                file.writelines(self.log_buffer)
            
            # Clear the buffer after writing
            self.log_buffer = []

    def BuildMIPs2(self):
        """
        Build the mathematical model using MIPSolver for ALNS.
        """
        if Constants.Debug: 
            print("\n We are in 'ALNS' Class -- BuildMIPs2")

        self.MIPSolver = MIPSolver( instance=self.Instance,
                                    model=Constants.Two_Stage,
                                    scenariotree=self.ScenarioTree,
                                    nrscenario=self.TreeStructure[1],
                                    givenACFEstablishment = self.GivenACFEstablishment,
                                    givenNrLandRescueVehicle = self.GivenNrLandRescueVehicle,
                                    givenBackupHospital = self.GivenBackupHospital,                                    
                                    logfile="NO")
        self.MIPSolver.BuildModel()

    def GenerateScenarios(self, scenariotree=None):

        if Constants.Debug:
            print("\n We are in 'ALNS' Class -- GenerateScenarios")

        if scenariotree is None:
            self.ScenarioTree = ScenarioTree(instance=self.Instance,
                                            tree_structure=self.TreeStructure,
                                            scenario_seed=self.TestIdentifier.ScenarioSeed,
                                            scenariogenerationmethod=self.TestIdentifier.ScenarioSampling)
        else:
            self.ScenarioTree = scenariotree

    def solve_relaxed_mip(self):

        print("Solving relaxed MIP...")
        self.MIPSolver.ChangeACFEstablishmentVarToContinuous()
        self.MIPSolver.ChangeLandRescueVehicleVarToContinuous()
        self.MIPSolver.ChangeBackupHospitalVarToContinuous()
        
        #Solve the model.
        relaxed_solution = self.MIPSolver.Solve(True)

        return relaxed_solution

    def fix_First_Stage_Variables(self, Rounded_x, Rounded_thetaVar, Rounded_w):
        
        self.fix_x_variables(Rounded_x)
        self.fix_thetaVar_variables(Rounded_thetaVar)
        self.fix_w_variables(Rounded_w)
    
    def fix_x_variables(self, Rounded_x):
        for w in self.scenario_set:
            for i in self.Instance.ACFSet:
                # Obtain the index
                index = self.MIPSolver.GetIndexACFEstablishmentVariable(w, i)

                # Retrieve the variable using the index
                variable = self.MIPSolver.ACFEstablishment_Var.get(index)  # Use .get() to avoid KeyError
                if variable:
                    variable.UB = Rounded_x[w][i]
                    variable.LB = Rounded_x[w][i]
                else:
                    print(f"No variable found for index: {index}")
        self.MIPSolver.LocAloc.update()

    def fix_thetaVar_variables(self, Rounded_thetaVar):
        for w in self.scenario_set:
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    # Obtain the index
                    index = self.MIPSolver.GetIndexLandRescueVehicleVariable(w, i, m)

                    # Retrieve the variable using the index
                    variable = self.MIPSolver.LandRescueVehicle_Var.get(index)
                    if variable:
                        variable.UB = Rounded_thetaVar[w][i][m]
                        variable.LB = Rounded_thetaVar[w][i][m]
                    else:
                        print(f"No variable found for index: {index}")
        self.MIPSolver.LocAloc.update()

    def fix_w_variables(self, Rounded_w):
        for w in self.scenario_set:
            for h in self.Instance.HospitalSet:
                for hprime in self.Instance.HospitalSet:
                    # Obtain the index
                    index = self.MIPSolver.GetIndexBackupHospitalVariable(w, h, hprime)

                    # Retrieve the variable using the index
                    variable = self.MIPSolver.BackupHospital_Var.get(index)
                    if variable:
                        variable.UB = Rounded_w[w][h][hprime]
                        variable.LB = Rounded_w[w][h][hprime]
                    else:
                        print(f"No variable found for index: {index}")
        self.MIPSolver.LocAloc.update()

    def solve_fixed_mip(self):
        """
        Solve the MIP with fixed first-stage variables.
        :return: Objective value of the fixed MIP.
        """
        #Solve the model.
        fixed_solution = self.MIPSolver.Solve(True)

        return fixed_solution

    def get_state_from_solution_old(self, x_solution):
        """Ensure state is always a tuple, whether x_solution is a dictionary or list."""
        if isinstance(x_solution, dict):
            return tuple(x_solution.values())  # Convert dict values to tuple
        return tuple(x_solution)  # Convert list to tuple

    def get_state_from_solution(self, x_solution, thetaVar_solution, w_solution):
        """
        Ensure state is always a tuple, whether x_solution, thetaVar_solution, or w_solution is a dictionary or list.
        This method will flatten and combine all three variables into a single state vector.
        """
        # Flatten the binary variables (x and w) into 1D arrays
        if isinstance(x_solution, dict):
            flattened_x = tuple(x_solution.values())  # Convert dict values to tuple
        else:
            flattened_x = tuple(x_solution)  # Convert list to tuple
        
        # Flatten the 2D w_solution (backup hospitals) into 1D array
        flattened_w = []
        for row in w_solution:
            flattened_w.extend(row)  # Flatten each row and add it to the list
        
        # Flatten the integer variables (thetaVar) and normalize by the maximum value for each i
        flattened_thetaVar = []
        for i in self.Instance.ACFSet:
            for m in self.Instance.RescueVehicleSet:
                # Normalize thetaVar values if necessary (e.g., to [0, 1])
                flattened_thetaVar.append(thetaVar_solution[i][m] / self.Instance.Number_Rescue_Vehicle_ACF[m])
        
        # Convert all parts to tuples for concatenation
        flattened_w = tuple(flattened_w)  # Convert flattened_w to tuple
        flattened_thetaVar = tuple(flattened_thetaVar)  # Ensure flattened_thetaVar is a tuple
        
        # Combine all parts into a single state vector
        state_vector = flattened_x + flattened_thetaVar + flattened_w
        
        # Convert the state vector into a tuple (or numpy array if needed)
        state = tuple(state_vector)  # Or use np.array(state_vector) if preferred
        
        return state

    def destroy_and_repair(self, x_solution, thetaVar_solution, w_solution):
        
        """
        Decide which combination of destroy and repair operators to use.
        Destroy_actions: {NoDestroy, Remove}
        Repair_actions: {Add, Swap}
        Integer_Destroy_actions: {NoChange, Decrease}
        Integer_Repair_actions: {NoChange, Increase}
        
        The combined operator is one of:
        0 -> (NoDestroy, Add, NoChange, NoChange)
        1 -> (NoDestroy, Add, NoChange, Increase)
        2 -> (NoDestroy, Add, Decrease, NoChange)
        3 -> (NoDestroy, Add, Decrease, Increase)
        4 -> (NoDestroy, Swap, NoChange, NoChange)
        5 -> (NoDestroy, Swap, NoChange, Increase)
        6 -> (NoDestroy, Swap, Decrease, NoChange)
        7 -> (NoDestroy, Swap, Decrease, Increase)
        8 -> (Remove, Add, NoChange, NoChange)
        9 -> (Remove, Add, NoChange, Increase)
        10 -> (Remove, Add, Decrease, NoChange)
        11 -> (Remove, Add, Decrease, Increase)
        12 -> (Remove, Swap, NoChange, NoChange)
        13 -> (Remove, Swap, NoChange, Increase)
        14 -> (Remove, Swap, Decrease, NoChange)
        15 -> (Remove, Swap, Decrease, Increase)
        
        Also, determine the number of changes (between 10% and 50% of FacilitySet, at least 1).
        Returns a tuple: (combined_action, destroy_action, repair_action, integer_destroy_action, integer_repair_action, num_changes)
        """
        # Get current state (if needed for RL)
        state = self.get_state_from_solution(x_solution[0], thetaVar_solution[0], w_solution[0])
        
        if self.use_RL:
            # The RL agent now selects among the combined actions
            combined_action = self.RL_Agent.select_action(state)
        else:
            # Roulette wheel selection based on operator weights (now length of combined actions)
            total_weight = sum(self.operator_weights)
            r = random.uniform(0, total_weight)
            cumulative = 0.0
            for idx, weight in enumerate(self.operator_weights):
                cumulative += weight
                if r <= cumulative:
                    combined_action = idx
                    break
        
        # Map the combined action integer to (binary_destroy_action, binary_repair_action, integer_destroy_action, integer_repair_action)
        mapping = {
            0: ("NoDestroy", "Add", "NoChange", "NoChange"),
            1: ("NoDestroy", "Add", "NoChange", "Increase"),
            2: ("NoDestroy", "Add", "Decrease", "NoChange"),
            3: ("NoDestroy", "Add", "Decrease", "Increase"),
            4: ("NoDestroy", "Swap", "NoChange", "NoChange"),
            5: ("NoDestroy", "Swap", "NoChange", "Increase"),
            6: ("NoDestroy", "Swap", "Decrease", "NoChange"),
            7: ("NoDestroy", "Swap", "Decrease", "Increase"),
            8: ("Remove", "Add", "NoChange", "NoChange"),
            9: ("Remove", "Add", "NoChange", "Increase"),
            10: ("Remove", "Add", "Decrease", "NoChange"),
            11: ("Remove", "Add", "Decrease", "Increase"),
            12: ("Remove", "Swap", "NoChange", "NoChange"),
            13: ("Remove", "Swap", "NoChange", "Increase"),
            14: ("Remove", "Swap", "Decrease", "NoChange"),
            15: ("Remove", "Swap", "Decrease", "Increase")
        }

        binary_destroy_action, binary_repair_action, integer_destroy_action, integer_repair_action = mapping[combined_action]

        # Determine number of changes (between 10% and 50% of FacilitySet, at least 1)
        num_changes = random.randint(
            max(1, int(0.1 * len(self.Instance.ACFSet))),
            int(0.5 * len(self.Instance.ACFSet))
        )

        return combined_action, binary_destroy_action, binary_repair_action, integer_destroy_action, integer_repair_action, num_changes

    def Apply_Action_Old(self, x_solution, destroy_action, repair_action, num_changes):
        """
        Apply the selected destroy and repair operators to the current solution.
        Depending on the combination:
        
        (NoDestroy, Add):
        • No destruction is done.
        • Repair: add ones to some zero entries.
        
        (NoDestroy, Swap):
        • No destruction is done.
        • Repair: swap a subset of ones and zeros.
        
        (Remove, Add):
        • Destroy: remove (set to 0) a subset of ones.
        • Repair: add (set to 1) a subset of zeros.
        
        (Remove, Swap):
        • Destroy: remove (set to 0) a subset of ones.
        • Repair: perform a swap on the modified solution.
        """
        # --- Destroy phase ---
        if destroy_action == "Remove":
            ones_indices = [i for i, x in enumerate(x_solution[0]) if x == 1]
            if ones_indices:
                indices_to_change = random.sample(ones_indices, min(len(ones_indices), num_changes))
                for i in indices_to_change:
                    for w in self.scenario_set:
                        x_solution[w][i] = 0
                #print(f"Remove operator: Converted {len(indices_to_change)} ones to zeros.")
        else:
            # "NoDestroy": do nothing.
            pass

        # --- Repair phase ---
        if repair_action == "Add":
            zeros_indices = [i for i, x in enumerate(x_solution[0]) if x == 0]
            if zeros_indices:
                indices_to_change = random.sample(zeros_indices, min(len(zeros_indices), num_changes))
                for i in indices_to_change:
                    for w in self.scenario_set:
                        x_solution[w][i] = 1
                #print(f"Add operator: Converted {len(indices_to_change)} zeros to ones.")
        elif repair_action == "Swap":
            ones_indices = [i for i, x in enumerate(x_solution[0]) if x == 1]
            zeros_indices = [i for i, x in enumerate(x_solution[0]) if x == 0]
            num_swaps = min(len(ones_indices), len(zeros_indices), num_changes)
            if ones_indices and zeros_indices and num_swaps > 0:
                indices_to_change_ones = random.sample(ones_indices, num_swaps)
                indices_to_change_zeros = random.sample(zeros_indices, num_swaps)
                for w in self.scenario_set:
                    for i in indices_to_change_ones:
                        x_solution[w][i] = 0
                    for i in indices_to_change_zeros:
                        x_solution[w][i] = 1
                #print(f"Swap operator: Swapped {num_swaps} ones and zeros.")

        if Constants.Debug: print("Modified x_solution:", x_solution)
            
        return x_solution

    def Apply_Action_on_x(self, x_solution, binary_destroy_action, binary_repair_action, num_changes):
        """
        Apply the selected destroy and repair operators to the binary variable x_solution.
        """
        # --- Destroy phase ---
        if binary_destroy_action == "Remove":
            ones_indices = [i for i, x in enumerate(x_solution[0]) if x == 1]
            if ones_indices:
                indices_to_change = random.sample(ones_indices, min(len(ones_indices), num_changes))
                for i in indices_to_change:
                    for w in self.scenario_set:
                        x_solution[w][i] = 0
                #print(f"Remove operator: Converted {len(indices_to_change)} ones to zeros.")
        else:
            # "NoDestroy": do nothing.
            pass

        # --- Repair phase ---
        if binary_repair_action == "Add":
            zeros_indices = [i for i, x in enumerate(x_solution[0]) if x == 0]
            if zeros_indices:
                indices_to_change = random.sample(zeros_indices, min(len(zeros_indices), num_changes))
                for i in indices_to_change:
                    for w in self.scenario_set:
                        x_solution[w][i] = 1
                #print(f"Add operator: Converted {len(indices_to_change)} zeros to ones.")
        elif binary_repair_action == "Swap":
            ones_indices = [i for i, x in enumerate(x_solution[0]) if x == 1]
            zeros_indices = [i for i, x in enumerate(x_solution[0]) if x == 0]
            num_swaps = min(len(ones_indices), len(zeros_indices), num_changes)
            if ones_indices and zeros_indices and num_swaps > 0:
                indices_to_change_ones = random.sample(ones_indices, num_swaps)
                indices_to_change_zeros = random.sample(zeros_indices, num_swaps)
                for w in self.scenario_set:
                    for i in indices_to_change_ones:
                        x_solution[w][i] = 0
                    for i in indices_to_change_zeros:
                        x_solution[w][i] = 1
                #print(f"Swap operator: Swapped {num_swaps} ones and zeros.")

        # To Check if the budget constraint is respected
        x_solution_i = self.round_x_variable(x_solution[0], RandomRemoval=True)
        x_solution = [x_solution_i[:] for _ in self.scenario_set]        
        if Constants.Debug: print("Modified x_solution:", x_solution)
        
        return x_solution

    def Apply_Action_on_thetaVar(self, x_solution, thetaVar_solution, integer_destroy_action, integer_repair_action, num_changes):
        """
        Apply the selected destroy and repair operators to the integer variable thetaVar_solution.
        """
        # Step 1: Check connection between x_solution and thetaVar_solution
        thetaVar_solution = self.check_connection_between_x_and_thetaVar(x_solution[0], thetaVar_solution[0])
        
        # Step 2: Apply the "Decrease" action (if applicable) on thetaVar_solution
        if integer_destroy_action == "Decrease":
            for i in self.Instance.ACFSet:
                for m in self.Instance.RescueVehicleSet:
                    if thetaVar_solution[i][m] > 0:
                        decrease_amount = random.randint(0, thetaVar_solution[i][m])  # Randomly decrease between 0 and current value
                        thetaVar_solution[i][m] -= decrease_amount
                        #print(f"Decreased {decrease_amount} vehicles for ACF {i}, vehicle type {m}.")
        else:
            # "NoChange": do nothing.
            pass

        # Step 3: Apply the "Increase" action (if applicable) on thetaVar_solution
        if integer_repair_action == "Increase":
            # First, calculate the total number of vehicles already assigned and the remaining vehicles
            remaining_vehicles = {m: self.Instance.Number_Rescue_Vehicle_ACF[m] - sum(thetaVar_solution[i][m] for i in self.Instance.ACFSet) 
                                for m in self.Instance.RescueVehicleSet}

            # Step 3.1: Assign 30% of the remaining vehicles to ACFs with no vehicles assigned yet
            for m in self.Instance.RescueVehicleSet:
                if remaining_vehicles[m] > 0:
                    for i in self.Instance.ACFSet:
                        if x_solution[0][i] == 1 and sum(thetaVar_solution[i][m] for m in self.Instance.RescueVehicleSet) == 0:
                            assign_amount = int(0.30 * remaining_vehicles[m])  # 30% of the remaining vehicles
                            thetaVar_solution[i][m] += assign_amount
                            remaining_vehicles[m] -= assign_amount
                            #print(f"Assigned {assign_amount} vehicles of type {m} to ACF {i} (30% allocation).")
                        
                            if remaining_vehicles[m] <= 0:
                                break

            # Step 3.2: Assign the remaining vehicles based on ACF capacities
            if any(remaining_vehicles[m] > 0 for m in self.Instance.RescueVehicleSet):
                
                # Sort ACFs by their capacity in descending order
                for m in self.Instance.RescueVehicleSet:
                    if remaining_vehicles[m] > 0:
                        # Distribute the remaining vehicles proportionally, but limit based on total remaining vehicles
                        for i in self.sorted_acfs:
                            if x_solution[0][i] == 1:  # Only assign vehicles to open ACFs
                                # Calculate the proportion of remaining vehicles for this ACF based on its capacity
                                capacity_ratio = self.Instance.ACF_Bed_Capacity[i] / sum(self.Instance.ACF_Bed_Capacity[j] for j in self.Instance.ACFSet if x_solution[0][j] == 1)
                                
                                # Limit the max number of vehicles that can be assigned to this ACF based on the total remaining vehicles
                                max_assignable = int(capacity_ratio * remaining_vehicles[m])
                                
                                if max_assignable > remaining_vehicles[m]:
                                    max_assignable = remaining_vehicles[m]  # Don't exceed the remaining vehicles available
                                
                                if max_assignable > 0:
                                    thetaVar_solution[i][m] += max_assignable
                                    remaining_vehicles[m] -= max_assignable
                                    #print(f"Assigned {max_assignable} vehicles of type {m} to ACF {i}.")
                                
                                if remaining_vehicles[m] <= 0:
                                    break
            
        else:
            # "NoChange": do nothing.
            pass
        
        thetaVar_solution_mi = self.check_limited_number_of_rescue_vehicles(thetaVar_solution)
        thetaVar_solution = [thetaVar_solution_mi[:][:] for _ in self.scenario_set]

        return thetaVar_solution

    def Apply_Action_on_w(self, w_solution, binary_destroy_action, binary_repair_action, num_changes):
        """
        Apply the selected destroy and repair operators to the binary variable x_solution.
        """

        # --- Destroy phase ---
        if binary_destroy_action == "Remove":
            ones_indices = [(h, hprime)
                            for h in range(len(w_solution[0]))              # i over first dimension
                            for hprime in range(len(w_solution[0][h]))          # m over second dimension
                            if w_solution[0][h][hprime] == 1]
            if ones_indices:
                # pick up to num_changes of those (i,m) pairs
                to_change = random.sample(ones_indices, min(len(ones_indices), num_changes))
                for h, hprime in to_change:
                    for w in self.scenario_set:
                        w_solution[w][h][hprime] = 0
        else:
            # "NoDestroy": do nothing.
            pass

        # --- Repair phase ---
        if binary_repair_action == "Add":
            # Collect all (h,hprime) pairs where in scenario 0 the entry is 0
            zeros_indices = [(h, hprime)
                            for h in range(len(w_solution[0]))
                            for hprime in range(len(w_solution[0][h]))
                            if w_solution[0][h][hprime] == 0]
            if zeros_indices:
                to_change = random.sample(zeros_indices, min(len(zeros_indices), num_changes))
                for h, hprime in to_change:
                    for w in self.scenario_set:
                        w_solution[w][h][hprime] = 1

        elif binary_repair_action == "Swap":
            # Find all ones and zeros in the base slice
            ones_indices = [(h, hprime)
                            for h in range(len(w_solution[0]))
                            for hprime in range(len(w_solution[0][h]))
                            if w_solution[0][h][hprime] == 1]
            zeros_indices = [(h, hprime)
                            for h in range(len(w_solution[0]))
                            for hprime in range(len(w_solution[0][h]))
                            if w_solution[0][h][hprime] == 0]

            num_swaps = min(len(ones_indices), len(zeros_indices), num_changes)
            if num_swaps > 0:
                swap_ones = random.sample(ones_indices, num_swaps)
                swap_zeros = random.sample(zeros_indices, num_swaps)
                for w in self.scenario_set:
                    for h, hprime in swap_ones:
                        w_solution[w][h][hprime] = 0
                    for h, hprime in swap_zeros:
                        w_solution[w][h][hprime] = 1

        # --- (Optional) re‑enforce feasibility by rounding your base slice and replicating ---
        w_base = self.round_w_variable(w_solution[0])
        for h in range(len(w_base)):
            w_base[h][h] = 0        
        w_solution = [[row[:] for row in w_base] for _ in self.scenario_set]

        if Constants.Debug: print("Modified w_solution:", w_solution)
        
        return w_solution
    
    def Apply_Action(self, x_solution, thetaVar_solution, w_solution, binary_destroy_action, binary_repair_action, integer_destroy_action, integer_repair_action, num_changes):
        """
        Apply the selected destroy and repair operators to the current solution for all variables.
        It first applies the actions to x, then to thetaVar, and finally to w.
        """
        # Apply actions to the binary variable x
        x_solution = self.Apply_Action_on_x(x_solution, binary_destroy_action, binary_repair_action, num_changes)
        
        # Apply actions to the integer variable thetaVar (we will define this in the next message)
        thetaVar_solution = self.Apply_Action_on_thetaVar(x_solution, thetaVar_solution, integer_destroy_action, integer_repair_action, num_changes)
        
        # Apply actions to the binary variable w (we will define this in the next message)
        if Constants.RandomALNSInitilization:
            w_solution = self.Apply_Action_on_w(w_solution, binary_destroy_action, binary_repair_action, num_changes)
        
        return x_solution, thetaVar_solution, w_solution

    def update_operator_weights(self):
        """
        Update the operator weights based on the scores obtained in the last segment.
        Here, we use a simple rule: new weight = 1 + accumulated score.
        After updating, reset the scores for the next segment.
        """
        for r in range(len(self.operator_weights)):
            self.operator_weights[r] = 1 + self.operator_scores[r]
            self.operator_scores[r] = 0
        print(f"Updated operator weights: {self.operator_weights}")

    def acceptance_check(self, new_cost, fixed_solution, T):
        """
        Check the acceptance criterion for the new solution using the Metropolis rule.
        
        If new_cost is lower than the current accepted cost (self.best_cost),
        the new solution is accepted unconditionally.
        
        Otherwise, the new solution is accepted with probability:
            exp(-(new_cost - self.best_cost) / T)
        
        If accepted, self.best_cost and self.current_solution_x are updated.
        Returns True if the new solution is accepted, False otherwise.
        """
        if new_cost < self.best_cost:
            print(f"New better solution accepted: {new_cost}")
            self.best_cost = new_cost
            self.current_solution_x = copy.deepcopy(fixed_solution.ACFEstablishment_x_wi)
            self.current_solution_thetaVar = copy.deepcopy(fixed_solution.LandRescueVehicle_thetaVar_wim)
            self.current_solution_w = copy.deepcopy(fixed_solution.BackupHospital_W_whhPrime)
            return True
        else:
            prob_accept = math.exp(-(new_cost - self.best_cost) / T)
            r = random.uniform(0, 1)
            if prob_accept > r:
                print(f"Worse solution accepted with probability {prob_accept:.4f} > {r:.4f}")
                self.best_cost = new_cost
                self.current_solution_x = copy.deepcopy(fixed_solution.ACFEstablishment_x_wi)
                self.current_solution_thetaVar = copy.deepcopy(fixed_solution.LandRescueVehicle_thetaVar_wim)
                self.current_solution_w = copy.deepcopy(fixed_solution.BackupHospital_W_whhPrime)
                return True
            else:
                print(f"Worse solution rejected with probability {prob_accept:.4f} <= {r:.4f}")
                return False

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

    def round_x_variable(self, x_var_i, RandomRemoval=False):
        """
        Rounds each value in a 2D variable using math.ceil and checks the budget constraint.
        Optimized by checking and adjusting the rounding for w=0, then copying to all w.
        """
        # Round the x_var using math.ceil
        Rounded_x_var = [math.ceil(val) for val in x_var_i]

        # Check and adjust the rounding for w=0
        Rounded_x_var = self.Check_ACF_Establishment_Budget_Constraint(x_var_i, Rounded_x_var)

        return Rounded_x_var

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

    def round_thetaVar_variable(self, Rounded_ACFEstablishment_x_i, thetaVar_var):
        """
        Rounds each value in a 3D variable using math.ceil and checks the land rescue vehicle allocation constraints.
        """
        Rounded_thetaVar = [[math.ceil(val) for val in inner] for inner in thetaVar_var]
        
        # Check the land rescue vehicle allocation constraints
        Rounded_thetaVar = self.Check_LandRescueVehicleAllocation_Constraints(Rounded_ACFEstablishment_x_i, Rounded_thetaVar)
        
        return Rounded_thetaVar
    
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

    def round_w_variable(self, w_var_hhprime):
        """
        Rounds each value in a 3D variable using math.ceil and checks the hospital compatibility constraint.
        """
        Rounded_w_hhprime = [[math.ceil(val) for val in inner] for inner in w_var_hhprime]

        # Now, call the Check_Hospitals_Compatibility_Constraint method to ensure compatibility
        Rounded_w_hhprime = self.Check_Hospitals_Compatibility_Constraint(Rounded_w_hhprime)
        
        return Rounded_w_hhprime

    def random_initial_solution(self):
        rng = random.Random(self.TestIdentifier.ScenarioSeed)
        # Dimensions
        num_scenario = self.TreeStructure[1]
        num_acf      = self.Instance.NrACFs
        num_res      = self.Instance.NrRescueVehicles
        num_hosp     = self.Instance.NrHospitals

        # --- initialize raw arrays with zeros ---
        raw_x0     = [0] * num_acf
        raw_theta0 = [[0] * num_res for _ in range(num_acf)]
        raw_w0     = [[0] * num_hosp for _ in range(num_hosp)]

        # 1) One raw x-vector (length = num_acf), binary {0,1}
        for i in self.Instance.ACFSet:
            raw_x0[i] = rng.randint(0, 1)

        # 2) One raw theta-matrix (num_acf × num_res), integer [0…capacity//5]
        for i in self.Instance.ACFSet:
            if raw_x0[i] == 1:
                # integer division to keep max_v an int
                for m in self.Instance.RescueVehicleSet:
                    max_v = self.Instance.Number_Rescue_Vehicle_ACF[m]
                    raw_theta0[i][m] = rng.randint(0, max_v)
            # else leave the row as zeros

        # 3) One raw w-matrix (num_hosp × num_hosp), binary {0,1}
        for h in self.Instance.HospitalSet:
            for hprime in self.Instance.HospitalSet:
                if h != hprime:
                    raw_w0[h][hprime] = rng.randint(0, 1)

        # --- replicate into each scenario ---
        Rounded_ACFEstablishment_x_wi = [raw_x0[:] for _ in range(num_scenario)]
        Rounded_LandRescueVehicle_thetaVar_wim = [[row[:] for row in raw_theta0] for _ in range(num_scenario)]
        Rounded_BackupHospital_W_whhprime = [[row[:] for row in raw_w0] for _ in range(num_scenario)]

        return Rounded_ACFEstablishment_x_wi, Rounded_LandRescueVehicle_thetaVar_wim, Rounded_BackupHospital_W_whhprime

    def Run(self):
        # To prevent obtaining the second-stage variables' values at every iteratrion of the ALNS
        Constants.Obtain_SecondStage_Solution = False        
        
        self.InitTrace()
        print("\nStarting ALNS...")

        if Constants.RandomALNSInitilization:
            # --- random init branch ---
            ACF_establishment_x_wi, landRescueVehicle_thetaVar_wim, backupHospital_W_whhPrime = self.random_initial_solution()

            # Step 2: Round relaxed solution to nearest integer (initial feasible solution)
            Rounded_ACFEstablishment_x_i = self.round_x_variable(ACF_establishment_x_wi[0])
            if Constants.Debug:print("Rounded_ACFEstablishment_x_wi:\n", Rounded_ACFEstablishment_x_i)

            Rounded_LandRescueVehicle_thetaVar_im = self.round_thetaVar_variable(Rounded_ACFEstablishment_x_i, landRescueVehicle_thetaVar_wim[0])
            if Constants.Debug:print("Rounded_LandRescueVehicle_thetaVar_im:\n", Rounded_LandRescueVehicle_thetaVar_im)

            Rounded_BackupHospital_W_hhprime = self.round_w_variable(backupHospital_W_whhPrime[0])
            if Constants.Debug:print("Rounded_BackupHospital_W_whhprime:\n", Rounded_BackupHospital_W_hhprime)

        else:
            # Step 1: Solve the relaxed MIP
            relaxed_solution = self.solve_relaxed_mip()
            if Constants.Debug:
                print("ACFEstablishment_x_wi:\n ", relaxed_solution.ACFEstablishment_x_wi)
                print("LandRescueVehicle_thetaVar_wim:\n ", relaxed_solution.LandRescueVehicle_thetaVar_wim)
                print("BackupHospital_W_whhPrime:\n ", relaxed_solution.BackupHospital_W_whhPrime)

            # Step 2: Round relaxed solution to nearest integer (initial feasible solution)
            ACF_establishment_x_wi = relaxed_solution.ACFEstablishment_x_wi
            Rounded_ACFEstablishment_x_i = self.round_x_variable(ACF_establishment_x_wi[0])
            if Constants.Debug:print("Rounded_ACFEstablishment_x_wi:\n", Rounded_ACFEstablishment_x_i)

            landRescueVehicle_thetaVar_wim = relaxed_solution.LandRescueVehicle_thetaVar_wim
            Rounded_LandRescueVehicle_thetaVar_im = self.round_thetaVar_variable(Rounded_ACFEstablishment_x_i, landRescueVehicle_thetaVar_wim[0])
            if Constants.Debug:print("Rounded_LandRescueVehicle_thetaVar_im:\n", Rounded_LandRescueVehicle_thetaVar_im)

            backupHospital_W_whhPrime = relaxed_solution.BackupHospital_W_whhPrime
            Rounded_BackupHospital_W_hhprime = self.round_w_variable(backupHospital_W_whhPrime[0])
            if Constants.Debug:print("Rounded_BackupHospital_W_whhprime:\n", Rounded_BackupHospital_W_hhprime)

        # Initialize empty lists for the 2D and 3D results
        Rounded_ACFEstablishment_x_wi = []
        Rounded_LandRescueVehicle_thetaVar_wim = []
        Rounded_BackupHospital_W_whhprime = []

        # Replicate the rounded values from scenario 0 to all scenarios in one loop
        for _ in self.scenario_set:
            # For x, we need a copy of the 1D list
            Rounded_ACFEstablishment_x_wi.append(Rounded_ACFEstablishment_x_i[:])
            # For thetaVar, which is 2D, make a deep copy of each row
            Rounded_LandRescueVehicle_thetaVar_wim.append([row[:] for row in Rounded_LandRescueVehicle_thetaVar_im])
            # For w, which is 2D, similarly make a deep copy
            Rounded_BackupHospital_W_whhprime.append([row[:] for row in Rounded_BackupHospital_W_hhprime])


        self.fix_First_Stage_Variables(Rounded_ACFEstablishment_x_wi, 
                                       Rounded_LandRescueVehicle_thetaVar_wim, 
                                       Rounded_BackupHospital_W_whhprime)
        fixed_solution = self.solve_fixed_mip()

        # Initialize the current solution, best solution, and global best solution.
        self.current_solution_x = copy.deepcopy(fixed_solution.ACFEstablishment_x_wi)
        self.current_solution_thetaVar = copy.deepcopy(fixed_solution.LandRescueVehicle_thetaVar_wim)
        self.current_solution_w = copy.deepcopy(fixed_solution.BackupHospital_W_whhPrime)
        self.best_solution = fixed_solution  # Current accepted solution
        self.best_cost = copy.deepcopy(fixed_solution.GRBCost)  # Cost of the current accepted solution
        # Global best solution (record of the best found overall)
        self.global_best_solution = fixed_solution
        self.global_best_cost = self.best_cost

        print(f"Initial solution cost: {self.best_cost}")

        # Initialize parameters for Simulated Annealing
        T = 1000.0           # Initial annealing temperature (adjust as needed)
        cooling_rate = 0.95  # Cooling rate (e.g., 0.99, adjust as needed)

        # Step 3: ALNS iterations
        self.CurrentIteration = 1
        no_improv_iters = 0

        while self.CurrentIteration <= self.MaxIterations and no_improv_iters < Constants.max_no_improv:
            # Check elapsed time
            elapsed_time = time.time() - self.StartTime_ALNS
            if elapsed_time >= Constants.AlgorithmTimeLimit:
                print(f"Time limit reached ({elapsed_time:.2f} seconds). Stopping ALNS.")
                break

            print(f"\n--- Iteration {self.CurrentIteration} ---")
            # Deep copy to preserve old state (for RL update if needed)
            old_x_solution = copy.deepcopy(self.current_solution_x[0])
            old_thetaVar_solution = copy.deepcopy(self.current_solution_thetaVar[0])
            old_w_solution = copy.deepcopy(self.current_solution_w[0])
            
            old_state = self.get_state_from_solution(old_x_solution, old_thetaVar_solution, old_w_solution)

            # Decide which operator to use and the number of changes
            combined_action, binary_destroy_action, binary_repair_action, integer_destroy_action, integer_repair_action, num_changes \
            = self.destroy_and_repair(self.current_solution_x, self.current_solution_thetaVar, self.current_solution_w)

            # Apply the chosen operator to a deep copy of the current solution
            modified_solution_x, \
            modified_solution_thetaVar, \
            modified_solution_w = self.Apply_Action(copy.deepcopy(self.current_solution_x), 
                                                                                    copy.deepcopy(self.current_solution_thetaVar),
                                                                                    copy.deepcopy(self.current_solution_w),
                                                                                    binary_destroy_action, 
                                                                                    binary_repair_action, 
                                                                                    integer_destroy_action, 
                                                                                    integer_repair_action, 
                                                                                    num_changes)

            # Fix modified solution and solve the MIP
            self.fix_First_Stage_Variables(modified_solution_x, 
                                           modified_solution_thetaVar, 
                                           modified_solution_w)
            fixed_solution = self.solve_fixed_mip()
            new_cost = copy.deepcopy(fixed_solution.GRBCost)

            # --- RL Update or Operator Score Update ---
            if self.use_RL:
                new_state = self.get_state_from_solution(modified_solution_x[0], modified_solution_thetaVar[0], modified_solution_w[0])
                reward = self.best_cost - new_cost
                if isinstance(self.RL_Agent, DQLAgent):
                    done = False  # Set to True if the episode ends, otherwise False
                    self.RL_Agent.update_q_value(old_state, combined_action, reward, new_state, done)
                else:
                    self.RL_Agent.update_q_value(old_state, combined_action, reward, new_state)
            else:
                # Update operator score for roulette selection if improvement is observed.
                reward = self.best_cost - new_cost
                if reward > 0:
                    self.operator_scores[combined_action] += reward

            # --- Acceptance Criterion Using Metropolis (Simulated Annealing) ---
            # Update global best if the new solution is strictly better.
            if new_cost < self.global_best_cost:
                print(f"New global best cost found: {new_cost}")
                self.global_best_cost = new_cost
                self.global_best_solution = fixed_solution
                no_improv_iters = 0
            else:
                no_improv_iters += 1

            # Optionally print when hitting the no‐improv threshold
            if no_improv_iters >= Constants.max_no_improv:
                print(f"No improvement in {Constants.max_no_improv} iterations. Stopping ALNS.")
                break 

            # Use the acceptance_check method to decide whether to accept the new solution for further exploration.
            self.acceptance_check(new_cost, fixed_solution, T)

            # Log iteration details including elapsed time
            trace_message = (
                "Iteration: %r, combined_action: %r, new_cost: %r, current_cost: %r, global_best_cost: %r, NoImprove: %r, elapsed_time: %.2f seconds\n"
                % (
                    self.CurrentIteration,
                    combined_action,
                    new_cost,
                    self.best_cost,
                    self.global_best_cost,
                    no_improv_iters,
                    elapsed_time,
                )
            )
            self.WriteInTraceFile(trace_message)

            # Update operator weights every 'segment_length' iterations if not using RL.
            if (not self.use_RL) and (self.CurrentIteration % self.segment_length == 0):
                self.update_operator_weights()

            # Update annealing temperature according to the cooling schedule.
            T *= cooling_rate

            # Increment iteration count
            self.CurrentIteration += 1

        # Up to now, for time-saving purposes, We have not saved the second-stage variables' values
        # Now, we only do it once, onstead of doing it every iteration in ALNS
        Constants.Obtain_SecondStage_Solution = True
        self.fix_First_Stage_Variables(self.global_best_solution.ACFEstablishment_x_wi, 
                                       self.global_best_solution.LandRescueVehicle_thetaVar_wim, 
                                       self.global_best_solution.BackupHospital_W_whhPrime)
        self.global_best_solution = self.solve_fixed_mip()
        completion_message = (
            f"ALNS completed. Global best cost: {self.global_best_cost}, Total time: {elapsed_time:.2f} seconds\n"
        )
        self.WriteInTraceFile(completion_message)
        print(completion_message)
        
        return self.global_best_solution