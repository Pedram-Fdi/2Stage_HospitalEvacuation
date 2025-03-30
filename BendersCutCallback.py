import gurobipy as gp
from gurobipy import GRB
from  Constants import Constants
import os

class BendersCutCallback:
    def __init__(self,
                 instance,
                 testidentifier,
                 scenarioTree,
                 master, 
                 sub, 
                 startFacilityEstablishmentVariables,
                 startthetaVariables,
                 theta_Var, 
                 facility_Establishment_Var,
                 facilityCapacityInfo, 
                 demandFlowInfo
                 ):

        self.Instance = instance
        self.TestIdentifier = testidentifier
        self.ScenarioTree = scenarioTree
        self.NrScenario = self.ScenarioTree.TreeStructure[1]
        self.ScenarioProbability = (1/self.NrScenario)    
        self.ScenarioSet = range(self.NrScenario)  

        self.master = master
        self.sub = sub

        self.StartFacilityEstablishmentVariables = startFacilityEstablishmentVariables
        self.StartthetaVariables = startthetaVariables
        self.theta_Var = theta_Var
        self.Facility_Establishment_Var = facility_Establishment_Var
        self.Facility_Establishment_CorePoint = [0 for i in self.Instance.FacilitySet] 

        # Unpack the facility capacity constraint information
        self.duals_FacilityCapacity = [[[0 
                                        for i in self.Instance.FacilitySet] 
                                        for t in self.Instance.TimeBucketSet]         
                                        for w in self.ScenarioSet] #it going to be [w][t][i]        
        self.FacilityCapacityConstraint_Objects = facilityCapacityInfo["constraints"]
        self.PIFacilityCapacityConstraint_Names = facilityCapacityInfo["pi_names"]
        self.Concerned_t_FacilityCapacityConstraint = facilityCapacityInfo["concerned_t"]
        self.Concerned_i_FacilityCapacityConstraint = facilityCapacityInfo["concerned_i"]
        
        # Unpack the demand flow constraint information
        self.duals_DemandFlow = [[[0 
                                        for j in self.Instance.DemandSet] 
                                        for t in self.Instance.TimeBucketSet] #it going to be [t][i]        
                                        for w in self.ScenarioSet] #it going to be [w][t][i]        
        self.DemandFlowConstraint_Objects = demandFlowInfo["constraints"]
        self.PIDemandFlowConstraint_Names = demandFlowInfo["pi_names"]
        self.Concerned_t_DemandFlowConstraint = demandFlowInfo["concerned_t"]
        self.Concerned_j_DemandFlowConstraint = demandFlowInfo["concerned_j"]

        self.TraceFileName = "./Temp/BBCtrace_%s_Evaluation_%s.txt" % (self.TestIdentifier.GetAsString(), Constants.Evaluation_Part)
        self.theta_Var_Values_Warm = {}

    def update_CorePoint_Facility_Establishment_Var(self, i, current_value, previous_value, core_point_coeff):
        """
        Update the core point using the convex combination:
        New CorePoint = CorePointCoeff * CurrentValue + (1 - CorePointCoeff) * PreviousValue

        Args:
            i (int): Facility index
            current_value (float): Current decision variable value from master problem
            previous_value (float): Previous core point value
            core_point_coeff (float): Multiplier for the convex combination
        """
        new_core_point = core_point_coeff * current_value + (1 - core_point_coeff) * previous_value
        self.Facility_Establishment_CorePoint[i] = new_core_point

    def update_FacilityCapacityConstraint_rhs(self, model):
        """Update the right-hand side of the capacity constraints based on the current master solution."""
        if(Constants.GenerateStrongCut):
            for t in self.Instance.TimeBucketSet:
                for i in self.Instance.FacilitySet:
                    CurrentValue = model.cbGetSolution(self.Facility_Establishment_Var[i])
                    PreviousValue = self.Facility_Establishment_CorePoint[i]
                    # Update the core point value
                    self.update_CorePoint_Facility_Establishment_Var(i, CurrentValue, PreviousValue, Constants.CorePointCoeff)                    
                    self.FacilityCapacityConstraint_Objects[t][i].RHS = -1.0 * (self.Instance.Facility_Capacity[i] * model.cbGetSolution(self.Facility_Establishment_Var[i])
                                                                                + Constants.Sherali_Multiplier * (self.Instance.Facility_Capacity[i] * self.Facility_Establishment_CorePoint[i]))
        else:
            for t in self.Instance.TimeBucketSet:
                for i in self.Instance.FacilitySet:
                    self.FacilityCapacityConstraint_Objects[t][i].RHS = -1.0 * self.Instance.Facility_Capacity[i] * model.cbGetSolution(self.Facility_Establishment_Var[i])

    def update_FacilityCapacityConstraint_rhs_Warm(self):
        """Update the right-hand side of the capacity constraints based on the current master solution."""
        for t in self.Instance.TimeBucketSet:
            for i in self.Instance.FacilitySet:
                index_var = self.GetIndexFacilityEstablishmentVariable(i)
                self.FacilityCapacityConstraint_Objects[t][i].RHS = -1.0 * self.Instance.Facility_Capacity[i] * self.Facility_Establishment_Var[index_var].X

    def update_DemandFlowConstraint_rhs(self, w):
        """Update the right-hand side of the demand flow constraints for scenario w."""
        for t in self.Instance.TimeBucketSet:
            for j in self.Instance.DemandSet:
                self.DemandFlowConstraint_Objects[t][j].RHS = self.ScenarioTree.Demand[w][t][j] 

    def write_lp_file(self, model, model_name, w):
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
            lp_filename = f"{model_name}__T_{T}_I_{I}_J_{J}_Scenario_{w}.lp"
            
            # Full path for the .lp file
            lp_full_path = os.path.join(lp_dir, lp_filename)

            # Write the model to the .lp file
            model.write(lp_full_path)

    def solve_subproblem(self, w):
        """Solve the subproblem after updating its constraints."""
        if Constants.Debug_lp_Files:
            self.write_lp_file(self.sub, "SubModel", w)
        self.sub.update()
        self.sub.optimize()

    def get_dual_FacilityCapacity_Const(self, w):
        # Reset all dual values to zero.
        for t in self.Instance.TimeBucketSet:
            for i in self.Instance.FacilitySet:
                self.duals_FacilityCapacity[w][t][i] = 0.0

        # Retrieve the duals from the constraints
        duals = [self.sub.getConstrByName(name).Pi for name in self.PIFacilityCapacityConstraint_Names]

        for du in range(len(duals)):
            t = self.Concerned_t_FacilityCapacityConstraint[du]
            i = self.Concerned_i_FacilityCapacityConstraint[du]
            self.duals_FacilityCapacity[w][t][i] = duals[du]

    def get_dual_Demandflow_Const(self, w):

        # Reset all dual values to zero.
        for t in self.Instance.TimeBucketSet:
            for j in self.Instance.DemandSet:
                self.duals_DemandFlow[w][t][j] = 0.0

        # Retrieve the duals from the constraints
        duals = [self.sub.getConstrByName(name).Pi for name in self.PIDemandFlowConstraint_Names]

        for du in range(len(duals)):
            t = self.Concerned_t_DemandFlowConstraint[du]
            j = self.Concerned_j_DemandFlowConstraint[du]
            self.duals_DemandFlow[w][t][j] = duals[du]

    def get_dual_variables(self, w):
        """
        Retrieve the dual variables from the subproblem's constraints.
        """
        self.get_dual_FacilityCapacity_Const(w)
        self.get_dual_Demandflow_Const(w)

    def GetIndexthetaVariable(self, w):
        if(Constants.UseMultiCut):
            return self.StartthetaVariables + w
        else:
            return self.StartthetaVariables
    
    def GetIndexFacilityEstablishmentVariable(self, i):
        # if Constants.Debug: print("We are in 'MIPSolver' Class -- GetIndexFacilityEstablishmentVariable")
        return self.StartFacilityEstablishmentVariables + i  
          
    def is_cut_violated(self, model, w, tol=1e-6):

        index = self.GetIndexthetaVariable(w)
        is_Violated = self.sub.objVal > model.cbGetSolution(self.theta_Var[index]) + tol
        return is_Violated

    def is_cut_violated_Warm(self, w, tol=1e-6):

        index = self.GetIndexthetaVariable(w)
        is_Violated = self.sub.objVal > self.theta_Var_Values_Warm[index] + tol
        return is_Violated
    
    def construct_multiple_cuts(self, w):
        # Coefficients from the capacity constraints:
        xcoef = {}
        ## Calculating the Coefficient of (x) in the optimality cut with out summation over [i] (YOU SUM OVER EVERY Dual's INDEX EXCEPT THE ones in common with the VARIABLE INDEX, here for example except i, in fact, you sum over (i) in add_constraint method)
        for i in self.Instance.FacilitySet:
            xcoef[i] = 0.0  # Initialize the summation for facility i
            for t in self.Instance.TimeBucketSet:
                xcoef[i] += -1.0 * self.Instance.Facility_Capacity[i] * self.duals_FacilityCapacity[w][t][i]


        # Right-hand side from the demand constraints:
        rhs = 0.0
        for t in self.Instance.TimeBucketSet:
            for j in self.Instance.DemandSet:
                rhs += self.ScenarioTree.Demand[w][t][j] * self.duals_DemandFlow[w][t][j]

        return xcoef, rhs

    def add_multiple_cuts_to_master(self, model, xcoef, rhs, w):
        """
        Add the derived optimality cut as a lazy constraint to the master problem.
        
        The cut is of the form: theta[s] - sum_p(xcoef[p]*fopen[p]) >= rhs.
        """
        index_theta = self.GetIndexthetaVariable(w)

        expr = self.theta_Var[index_theta] - gp.quicksum(xcoef[i] * self.Facility_Establishment_Var[self.GetIndexFacilityEstablishmentVariable(i)] for i in self.Instance.FacilitySet)
        model.cbLazy(expr >= rhs)
    
    def add_multiple_cuts_to_master_Warm(self, xcoef, rhs, w):

        index_theta = self.GetIndexthetaVariable(w)

        expr = self.theta_Var[index_theta] - gp.quicksum(xcoef[i] * self.Facility_Establishment_Var[self.GetIndexFacilityEstablishmentVariable(i)] for i in self.Instance.FacilitySet)
        print("rhs: ", rhs)
        self.master.addConstr(expr >= rhs)

    def construct_single_cut(self):

        xcoef = {}        
        ## Calculating the Coefficient of (x) in the optimality cut with out summation over [i] (YOU SUM OVER EVERY Dual's INDEX EXCEPT THE ones in common with the VARIABLE INDEX, here for example except i, in fact, you sum over (i) in add_constraint method)
        for i in self.Instance.FacilitySet:
            xcoef[i] = 0.0
            for w in self.ScenarioSet:
                for t in self.Instance.TimeBucketSet:
                    xcoef[i] += -1.0 * self.ScenarioProbability * self.Instance.Facility_Capacity[i] * self.duals_FacilityCapacity[w][t][i]
            
        ### Calculating the RHS of the Optimality Cut
        rhs = 0.0
        for w in self.ScenarioSet:
            for t in self.Instance.TimeBucketSet:
                for j in self.Instance.DemandSet:
                    rhs += self.ScenarioProbability * self.ScenarioTree.Demand[w][t][j] * self.duals_DemandFlow[w][t][j]


        return xcoef, rhs

    def add_single_cut_to_master(self, model, xcoef, rhs):
        """
        Add the derived single optimality cut as a lazy constraint to the master problem.
        
        The cut is of the form: theta >= sum_w self.ScenarioProbability * (objective_w - facility terms).
        """
        index_theta = self.GetIndexthetaVariable(0)
        expr = self.theta_Var[index_theta] - gp.quicksum(xcoef[i] * self.Facility_Establishment_Var[self.GetIndexFacilityEstablishmentVariable(i)]
                                                            for i in self.Instance.FacilitySet)
        model.cbLazy(expr >= rhs)
    
    def add_single_cut_to_master_Warm(self, xcoef, rhs):

        index_theta = self.GetIndexthetaVariable(0)
        expr = self.theta_Var[index_theta] - gp.quicksum(xcoef[i] * self.Facility_Establishment_Var[self.GetIndexFacilityEstablishmentVariable(i)]
                                                            for i in self.Instance.FacilitySet)
        self.master.addConstr(expr >= rhs)

    def __call__(self, model, where):
        """The callback method that will be invoked by the master model."""
        if where == GRB.Callback.MIPSOL:

            if self.check_stopping_criteria(model):
                print("Stopping criteria reached. Terminating model.")
                model.terminate()
                return
            
            # First, update the capacity constraints using the current master solution.
            self.update_FacilityCapacityConstraint_rhs(model)

            if (Constants.UseMultiCut):
                # Process each scenario in the subproblem.
                for w in self.ScenarioSet:
                    # Update the demand flow constraints based on each scenario
                    self.update_DemandFlowConstraint_rhs(w)
                    self.solve_subproblem(w)
                    self.get_dual_variables(w)

                    if self.is_cut_violated(model, w):
                        xcoef, rhs = self.construct_multiple_cuts(w)
                        self.add_multiple_cuts_to_master(model, xcoef, rhs, w)
            else:
                # Apply Single-Cut Version
                for w in self.ScenarioSet:
                    self.update_DemandFlowConstraint_rhs(w)
                    self.solve_subproblem(w)
                    self.get_dual_variables(w)

                # Construct and add a single aggregated cut
                xcoef, rhs = self.construct_single_cut()
                self.add_single_cut_to_master(model, xcoef, rhs)                

    def ChangeACFEstablishmentVarToContinuouse(self):

        for i in self.Instance.FacilitySet:

                Index_Var = self.GetIndexFacilityEstablishmentVariable(i)

                variable_to_update = self.Facility_Establishment_Var[Index_Var]

                variable_to_update.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)
                variable_to_update.setAttr(GRB.Attr.LB, 0.0)
                variable_to_update.setAttr(GRB.Attr.UB, 1.0)
                self.master.update()

    def ChangeACFEstablishmentVarToBinary(self):

        for i in self.Instance.FacilitySet:

                Index_Var = self.GetIndexFacilityEstablishmentVariable(i)

                variable_to_update = self.Facility_Establishment_Var[Index_Var]

                variable_to_update.setAttr(GRB.Attr.VType, GRB.BINARY)
                variable_to_update.setAttr(GRB.Attr.LB, 0.0)
                variable_to_update.setAttr(GRB.Attr.UB, 1.0)
                self.master.update()

    def WarmStart_Master_Model(self):
           #### Start Solving the Master problem linearly for a while and after that going for Branch-and-Benders Cut
            Warm_Iter = 1
            Total_WarmCut = 0
            cutfound = 1
            self.ChangeACFEstablishmentVarToContinuouse()
            while (Warm_Iter <= Constants.NrIterationWarmUp) and (cutfound == 1):
                print("------------- Warm Iter: ", Warm_Iter, " Cut added: ", Total_WarmCut)
                final_trace_str = ("\n------------- Warm Iter: {}, Cut added: {}, Total_WarmCut: {}\n").format(Warm_Iter, Total_WarmCut, Total_WarmCut)
                self.WriteInTraceFile(final_trace_str)

                Warm_Iter += 1
                cutfound = 0

                self.master.optimize()

                self.update_FacilityCapacityConstraint_rhs_Warm()
                if (Constants.UseMultiCut):
                    for w in self.ScenarioSet:
                        index = self.GetIndexthetaVariable(w)                    
                        # Store theta values using a dictionary
                        self.theta_Var_Values_Warm[index] = self.theta_Var[index].X
                    
                    for w in self.ScenarioSet:
                        # Update the demand flow constraints based on each scenario
                        self.update_DemandFlowConstraint_rhs(w)
                        self.solve_subproblem(w)
                        self.get_dual_variables(w)

                        if self.is_cut_violated_Warm(w):
                            xcoef, rhs = self.construct_multiple_cuts(w)
                            self.add_multiple_cuts_to_master_Warm(xcoef, rhs, w)
                            self.write_lp_file(self.master, "MasterWarm", w)
                            Total_WarmCut += 1
                            cutfound = 1
                else:
                    index = self.GetIndexthetaVariable(0)                    
                    # Store theta values using a dictionary
                    self.theta_Var_Values_Warm[index] = self.theta_Var[index].X                    
                    # Apply Single-Cut Version
                    for w in self.ScenarioSet:
                        self.update_DemandFlowConstraint_rhs(w)
                        self.solve_subproblem(w)
                        self.get_dual_variables(w)

                    # Construct and add a single aggregated cut
                    xcoef, rhs = self.construct_single_cut()
                    self.add_single_cut_to_master_Warm(xcoef, rhs)


            self.ChangeACFEstablishmentVarToBinary()

    def WriteInTraceFile(self, string):
        if Constants.Debug: print("\n We are in 'BranchandBendersCut' Class -- WriteInTraceFile")

        if Constants.PrintPHATrace:
            self.TraceFile = open(self.TraceFileName, "a")
            self.TraceFile.write(string)
            self.TraceFile.close()

    def check_stopping_criteria(self, model):
        """
        Checks if either the runtime exceeds the allowed time limit 
        or the optimality gap is smaller than the specified tolerance.
        Returns True if termination criteria are met, otherwise False.
        """
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        try:
            lower_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        except gp.GurobiError:
            lower_bound = float('inf')
        upper_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        
        # Prevent division by zero.
        if upper_bound == 0:
            current_gap = float('inf')
        else:
            current_gap = (upper_bound - lower_bound) / upper_bound

        node_count = model.cbGet(GRB.Callback.MIPSOL_NODCNT)

        # Prepare the string to write into the trace file including runtime, LB, UB, and gap.
        trace_str = ("Runtime = {:.2f} sec, LB = {:.4f}, "
                    "UB = {:.4f}, Gap = {:.2%}, ExploredNodes: {}\n"
                    ).format(runtime, lower_bound, upper_bound, current_gap, node_count)
        
        # Write the string into the trace file.
        self.WriteInTraceFile(trace_str)
        
        if runtime >= Constants.AlgorithmTimeLimit or current_gap < Constants.My_EpGap_BBC:
            return True
        return False