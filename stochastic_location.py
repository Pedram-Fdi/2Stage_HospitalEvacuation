import numpy as np
import gurobipy as gp
from gurobipy import GRB


# Class for Stochastic Location Allocation problem
class StochasticLocationAllocation:

    def __init__(self, I, J, S, random_params):
        """
        Initialize the problem instance with pre-generated random parameters.
        :param I: Number of facilities.
        :param J: Number of demand locations.
        :param S: Number of demand scenarios.
        :param random_params: Pre-generated random parameters as a dictionary.
        """
        self.I = I
        self.J = J
        self.S = S

        # Use pre-generated random parameters
        self.fixed_costs = random_params["fixed_costs"]
        self.facility_capacity = random_params["facility_capacity"]
        self.transport_costs = random_params["transport_costs"]
        self.penalty_cost = random_params["penalty_cost"]
        self.demands = random_params["demands"]

        # Gurobi model instance
        self.model = None
        self.variables = {}  # Hold variable references for easy access

    def build_model(self):
        """
        Build the integrated Gurobi model for MIP and ALNS solvers.
        """
        self.model = gp.Model("StochasticLocationAllocation")

        # Decision variables
        for i in range(self.I):
            self.variables[f"x_{i}"] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}")

        for i in range(self.I):
            for j in range(self.J):
                for s in range(self.S):
                    index = f"y_{i}_{j}_{s}"
                    self.variables[index] = self.model.addVar(vtype=GRB.CONTINUOUS, name=index)

        for j in range(self.J):
            for s in range(self.S):
                index = f"z_{j}_{s}"
                self.variables[index] = self.model.addVar(vtype=GRB.CONTINUOUS, name=index)

        # Probability of each scenario
        scenario_prob = 1 / self.S

        # Objective function: Minimize fixed costs + transport + penalties
        obj = gp.quicksum(self.fixed_costs[i] * self.variables[f"x_{i}"] for i in range(self.I)) \
              + scenario_prob * gp.quicksum(self.transport_costs[i, j] * self.variables[f"y_{i}_{j}_{s}"]
                                            for i in range(self.I)
                                            for j in range(self.J)
                                            for s in range(self.S)) \
              + scenario_prob * gp.quicksum(self.penalty_cost * self.variables[f"z_{j}_{s}"]
                                            for j in range(self.J)
                                            for s in range(self.S))

        self.model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        for j in range(self.J):
            for s in range(self.S):
                self.model.addConstr(
                    gp.quicksum(self.variables[f"y_{i}_{j}_{s}"] for i in range(self.I)) + self.variables[f"z_{j}_{s}"] == self.demands[j, s],
                    name=f"DemandConstr_{j}_{s}")

        for i in range(self.I):
            for s in range(self.S):
                self.model.addConstr(
                    gp.quicksum(self.variables[f"y_{i}_{j}_{s}"] for j in range(self.J)) <= self.facility_capacity[i] * self.variables[f"x_{i}"],
                    name=f"CapacityConstr_{i}_{s}")

        self.model.update()

    def build_model_for_scenario(self, scenario_index):
        """
        Build a Gurobi model for a specific scenario.
        :param scenario_index: Index of the scenario to build the model for.
        """
        model = gp.Model(f"StochasticLocationAllocation_Scenario_{scenario_index}")

        # First-stage decision variables (binary facility decisions)
        x_vars = {i: model.addVar(vtype=GRB.BINARY, name=f"x[{i}]") for i in range(self.I)}

        # Second-stage decision variables (continuous allocation and shortages)
        y_vars = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, name=f"y[{i},{j}]_{scenario_index}") for i in range(self.I) for j in range(self.J)}
        z_vars = {j: model.addVar(vtype=GRB.CONTINUOUS, name=f"z[{j}]_{scenario_index}") for j in range(self.J)}

        # Objective function
        obj = (gp.quicksum(self.fixed_costs[i] * x_vars[i] for i in range(self.I)) +
                gp.quicksum(self.transport_costs[i, j] * y_vars[i, j] for i in range(self.I) for j in range(self.J)) +
                gp.quicksum(self.penalty_cost * z_vars[j] for j in range(self.J)))

        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints
        for j in range(self.J):
            model.addConstr(
                gp.quicksum(y_vars[i, j] for i in range(self.I)) + z_vars[j] == self.demands[j, scenario_index],
                name=f"DemandConstr_{j}_{scenario_index}")

        for i in range(self.I):
            model.addConstr(
                gp.quicksum(y_vars[i, j] for j in range(self.J)) <= self.facility_capacity[i] * x_vars[i],
                name=f"CapacityConstr_{i}_{scenario_index}")

        model.update()
        return model

    def solve_mip(self):
        lp_filename = f"MIPModel.lp"
        self.model.write(lp_filename)

        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal solution found with objective value: {self.model.objVal}")
        else:
            print(f"Model was not solved to optimality. Status: {self.model.status}")

    def show_variable_values(self):
        """
        Display the final values of decision variables after optimization.
        """
        if self.model.status != GRB.OPTIMAL:
            print("Model is not solved to optimality. Cannot display variable values.")
            return
        
        print("\nobj: ", self.model.ObjVal)

        print("\nFinal Variable Values:")
        # Display x variables
        print("First-stage decision variables (x):")
        for i in range(self.I):
            value = self.variables[f"x_{i}"].X
            if value != 0:
                print(f"x_{i} = {value}")

        # Display y variables
        print("\nSecond-stage decision variables (y):")
        for i in range(self.I):
            for j in range(self.J):
                for s in range(self.S):
                    var_name = f"y_{i}_{j}_{s}"
                    value = self.variables[var_name].X
                    if value != 0:
                        print(f"{var_name} = {value}")

        # Display z variables
        print("\nShortage variables (z):")
        for j in range(self.J):
            for s in range(self.S):
                var_name = f"z_{j}_{s}"
                value = self.variables[var_name].X
                if value != 0:
                    print(f"{var_name} = {value}")

    def solve_relaxed_mip(self):
        # Set x_i as continuous
        for i in range(self.I):
            self.variables[f"x_{i}"].setAttr(GRB.Attr.VType, GRB.CONTINUOUS)

        # Solve the relaxed model
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            # Get and round the solution for x_i
            x_solution = {i: round(self.variables[f"x_{i}"].X) for i in range(self.I)}
            print("Relaxed x_i solution (rounded):", x_solution)
            return x_solution
        else:
            print(f"Relaxed model was not solved to optimality. Status: {self.model.status}")
            return {i: 0 for i in range(self.I)}  # Default to all facilities closed

    def fix_x_variables(self, x_solution):
        # Fix x_i to the rounded values and convert to continuous for later optimization
        for i in range(self.I):
            var = self.variables[f"x_{i}"]
            var.setAttr(GRB.Attr.LB, x_solution[i])
            var.setAttr(GRB.Attr.UB, x_solution[i])
            var.setAttr(GRB.Attr.VType, GRB.CONTINUOUS)
        self.model.update()

    def solve_fixed_mip(self):
        # Solve the MIP with fixed x values
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            print(f"Fixed MIP solved with objective value: {self.model.objVal}")
            return self.model.objVal
        else:
            print(f"Fixed MIP was not solved to optimality. Status: {self.model.status}")
            return float('inf')  # Assign a high cost if not solved

