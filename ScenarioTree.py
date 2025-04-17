import numpy as np
from scipy.stats import qmc
from Scenario import Scenario
from Constants import Constants
from sklearn.cluster import KMeans
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import copy
import pprint

class ScenarioTree:
    
    def __init__(self, 
                 instance, 
                 tree_structure, 
                 scenario_seed, 
                 givenscenarioset = False,
                 CopyscenariofromMulti_Stage = False,
                 evaluationscenario = False, 
                 averagescenariotree=False, 
                 scenariogenerationmethod="MC"):
        """
        Initialize the ScenarioTree object for generating uniformly distributed uncertain parameters.

        Parameters:
        - instance: Contains dimensions like time buckets, locations, scenarios, etc.
        - tree_structure: The scenario tree structure (not used in two-stage, kept for flexibility).
        - scenario_seed: Seed for reproducibility.
        - averagescenariotree: Flag for deterministic average-based scenario generation.
        - scenariogenerationmethod: "MC", "QMC", or "RQMC" for Monte Carlo, Quasi-Monte Carlo, or Randomized QMC.
        """
        self.Instance = instance
        self.TreeStructure = tree_structure
        self.ScenarioSeed = scenario_seed
        self.AverageScenarioTree = averagescenariotree
        self.EvaluationScenrio = evaluationscenario
        self.ScenarioGenerationMethod = scenariogenerationmethod

        # Initialize variables for storing generated uncertain parameters
        self.CasualtyDemand = None  
        self.HospitalDisruption = None  
        self.PatientDemand = None  
        self.PatientDischargedPercentage = None 
        self.CopyscenariofromMulti_Stage = CopyscenariofromMulti_Stage

        if self.CopyscenariofromMulti_Stage:
            ### This part is only used in PHA to separate scenarios from each other, while all methods (including PHA) starts from the "else" part!
            self.GenerateDemandToFollowFromScenarioSet(givenscenarioset)
        else:
            nrscenario = tree_structure[1]
            self.Probability = (1/nrscenario)

            ###################### Generate Casualty demand scenarios
            CasualtyDemand_param_dim = (self.Instance.NrTimeBucket, self.Instance.NrInjuries, self.Instance.NrDisasterAreas)
            self.CasualtyDemand = self.generate_uncertain_parameter_scenarios(param_dim = CasualtyDemand_param_dim, 
                                                                              Avg = self.Instance.ForecastedAvgCasualtyDemand, 
                                                                              STD = self.Instance.ForecastedSTDCasualtyDemand,
                                                                              Clustering = Constants.ClusteringMethod)            
            self.CasualtyDemand_LBF = self.compute_average_uncertain_parameter_scenario(self.CasualtyDemand, rounding='float')
            if self.AverageScenarioTree:
                self.CasualtyDemand = self.compute_average_uncertain_parameter_scenario(self.CasualtyDemand, rounding='int')

            ###################### Generate Patient Discharged Percentage scenarios
            HospitalDisruption_param_dim = (self.Instance.NrHospitals,)
            self.HospitalDisruption = self.generate_uncertain_parameter_scenarios(param_dim = HospitalDisruption_param_dim, 
                                                                                  Avg = self.Instance.ForecastedAvgHospitalDisruption, 
                                                                                  STD = self.Instance.ForecastedSTDHospitalDisruption, 
                                                                                  rounding='int',
                                                                                  Clustering = Constants.ClusteringMethod)          
            self.HospitalDisruption_LBF = self.compute_average_uncertain_parameter_scenario(self.HospitalDisruption, rounding='float')
            if self.AverageScenarioTree:
                self.HospitalDisruption = self.compute_average_uncertain_parameter_scenario(self.HospitalDisruption, rounding='int')

            ###################### Generate Patient demand scenarios
            PatientDemand_param_dim = (self.Instance.NrInjuries, self.Instance.NrHospitals)
            self.PatientDemand = self.generate_uncertain_parameter_scenarios(param_dim = PatientDemand_param_dim, 
                                                                             Avg = self.Instance.ForecastedAvgPatientDemand, 
                                                                             STD = self.Instance.ForecastedSTDPatientDemand,
                                                                             Clustering = Constants.ClusteringMethod)            
            self.PatientDemand_LBF = self.compute_average_uncertain_parameter_scenario(self.PatientDemand, rounding='float')
            if self.AverageScenarioTree:
                self.PatientDemand = self.compute_average_uncertain_parameter_scenario(self.PatientDemand, rounding='int')

            ###################### Generate Patient Discharged Percentage scenarios
            PatientDischargedPercentage_param_dim = (self.Instance.NrTimeBucket, self.Instance.NrInjuries, self.Instance.NrMedFacilities)
            self.PatientDischargedPercentage = self.generate_uncertain_parameter_scenarios(param_dim = PatientDischargedPercentage_param_dim, 
                                                                                           Avg = self.Instance.ForecastedAvgPercentagePatientDischarged, 
                                                                                           STD = self.Instance.ForecastedSTDPercentagePatientDischarged, 
                                                                                           rounding='float',
                                                                                           non_negative=True,
                                                                                           Clustering = Constants.ClusteringMethod)            
            self.PatientDischargedPercentage_LBF = self.compute_average_uncertain_parameter_scenario(self.PatientDischargedPercentage, rounding='float')
            if self.AverageScenarioTree:
                self.PatientDischargedPercentage = self.compute_average_uncertain_parameter_scenario(self.PatientDischargedPercentage, rounding='float')

            if (Constants.Evaluation_Part == False) and (Constants.ClusteringMethod == 'DB'):
                self.Scenario_DB = copy.copy(self)
                selected_indices = self._decision_based_reduction(nrscenario, 
                                                                    self.Scenario_DB,
                                                                    self.CasualtyDemand,
                                                                    self.HospitalDisruption,
                                                                    self.PatientDemand,
                                                                    self.PatientDischargedPercentage)
                if(Constants.Debug):print("selected_indices: ", selected_indices)
                ## Keeping only selected scenarios in the generated ones!
                self.CasualtyDemand               = self.CasualtyDemand[selected_indices, ...]
                self.HospitalDisruption           = self.HospitalDisruption[selected_indices, ...]
                self.PatientDemand                = self.PatientDemand[selected_indices, ...]
                self.PatientDischargedPercentage  = self.PatientDischargedPercentage[selected_indices, ...]
                
    def generate_uncertain_parameter_scenarios(self, param_dim, Avg, STD, rounding='int', 
                                               non_negative=False, decimal_places = 3,
                                               Clustering = Constants.ClusteringMethod):
        """
        Generates uncertain parameter scenarios with flexibility to handle different dimensions.

        :param param_dim: Tuple representing the shape of the parameter (e.g., (T, L) or (T, I, L)).
        :param rounding: str, choose 'int' (default) for integer output, 'float' for raw float values.
        :param non_negative: bool, if True, ensures all generated values are non-negative.
        :param decimal_places: int, number of decimal places for floating-point output (default=2).
        :return: Generated uncertain parameter scenarios in the given shape.
        """

        if Clustering != 'DB':
            num_scenarios = self.TreeStructure[1]  # Final number of representative scenarios
        else:
            num_scenarios = Constants.Multiplier_NumberofOriginalScenarios * self.TreeStructure[1]

        # If no clustering is selected, generate scenarios directly
        if Clustering == 'NoC':
            scenarios = self.generate_samples(num_scenarios=num_scenarios,
                                                param_dim=param_dim,
                                                Average=Avg,
                                                StandardDeviation=STD,
                                                sampling_method=self.ScenarioGenerationMethod)
        elif Clustering == 'DB':
            scenarios = self.generate_samples(num_scenarios=num_scenarios,
                                            param_dim=param_dim,
                                            Average=Avg,
                                            StandardDeviation=STD,
                                            sampling_method=self.ScenarioGenerationMethod)                        
        else:  # If clustering is selected (K-Means, K-Means++, or SOM)
            num_original_scenarios = Constants.Multiplier_NumberofOriginalScenarios * num_scenarios
            # Step 1: Generate a large number of scenarios
            original_scenarios = self.generate_samples(num_scenarios=num_original_scenarios,
                                                        param_dim=param_dim,
                                                        Average=Avg,
                                                        StandardDeviation=STD,                
                                                        sampling_method=self.ScenarioGenerationMethod).reshape(num_original_scenarios, -1)  # Flatten each scenario for clustering
            # Step 2: Apply Clustering Based on User Selection
            if Clustering == 'KMPP':  # K-Means++
                kmeans = KMeans(n_clusters=num_scenarios, init="k-means++", n_init=10, random_state=self.ScenarioSeed)
                kmeans.fit(original_scenarios)
                cluster_centers = kmeans.cluster_centers_

            elif Clustering == 'KM':  # K-Means
                kmeans = KMeans(n_clusters=num_scenarios, init="random", n_init=10, random_state=self.ScenarioSeed)
                kmeans.fit(original_scenarios)
                cluster_centers = kmeans.cluster_centers_

            elif Clustering == 'SOM':  # **Self-Organizing Map (SOM)**
                np.random.seed(self.ScenarioSeed)  # Set the seed for reproducibility

                # Step 2.1: Scale the features (SOM works better with normalized data)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_features = scaler.fit_transform(original_scenarios)

                print("----- Applying SOM for Scenario Reduction -----")

                # Step 2.2: Train the SOM with a fixed seed
                som = MiniSom(x=num_scenarios, y=1, input_len=len(scaled_features[0]), sigma=1.0, learning_rate=0.5, random_seed=self.ScenarioSeed)
                som.random_weights_init(scaled_features)
                som.train_random(scaled_features, num_iteration=100)

                # Step 2.3: Assign each scenario to a SOM cluster
                winner_coordinates = np.array([som.winner(x) for x in scaled_features])
                winner_indices = np.ravel_multi_index(winner_coordinates.T, (num_scenarios, 1))

                # Step 2.4: Select representative scenarios
                selected_scenarios = []
                for i in range(num_scenarios):
                    indices_in_cluster = np.where(winner_indices == i)[0]
                    if len(indices_in_cluster) > 0:
                        selected_scenarios.append(indices_in_cluster[0])  # Take the first scenario in each cluster
                    else:
                        # If a cluster is empty, find the closest scenario to the neuron
                        closest_scenario = np.argmin([np.linalg.norm(scaled_features - som.get_weights()[i]) for _ in scaled_features])
                        selected_scenarios.append(closest_scenario)

                # Ensure we select exactly `num_scenarios`
                selected_scenarios = selected_scenarios[:num_scenarios]
                cluster_centers = original_scenarios[selected_scenarios]  # Use selected scenarios
            else:
                raise ValueError(f"Unsupported clustering method: {Clustering}")

            # Step 3: Assign the clustered parameters as final scenarios
            scenarios = cluster_centers  # Keep as float before rounding decision

        # Step 4: Convert to integer if requested
        if rounding == 'int':
            scenarios = np.round(scenarios).astype(int)  # Convert to integers like before

        #**Ensure non-negative values if required**
        if non_negative:
            scenarios = np.maximum(scenarios, 0)

        #**Ensure only up to `decimal_places` decimals if rounding='float'**
        if rounding == 'float':
            scenarios = np.round(scenarios, decimal_places)

        # Step 5: Reshape `scenarios` to match the expected shape `param_dim`
        return scenarios.reshape(num_scenarios, *param_dim)


    def _decision_based_reduction(self,
                                nrscenario: int,
                                tree_for_db,
                                casualty_pool: np.ndarray,
                                hospital_pool: np.ndarray,
                                patient_pool: np.ndarray,
                                discharged_pool: np.ndarray):
        from MIPSolver import MIPSolver

        ########### Start obtainng the Model Solutions for each Scenario
        self.solutions_DB = []
        N = casualty_pool.shape[0]   # e.g. 10
        for i in range(N):
            # 1) overwrite the tree’s attributes so it now holds *only* scenario i
            tree_for_db.CasualtyDemand               = casualty_pool[i : i+1]
            tree_for_db.HospitalDisruption           = hospital_pool[i : i+1]
            tree_for_db.PatientDemand                = patient_pool[i : i+1]
            tree_for_db.PatientDischargedPercentage  = discharged_pool[i : i+1]

            # 2) solve exactly that single‐scenario tree
            MIPSolver_DB = MIPSolver(instance = self.Instance,
                                        model= Constants.Two_Stage,
                                        scenariotree = tree_for_db,
                                        nrscenario = 1,
                                        linearRelaxation = True,
                                        logfile = "NO")
            MIPSolver_DB.BuildModel()
            self.solutions_DB.append(MIPSolver_DB.Solve(True))

        ########### Start evaluating the solution obtained by each scenario for other scenarios
        self.eval_results = [[None]*N for _ in range(N)]
        for i in range(N):
            # 1) overwrite the tree’s attributes so it now holds *only* scenario i
            for j in range(N):
                if j != i:
                    tree_for_db.CasualtyDemand               = casualty_pool[j : j+1]
                    tree_for_db.HospitalDisruption           = hospital_pool[j : j+1]
                    tree_for_db.PatientDemand                = patient_pool[j : j+1]
                    tree_for_db.PatientDischargedPercentage  = discharged_pool[j : j+1]

                    MIPSolver_Eval = MIPSolver( instance = self.Instance, 
                                                model = Constants.Two_Stage, 
                                                scenariotree = tree_for_db,
                                                nrscenario = 1,
                                                evaluatesolution=True,
                                                givenACFEstablishment=self.solutions_DB[i].ACFEstablishment_x_wi,
                                                givenNrLandRescueVehicle=self.solutions_DB[i].LandRescueVehicle_thetaVar_wim,
                                                givenBackupHospital=self.solutions_DB[i].BackupHospital_W_whhPrime,
                                                linearRelaxation = True)                       
                    MIPSolver_Eval.BuildModel()
                    Solution_Eval = MIPSolver_Eval.Solve(True)
                    self.eval_results[i][j] = Solution_Eval.GRBCost

        # Now build the dis‐similarity matrix
        self.dissimilarity = [[None]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                # you may choose to leave D[i,i]=0 or np.nan
                if i == j:
                    self.dissimilarity[i][j] = 0
                else:
                    self.dissimilarity[i][j] = (self.eval_results[i][j] + self.eval_results[j][i])
                    

        # convert to NumPy array for ease of passing around
        D = np.array(self.dissimilarity)

        # now pick the k = nrscenario most‐dissimilar indices
        selected_indices = self.select_maxmin(D, nrscenario)

        # you can return them, or store them on self:
        self.selected_scenarios = selected_indices
        return selected_indices
    
    def select_maxmin(self, D: np.ndarray, k: int):
        """
        Greedy farthest‐point sampling: start with the pair (i,j) of largest D[i,j],
        then iteratively add the point whose minimum distance to the existing set
        is maximal.
        """
        N = D.shape[0]
        # 1) find the pair (i,j) with maximum dissimilarity
        i, j = np.unravel_index(np.argmax(D, axis=None), D.shape)
        S = {i, j}

        # 2) iteratively add the point farthest from current S
        while len(S) < k:
            remaining = list(set(range(N)) - S)
            min_dists = {
                l: min(D[l, s] for s in S)
                for l in remaining
            }
            l_star = max(min_dists, key=min_dists.get)
            S.add(l_star)

        return sorted(S)

    def compute_average_uncertain_parameter_scenario(self, parameter, rounding='int'):
        """
        Compute the average uncertain parameter over all scenarios (w) and store it as a single scenario.

        :param parameter: A 3D numpy array of shape (num_scenarios, *param_dim).
        :param rounding: str, choose 'int' (default) for integer output, 'float' for raw float values.
        :return: A single averaged scenario with shape (1, *param_dim).
        """
        # Compute the mean over all scenarios (axis=0) -> result shape: (*param_dim)
        avg_parameter_array = np.mean(parameter, axis=0)

        # Convert to integer if requested
        if rounding == 'int':
            avg_parameter_array = np.round(avg_parameter_array).astype(int)

        # Expand to maintain 3D shape (1, *param_dim)
        avg_parameter_array = np.expand_dims(avg_parameter_array, axis=0)

        return avg_parameter_array

    def _generate_average_demand_Old(self, param_dim):
        """
        Generate a single deterministic scenario where demands are equal to the average value.
        """
        # All demands are equal to the forecasted average demand
        average_demand = self.Instance.ForecastedAverageDemand
        return np.expand_dims(np.ceil(average_demand).astype(int), axis=0)  # Add scenario dimension

    def generate_samples(self, num_scenarios, param_dim, Average, StandardDeviation, sampling_method):
        """
        Generate uniform samples using MC, QMC, or RQMC.

        Parameters:
        - num_scenarios: Number of scenarios to generate.
        - param_dim: Dimensions of the uncertain parameter (e.g., time buckets x locations).
        - Average of the targeted parameter.
        - STD of the targeted parameter.
        - sampling_method: "MC", "QMC", or "RQMC".

        Returns:
        - Generated samples as a NumPy array.
        """
        np.random.seed(self.ScenarioSeed)  # Set seed for reproducibility
        dimension = np.prod(param_dim)  # Flatten the parameter dimension for sampling

        if sampling_method == "MC":
            # Monte Carlo Sampling
            probabilities = np.random.uniform(size=(num_scenarios, dimension))
        elif sampling_method in ["QMC", "RQMC"]:
            # Quasi-Monte Carlo or Randomized Quasi-Monte Carlo Sampling
            scramble = True if sampling_method == "RQMC" else False
            sampler = qmc.Halton(dimension, scramble=scramble, seed=self.ScenarioSeed)
            probabilities = sampler.random(num_scenarios)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")

        # Uniformly scale probabilities to the range [average - std, average + std]
        lower_bounds = (Average - StandardDeviation).flatten()
        upper_bounds = (Average + StandardDeviation).flatten()
        
        # Apply uniform scaling
        samples = np.zeros_like(probabilities)
        for i in range(dimension):
            if upper_bounds[i] - lower_bounds[i] > 0:  # Avoid division by zero
                samples[:, i] = lower_bounds[i] + probabilities[:, i] * (upper_bounds[i] - lower_bounds[i])
            else:
                # Handle cases where standard deviation is 0 (constant value)
                samples[:, i] = lower_bounds[i]

        return samples.reshape(num_scenarios, *param_dim)

    def get_scenarios(self):
        """
        Return the generated demand scenarios.
        """
        return self.Demand  
 
    def GetAllScenarioSet(self):
        """
        Separates scenarios from the ScenarioTree into individual Scenario objects.

        Returns:
            list: A list of Scenario objects with attributes copied from the ScenarioTree.
        """
        # Extract number of scenarios from the tree structure

        if (Constants.Evaluation_Part == False):
            if hasattr(self.CasualtyDemand, 'shape'):
                num_scenarios = self.CasualtyDemand.shape[0]
            else:
                num_scenarios = len(self.CasualtyDemand)            
        else:
            num_scenarios = self.TreeStructure[1]

        # Get all attributes of ScenarioTree except Demand
        scenario_tree_attributes = vars(self).copy()  # Get all attributes as a dictionary
        
        scenario_tree_attributes.pop('CasualtyDemand', None)  # Remove 'CasualtyDemand' since it will be handled separately
        if(Constants.Debug): print("self.CasualtyDemand:\n", self.CasualtyDemand)
        
        scenario_tree_attributes.pop('HospitalDisruption', None)
        if(Constants.Debug): print("self.HospitalDisruption:\n", self.HospitalDisruption)

        scenario_tree_attributes.pop('PatientDemand', None)
        if(Constants.Debug): print("self.PatientDemand:\n", self.PatientDemand)

        scenario_tree_attributes.pop('PatientDischargedPercentage', None)
        if(Constants.Debug): print("self.PatientDischargedPercentage:\n", self.PatientDischargedPercentage)
        
        # Create a list of Scenario objects
        scenario_set = []
        for scenario_idx in range(num_scenarios):
            casualtyDemand = self.CasualtyDemand[scenario_idx]  # Get the CasualtyDemand for this scenario
            hospitalDisruption = self.HospitalDisruption[scenario_idx]  # Get the HospitalDisruption for this scenario
            patientDemand = self.PatientDemand[scenario_idx]  # Get the PatientDemand for this scenario
            patientDischargedPercentage = self.PatientDischargedPercentage[scenario_idx]  # Get the PatientDischargedPercentage for this scenario
            scenario = Scenario(casualtyDemand = casualtyDemand, 
                                hospitalDisruption = hospitalDisruption, 
                                patientDemand = patientDemand, 
                                patientDischargedPercentage = patientDischargedPercentage, 
                                **scenario_tree_attributes)  # Pass all other attributes dynamically
            scenario_set.append(scenario)  # Append to the ScenarioSet

        return scenario_set
    
    def GenerateDemandToFollowFromScenarioSet(self, scenarioset):
        if Constants.Debug: print("\n We are in 'ScenarioTree' Class -- GenerateDemandToFollowFromScenarioSet")
        if Constants.Debug: print("scenarioset:",scenarioset)
        nrscenario = len(scenarioset)
        if Constants.Debug: print("nrscenario:",nrscenario)
        self.CasualtyDemand = [[[[scenarioset[s].CasualtyDemand[t][j][l]
                                                for l in self.Instance.DisasterAreaSet]
                                                for j in self.Instance.InjuryLevelSet]
                                                for t in self.Instance.TimeBucketSet]
                                                for s in range(nrscenario)]
        
        self.HospitalDisruption = [[scenarioset[s].HospitalDisruption[h]
                                                for h in self.Instance.HospitalSet]
                                                for s in range(nrscenario)]
        
        self.PatientDemand = [[[scenarioset[s].PatientDemand[j][h]
                                                for h in self.Instance.HospitalSet]
                                                for j in self.Instance.InjuryLevelSet]
                                                for s in range(nrscenario)]
        
        self.PatientDischargedPercentage = [[[[scenarioset[s].PatientDischargedPercentage[t][j][u]
                                                for u in self.Instance.MedFacilitySet]
                                                for j in self.Instance.InjuryLevelSet]
                                                for t in self.Instance.TimeBucketSet]
                                                for s in range(nrscenario)]
        
        self.Probability = [scenarioset[s].Probability for s in range(nrscenario)]

        if Constants.Debug: 
            print("self.CasualtyDemand:\n" , self.CasualtyDemand)
            print("self.HospitalDisruption:\n" , self.HospitalDisruption)
            print("self.PatientDemand:\n" , self.PatientDemand)
            print("self.PatientDischargedPercentage:\n" , self.PatientDischargedPercentage)
            print("self.Probability:" , self.Probability)