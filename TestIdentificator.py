from Constants import Constants


class TestIdentificator:
    """
    This class identifies the test settings and stores the configuration for the stochastic location allocation model.
    """

    def __init__(self, 
                 instance_name, 
                 model, 
                 solver, 
                 nrScenario, 
                 seed, 
                 sampling,
                 phaobj,
                 phapenalty,
                 alnsRL,
                 alnsRL_DeepQ,
                 rlSelectionMethod,
                 bbcsetting,
                 clustering
                 ):
        
        """
        Initialize the TestIdentificator with the given parameters.
        :param instance_name: Name of the instance being solved.
        :param model: Model type.
        :param solver: Solver type (MIP, ALNS, or PHA).
        :param nrScenario: Number of scenarios.
        :param seed: Random seed for reproducibility.
        :param sampling: Sampling method for generating scenarios
        """
        self.InstanceName = instance_name
        self.Model = model
        self.Solver = solver
        self.NrScenario = nrScenario
        self.ScenarioSeed = seed
        self.ScenarioSampling = sampling
        self.PHAObj = phaobj
        self.PHAPenalty = phapenalty
        self.ALNSRL = alnsRL
        self.ALNSRL_DeepQ = alnsRL_DeepQ
        self.RLSelectionMethod = rlSelectionMethod
        self.BBCSetting = bbcsetting
        self.Clustering = clustering

    # The following list wil be appear only for the naming the stored files
    def get_as_string_list(self):
        """
        Return the test settings as a list of strings.
        """
        return [
            self.InstanceName,
            self.Model,
            self.Solver,
            self.NrScenario,
            self.ScenarioSampling,
            self.ScenarioSeed,
            self.PHAObj,
            self.PHAPenalty,
            self.ALNSRL,
            self.ALNSRL_DeepQ,
            self.BBCSetting,
            self.Clustering
        ]

    # The following list wil be appear IN the stored files
    def GetAsString(self):
        """
        Return the test settings as a single formatted string.
        """
        return "_".join(str(item) for item in self.get_as_string_list())

    def GetAsStringList(self):
        if Constants.Debug: print("\n We are in 'TestIdentificator' Class -- GetAsStringList")
        result = [self.InstanceName,
                self.Model,
                self.Solver,
                self.ScenarioSampling,
                self.NrScenario,
                "%s"%self.ScenarioSeed,
                self.PHAObj,
                self.PHAPenalty,
                "%s"%self.ALNSRL,
                "%s"%self.ALNSRL_DeepQ,
                self.RLSelectionMethod,
                self.BBCSetting,
                self.Clustering
                ]
        return result
    
    def print_attributes(self):
        """
        Print the attributes of the test identifier.
        """
        print("Test Identifier Details:")
        print(f"  Instance Name: {self.InstanceName}")
        print(f"  Model: {self.Model}")
        print(f"  Solver: {self.Solver}")
        print(f"  NrScenario: {self.NrScenario}")
        print(f"  Seed: {self.ScenarioSeed}")
        print(f"  ScenarioSampling: {self.ScenarioSampling}")