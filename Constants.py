class Constants( object ):

    #Model
    Average = "Average"
    Two_Stage = "2Stage"

    PathInstances = "./Instances/"
    PathCPLEXLog = "./CPLEXLog/"

    #low-discrepancy sequence type to generate QMC or RQMC: (This part is inactive now)
    SequenceTypee = "Halton"                     # Do not Change it, You can Modify it in main.py

    #Scenario sampling methods:
    MonteCarlo = "MC"
    QMC = "QMC"
    RQMC = "RQMC"
    All = "all"


    #Solver
    MIP = "MIP"
    ProgressiveHedging = "PHA"
    ALNS = "ALNS"
    BBC = "BBC"

    #Demand distributions:
    Lumpy = "Lumpy"
    SlowMoving = "SlowMoving"
    Uniform = "Uniform"
    Normal = "Normal"
    Binomial = "Binomial"
    NonStationary = "NonStationary"
    
    # Risk
    #Risk = "Constant"
    #Risk = "Linear"
    Risk = "Exponential"

    #The set of seeds used for random number generator
    SeedArray = [42]#[2934, 875, 3545, 765, 546, 768, 242, 375, 142, 236, 788]
    EvaluationScenarioSeed = 3545
    
    Debug = False
    Obtain_SecondStage_Solution  = True         # Do NOT CHANGE IT! I will use it when I am working on ALNS and wanna save time, avoiding generating ALNS solutions of the second-stage at each iteration!
    PrintSolutionFileToExcel = False            # (Defaul: False) If you set this 'True', then all final values of variables and objective function will be printed in Excel too.
    PrintSolutionFileToPickle = True            # (Defaul: True)
    PrintDetailsExcelFiles = False               # (Defaul: False) Here, you save some statistics which is useful for analytical part of the paper!
    RunEvaluationInSeparatedJob = False
    PrintOnlyFirstStageDecision = True
    PrintPHATrace = True
    LauchEvalAfterSolve = True 
    Debug_lp_Files = False
    PrintSolutionFileInTMP = False
    Evaluation_Part = False

    #Code parameter
    Infinity = 9999999999999999
    AlgorithmTimeLimit = 12 * 3600     #Whatever you have here, then in the SDDP algorithm, (AlgorithmTimeLimit * |T|) will be used as time limit.
    
    
    MIPTimeLimit = 12 * 3600            #This is only a time limit to solve the extended model via MIP.
    ModelOutputFlag = 0                 #If it is 0, Prevents Gurobi from showing the optimization process!

    ####################### PHA Algorithm
    We_Are_in_PHA = False                            # Do NOT CHANGE IT!
    PHIterationLimit = 100000                       # Previosly set to 10000
    PHConvergenceTolerence = 0.01        
    Rho_PH_PenaltyParameter= 0.01                    # The best: 0.1
    PHCoeeff_QuadraticPart = 0.5                     # (Do not Change it) (Usually you see (1/2) in mathematical models)
    Dynamic_rho_PenaltyParameter = False             # (Do not Change it) True: Just Increases the penalty parameter (rho) every 10 iterations
    Increase_rate_dynamic_rho = 0.05
    Dynamic_Learning_rho_PenaltyParameter = False    # (Do not Change it) True: Adgust penalty parameter rho based on: (Ref: A progressive hedging method for the optimization of social engagement and opportunistic IoT problems) article
    Quadratic_to_Linear_PHA = False                  # (Do not Change it) True: There is not Quadratic part in the Objective function of the PHA anymore based on: (A progressive hedging method for the optimization of social engagement and opportunistic IoT problems) article

    ####################### ALNS Algorithm
    Max_ALNS_Iterations = 100000

    ####################### BBC Algorithm
    My_EpGap_BBC = 0.01
    UseMultiCut = False                  # Do NOt Change!
    GenerateStrongCut = False            # Do NOt Change!
    UseLBF = False                       # Do NOt Change! (Lower bounding Functional (LBF))
    UseWarmUp = False                    #Do NOt Change!
    NrIterationWarmUp = 10            # 10 to 50, for Warm-up  
    Sherali_Multiplier = 1e-9           # For Strong Cut acceleration
    CorePointCoeff = 0.5                # For Strong Cut acceleration (# The coefficient of the previous core point value, which should be less than 1!)

    ####################### Scenario Clustering
    ClusteringMethod = "Noc"                    # Do NOT CHANGE IT
    Multiplier_NumberofOriginalScenarios = 10   # If you set it to 10, and then choose any type of scenario clustering, then the model cluseter "NrScenario" scenairos from "10 * NrScenario" scenarios!

    @staticmethod
    def IsDeterministic(s):
        result = s == Constants.Average
        return result
    
    @staticmethod
    def IsQMCMethos(s):
       result = s in [Constants.QMC, Constants.RQMC]
       return result
    
    @staticmethod
    def GetEvaluationFolder():
        if Constants.PrintSolutionFileInTMP:
            return "/tmp/Evaluations/"
        else:
            return "./Evaluations/"    