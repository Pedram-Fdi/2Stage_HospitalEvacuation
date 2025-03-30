
# CRP-PLT-SDDP

**Resilient Casualty Response Planning Considering Hospital Evacuation**

## Overview

This project focuses on the optimization and evaluation of instances related to Casualty Response Planning (CRP) and Platelet Inventory Management using various methods such as MIP, NBD, SDDP, PH, Hybrid, and MLLocalSearch. Below is a detailed explanation of each component and its role within the script.

## Components and Actions

### 1. Action: Solve

This action is used for solving different models within the project. Here’s a step-by-step guide:

#### a. Generate Instances

Before solving any problem, you must generate instances using:

\`\`\`bash
--Action="GenerateInstances"
\`\`\`

#### b. Choose an Instance

After generating instances, select the specific instance to solve:

\`\`\`bash
--Instance="Name_of_your_desired_instance"
\`\`\`

#### c. Choose a Model

- **2Stage**:
  - Solved exactly via Gurobi with the method set to MIP.
  - Scenarios are generated based on \`NrScenario\` and the number of periods (T), e.g., T^4.

- **Average Model**:
  - Solves the MIP model for a single scenario using the average value of uncertain parameters.

#### d. Choose an algorithm

- **Multi_Stage Model**:
  - The primary functionality with multiple solving methods:
    - \`--method=MIP\`: Solves using Gurobi with non-anticipativity constraints.
    - \`--method=ALNS\`: Solves the two-stage model using Adaptive Large Neighbourhood Search.
    - \`--method=PHA\`: Solves the two-stage model using Progressive Hedging Algorithm.
    - \`--method=BBC\`: Solves the two-stage model using Branch-and-Benders Cut Algorithm.



#### e. Scenario and Evaluation Settings

- **NrScenario**: Specifies the number of scenarios for each time period.
- **ScenarioGeneration**: Defines the scenario generation method for in-sample scenarios. Monte Carlo is always used for out-of-sample evaluation.
- **nrevaluation**: Number of out-of-sample scenarios used for evaluation.
