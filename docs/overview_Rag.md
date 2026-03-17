# Overview

This project addresses **Casualty Response Planning (CRP)** and **Platelet Inventory Management** through advanced optimization methods, particularly **two-stage stochastic integer programming**, and a variety of solution techniques including exact solvers, matheuristics, and learning-enhanced heuristics.

It integrates models for hospital evacuation, temporary hospital (ACF) placement, and coordinated patient transport using land and aerial vehicles.

## Choose a Model

### Stochastic Model:
Captures uncertainty through multiple possible future scenarios.

- **Use when**: Patient needs, facility status, or resources are unpredictable.
- **Method**: Solved via:
  - MIP (exact)
  - ALNS (metaheuristic)
  - PHA (scenario decomposition)
- **Features**:
  - Multi-scenario planning
  - Explicit risk modeling (threat + transport)
  - Requires scenario generation and potentially clustering

### Deterministic Model:
Solves a single-scenario problem using average expected values.

- **Use when**: You need a quick, interpretable baseline.
- **Method**: Always solved by MIP.
- **Features**:
  - Fast to compute
  - No uncertainty modeling
  - May underestimate risk and resource needs