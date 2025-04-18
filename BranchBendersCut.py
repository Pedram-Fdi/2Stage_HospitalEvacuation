import gurobipy as gp
from gurobipy import GRB

### Change the values of these two parameters to change the algorithm settings:

## Set usesubprobcuts = 1 to add valid inequalities to subproblems, 0 otherwise
usesubprobcuts = 1

## set cutfound = 1 to initialize the algorithm with Benders cuts derived from LP relaxation, 0 otherwise
cutfound = 1


#### This part of file declares the data. (You can instead read the data from a different file to solve a different instance) 


Cset = ['C1','C2','C3','C4','C5']
Sset = ['S1','S2']
Fset = ['F1','F2','F3','F4','F5']

#demand values in each scenario
demand = { ('C1','S1'): 13,
           ('C2','S1'): 8,  
           ('C3','S1'): 6,  
           ('C4','S1'): 11,  
           ('C5','S1'): 15,  
           ('C1','S2'): 8,  
           ('C2','S2'): 12,
           ('C3','S2'): 7,  
           ('C4','S2'): 6,
           ('C5','S2'): 4 }

# capacity of each facility, if opened
capacity = { 'F1': 26,
             'F2': 25,
             'F3': 16,
				 'F4': 8,
				 'F5': 10}

# fixed cost of opening each facility
fixedCost = {'F1': 120,
             'F2': 100,
             'F3': 90,
				 'F4': 40,
				 'F5': 60}


# Transportation costs per thousand units
transCost = { ('F1','C1'): 2,
              ('F1','C2'): 7,
              ('F1','C3'): 6,
              ('F1','C4'): 9,
              ('F1','C5'): 6,
              ('F2','C1'): 7,
              ('F2','C2'): 3,
              ('F2','C3'): 8,
              ('F2','C4'): 5,
              ('F2','C5'): 9,
              ('F3','C1'): 6,
              ('F3','C2'): 8,
              ('F3','C3'): 4,
              ('F3','C4'): 10,
              ('F3','C5'): 3,
			  ('F4','C1'): 2,
              ('F4','C2'): 7,
              ('F4','C3'): 6,
              ('F4','C4'): 2,
              ('F4','C5'): 5,
              ('F5','C1'): 2,
              ('F5','C2'): 9,
              ('F5','C3'): 6,
              ('F5','C4'): 7,
              ('F5','C5'): 12 }

# penalties per unit of unmet customer demand
penalties = {'C1': 30, 'C2': 30, 'C3':30, 'C4': 30, 'C5': 30 }



# main Problem  and main decision variables
main = gp.Model("main")

# Plant open decision variables: open[p] == 1 if plant p is open.
### Here we start with LP relaxation, so varibles are declared as continuous, with bounds [0,1]
fopen = main.addVars(Fset, ub=1.0, obj=fixedCost, name="Open")

# Value function decision variables
thetacoefs = {}
for s in Sset:
    thetacoefs[s] = 1/len(Sset)
theta = main.addVars(Sset, vtype=GRB.CONTINUOUS, obj=thetacoefs, name="theta")

main.modelSense = GRB.MINIMIZE
main.update()



## Subproblem and subproblem decision variables
sub = gp.Model("facility")
sub.params.logtoconsole = 1  ## turns off display of Gurobi output when solving subproblems

# Transportation decision variables: how much to transport from
# a plant p to a customer w
transport = sub.addVars(Fset, Cset, obj=transCost, name="Transport")


unmet = sub.addVars(Cset, obj=penalties, name="Unmet")

# The objective is to minimize the total fixed and variable costs
sub.modelSense = GRB.MINIMIZE 
sub.update() 

# Subproblem production constraints
# For now just set right-hand side to capacity[p] -- it will be reset during algorithm
# based on main problem solution
capcon = sub.addConstrs((gp.quicksum(transport[p,w] for w in Cset) <= capacity[p] for p in Fset),
           "Capacity")

# Demand constraints
# For now just use scenario 0 data -- it will be reset during algorithm as it loops through scnearios
demcon = sub.addConstrs((gp.quicksum(transport[p,w] for p in Fset) + unmet[w] >= demand[w,'S1'] for w in Cset),
                    "Demand")

## Upper bound constraints (optional -- these strengthen the subproblem relaxation)
## upper bounds need to be updated to demand[w,p]*fopen[p].x

if usesubprobcuts:
    trub = sub.addConstrs(transport[p,w] <= demand[w,'S1']  for p in Fset for w in Cset)

sub.update()


# Begin the LP cutting plane loop

iter = 1
totcuts = 0
while cutfound:

    print('================ Iteration ', iter, ' ===================')
    iter = iter+1
    # Solve current main problem
    cutfound = 0 
    main.optimize() 
 
    print('current LP main objval = ', main.objVal)
    openvals = main.getAttr('x',fopen)
    thetavals = main.getAttr('x',theta)

	 # Fix the right-hand side in subproblem constraints according to each scenario and main solution, then solve
    for s in Sset:
        for p in Fset:
            capcon[p].RHS = openvals[p]*capacity[p]
        for w in Cset:
            demcon[w].RHS = demand[w,s]
        if usesubprobcuts:
            for w in Cset:
                for p in Fset:
                    trub[p,w].RHS = demand[w,s]*openvals[p]
        sub.update()	
        sub.optimize()
        
        cappivals = sub.getAttr('Pi',capcon)
        dempivals = sub.getAttr('Pi',demcon)
        if usesubprobcuts:
            trubpivals = sub.getAttr('Pi',trub)

        if sub.objVal > thetavals[s] + 0.000001:  ### violation tolerance
            totcuts += 1
            xcoef = {} 
            for p in Fset:
                xcoef[p] = capacity[p]*cappivals[p]
            rhs = 0.0 
            for w in Cset:
                rhs += demand[w,s] * dempivals[w]
            if usesubprobcuts:
                for p in Fset:
                    for w in Cset:
                        xcoef[p] += demand[w,s]*trubpivals[p,w]
            main.addConstr(theta[s] - gp.quicksum(xcoef[p]*fopen[p] for p in Fset) >= rhs)
            cutfound = 1 

print('Benders cuts in LP main problem: ', totcuts)

### Now declare the variables to be binary
for p in Fset:
    fopen[p].vType = GRB.BINARY


### Define the callback function that Gurobi will call when it finds an integer feasible solution 
### This is where you need to search for more Benders cuts and add them if any violated
### See Gurobi's "callback.py" and "tsp.py" examples for more details


def BendersCallback(model, where):
    if where == GRB.Callback.MIPSOL:

        ## Set up and solve the Benders subproblems, just like in cutting plane loop

        # Fix the right-hand side in subproblem constraints according to each scenario and main solution, then solve
        for p in Fset:
            capcon[p].RHS = model.cbGetSolution(fopen[p])*capacity[p] 

        for s in Sset:
            for w in Cset:
                demcon[w].RHS = demand[w,s]
            if usesubprobcuts:
                for w in Cset:
                    for p in Fset:
                        trub[p,w].RHS = demand[w,s]*model.cbGetSolution(fopen[p])
            sub.update()	
            sub.optimize()
            
            cappivals = sub.getAttr('Pi',capcon)
            dempivals = sub.getAttr('Pi',demcon)
            if usesubprobcuts:
                trubpivals = sub.getAttr('Pi',trub)

            if sub.objVal > model.cbGetSolution(theta[s]) + 0.000001:  ### violation tolerance
                xcoef = {} 
                for p in Fset:
                    xcoef[p] = capacity[p]*cappivals[p]
                rhs = 0.0 
                for w in Cset:
                    rhs += demand[w,s]*dempivals[w]
                if usesubprobcuts:
                    for p in Fset:
                        for w in Cset:
                            xcoef[p] += demand[w,s]*trubpivals[p,w]
                model.cbLazy(theta[s] - gp.quicksum(xcoef[p]*fopen[p] for p in Fset) >= rhs)


### Pass BendersCallback as argument to the optimize function on the main
main.Params.lazyConstraints = 1

main.optimize(BendersCallback)

print('current optimal solution:')
for p in Fset:
    print('fopen[', p, ']=', fopen[p].x)
for k in Sset:
    print('theta[', k, ']=', theta[k].x)
print('objval = ', main.objVal)