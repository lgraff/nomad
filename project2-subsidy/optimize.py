# libraries
from pathlib import Path
import functools
import itertools
import time
import os
import ast
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import geopandas as gpd

import multiprocessing as mp

from conf import config

def calculate_expense_less_pt(row):
    if row['transit_included'] == True:
        if row['expense'] % 2.75 == 0:
            return 0
        else:
            return row['expense'] - 2.75
    else:
        return row['expense']

# Pre-processing
def preprocess(od_matrix, poverty_df, jobs_df, config):
    od_matrix['mode_subset'] = od_matrix['mode_subset'].apply(ast.literal_eval) # convert string representation to list

    # For double transfer trips, subtract 2.75. Solution below is a hack. Could fix properly when calc trip expense by checking node costs - but that takes a while
    od_matrix['expense_less_pt'] = od_matrix.apply(calculate_expense_less_pt, axis=1)
    od_matrix['access_ind'] = (od_matrix['travel_time'] <= config['demographics']['TRAVEL_TIME_THRESHOLD']).astype(int) 

    # Get population and job center data
    poverty_df['org_geo'] = poverty_df['GEOID'].astype(str).str[5:]
    poverty_df['org'] = 'org' + poverty_df['org_geo']
    jobs_df.loc[jobs_df['opp_jobs_total'].isna(), 'opp_jobs_total'] = 0
    jobs_df['dst_geo'] = jobs_df['GEOID'].astype(str).str[5:]
    jobs_df['dst'] = 'dst' + jobs_df['dst_geo']
    
    return (od_matrix, poverty_df, jobs_df)

def get_idx_mappings(od_matrix):
    # Get name-to-idx mappings for orgs, dsts, and modes
    orgs = sorted(set(od_matrix['org']))
    dsts = sorted(set(od_matrix['dst']))
    org2idx = dict(zip(orgs, range(len(orgs))))
    dst2idx = dict(zip(dsts, range(len(dsts))))
    modes = od_matrix['mode_subset'].unique()  # od_matrix correctly does not include transit 
    mode2idx = dict(zip(modes, range(len(modes))))

    return (org2idx, dst2idx, mode2idx)

def get_sets(org2idx, dst2idx, mode2idx):
    # Sets
    I = len(org2idx)
    J = len(dst2idx)
    M = len(mode2idx)

    return (I, J, M)

def get_params(od_matrix, poverty_df, jobs_df):
    # Convert org, dst, and mode to index form
    org2idx, dst2idx, mode2idx = get_idx_mappings(od_matrix)
    od_matrix['org_idx'] = od_matrix['org'].map(org2idx)
    od_matrix['dst_idx'] = od_matrix['dst'].map(dst2idx)
    od_matrix['mode_idx'] = od_matrix['mode_subset'].apply(lambda x: mode2idx[x])  #  "map" function does not work
    poverty_df = poverty_df[poverty_df['org'].isin(list(org2idx.keys()))]
    jobs_df = jobs_df[jobs_df['dst'].isin(list(dst2idx.keys()))]
    poverty_df['org_idx'] = poverty_df['org'].map(org2idx).astype(int)
    jobs_df['dst_idx'] = jobs_df['dst'].map(dst2idx).astype(int)

    # Parameters 
    p = dict(zip(poverty_df['org_idx'], poverty_df['total_eligible'])) # population
    o = dict(zip(jobs_df['dst_idx'], jobs_df['opp_jobs_total'])) # job opportunities
    #T = dict(zip(list(zip(od_matrix.org_idx, od_matrix.dst_idx, od_matrix.mode_idx)), od_matrix['travel_time'])) # travel time dict
    c = dict(zip(list(zip(od_matrix['org_idx'], od_matrix['dst_idx'], od_matrix['mode_idx'])), od_matrix['expense_less_pt'])) # expense dict (subtract out cost of pt)
    N = 40 # number of work trips a month (hardwired here, could make it user-defined in config)
    V = dict(zip(list(zip(od_matrix.org_idx, od_matrix.dst_idx, od_matrix.mode_idx)), od_matrix['access_ind']))  # 'V' is the access indicator parameter

    MAX_JOBS_REACHABLE = jobs_df['opp_jobs_total'].sum() * poverty_df['total_eligible'].sum()

    # Budget parameter
    TOTAL_ELIGIBLE = poverty_df['total_eligible'].sum()
    r = 97.50  # MONTHLY_TRANSIT_PASS, could move to config

    I, J, M = get_sets(org2idx, dst2idx, mode2idx)

    params = {'I':I, 'J':J, 'M':M, 'p':p, 'o':o, 'c':c, 'V':V, 'N':N, 'r':r, 'MAX_JOBS_REACHABLE':MAX_JOBS_REACHABLE, 'TOTAL_ELIGIBLE':TOTAL_ELIGIBLE}

    return params, org2idx, dst2idx

def get_b_star(total_elig, r, budget_pp):
    '''Return the value of the remaining budget (B_STAR) after providing all eligible participants with a transit pass.'''
    total_budget = total_elig * budget_pp
    b_star = total_budget - (total_elig * r)
    return b_star

def define_model(params, dev_pct, budget_pp, time_limit):
    # Extract Parameters
    I = params['I']
    J = params['J']
    M = params['M']
    c = params['c']
    V = params['V']
    r = params['r']
    p = params['p']
    o = params['o']
    N = params['N']
    TOTAL_ELIGIBLE = params['TOTAL_ELIGIBLE']
    MAX_JOBS_REACHABLE = params['MAX_JOBS_REACHABLE']
    # dev_pct = d_b_sensitivity['dev_pct']
    # budget_pp = d_b_sensitivity['budget_pp']

    # Set environment variables
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()

    # MIP  model formulation
    model = gp.Model("UBM", env=env)
    model.Params.FeasibilityTol = 1e-4
    model.Params.MIPGap = 1e-4
    model.Params.TimeLimit = time_limit # seconds
    model.Params.JSONSolDetail = 1  # include data for all vars, even those with a value of 0
    #model.Params.MIPGapAbs = 0.005
    #model.Params.DualReductions = 0
    
    # decision variables
    y = model.addVars(range(I), vtype=GRB.CONTINUOUS, name='y') # daily allowance provided to neighborhood "i"
    # auxiliary variables
    x = model.addVars(range(I), range(J), range(M), vtype=GRB.BINARY, name='x')
    u = model.addVars(range(I), range(J), range(M), vtype=GRB.BINARY, name='u')
    z = model.addVars(range(I), range(J), vtype=GRB.BINARY, name='z')
    q = model.addVars(range(I), range(J), vtype=GRB.CONTINUOUS, name='q')

    A = model.addVars(range(I), vtype=GRB.CONTINUOUS, name='A') # accessed by any mode

    # Constraint: definition of u: if y[i] >= C[i,j,m] then u[i,j,m] = 1 otherwise 0. See: https://support.gurobi.com/hc/en-us/articles/4414392016529-How-do-I-model-conditional-statements-in-Gurobi
    BIG_M = 1000
    EPS = 0.0001
    model.addConstrs(y[i] >= c[(i,j,m)] - BIG_M * (1-u[i,j,m]) for i in range(I) for j in range(J) for m in range(M))
    model.addConstrs(y[i] <= c[i,j,m] - EPS + BIG_M * u[i,j,m] for i in range(I) for j in range(J) for m in range(M)) # subtract EPS?

    # Constraint: definition of x: if u = 1 & v = 1 then x = 1, otherwise 0
    model.addConstrs(0 <= u[i,j,m] + V[(i,j,m)] - 2*x[i,j,m] for i in range(I) for j in range(J) for m in range(M))
    model.addConstrs(u[i,j,m] + V[(i,j,m)] - 2*x[i,j,m] <= 1 for i in range(I) for j in range(J) for m in range(M))

    # Constraint: definition of q (number of modal candidate paths for which monetary and travel time budgets are satisfied)
    model.addConstrs(q[i,j] == gp.quicksum(x[i,j,m] for m in range(M)) for i in range(I) for j in range(J))

    # Constraint: definition of z. See: https://support.gurobi.com/hc/en-us/community/posts/360074960552-Binary-Variable-to-check-if-another-variable-is-0
    model.addConstrs(q[i,j] >= z[i,j] for i in range(I) for j in range(J))
    model.addConstrs(q[i,j] <= BIG_M * z[i,j] for i in range(I) for j in range(J))
                     
    # Constraint: definition of A
    model.addConstrs(A[i] == gp.quicksum(o[j] * z[i,j] for j in range(J)) for i in range(I))

    # Constraints: fairness
    # Variables to define min and max
    minA = model.addVar(name='minJobs')
    maxA = model.addVar(name='maxJobs')
    # Constraints to ensure min and max
    model.addGenConstrMin(minA, A, name='minJobs_constr') # The addGenConstrMin() method of the model object m adds a new general constraint that determines the minimum value among a set of variables.
    model.addGenConstrMax(maxA, A, name='maxJobs_constr')

    #Constraint: ensure deviation between maxJobs and minJobs is below a threshold
    if dev_pct == 0.0:
        pass  # let's see what happens if we strictly remove the constraint 
    else:
        model.addConstr(minA >= dev_pct * maxA, name='deviations')

    B_STAR = get_b_star(TOTAL_ELIGIBLE, r, budget_pp)    
    model.addConstr( N * y.prod(p) <= B_STAR, name="budget")

    # maximize reachable jobs
    model.setObjective(gp.quicksum(A[i] * p[i] for i in range(I)) / MAX_JOBS_REACHABLE, GRB.MAXIMIZE)
    
    return model

def solve_model(model):
    model.optimize()
    #print(D)
    #print('solved') if model.Status == GRB.OPTIMAL else print('infeasible')
    return(model)

def read_sol_file(filepath, params, dev_pct, budget_pp, time_limit=300):
    m = define_model(params, dev_pct, budget_pp, time_limit)
    m.update()
    m.read(filepath)
    #m.params.SolutionLimit = 1
    #m.Params.MIPGap = 1e-3
    #m.optimize()
    return m 

# def sensitivity_analysis(params, d_LB, d_UB, b_LB, b_UB, solutions_folder):
#     #read_sol_file(filepath, params, time_limit=300)
#     dev_pct_values = [round(0.1 * d, 1) for d in range(d_LB, d_UB)]  # rerun
#     budget_pp_values = [b for b in range(b_LB, b_UB, 10)]
#     dev_budget_combinations = [{'dev_pct': dev_pct, 'budget_pp': budget_pp} for dev_pct in dev_pct_values for budget_pp in budget_pp_values]

#     for combo in dev_budget_combinations:
#         dev_pct = combo['dev_pct']
#         budget_pp = combo['budget_pp']
        
#         folder = Path().absolute() / solutions_folder   

#         # Check if a more constrained version of this problem (lower budget) has been solved
#         filename = 'model_' + str(dev_pct) + '_' + str(budget_pp - 10) 
#         filepath = str(folder / filename) + '.sol'
#         if Path(filepath).is_file():
#             model = read_sol_file(filepath, params, dev_pct, budget_pp, time_limit=60*60)
#             model = solve_model(model)
        
#         # Otherwise, solve as new
#         else:
#             # Define and solve model
#             model = define_model(params, dev_pct, budget_pp, time_limit=60*60)
#             model = solve_model(model)

#         # Write results to a sol file
#         filename = 'model_' + str(dev_pct) + '_' + str(budget_pp) 
#         filepath = str(folder / filename) + '.sol'

#         print(params)
#         if model.Status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
#             print('Model infeasible. Cannot write a solution.')
#         else: 
#             print('Model solved.')
#             model.write(filepath)

def solve_instance(params, time_limit, d, b, filepath):
    print('Solving for d=', str(d), 'and b=', str(b))

    # Check if a more constrained version of this problem (higher deviation pct) has been solved
    solution_folder = str(Path.cwd() / 'project2-subsidy' / 'solutions_30min_v2')
    filename = f"model_{d+0.1}_{b}.sol"
    constrained_filepath = os.path.join(solution_folder, filename)
    if Path(constrained_filepath).is_file():
        model = read_sol_file(constrained_filepath, params, d, b, time_limit=60*60)
        model = solve_model(model)

    # Otherwise solve as new
    else:
        model = define_model(params, d, b, time_limit=time_limit)
        model = solve_model(model)

    if model.Status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
        print('Model infeasible. Cannot write a solution.')
    elif model.ObjVal > 0:
        print('Model solved.', "d=", str(d), "b=", str(b))
        model.write(filepath)

def main():
    od_matrix = pd.read_csv(Path.cwd() / 'project2-subsidy' / 'modal_travel_costs_2.csv')
    poverty_df = gpd.read_file(config['paths']['data']['poverty_pop'])
    jobs_df = gpd.read_file(config['paths']['data']['opp_jobs'])
    od_matrix, poverty_df, jobs_df = preprocess(od_matrix, poverty_df, jobs_df, config)

    params, org2idx, dst2idx = get_params(od_matrix, poverty_df, jobs_df)

    # # TEST a single case
    # model = define_model(params, 0.4, 260, time_limit=60*60)
    # model = solve_model(model)
    # model.write('solved_0.4_260.sol')
    # print('model solved')

    # # Sensitivity analysis. Prepare arguments for parallel processing
    # solution_folder = str(Path.cwd() / 'project2-subsidy' / 'solutions_30min_v2')
    # d_LB, d_UB = 0, 9
    # b_LB, b_UB = 100, 410
    # dev_pct_values = [round(0.1 * d, 1) for d in range(d_LB, d_UB)]  
    # budget_pp_values = [b for b in range(b_LB, b_UB, 10)]
    # time_limit = 60*60*3  # 3 hour limit per problem
    # jobs = []

    # for d, b in itertools.product(dev_pct_values[::-1], budget_pp_values):
    #     # Construct filepath
    #     filename = f"model_{d}_{b}.sol"
    #     filepath = os.path.join(solution_folder, filename)
    #     jobs.append((d, b, filepath))

    # start_time = time.time()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    # # Fix params and time_limit using functools.partial
    # solve_instance_partial = functools.partial(solve_instance, params, time_limit)
    # with mp.Pool(mp.cpu_count()-1) as pool:
    #     results = pool.starmap(solve_instance_partial, jobs)

    # end_time = time.time()
    # elapsed = (end_time - start_time)/60
    # print(f"Elapsed time: {elapsed:.1f} minutes")

    # #dev_budget_combinations = [{'dev_pct': dev_pct, 'budget_pp': budget_pp} for dev_pct in dev_pct_values for budget_pp in budget_pp_values]
    # # sensitivity_analysis(PARAMS, 0, 8, 100, 510, Path.cwd() / 'project2-subsidy' / 'solutions_30min_v2')

    # Save for future mapping
    org2idx_df = pd.DataFrame(list(org2idx.items()), columns=['org_geo','org_idx'])
    dst2idx_df = pd.DataFrame(list(dst2idx.items()), columns=['dst_geo','dst_idx'])
    org2idx_df.to_csv(Path.cwd() / 'project2-subsidy' / 'org2idx.csv', index=False)
    dst2idx_df.to_csv(Path.cwd() / 'project2-subsidy' / 'dst2idx.csv', index=False)

if __name__ == "__main__":
    main()


