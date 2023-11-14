import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

#Extracting required data
def import_data(str):
    df = pd.read_csv(str)
    sub_df = df[df['Innings'] == 1]
    remaining_runs = sub_df['Runs.Remaining'].values
    overs_left = 50 - sub_df['Over'].values
    wickets_in_hand    = sub_df['Wickets.in.Hand'].values
    return remaining_runs, overs_left, wickets_in_hand

def loss_function(parameters, args):
    SE = 0
    remaining_runs = args[0]
    overs_left = args[1]
    wickets_in_hand = args[2]
    l_param = parameters[10]
    
    for i in range(len(wickets_in_hand)):
        runs_remaining = remaining_runs[i]
        overs_left_val = overs_left[i] 
        wickets = wickets_in_hand[i]
        Z0 = parameters[wickets - 1]
        rpf = Z0 * (1 - np.exp(-l_param * overs_left_val / Z0))
        SE += (rpf - runs_remaining) ** 2
    
    return SE

def optimize(remaining_runs, overs_left, wickets_in_hand, parameters, model):
    result = sp.minimize(loss_function, parameters, args=[remaining_runs, overs_left, wickets_in_hand], method=model)
    return result['fun'], result['x']

def plot(result,model):
    plt.figure()
    x = np.linspace(1,50,100)
    for i in range(len(result)-1):
        y = result[i]*(1-np.exp(-result[-1]*x/result[i]))
        plt.plot(x,y,label='Wickets in hand:' + str(i+1))
    plt.title(f'Run Production function using {model}')
    plt.xlim(0,50)
    plt.xlabel('Overs left')
    plt.ylabel('Average Runs Obtainable')
    plt.legend()
    #plt.savefig(f'RPF using {model}.png')
    plt.show()
    
remaining_runs, overs_left, wickets_in_hand = import_data("04_cricket_1999to2011.csv")
initial_guess = [10, 30, 40, 60, 90, 125, 150, 170, 190, 200,10]   #INITIAL GUESS FOR VALUES OF PARAMETERS    
model_list = ['trust-constr']
for model in model_list:
    loss,result = optimize(remaining_runs,overs_left,wickets_in_hand,initial_guess,model)
    #print(f'Total loss using {model} solver is: {loss}')
    #for i in range(len(result)-1):
        #print(f'Z[{i+1}]: {result[i]}')
    #print(f'L: {result[-1]}')
    #plot(result,model)
#plt.close('all')