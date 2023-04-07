import numpy as np
import matplotlib.pyplot as plt
import random

# all states
N_STATES = 19

# discount
GAMMA = 1

states = np.arange(1, N_STATES + 1)

# start from the middle state
START_STATE = 10

END_STATES = [0, N_STATES + 1]

def temporalDifference(stateValues, n, alpha):

    currentState = START_STATE

    Td_error=0
    no_td_err=0

   
    states = [currentState]
    rewards = [0]

    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        
        time += 1

        no_td_err+=1

        if time < T:
            # choose an action randomly
            if np.random.binomial(1, 0.5) == 1:
                newState = currentState + 1
            else:
                newState = currentState - 1
            if newState == 0:
                reward = -1
            elif newState == 20:
                reward = 1
            else:
                reward = 0

            states.append(newState)
            rewards.append(reward)

            if newState in END_STATES:
                T = time

      
        updateTime = time - n
        if updateTime >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(updateTime + 1, min(T, updateTime + n) + 1):
                returns += pow(GAMMA, t - updateTime - 1) * rewards[t]
            # add state value to the return
            if updateTime + n <= T:
                returns += pow(GAMMA, n) * stateValues[states[(updateTime + n)]]
            stateToUpdate = states[updateTime]
            if not stateToUpdate in END_STATES:
                stateValues[stateToUpdate] += alpha * (returns - stateValues[stateToUpdate])
        if updateTime == T - 1:
            Td_error=pow((returns - stateValues[stateToUpdate]),2)
            break
        currentState = newState
    return Td_error 

def avgErr(n):
    alpha=0.6
    step=n
    # each run has 10 episodes
    episodes = 10
    no_seed=20      
    runs=100
    mse=0
    # initial state values
    
    for k in range(no_seed):
        rand=random.randint(1, 1000)
        for run in range(0, runs):
            random.seed(rand)
            stateValues = np.zeros(N_STATES + 2)
            currentStateValues = np.copy(stateValues)
            for ep in range(0, episodes):
                td_err=temporalDifference(currentStateValues, step, alpha)
        # print("td err",td_err)
        mse+=td_err            
            
    mse /= (no_seed)
    return mse


def n_Itr():
  n_val=[]
  J_dl_val=[]
  f=open("n_step_logFile_n1_10_pt6","w+")
#   f.write('itr_no,n_val,rmse\n')
  for n in range(10):
    n=n+1
    n_val.append(n)
    J_dl=avgErr(n)       
    print("Value of J_dl:",round(J_dl,6))
    J_dl_val.append(round(J_dl,6))
    f.write("{}\t{}\n".format(n,round(J_dl,6))) 
                    

    plot1 = plt.figure(1)
    plt.plot(n_val,J_dl_val,'r')
    plt.xlabel("Value of n")
    plt.ylabel("MSE")
    plt.savefig("./n_converged_MSE_n1_10_pt6")

if __name__ == "__main__":
    n_Itr()