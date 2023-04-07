import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

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
    
    states = [currentState]
    rewards = [0]

 
    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        
        time += 1
        if time < T:
            
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
            
            for t in range(updateTime + 1, min(T, updateTime + n) + 1):
                returns += pow(GAMMA, t - updateTime - 1) * rewards[t]
            
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

def avgErr(n,k):
    alpha=0.2
    step=n
    # each run has 10 episodes
    episodes = 10
          
    runs=50
    mse=0
    stateValues = np.zeros(N_STATES + 2)
    for run in range(0, runs):
        currentStateValues = np.copy(stateValues)
        for ep in range(0, episodes):
            td_err=temporalDifference(currentStateValues, step, alpha)
        mse+=td_err    
    mse /= (runs)
    return np.sqrt(mse)


def proj_val(l_val):
        print("value of l_val",l_val)
        j,d=divmod(l_val,1)
        d = float("{0:.2f}".format(d))
        print("Value of j and d",j,d)
        y=random.choices([j,j+1],[1-d,d])
        return int(y[0])

def n_update():
  n=2
  k_val=[]
  f=open("n_step_logFile_n2_pt2_avg","w+")
  f.write('itr_no,n_val,rmse\n')
  n_deque=deque(maxlen=1000)
  J_dl_deque=deque(maxlen=1000)
  n_vals = np.zeros(20000)
  J_dl_vals = np.zeros(20000)
  for k in range(20000):
    delta=0.6          
    n_update_lr=0.19
    # if ((k==0) or ((k+1)%100==0)):
    k_val.append(k+1)
    #     n_val.append(min(max(1,round(n)),256))
    n_deque.append(min(max(1,round(n)),256))
    n_vals[k] = np.mean(n_deque)
    print("value of n",n_vals[k])
    n_prev=n
    if k%2==0:
      n1=n  
      n1=min(max(1,proj_val(n1+delta)),256)
      J_dl=avgErr(n1,k)      
      print("Value of J_dl:",round(J_dl,6))
      if(7.5<=n<=8.5):
        n_update_lr/=100
      n=n-(n_update_lr*(J_dl/delta))
         
      J_dl_deque.append(round(J_dl,6))
      J_dl_vals[k] = np.mean(J_dl_deque)
      print("value of J_dl",J_dl_vals[k])
      f.write("{}\t{}\t{}\n".format(k,n_vals[k],J_dl_vals[k]))               
          
    else:
      n2=n              
      n2=min(max(1,proj_val(n2-delta)),256)                
      J_dl=avgErr(n2,k)
      print("Value of J_dl:",round(J_dl,6))
      if(7.5<=n<=8.5):
        n_update_lr/=100
      n= n+(n_update_lr*(J_dl/delta))
    
      J_dl_deque.append(round(J_dl,6))
      J_dl_vals[k] = np.mean(J_dl_deque)
      print("value of J_dl",J_dl_vals[k])
      f.write("{}\t{}\t{}\n".format(k,n_vals[k],J_dl_vals[k])) 

  plot1 = plt.figure(1)
  plt.plot(k_val,J_dl_vals,'r')
  plt.xlabel("No. of n-training iterations")
  plt.ylabel("RMSE")
  plt.savefig("./n_training_avg_RMSE_n2_pt2")

  plot2 = plt.figure(2)    
  plt.plot(k_val,n_vals,'g')
  plt.xlabel("No. of n-training iterations")
  plt.ylabel("Value of n")
  plt.savefig("./Projected_avg_n_val_n2_pt2")

if __name__ == "__main__":
    n_update()