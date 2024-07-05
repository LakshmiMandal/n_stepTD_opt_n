import numpy as np
import matplotlib.pyplot as plt
import random
import os
import grid_world_env

env = grid_world_env.GridWorldEnv()
if not os.path.exists("./plot"):   
        os.makedirs("./plot")
# env.render()
state_size = env.grid_size[0]* env.grid_size[1]#24 #env.observation_space.shape[0]
action_size = env.action_space.n
max_ep_len=200
gamma=0.99
alpha=0.4
num_episodes=20

def n_step_TD(env, n, num_episodes, alpha, gamma):
    # Initialize Q table
    Q = np.zeros((state_size, action_size))
    ep_Td_error=0
    for _ in range(num_episodes):
        # Reset the environment to start a new episode
        obs = env.reset()
        state=env.grid_size[0]*(np.where(obs==1)[0][0])+np.where(obs==1)[1][0]
        done = False
        
        # Initialize variables
        T = float('inf')
        t = 0
        tau = 0
        rewards = []
        states = [state]

        Td_error=0
        no_td_err=0
        
        while (done==False and (t < T - 1) and no_td_err<max_ep_len):
            no_td_err+=1
            if t < T:
                # Take an action according to epsilon-greedy policy
                action = epsilon_greedy_policy(Q[state], epsilon=0.1)
                obs, reward, done= env.step(action)
                
                next_state=(env.grid_size[0]*(np.where(obs==1)[0][0]))+np.where(obs==1)[1][0]
                
                # Store reward
                rewards.append(reward)
                states.append(next_state)
                # print("rewards,states",rewards,states)
                
                if done:
                    T = t + 1
                else:
                    # Continue the episode
                    next_action = epsilon_greedy_policy(Q[next_state], epsilon=0.1)
                    tau = t - n + 1
                    
                    if tau >= 0:
                        G = sum([(gamma**(i-tau-1)) * rewards[i] for i in range(tau+1, min(tau+n, T))])
                        # print("G",G)
                        if tau + n < T:
                            Td_error+=pow((G - Q[states[tau]][action]),2)
                            G += gamma**n * Q[states[tau+n]][next_action]
                        Q[states[tau]][action] += alpha * (G - Q[states[tau]][action])
                        
            # Update state and time step
            state = next_state
            t += 1
            # print(no_td_err)
        ep_Td_error+=np.sqrt(Td_error/no_td_err)    
            
    return Q,ep_Td_error/200

def epsilon_greedy_policy(Q_state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q_state))
    else:
        return np.argmax(Q_state)

def avgErr(n,k):
    alpha=0.2
    # each run has 10 episodes
    # episodes = 500
          
    runs=50
    rmse=0
    for run in range(0, runs):
            Q_val,error=n_step_TD(env, n, num_episodes, alpha, gamma)
            rmse+=error
            
    rmse /= (runs)
    return rmse


def proj_val(l_val):
        print("value of l_val",l_val)
        j,d=divmod(l_val,1)
        d = float("{0:.2f}".format(d))
        print("Value of j and d",j,d)
        y=random.choices([j,j+1],[1-d,d])
        return int(y[0])

def n_update():
  n=16
  k_val=[]
  n_val=[]
  J_dl_val=[]
  ep_val=[]
  avg_score_outer=0
  avg_scores=[]
  f=open("n_step_logFile_n16_pt4","w+")
  f.write('itr_no,n_val,rmse\n')
  for k in range(5000):
    delta=0.06          
    n_update_lr=0.2
    if ((k==0) or ((k+1)%100==0)):
        k_val.append(k+1)
        n_val.append(min(max(1,round(n)),256))
    n_prev=n
    if k%2==0:
      n1=n  
      n1=min(max(1,proj_val(n1+delta)),256)
      J_dl=avgErr(n1,k)
    #   avg_score_outer=(0.2*avg_score_outer+0.8 *g_sc)        
      print("Value of J_dl:",round(J_dl,6))
      
      n=n-(n_update_lr*(J_dl/delta))
      if ((k==0) or ((k+1)%100==0)):
        J_dl_val.append(round(J_dl,6))
        # avg_scores.append(round(avg_score_outer,4))
        f.write("{}\t{}\t{}\n".format(k,n_prev,round(J_dl,6))) 
                    
          
    else:
      n2=n              
      n2=min(max(1,proj_val(n2-delta)),256)                
      J_dl=avgErr(n2,k)
    #   avg_score_outer=(0.2*avg_score_outer+0.8 *g_sc)
      print("Value of J_dl:",round(J_dl,6))
     
      n=n+(n_update_lr*(J_dl/delta))
    
      if ((k==0) or ((k+1)%100==0)):
        J_dl_val.append(round(J_dl,6))
        # avg_scores.append(round(avg_score_outer,4))
        f.write("{}\t{}\t{}\n".format(k,n_prev,round(J_dl,6)))  
    print("Value of k, n:",k,n)

    
    if not os.path.exists("./plot"):   
        os.makedirs("./plot")
    

if __name__ == "__main__":
    n_update()
    #avgScore()    
