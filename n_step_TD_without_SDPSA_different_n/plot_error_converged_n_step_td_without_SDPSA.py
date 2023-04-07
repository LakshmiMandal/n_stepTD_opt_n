import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
# path="./n_step_logFile_n1_256_pt2"
# path1="./n_step_logFile_n1_256_pt4"
# path2="./n_step_logFile_n1_256_pt6"
df = pd.read_csv("n_step_logFile_n1_10_pt2",delimiter='\t',header = None)
df1 = pd.read_csv("n_step_logFile_n1_10_pt4",delimiter='\t',header = None)
df2 = pd.read_csv("n_step_logFile_n1_10_pt6",delimiter='\t',header = None) 
# print("len of df", len(df)) 
# print(df)
x=[]
# l=len(df)
y=[]
y1=[]
y2=[]
# l=max(len(df),len(df1),len(df2))
for i in range(df.shape[0]):
       # print (i+1)
       x.append(i+1)
       y.append(np.sqrt(df.iat[i,1]))
       y1.append(np.sqrt(df1.iat[i,1]))
       y2.append(np.sqrt(df2.iat[i,1]))
# print("len of x", len(x))
plt.plot(x,y, color = 'b', linestyle = '-',label = r'$\alpha$=0.2')
plt.plot(x,y1, color = 'g', linestyle = '-',label = r'$\alpha$=0.4')
plt.plot(x,y2, color = 'c', linestyle = '-',label = r'$\alpha$=0.6')
plt.xticks(rotation = 25)
plt.xlabel('Value of n')
plt.ylabel('RMSE')
plt.title('', fontsize = 20)
# plt.grid()
plt.legend()
#plt.show()
#plt.savefig('Agerage_Error.png')
plt.savefig('./RMSE_plot.png')
plt.savefig('./RMSE_plot.svg')