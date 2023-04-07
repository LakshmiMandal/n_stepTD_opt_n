import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv

df = pd.read_csv("n_step_logFile_n8_pt6",delimiter='\t')
# print(df.shape)
# print(data.shape)
x=df.iloc[:,[0]]
# x=np.array(x)
y=[]
avg_rmse=0
# print(avg_rmse)
for i in range(df.shape[0]):
    err=df.iat[i,2]
#     print("err:",err)
    if(i==0):
        avg_rmse=err
    else:
        avg_rmse=((0.9*avg_rmse)+ (0.1*err))
#     print("avg_rmse:",avg_rmse)
    y.append(avg_rmse)

# y=np.array(y)
plot2 = plt.figure(2)
plt.plot(x,y,'r')
# plt.xticks(rotation = 25)
plt.xlabel('No. of n-training Iterations')
plt.ylabel('RMSE')
# plt.title('', fontsize = 20)
# plt.grid()
# plt.show()
plt.savefig("./n_step_RMSE_n8_pt6")