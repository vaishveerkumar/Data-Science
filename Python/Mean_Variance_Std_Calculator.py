import numpy as np

l=[]
for i in range(9):
    while True:
        try:
            a = int(input(f"Enter number {i+1}: "))
            l.append(a)
            break 
        except ValueError:
            print("Invalid input! Please enter a valid number.")

l=np.array(l).reshape(3,3)
print ("The input values are: \n", l)

column_means = np.mean(l, axis=0)
row_means = np.mean(l, axis=1)
mean=np.mean(l)

column_variance = np.var(l, axis=0)
row_variance = np.var(l, axis=1)
variance=np.var(l)

column_std = np.std(l, axis=0)
row_std = np.std(l, axis=1)
std=np.std(l)

column_max = np.max(l, axis=0)
row_max = np.max(l, axis=1)
max=np.max(l)

column_min = np.min(l, axis=0)
row_min = np.min(l, axis=1)
min=np.min(l)

column_sum = np.sum(l, axis=0)
row_sum = np.sum(l, axis=1)
sum=np.sum(l)

d={}
def calculate(l):
  d = {"Mean":[column_means,row_means,mean],"Variance":[column_variance,row_variance,variance], "Standard Deviation":[column_std,row_std,std], "Max":[column_max,row_max,max], "Min":[column_min,row_min,min], "Sum":[column_min,row_min,min]}
  return d

result = calculate(l)
print(result)