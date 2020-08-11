import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#testtt
def readdata():
	data = pd.read_csv("train.csv", delimiter=',', skiprows=0)

	pid = data['PassengerId']
	surv = np.asarray(data['Survived'])
	pclass = np.asarray(data['Pclass'])
	name = data['Name']
	sex = data['Sex']
	age = data['Age']
	sib = data['SibSp']
	parch = data['Parch']
	ticket = data['Ticket']
	fare = data['Fare']
	cabin = data['Cabin']
	embark = data['Embarked']
	return pid, surv, pclass, name, sex, age, sib, parch, ticket, fare, cabin, embark

def p_pclass(surv, pclass) :
	P = np.transpose(np.array([[1 ,2 ,3], [0, 0 ,0], [0, 0, 0]])) #class, survived, all
	for i, n in enumerate(pclass):
		if surv[i] ==1: P[n-1, 1] += 1
		P[n-1, 2] += 1
	return P #class, survived, all

def p_sex(surv, sex) :
	P = np.transpose(np.array([[0, 0], [0, 0]])) #survived, all
	for i, n in enumerate(sex):
		if n=='female': 
			P[0, 1] += 1
			if surv[i]==1: P[0, 0] += 1
		if n=='male':
			P[1, 1] += 1 
			if surv[i]==1: P[1, 0] += 1		
	return P #survived, all
	
def p_class_sex(surv, pclass, sex):
	Pf  = np.transpose(np.array([[1 ,2 ,3], [0, 0 , 0], [0, 0, 0]])) #female: class, survived, all
	Pm  = np.transpose(np.array([[1 ,2 ,3], [0, 0 , 0], [0, 0, 0]])) #male: class, survived, all
	#for i, n in enumerate(sex):
	#	if n=='female': 
	#		pclass[i]-1
	#	if n=='male':
		
def main():
	pid, surv, pclass, name, sex, age, sib, parch, ticket, fare, cabin, embark = readdata()
	P_CLASS = p_pclass(surv, pclass)
	print("\nLikelihood of survival in 1. class: " + str(P_CLASS[0, 1]/P_CLASS[0, 2]))
	print("Likelihood of survival in 2. class: " + str(P_CLASS[1, 1]/P_CLASS[1, 2]))
	print("Likelihood of survival in 3. class: " + str(P_CLASS[2, 1]/P_CLASS[2, 2]))
	P_SEX = p_sex(surv, sex)
	print("\nLikelihood of survival F: " + str(P_SEX[0, 0]/P_SEX[0, 1]))
	print("Likelihood of survival M: " + str(P_SEX[1, 0]/P_SEX[1, 1])+"\n")
	P_CLASS_SEX = p_class_sex(surv, pclass, sex)
	
	#print("\nLikelihood of survival F in the 1 calss: " + str(P_SEX[0, 0]/P_SEX[0, 1]))
	#print("Likelihood of survival M: " + str(P_SEX[1, 0]/P_SEX[1, 1])+"\n")
	
	
	
	#plt.figure(1)
	#plt.plot(pid, pclass, 'g')
	#plt.show()

if __name__=="__main__":
	main()
