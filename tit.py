import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def readdata():
	data = pd.read_csv("train.csv", delimiter=',', skiprows=0)

	pid = data['PassengerId']
	surv = data['Survived']
	pclass = data['Pclass']
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
	for i, n in enumerate(sex):
		if n=='female': 
			Pf[pclass[i]-1, 2] += 1
			if surv[i]==1: Pf[pclass[i]-1, 1] += 1
		if n=='male':
			Pm[pclass[i]-1, 2] += 1
			if surv[i]==1: Pm[pclass[i]-1, 1] += 1
	return Pf, Pm 
	
def p_age(surv, age):
	P = np.transpose(np.array([range(0, 121, 1), [0]*121, [0]*121])) #age [y], survived, all
	Pbaby = np.transpose(np.array([[0]*12, [0]*12, [0]*12])) #baby age [months], survived, all
	
	for i, n in enumerate(age):
		if n>=1.0: #over 1 y olds
			try:
				P[round(n-1), 2] += 1
				if surv[i]==1: P[round(n-1), 1] += 1
			except:
				pass
		else: #babies
			pass
	return P #age, survived, all
	
def main():
	pid, surv, pclass, name, sex, age, sib, parch, ticket, fare, cabin, embark = readdata()
	P_CLASS = p_pclass(surv, pclass)
	#print("\nLikelihood of survival in 1. class: " + str(P_CLASS[0, 1]/P_CLASS[0, 2]))
	#print("Likelihood of survival in 2. class: " + str(P_CLASS[1, 1]/P_CLASS[1, 2]))
	#print("Likelihood of survival in 3. class: " + str(P_CLASS[2, 1]/P_CLASS[2, 2]))
	P_SEX = p_sex(surv, sex)
	#print("\nLikelihood of survival F: " + str(P_SEX[0, 0]/P_SEX[0, 1]))
	#print("Likelihood of survival M: " + str(P_SEX[1, 0]/P_SEX[1, 1]))
	P_CLASS_SEX_F, P_CLASS_SEX_M = p_class_sex(surv, pclass, sex)
	#print("\nLikelihood of survival F class 1: " + str(P_CLASS_SEX_F[0, 1]/P_CLASS_SEX_F[0, 2]))
	#print("Likelihood of survival F class 2: " + str(P_CLASS_SEX_F[1, 1]/P_CLASS_SEX_F[1, 2]))
	#print("Likelihood of survival F class 3: " + str(P_CLASS_SEX_F[2, 1]/P_CLASS_SEX_F[2, 2]))
	#print("Likelihood of survival M class 1: " + str(P_CLASS_SEX_M[0, 1]/P_CLASS_SEX_M[0, 2]))
	#print("Likelihood of survival M class 2: " + str(P_CLASS_SEX_M[1, 1]/P_CLASS_SEX_M[1, 2]))
	#print("Likelihood of survival M class 3: " + str(P_CLASS_SEX_M[2, 1]/P_CLASS_SEX_M[2, 2]))
	
	#check that likelihood of survival all F is the same that is counted from class;
	#check posterio probability ? 
	P_AGE = p_age(surv, age)
	# spline to age curve? how to estimate accuracy of individual survivals
	
	plt.figure(1)
	plt.plot(P_AGE[:, 0], P_AGE[:, 1], 'g')
	plt.plot(P_AGE[:, 0], P_AGE[:, 2], 'b')
	#print(P_AGE)
	plt.show()
	
if __name__=="__main__":
	main()
