import numpy as np
import matplotlib.pyplot as plt
from math import *
import time


# Partie 1 : Programmation des méthodes itératives

def MIGeneral (M, N, b, x_0, epsilon, Nitermax):
	Niter = 0
	x = x_0
	erreur = 2*epsilon
	while (erreur >= epsilon):
		if (Niter == Nitermax):
			print("nombre d'itération max atteint")
			break
		else:
			u = np.dot(N,x) +b
			xp = np.linalg.solve(M,u)
			erreur = np.linalg.norm(xp - x)
			x = xp
			Niter += 1
	return (xp, Niter, erreur)

def MIJacobi (A,b,x_0, epsilon, Nitermax):
	Nitermax = Nitermax
	M = np.diag(np.diag(A))
	N = M - A
	MT = np.transpose(M)
	MN = np.dot(MT,N)
	normMN = np.linalg.norm(MN)
	xp, Niter, erreur = MIGeneral(M,N,b,x_0,epsilon, Nitermax)
	return(xp, Niter, erreur, normMN)

def MIGS (A,b,x_0, epsilon, Nitermax):
	Nitermax = Nitermax
	M = np.tril(A)
	F = -np.triu(A,1)
	N = F
	MT = np.transpose(M)
	MN = np.dot(MT,N)
	normMN = np.linalg.norm(MN)
	xp, Niter, erreur = MIGeneral(M,N,b,x_0,epsilon,Nitermax)
	return(xp,Niter,erreur,normMN)

def MIRelaxation(A,b,omega,x_0,epsilon,Nitermax):
	D = np.diag(np.diag(A))
	D_1 = (1/omega)*D
	E = -np.tril(A,-1)
	M = D_1 - E
	D_2 = ((1/omega) - 1)*D
	F = -np.triu(A,1)
	N = D_2 + F
	MT = np.transpose(M)
	MN = np.dot(MT,N)
	normMN = np.linalg.norm(MN)
	xp, Niter, erreur = MIGeneral(M,N,b,x_0,epsilon,Nitermax)
	return (xp, Niter, erreur, normMN)

#Partie 2 : Expérimentation des méthodes 

def A_B (n):
	A = np.zeros((n,n))
	b = np.zeros((n,1))
	for i in range(n):
		b[i]  = np.cos(i/8)
		for j in range(n):
			if j==i :
				A[i,j] = 3
			else :
				A[i,j] = 1/(12+(3*i - 5*j)**2)
	return(A,b)

def Q1P2 ():
	print('Question 1')
	A,b = A_B(n)
	x_0 = np.zeros((n,1))
	EP = 10.**(-np.arange(1,15))
	Iter_J= list()
	Iter_GS = list()
	P_J = list()
	P_GS = list()
	normXp = list()
	normXp_2 = list()
	Nitermax = 100
	TIME_J = list()
	TIME_GS = list()
	for epsilon in EP:
		start_J = time.time()
		xp, Niter, erreur, normMN_1 = MIJacobi(A,b,x_0,epsilon,Nitermax)
		end_J = time.time()
		total_time_J = end_J - start_J
		TIME_J.append(total_time_J)
		start_GS = time.time()
		xp_2, Niter_2, erreur_2, normMN_2 = MIGS (A,b, x_0, epsilon,Nitermax)
		end_GS = time.time()
		total_time_GS = end_GS - start_GS
		TIME_GS.append(total_time_GS)
		normXp.append(np.linalg.norm(xp))
		normXp_2.append(np.linalg.norm(xp_2))
		Iter_J.append(Niter)
		Iter_GS.append(Niter_2)
		P_J.append(erreur)
		P_GS.append(erreur_2)


	if normMN_1 < 1:
		print(normMN_1)
		print('La méthode de Jacobi converge')
	else:
		print(normMN_1)
		print('La méthode de Jaconbi ne converge pas')

	if normMN_2 <1:
		print(normMN_2)
		print('La méthode GS converge')
	else:
		print(normMN_2)
		print('la méthode GS ne converge pas')


	plt.figure(1)
	plt.semilogx(P_J, Iter_J, label='Précision Jacobi', color ='green', linestyle = 'dashed')
	plt.semilogx(P_GS, Iter_GS, label = 'Précision GS', color = 'blue', linestyle = 'dashed')
	plt.legend()
	plt.title('Comparaison Jacobi et GS question 1')
	plt.grid(True)

	plt.figure(2)
	plt.semilogx(P_J, TIME_J, label='Temps Jacobi', color = 'green')
	plt.semilogx(P_GS, TIME_GS, label ='Temps GS', color = 'blue')
	plt.legend()
	plt.title('Comparaison vitesse Jacobi et GS question 1')
	plt.grid(True)
	plt.show()

def A_B_2 (n):
	A = np.zeros((n,n))
	b = np.zeros((n,1))
	for i in range(n):
		b[i]  = np.cos(i/8)
		for j in range(n):
			if j==i :
				A[i,j] = 3
			else :
				A[i,j] = 1/(12+3*abs(i - j))
	return(A,b)

def Q2P2 ():
	print('Question 2')
	A,b = A_B_2(n)
	x_0 = np.zeros((n,1))
	EP = 10.**(-np.arange(1,15))
	Iter_J= list()
	Iter_GS = list()
	P_J = list()
	P_GS = list()
	Nitermax = 100
	TIME_J = list()
	TIME_GS = list()
	for epsilon in EP:
		start_J = time.time()
		xp, Niter, erreur, normMN_1 = MIJacobi(A,b,x_0,epsilon,Nitermax)
		end_J = time.time()
		total_time_J = end_J - start_J
		start_GS = time.time()
		xp_2, Niter_2, erreur_2, normMN_2 = MIGS (A,b, x_0, epsilon,Nitermax)
		end_GS = time.time()
		total_time_GS = end_GS - start_GS
		TIME_J.append(total_time_J)
		TIME_GS.append(total_time_GS)
		Iter_J.append(Niter)
		Iter_GS.append(Niter_2)
		P_J.append(erreur)
		P_GS.append(erreur_2)


	if normMN_1 < 1:
		print(normMN_1)
		print('La méthode de Jacobi converge')
	else:
		print(normMN_1)
		print('La méthode de Jaconbi ne converge pas')

	if normMN_2 <1:
		print(normMN_2)
		print('La méthode GS converge')
	else:
		print(normMN_2)
		print('la méthode GS ne converge pas')

	plt.figure(3)
	plt.semilogx(P_J, Iter_J, label='Précision Jacobi', color ='purple', linestyle = 'dashed')
	plt.semilogx(P_GS, Iter_GS, label = 'Précision GS', color = 'orange', linestyle = 'dashed')
	plt.legend()
	plt.title('Comparaison Jacobi et GS question 2')
	plt.grid(True)
	plt.show()

	plt.figure(4)
	plt.semilogx(P_J, TIME_J, label='Temps Jacobi', color='purple')
	plt.semilogx(P_GS, TIME_GS, label='Temps GS', color='orange')
	plt.legend()
	plt.title('Comparaison vitesse Jacobi et GS question 2')
	plt.grid(True)
	plt.show()

def Q3P2part1(omega):
	omega = omega
	A,b = A_B(n)
	x_0 = np.zeros((n,1))
	EP = 10.**(-np.arange(1,15))
	Iter_J= list()
	Iter_GS = list()
	Iter_relax = list()
	P_J = list()
	P_GS = list()
	P_relax = list()
	Nitermax = 100
	TIME_J = list()
	TIME_GS = list()
	TIME_relax = list()
	for epsilon in EP:
		start_J = time.time()
		xp, Niter, erreur, normMN_1 = MIJacobi(A,b,x_0,epsilon,Nitermax)
		end_J = time.time()
		total_time_J = end_J - start_J
		start_GS = time.time()
		xp_2, Niter_2, erreur_2, normMN_2 = MIGS (A,b, x_0, epsilon,Nitermax)
		end_GS = time.time()
		total_time_GS = end_GS - start_GS
		start_relax = time.time()
		xp_3, Niter_3, erreur_3, normMN_3 = MIRelaxation(A,b,omega,x_0,epsilon,Nitermax)
		end_relax = time.time()
		total_time_relax = end_relax - start_relax
		TIME_J.append(total_time_J)
		TIME_GS.append(total_time_GS)
		TIME_relax.append(total_time_relax)
		Iter_J.append(Niter)
		Iter_GS.append(Niter_2)
		Iter_relax.append(Niter_3)
		P_J.append(erreur)
		P_GS.append(erreur_2)
		P_relax.append(erreur_3)


	if normMN_1 < 1:
		print(normMN_1)
		print('La méthode de Jacobi converge')
	else:
		print(normMN_1)
		print('La méthode de Jaconbi ne converge pas')

	if normMN_2 <1:
		print(normMN_2)
		print('La méthode GS converge')
	else:
		print(normMN_2)
		print('la méthode GS ne converge pas')

	if normMN_3 <1:
		print(normMN_3)
		print('la méthode de relaxation converge')
	else:
		print(normMN_3)
		print('la méthode de relaxation ne converge pas')


	plt.figure(5)
	plt.semilogx(P_J, Iter_J, label='Précision Jacobi', color ='orangered', linestyle = 'dashed')
	plt.semilogx(P_GS, Iter_GS, label = 'Précision GS', color = 'royalblue', linestyle = 'dashed')
	plt.semilogx(P_relax, Iter_relax, label='Précision relaxation', color='springgreen', linestyle='dashed')
	plt.legend()
	plt.title('Comparaison Jacobi, GS  et relaxation système 1')
	plt.grid(True)

	plt.figure(6)
	plt.semilogx(P_J, TIME_J, label='Temps Jacobi', color = 'orangered')
	plt.semilogx(P_GS, TIME_GS, label ='Temps GS', color = 'royalblue')
	plt.semilogx(P_relax, TIME_relax, label='Temps relaxation', color='springgreen')
	plt.legend()
	plt.title('Comparaison vitesse Jacobi, GS et relaxation question 1')
	plt.grid(True)
	plt.show()

def Q3P2part2(omega):
	omega = omega
	A,b = A_B_2(n)
	x_0 = np.zeros((n,1))
	EP = 10.**(-np.arange(1,15))
	Iter_J= list()
	Iter_GS = list()
	Iter_relax = list()
	P_J = list()
	P_GS = list()
	P_relax = list()
	Nitermax = 100
	TIME_J = list()
	TIME_GS = list()
	TIME_relax = list()
	for epsilon in EP:
		start_J = time.time()
		xp, Niter, erreur, normMN_1 = MIJacobi(A,b,x_0,epsilon,Nitermax)
		end_J = time.time()
		total_time_J = end_J - start_J
		start_GS = time.time()
		xp_2, Niter_2, erreur_2, normMN_2 = MIGS (A,b, x_0, epsilon,Nitermax)
		end_GS = time.time()
		total_time_GS = end_GS - start_GS
		start_relax = time.time()
		xp_3, Niter_3, erreur_3, normMN_3 = MIRelaxation(A,b,omega,x_0,epsilon,Nitermax)
		end_relax = time.time()
		total_time_relax = end_relax - start_relax
		TIME_J.append(total_time_J)
		TIME_GS.append(total_time_GS)
		TIME_relax.append(total_time_relax)
		Iter_J.append(Niter)
		Iter_GS.append(Niter_2)
		Iter_relax.append(Niter_3)
		P_J.append(erreur)
		P_GS.append(erreur_2)
		P_relax.append(erreur_3)


	if normMN_1 < 1:
		print(normMN_1)
		print('La méthode de Jacobi converge')
	else:
		print(normMN_1)
		print('La méthode de Jaconbi ne converge pas')

	if normMN_2 <1:
		print(normMN_2)
		print('La méthode GS converge')
	else:
		print(normMN_2)
		print('la méthode GS ne converge pas')

	if normMN_3 <1:
		print(normMN_3)
		print('la méthode de relaxation converge')
	else:
		print(normMN_3)
		print('la méthode de relaxation ne converge pas')


	plt.figure(5)
	plt.semilogx(P_J, Iter_J, label='Précision Jacobi', color ='salmon', linestyle = 'dashed')
	plt.semilogx(P_GS, Iter_GS, label = 'Précision GS', color = 'steelblue', linestyle = 'dashed')
	plt.semilogx(P_relax, Iter_relax, label='Précision relaxation', color='lightgreen', linestyle='dashed')
	plt.legend()
	plt.title('Comparaison Jacobi, GS  et relaxation système 1')
	plt.grid(True)

	plt.figure(6)
	plt.semilogx(P_J, TIME_J, label='Temps Jacobi', color = 'salmon')
	plt.semilogx(P_GS, TIME_GS, label ='Temps GS', color = 'steelblue')
	plt.semilogx(P_relax, TIME_relax, label='Temps relaxation', color='lightgreen')
	plt.legend()
	plt.title('Comparaison vitesse Jacobi, GS et relaxation question 1')
	plt.grid(True)
	plt.show()


n =100
OM = range(1,5,1)
for omega in OM:
	print(omega)
	Q3P2part2(omega)







