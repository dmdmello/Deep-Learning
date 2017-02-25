import sys
import os
import time
import numpy as np

lista_shape = [200, 600, 1000, 1100, 1150]
lista_batch = [30, 80, 200, 400, 800, 1000, 1100]

sys.stdout.flush()

print ("-----------------------------------------------------------")
print("\n" * 2)
print("Operatation: Inverse")
print("\n" * 2)
print ("-----------------------------------------------------------")
sys.stdout.flush()

for i in lista_shape:	
	for j in lista_batch:
		op = "inv_operation.py"
		print ("----------------------------------------------------------")
		print("Operatation: Inverse")
		print("Device: cpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s cpu %d %d" % (op, i, j))
		print ("----------------------------------------------------------")
		print("Operatation: Inverse")
		print("Device: gpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s gpu %d %d" % (op, i, j))
sys.stdout.flush()



print("\n" * 10)
print ("-----------------------------------------------------------")
print("\n" * 2)
print("Operatation: Matrix mult.")
print("\n" * 2)
print ("-----------------------------------------------------------")
for i in lista_shape:	
	for j in lista_batch:
		op = "matmul.py"
		print ("----------------------------------------------------------")
		print("Operatation: Matrix mult.")
		print("Device: cpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s cpu %d %d" % (op, i, j))
		print ("----------------------------------------------------------")
		print("Operatation: Matrix mult.")
		print("Device: gpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s gpu %d %d" % (op, i, j))





print("\n" * 10)
print ("-----------------------------------------------------------")
print("\n" * 2)
print("Operatation: Matrix Mult. Element-wise")
print("\n" * 2)
print ("-----------------------------------------------------------")
for i in lista_shape:	
	for j in lista_batch:
		op = "mult_el_wise.py"
		print ("----------------------------------------------------------")
		print("Operatation: Matrix Mult. Element-wise")
		print("Device: cpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s cpu %d %d" % (op, i, j))
		print ("----------------------------------------------------------")
		print("Operatation: Matrix Mult. Element-wise")
		print("Device: gpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s gpu %d %d" % (op, i, j))



print("\n" * 10)
print ("-----------------------------------------------------------")
print("\n" * 2)
print("Operatation: Matrix Sum")
print("\n" * 2)
print ("-----------------------------------------------------------")
for i in lista_shape:	
	for j in lista_batch:
		op = "sum_operation.py"
		print ("----------------------------------------------------------")
		print("Operatation: Matrix Sum")
		print("Device: cpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s cpu %d %d" % (op, i, j))
		print ("----------------------------------------------------------")
		print("Operatation: Matrix Sum")
		print("Device: gpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s gpu %d %d" % (op, i, j))


print("\n" * 10)
print ("-----------------------------------------------------------")
print("\n" * 2)
print("Operatation: Matrix Transpose")
print("\n" * 2)
print ("-----------------------------------------------------------")
for i in lista_shape:	
	for j in lista_batch:
		op = "transp_operation.py"
		print ("----------------------------------------------------------")
		print("Operatation: Matrix Transpose")
		print("Device: cpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s cpu %d %d" % (op, i, j))
		print ("----------------------------------------------------------")
		print("Operatation: Matrix Transpose")
		print("Device: gpu")
		print("Shape: %d" % i)
		print("Batchs: %d" % j)
		print ("----------------------------------------------------------")
		os.system("python %s gpu %d %d" % (op, i, j))
