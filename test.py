import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gstools as gs
from collections import defaultdict
#Ⅳ1
# conditioning data (x, y, value)
cond_pos1_1 = [[419], [346]]
cond_val1_1 = [0.55]
#--------------------------------------
with open( 'unit coordinates_for1.txt' ,'r') as f1_1:
	data1_1 = []
	line1_1 = f1_1.readlines()
	for i, rows in enumerate(line1_1):
		if i in range(0,len(line1_1)) :
			data1_1.append(rows)
dict_x1_1 = defaultdict(list) 
dict_y1_1 = defaultdict(list)
dict_ncs1_1 = defaultdict(list)
for i in range(len(data1_1)):
	a = data1_1[i].split(',')[0:1]
	b = int(a[0])
	c = data1_1[i].split(',')[1:2]
	d = float(c[0])
	dict_x1_1[b].append(d)
	dict_ncs1_1[b].append(d)
	e = data1_1[i].split(',')[2:3]
	f = float(e[0])
	dict_y1_1[b].append(f)
	dict_ncs1_1[b].append(f)
func = lambda x: round(x,0)
x1_1_min = list(map(func, min(dict_x1_1.values())))
x1_1_max = list(map(func, max(dict_x1_1.values())))
y1_1_min = list(map(func, min(dict_y1_1.values())))
y1_1_max = list(map(func, max(dict_y1_1.values())))
elements1_1 = len(data1_1)
with open( 'unit coordinates_fordan.txt' ,'r') as f1_2:
	data1_2 = []
	line1_2 = f1_2.readlines()
	for i, rows in enumerate(line1_2):
		if i in range(0,len(line1_2)) :
			data1_2.append(rows)
dict_x1_2 = defaultdict(list) 
dict_y1_2 = defaultdict(list)
dict_ncs1_2 = defaultdict(list)
for i in range(len(data1_2)):
	a = data1_2[i].split(',')[0:1]
	b = int(a[0])
	c = data1_2[i].split(',')[1:2]
	d = float(c[0])
	dict_x1_2[b].append(d)
	dict_ncs1_2[b].append(d)
	e = data1_2[i].split(',')[2:3]
	f = float(e[0])
	dict_y1_2[b].append(f)
	dict_ncs1_2[b].append(f)
x1_2_min = list(map(func, min(dict_x1_2.values())))
x1_2_max = list(map(func, max(dict_x1_2.values())))
y1_2_min = list(map(func, min(dict_y1_2.values())))
y1_2_max = list(map(func, max(dict_y1_2.values())))
elements1_2 = len(data1_2)
x1_min = min(int(x1_1_min[0]), int(x1_2_min[0]))
x1_max = max(int(x1_1_max[0]), int(x1_2_max[0]))
y1_min = min(int(y1_1_min[0]), int(y1_2_min[0]))
y1_max = max(int(y1_1_max[0]), int(y1_2_max[0]))

# grid definition for output field
x_1 = np.arange(x1_min, x1_max, 1)
y_1 = np.arange(y1_min, y1_max, 1)

model1_1 = gs.Gaussian(dim=2, var=0.01, len_scale=[15,0.1], anis=0.5, angles=-0.5)
krige1_1 = gs.Krige(model1_1, cond_pos=cond_pos1_1, cond_val=cond_val1_1)
cond_srf1_1 = gs.CondSRF(krige1_1)
cond_srf1_1.set_pos([x_1, y_1], "structured")

seed = gs.random.MasterRNG(20170519)
ens_no = 1
dict_C_for1, dict_C_fordan, dict_Fai_for1, dict_Fai_fordan = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
C_for1, C_fordan, Fai_for1, Fai_fordan = [],[],[],[]
for i in range(ens_no):
	cond_srf1_1(seed=seed(), store=[f"fld{i}", False, False])
	for j in range(elements1_1):
		m1_1 = data1_1[j].split(',')[0:1]
		keyword = int(m1_1[0])
		a1_1 = int(round(dict_ncs1_1[keyword][0],0))
		b1_1 = int(round(dict_ncs1_1[keyword][1],0))
		c1_1 = a1_1-int(x1_1_min[0])
		d1_1 = b1_1-int(y1_1_min[0])
		dict_C_for1[i].append(round(cond_srf1_1[i][c1_1][d1_1], 2)*1e6)
	C_for1.append(','.join(str(m) for m in dict_C_for1[i]))
	for k in range(elements1_2):
		m1_2 = data1_2[k].split(',')[0:1]
		keyword = int(m1_2[0])
		a1_2 = int(round(dict_ncs1_2[keyword][0],0))
		b1_2 = int(round(dict_ncs1_2[keyword][1],0))
		c1_2 = a1_2-int(x1_2_min[0])
		d1_2 = b1_2-int(y1_2_min[0])
		dict_C_fordan[i].append(round(cond_srf1_1[i][c1_2][d1_2], 2)*1e6)
	C_fordan.append(','.join(str(n) for n in dict_C_fordan[i]))

ele_m1, ele_st1 = 35, 2.8
#--------------------------------------
mat = np.random.normal(size=(ens_no, 1))
sdln1_2 = math.log(1+(ele_st1*ele_st1)/(ele_m1*ele_m1))
sdln1 = math.sqrt(sdln1_2)
mln1 = math.log(ele_m1) - 0.5*sdln1_2
mat = math.e**(mln1+sdln1*mat)
mat = np.where(mat == 0., ele_m1, mat)
for1 = np.ones((1,elements1_1))
fordan = np.ones((1,elements1_2))
mat1_1 = np.dot(mat, for1)
mat1_2 = np.dot(mat, fordan)
my_for1 = np.around(mat1_1, decimals=2)
my_fordan = np.around(mat1_2, decimals=2)
for i in range(ens_no):
	for j in range(elements1_1):
		dict_Fai_for1[i].append(round(my_for1[i][j],2))
	Fai_for1.append(','.join(str(n) for n in dict_Fai_for1[i]))
	for k in range(elements1_2):
		dict_Fai_fordan[i].append(round(my_fordan[i][k],2))
	Fai_fordan.append(','.join(str(n) for n in dict_Fai_fordan[i]))
with open( 'TXT/txt(15_0.1)/myC_for1.txt' , 'w') as fileObject1_1, open( 'TXT/txt(15_0.1)/myC_fordan.txt' , 'w') as fileObject1_2, \
	open( 'TXT/txt(15_0.1)/myFai_for1.txt' , 'w') as fileObject1_3, open( 'TXT/txt(15_0.1)/myFai_fordan.txt' , 'w') as fileObject1_4:
	ucs1_1 = '\n'.join(C_for1)
	fileObject1_1.write(ucs1_1)
	ucs1_2 = '\n'.join(C_fordan)
	fileObject1_2.write(ucs1_2)
	ucs1_3 = '\n'.join(Fai_for1)
	fileObject1_3.write(ucs1_3)
	ucs1_4 = '\n'.join(Fai_fordan)
	fileObject1_4.write(ucs1_4)
#--------------------------------------
#Ⅲ2
# conditioning data (x, y, value)
cond_pos2_1 = [[341, 360], [402, 402]]
cond_val2_1 = [1.3, 0.43]
#--------------------------------------
with open( 'unit coordinates_thi2.txt' ,'r') as f2_1:
	data2_1 = []
	line2_1 = f2_1.readlines()
	for i, rows in enumerate(line2_1):
		if i in range(0,len(line2_1)) :
			data2_1.append(rows)
dict_x2_1 = defaultdict(list) 
dict_y2_1 = defaultdict(list)
dict_ncs2_1 = defaultdict(list)
for i in range(len(data2_1)):
	a = data2_1[i].split(',')[0:1]
	b = int(a[0])
	c = data2_1[i].split(',')[1:2]
	d = float(c[0])
	dict_x2_1[b].append(d)
	dict_ncs2_1[b].append(d)
	e = data2_1[i].split(',')[2:3]
	f = float(e[0])
	dict_y2_1[b].append(f)
	dict_ncs2_1[b].append(f)
func = lambda x: round(x,0)
x2_1_min = list(map(func, min(dict_x2_1.values())))
x2_1_max = list(map(func, max(dict_x2_1.values())))
y2_1_min = list(map(func, min(dict_y2_1.values())))
y2_1_max = list(map(func, max(dict_y2_1.values())))
elements2_1 = len(data2_1)
with open( 'unit coordinates_thi2dan.txt' ,'r') as f2_2:
	data2_2 = []
	line2_2 = f2_2.readlines()
	for i, rows in enumerate(line2_2):
		if i in range(0,len(line2_2)) :
			data2_2.append(rows)
dict_x2_2 = defaultdict(list) 
dict_y2_2 = defaultdict(list)
dict_ncs2_2 = defaultdict(list)
for i in range(len(data2_2)):
	a = data2_2[i].split(',')[0:1]
	b = int(a[0])
	c = data2_2[i].split(',')[1:2]
	d = float(c[0])
	dict_x2_2[b].append(d)
	dict_ncs2_2[b].append(d)
	e = data2_2[i].split(',')[2:3]
	f = float(e[0])
	dict_y2_2[b].append(f)
	dict_ncs2_2[b].append(f)
x2_2_min = list(map(func, min(dict_x2_2.values())))
x2_2_max = list(map(func, max(dict_x2_2.values())))
y2_2_min = list(map(func, min(dict_y2_2.values())))
y2_2_max = list(map(func, max(dict_y2_2.values())))
elements2_2 = len(data2_2)
x2_min = min(int(x2_1_min[0]), int(x2_2_min[0]))
x2_max = max(int(x2_1_max[0]), int(x2_2_max[0]))
y2_min = min(int(y2_1_min[0]), int(y2_2_min[0]))
y2_max = max(int(y2_1_max[0]), int(y2_2_max[0]))

# grid definition for output field
x_2 = np.arange(x2_min, x2_max, 1)
y_2 = np.arange(y2_min, y2_max, 1)

model2_1 = gs.Gaussian(dim=2, var=0.0169, len_scale=[15,0.1], anis=0.5, angles=-0.5)
krige2_1 = gs.Krige(model2_1, cond_pos=cond_pos2_1, cond_val=cond_val2_1)
cond_srf2_1 = gs.CondSRF(krige2_1)
cond_srf2_1.set_pos([x_2, y_2], "structured")

seed = gs.random.MasterRNG(20170519)
dict_C_thi2, dict_C_thi2dan, dict_Fai_thi2, dict_Fai_thi2dan = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
C_thi2, C_thi2dan, Fai_thi2, Fai_thi2dan = [],[],[],[]
for i in range(ens_no):
	cond_srf2_1(seed=seed(), store=[f"fld{i}", False, False])
	for j in range(elements2_1):
		m2_1 = data2_1[j].split(',')[0:1]
		keyword = int(m2_1[0])
		a2_1 = int(round(dict_ncs2_1[keyword][0],0))
		b2_1 = int(round(dict_ncs2_1[keyword][1],0))
		c2_1 = a2_1-int(x2_1_min[0])       #index
		d2_1 = b2_1-int(y2_1_min[0])
		dict_C_thi2[i].append(round(cond_srf2_1[i][c2_1][d2_1], 2)*1e6)
	C_thi2.append(','.join(str(m) for m in dict_C_thi2[i]))
	for k in range(elements2_2):
		m2_2 = data2_2[k].split(',')[0:1]
		keyword = int(m2_2[0])
		a2_2 = int(round(dict_ncs2_2[keyword][0],0))
		b2_2 = int(round(dict_ncs2_2[keyword][1],0))
		c2_2 = a2_2-int(x2_2_min[0])
		d2_2 = b2_2-int(y2_2_min[0])
		dict_C_thi2dan[i].append(round(cond_srf2_1[i][c2_2][d2_2], 2)*1e6)
	C_thi2dan.append(','.join(str(n) for n in dict_C_thi2dan[i]))

ele_m2, ele_st2 = 45.6, 4.56
#--------------------------------------
mat = np.random.normal(size=(ens_no, 1))
sdln2_2 = math.log(1+(ele_st2*ele_st2)/(ele_m2*ele_m2))
sdln2 = math.sqrt(sdln2_2)
mln2 = math.log(ele_m2) - 0.5*sdln2_2
mat = math.e**(mln2+sdln2*mat)
mat = np.where(mat == 0., ele_m2, mat)
thi2 = np.ones((1,elements2_1))
thi2dan = np.ones((1,elements2_2))
mat2_1 = np.dot(mat, thi2)
mat2_2 = np.dot(mat, thi2dan)
my_thi2 = np.around(mat2_1, decimals=2)
my_thi2dan = np.around(mat2_2, decimals=2)
for i in range(ens_no):
	for j in range(elements2_1):
		dict_Fai_thi2[i].append(round(my_thi2[i][j],2))
	Fai_thi2.append(','.join(str(n) for n in dict_Fai_thi2[i]))
	for k in range(elements2_2):
		dict_Fai_thi2dan[i].append(round(my_thi2dan[i][k],2))
	Fai_thi2dan.append(','.join(str(n) for n in dict_Fai_thi2dan[i]))
with open( 'TXT/txt(15_0.1)/myC_thi2.txt' , 'w') as fileObject2_1, open( 'TXT/txt(15_0.1)/myC_thi2dan.txt' , 'w') as fileObject2_2, \
	open( 'TXT/txt(15_0.1)/myFai_thi2.txt' , 'w') as fileObject2_3, open( 'TXT/txt(15_0.1)/myFai_thi2dan.txt' , 'w') as fileObject2_4:
	ucs2_1 = '\n'.join(C_thi2)
	fileObject2_1.write(ucs2_1)
	ucs2_2 = '\n'.join(C_thi2dan)
	fileObject2_2.write(ucs2_2)
	ucs2_3 = '\n'.join(Fai_thi2)
	fileObject2_3.write(ucs2_3)
	ucs2_4 = '\n'.join(Fai_thi2dan)
	fileObject2_4.write(ucs2_4)
#--------------------------------------
#Ⅲ1
# conditioning data (x, y, value)
cond_pos3_1 = [[418, 365], [290, 346]]
cond_val3_1 = [1.85, 1.18]
cond_pos3_2 = [[418, 365], [290, 346]]
cond_val3_2 = [43.53, 42.92]
#--------------------------------------
with open( 'unit coordinates_thi1L.txt' ,'r') as f3_1:
	data3_1 = []
	line3_1 = f3_1.readlines()
	for i, rows in enumerate(line3_1):
		if i in range(0,len(line3_1)) :
			data3_1.append(rows)
dict_x3_1 = defaultdict(list) 
dict_y3_1 = defaultdict(list)
dict_ncs3_1 = defaultdict(list)
for i in range(len(data3_1)):
	a = data3_1[i].split(',')[0:1]
	b = int(a[0])
	c = data3_1[i].split(',')[1:2]
	d = float(c[0])
	dict_x3_1[b].append(d)
	dict_ncs3_1[b].append(d)
	e = data3_1[i].split(',')[2:3]
	f = float(e[0])
	dict_y3_1[b].append(f)
	dict_ncs3_1[b].append(f)
func = lambda x: round(x,0)
x3_1_min = list(map(func, min(dict_x3_1.values())))
x3_1_max = list(map(func, max(dict_x3_1.values())))
y3_1_min = list(map(func, min(dict_y3_1.values())))
y3_1_max = list(map(func, max(dict_y3_1.values())))
elements3_1 = len(data3_1)
with open( 'unit coordinates_thi1M.txt' ,'r') as f3_2:
	data3_2 = []
	line3_2 = f3_2.readlines()
	for i, rows in enumerate(line3_2):
		if i in range(0,len(line3_2)) :
			data3_2.append(rows)
dict_x3_2 = defaultdict(list) 
dict_y3_2 = defaultdict(list)
dict_ncs3_2 = defaultdict(list)
for i in range(len(data3_2)):
	a = data3_2[i].split(',')[0:1]
	b = int(a[0])
	c = data3_2[i].split(',')[1:2]
	d = float(c[0])
	dict_x3_2[b].append(d)
	dict_ncs3_2[b].append(d)
	e = data3_2[i].split(',')[2:3]
	f = float(e[0])
	dict_y3_2[b].append(f)
	dict_ncs3_2[b].append(f)
x3_2_min = list(map(func, min(dict_x3_2.values())))
x3_2_max = list(map(func, max(dict_x3_2.values())))
y3_2_min = list(map(func, min(dict_y3_2.values())))
y3_2_max = list(map(func, max(dict_y3_2.values())))
elements3_2 = len(data3_2)
with open( 'unit coordinates_thi1R.txt' ,'r') as f3_3:
	data3_3 = []
	line3_3 = f3_3.readlines()
	for i, rows in enumerate(line3_3):
		if i in range(0,len(line3_3)) :
			data3_3.append(rows)
dict_x3_3 = defaultdict(list) 
dict_y3_3 = defaultdict(list)
dict_ncs3_3 = defaultdict(list)
for i in range(len(data3_3)):
	a = data3_3[i].split(',')[0:1]
	b = int(a[0])
	c = data3_3[i].split(',')[1:2]
	d = float(c[0])
	dict_x3_3[b].append(d)
	dict_ncs3_3[b].append(d)
	e = data3_3[i].split(',')[2:3]
	f = float(e[0])
	dict_y3_3[b].append(f)
	dict_ncs3_3[b].append(f)
x3_3_min = list(map(func, min(dict_x3_3.values())))
x3_3_max = list(map(func, max(dict_x3_3.values())))
y3_3_min = list(map(func, min(dict_y3_3.values())))
y3_3_max = list(map(func, max(dict_y3_3.values())))
elements3_3 = len(data3_3)
x3_min = min(int(x3_1_min[0]), int(x3_2_min[0]), int(x3_3_min[0]))
x3_max = max(int(x3_1_max[0]), int(x3_2_max[0]), int(x3_3_max[0]))
y3_min = min(int(y3_1_min[0]), int(y3_2_min[0]), int(y3_3_min[0]))
y3_max = max(int(y3_1_max[0]), int(y3_2_max[0]), int(y3_3_max[0]))

# grid definition for output field
x_3 = np.arange(x3_min, x3_max, 1)
y_3 = np.arange(y3_min, y3_max, 1)

model3_1 = gs.Gaussian(dim=2, var=0.1296, len_scale=[15,0.1], anis=0.5, angles=-0.5)
krige3_1 = gs.Krige(model3_1, cond_pos=cond_pos3_1, cond_val=cond_val3_1)
model3_2 = gs.Gaussian(dim=2, var=71.57, len_scale=[15,0.1], anis=0.5, angles=-0.5)
krige3_2 = gs.Krige(model3_2, cond_pos=cond_pos3_2, cond_val=cond_val3_2)
cond_srf3_1 = gs.CondSRF(krige3_1)
cond_srf3_1.set_pos([x_3, y_3], "structured")
cond_srf3_2 = gs.CondSRF(krige3_2)
cond_srf3_2.set_pos([x_3, y_3], "structured")

seed = gs.random.MasterRNG(20170519)
dict_C_thi1L, dict_C_thi1M, dict_C_thi1R, dict_Fai_thi1L, dict_Fai_thi1M, dict_Fai_thi1R = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
C_thi1L, C_thi1M, C_thi1R, Fai_thi1L, Fai_thi1M, Fai_thi1R = [],[],[],[],[],[]
for i in range(ens_no):
	cond_srf3_1(seed=seed(), store=[f"fld{i}", False, False])
	cond_srf3_2(seed=seed(), store=[f"fld{i}", False, False])
	for j in range(elements3_1):
		m3_1 = data3_1[j].split(',')[0:1]
		keyword = int(m3_1[0])
		a3_1 = int(round(dict_ncs3_1[keyword][0],0))
		b3_1 = int(round(dict_ncs3_1[keyword][1],0))
		c3_1 = a3_1-int(x3_1_min[0])       #index
		d3_1 = b3_1-int(y3_1_min[0])
		dict_C_thi1L[i].append(round(cond_srf3_1[i][c3_1][d3_1], 2)*1e6)
		dict_Fai_thi1L[i].append(round(cond_srf3_2[i][c3_1][d3_1], 2))
	C_thi1L.append(','.join(str(m) for m in dict_C_thi1L[i]))
	Fai_thi1L.append(','.join(str(m) for m in dict_Fai_thi1L[i]))
	for k in range(elements3_2):
		m3_2 = data3_2[k].split(',')[0:1]
		keyword = int(m3_2[0])
		a3_2 = int(round(dict_ncs3_2[keyword][0],0))
		b3_2 = int(round(dict_ncs3_2[keyword][1],0))
		c3_2 = a3_2-int(x3_2_min[0])       #index
		d3_2 = b3_2-int(y3_2_min[0])
		dict_C_thi1M[i].append(round(cond_srf3_1[i][c3_2][d3_2], 2)*1e6)
		dict_Fai_thi1M[i].append(round(cond_srf3_2[i][c3_2][d3_2], 2))
	C_thi1M.append(','.join(str(n) for n in dict_C_thi1M[i]))
	Fai_thi1M.append(','.join(str(n) for n in dict_Fai_thi1M[i]))
	for l in range(elements3_3):
		m3_3 = data3_3[l].split(',')[0:1]
		keyword = int(m3_3[0])
		a3_3 = int(round(dict_ncs3_3[keyword][0],0))
		b3_3 = int(round(dict_ncs3_3[keyword][1],0))
		c3_3 = a3_3-int(x3_3_min[0])       #index
		d3_3 = b3_3-int(y3_3_min[0])
		dict_C_thi1R[i].append(round(cond_srf3_1[i][c3_3][d3_3], 2)*1e6)
		dict_Fai_thi1R[i].append(round(cond_srf3_2[i][c3_3][d3_3], 2))
	C_thi1R.append(','.join(str(mn) for mn in dict_C_thi1R[i]))
	Fai_thi1R.append(','.join(str(mn) for mn in dict_Fai_thi1R[i]))
with open( 'TXT/txt(15_0.1)/myC_thi1L.txt' , 'w') as fileObject3_1, open( 'TXT/txt(15_0.1)/myC_thi1M.txt' , 'w') as fileObject3_2, open( 'TXT/txt(15_0.1)/myC_thi1R.txt' , 'w') as fileObject3_3, \
	open( 'TXT/txt(15_0.1)/myFai_thi1L.txt' , 'w') as fileObject3_4, open( 'TXT/txt(15_0.1)/myFai_thi1M.txt' , 'w') as fileObject3_5, open( 'TXT/txt(15_0.1)/myFai_thi1R.txt' , 'w') as fileObject3_6:
	ucs3_1 = '\n'.join(C_thi1L)
	fileObject3_1.write(ucs3_1)
	ucs3_2 = '\n'.join(C_thi1M)
	fileObject3_2.write(ucs3_2)
	ucs3_3 = '\n'.join(C_thi1R)
	fileObject3_3.write(ucs3_3)
	ucs3_4 = '\n'.join(Fai_thi1L)
	fileObject3_4.write(ucs3_4)
	ucs3_5 = '\n'.join(Fai_thi1M)
	fileObject3_5.write(ucs3_5)
	ucs3_6 = '\n'.join(Fai_thi1R)
	fileObject3_6.write(ucs3_6)
#--------------------------------------
#f
#--------------------------------------
with open( 'unit coordinates_f.txt' ,'r') as f4:
	data4 = []
	line4 = f4.readlines()
	for i, rows in enumerate(line4):
		if i in range(0,len(line4)) :
			data4.append(rows)
dict_C_f, dict_Fai_f = defaultdict(list), defaultdict(list)
C_f, Fai_f= [],[]
elements4 = len(data4)
ele_m4_1, ele_st4_1, ele_m4_2, ele_st4_2 = 0.3, 0.05, 25.2, 0.5
#--------------------------------------
mat1 = np.random.normal(size=(ens_no, 1))
mat2 = np.random.normal(size=(ens_no, 1))
sdln4_1_2 = math.log(1+(ele_st4_1*ele_st4_1)/(ele_m4_1*ele_m4_1))
sdln4_2_2 = math.log(1+(ele_st4_2*ele_st4_2)/(ele_m4_2*ele_m4_2))
sdln4_1 = math.sqrt(sdln4_1_2)
sdln4_2 = math.sqrt(sdln4_2_2)
mln4_1 = math.log(ele_m4_1) - 0.5*sdln4_1_2
mln4_2 = math.log(ele_m4_2) - 0.5*sdln4_2_2
mat4_1 = math.e**(mln4_1+sdln4_1*mat1)
mat4_2 = math.e**(mln4_2+sdln4_2*mat2)
mat4_1 = np.where(mat4_1 == 0., ele_m4_1, mat4_1)
mat4_2 = np.where(mat4_2 == 0., ele_m4_2, mat4_2)
f_Fai = np.ones((1,elements4))
f_C = np.ones((1,elements4))
mat4_1 = np.dot(mat4_1, f_Fai)
mat4_2 = np.dot(mat4_2, f_C)
my_C_f = np.around(mat4_1, decimals=2)
my_Fai_f = np.around(mat4_2, decimals=2)
for i in range(ens_no):
	for j in range(elements4):
		dict_C_f[i].append(round(my_C_f[i][j],2)*1e6)
		dict_Fai_f[i].append(round(my_Fai_f[i][j],2))
	C_f.append(','.join(str(n) for n in dict_C_f[i]))
	Fai_f.append(','.join(str(n) for n in dict_Fai_f[i]))
with open( 'TXT/txt(15_0.1)/myC_f.txt' , 'w') as fileObject4_1, open( 'TXT/txt(15_0.1)/myFai_f.txt' , 'w') as fileObject4_2:
	ucs4_1 = '\n'.join(C_f)
	fileObject4_1.write(ucs4_1)
	ucs4_2 = '\n'.join(Fai_f)
	fileObject4_2.write(ucs4_2)
#--------------------------------------
#x
#--------------------------------------
with open( 'unit coordinates_x.txt' ,'r') as f5:
	data5 = []
	line5 = f5.readlines()
	for i, rows in enumerate(line5):
		if i in range(0,len(line5)) :
			data5.append(rows)
dict_C_x, dict_Fai_x = defaultdict(list), defaultdict(list)
C_x, Fai_x= [],[]
elements5 = len(data5)
ele_m5_1, ele_st5_1, ele_m5_2, ele_st5_2 = 0.4, 0.08, 31, 2.325
#--------------------------------------
mat1 = np.random.normal(size=(ens_no, 1))
mat2 = np.random.normal(size=(ens_no, 1))
sdln5_1_2 = math.log(1+(ele_st5_1*ele_st5_1)/(ele_m5_1*ele_m5_1))
sdln5_2_2 = math.log(1+(ele_st5_2*ele_st5_2)/(ele_m5_2*ele_m5_2))
sdln5_1 = math.sqrt(sdln5_1_2)
sdln5_2 = math.sqrt(sdln5_2_2)
mln5_1 = math.log(ele_m5_1) - 0.5*sdln5_1_2
mln5_2 = math.log(ele_m5_2) - 0.5*sdln5_2_2
mat5_1 = math.e**(mln5_1+sdln5_1*mat1)
mat5_2 = math.e**(mln5_2+sdln5_2*mat2)
mat5_1 = np.where(mat5_1 == 0., ele_m5_1, mat5_1)
mat5_2 = np.where(mat5_2 == 0., ele_m5_2, mat5_2)
x_C = np.ones((1,elements5))
x_Fai = np.ones((1,elements5))
mat5_1 = np.dot(mat5_1, x_C)
mat5_2 = np.dot(mat5_2, x_Fai)
my_C_x = np.around(mat5_1, decimals=2)
my_Fai_x = np.around(mat5_2, decimals=2)
for i in range(ens_no):
	for j in range(elements5):
		dict_C_x[i].append(round(my_C_x[i][j],2)*1e6)
		dict_Fai_x[i].append(round(my_Fai_x[i][j],2))
	C_x.append(','.join(str(n) for n in dict_C_x[i]))
	Fai_x.append(','.join(str(n) for n in dict_Fai_x[i]))
with open( 'TXT/txt(15_0.1)/myC_x.txt' , 'w') as fileObject5_1, open( 'TXT/txt(15_0.1)/myFai_x.txt' , 'w') as fileObject5_2:
	ucs5_1 = '\n'.join(C_x)
	fileObject5_1.write(ucs5_1)
	ucs5_2 = '\n'.join(Fai_x)
	fileObject5_2.write(ucs5_2)
#--------------------------------------
#Ⅱ
#--------------------------------------
with open( 'unit coordinates_sec.txt' ,'r') as f6:
	data6 = []
	line6 = f6.readlines()
	for i, rows in enumerate(line6):
		if i in range(0,len(line6)) :
			data6.append(rows)
dict_C_sec, dict_Fai_sec = defaultdict(list), defaultdict(list)
C_sec, Fai_sec= [],[]
elements6 = len(data6)
ele_m6_1, ele_st6_1, ele_m6_2, ele_st6_2 = 2., 0.26, 53.5, 8.025
#--------------------------------------
mat1 = np.random.normal(size=(ens_no, 1))
mat2 = np.random.normal(size=(ens_no, 1))
sdln6_1_2 = math.log(1+(ele_st6_1*ele_st6_1)/(ele_m6_1*ele_m6_1))
sdln6_2_2 = math.log(1+(ele_st6_2*ele_st6_2)/(ele_m6_2*ele_m6_2))
sdln6_1 = math.sqrt(sdln6_1_2)
sdln6_2 = math.sqrt(sdln6_2_2)
mln6_1 = math.log(ele_m6_1) - 0.5*sdln6_1_2
mln6_2 = math.log(ele_m6_2) - 0.5*sdln6_2_2
mat6_1 = math.e**(mln6_1+sdln6_1*mat1)
mat6_2 = math.e**(mln6_2+sdln6_2*mat2)
mat6_1 = np.where(mat6_1 == 0., ele_m6_1, mat6_1)
mat6_2 = np.where(mat6_2 == 0., ele_m6_2, mat6_2)
sec_C = np.ones((1,elements6))
sec_Fai = np.ones((1,elements6))
mat6_1 = np.dot(mat6_1, sec_C)
mat6_2 = np.dot(mat6_2, sec_Fai)
my_C_sec = np.around(mat6_1, decimals=2)
my_Fai_sec = np.around(mat6_2, decimals=2)
for i in range(ens_no):
	for j in range(elements6):
		dict_C_sec[i].append(round(my_C_sec[i][j],2)*1e6)
		dict_Fai_sec[i].append(round(my_Fai_sec[i][j],2))
	C_sec.append(','.join(str(n) for n in dict_C_sec[i]))
	Fai_sec.append(','.join(str(n) for n in dict_Fai_sec[i]))
with open( 'TXT/txt(15_0.1)/myC_sec.txt' , 'w') as fileObject6_1, open( 'TXT/txt(15_0.1)/myFai_sec.txt' , 'w') as fileObject6_2:
	ucs6_1 = '\n'.join(C_sec)
	fileObject6_1.write(ucs6_1)
	ucs6_2 = '\n'.join(Fai_sec)
	fileObject6_2.write(ucs6_2)