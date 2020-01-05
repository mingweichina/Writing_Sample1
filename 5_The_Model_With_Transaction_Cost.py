#Import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sympy
import os


"""
0 Parameters
"""
# 0.1 basic parameters
gamma = 10
rho = 0.025
beta = 0.7
r = 0.015
alpha_s = 0.077
sigma_s = 0.1655
ibtio = 0.10  #Cost of moving
k = 0.03
kesei = 0
sigmap1 = 0.0383
sigmap2 = 0.05
miul = -0.0134*4
mium = 0.0017*4
miuh = 0.0131*4
lamdall = 0.7466
lamdalm = 0.2533
lamdalh = 1-lamdall-lamdalm
lamdaml = 0.0592
lamdamm = 0.9249
lamdamh = 1-lamdamm-lamdaml
lamdahl = 0
lamdahm = 0.0322
lamdahh = 1 -lamdahl-lamdahm 
sigmap= pow(sigmap1**2+sigmap2**2,0.5)
roups = sigmap1/sigmap
sigmap_fang= pow(sigmap,2)
rousigmasigma = roups*sigma_s*sigmap

# 0.2 derived parameters
roul = 0.5*(-2*rho-2*(gamma-1)*(miul-kesei+beta*(gamma-1)*(1+beta*(gamma-1))*sigmap_fang))
roum = 0.5*(-2*rho-2*(gamma-1)*(mium-kesei+beta*(gamma-1)*(1+beta*(gamma-1))*sigmap_fang))
rouh = 0.5*(-2*rho-2*(gamma-1)*(miuh-kesei+beta*(gamma-1)*(1+beta*(gamma-1))*sigmap_fang))
para_1_l = r+kesei-miul+sigmap_fang*(1+beta*(gamma-1))
para_1_m = r+kesei-mium+sigmap_fang*(1+beta*(gamma-1))
para_1_h = r+kesei-miuh+sigmap_fang*(1+beta*(gamma-1))
para_2 = alpha_s-r-(1+beta*(gamma-1))*rousigmasigma
omega = (alpha_s-r+(1-beta*(1-gamma))*roups*sigmap)/(sigma_s**2)


"""
1 Equations
"""
(v_l,v_m,v_h,v_z_l,v_z_m,v_z_h,v_zz_l,v_zz_m,v_zz_h) = sympy.symbols('v_l,v_m,v_h,v_z_l,v_z_m,v_z_h,v_zz_l,v_zz_m,v_zz_h')
(c_l,c_m,c_h,seta_l,seta_m,seta_h,z)=sympy.symbols('c_l,c_m,c_h,seta_l,seta_m,seta_h,z')
(u_l,u_m,u_h)=sympy.symbols('u_l,u_m,u_h')
(D_l,D_m,D_h)=sympy.symbols('D_l,D_m,D_h')
(eq_l,eq_m,eq_h)=sympy.symbols('eq_l,eq_m,eq_h')
c_l=pow(v_z_l/beta,1/(beta*(1-gamma)-1))
c_m=pow(v_z_m/beta,1/(beta*(1-gamma)-1))
c_h=pow(v_z_h/beta,1/(beta*(1-gamma)-1))
seta_l = -omega*v_z_l/v_zz_l+roups*sigmap*(z-1)/sigma_s
seta_m = -omega*v_z_m/v_zz_m+roups*sigmap*(z-1)/sigma_s
seta_h = -omega*v_z_h/v_zz_h+roups*sigmap*(z-1)/sigma_s
u_l = pow(c_l,beta*(1-gamma))/(1-gamma)
u_m = pow(c_m,beta*(1-gamma))/(1-gamma)
u_h = pow(c_h,beta*(1-gamma))/(1-gamma)
D_l = ((z-1)*para_1_l+seta_l*para_2-c_l)*v_z_l+0.5*(pow((z-1)*sigmap,2)-2*(z-1)*seta_l*rousigmasigma+(seta_l*sigma_s)**2)*v_zz_l
D_m = ((z-1)*para_1_m+seta_m*para_2-c_m)*v_z_m+0.5*(pow((z-1)*sigmap,2)-2*(z-1)*seta_m*rousigmasigma+(seta_m*sigma_s)**2)*v_zz_m
D_h = ((z-1)*para_1_h+seta_h*para_2-c_h)*v_z_h+0.5*(pow((z-1)*sigmap,2)-2*(z-1)*seta_h*rousigmasigma+(seta_h*sigma_s)**2)*v_zz_h

Ml = 1.5288*(10**12)
Mm = 7.06748*(10**11)
Mh = 8.086484*(10**11)
eq_l = -(roul+k)*v_l+u_l+D_l+lamdalm*(v_m-v_l)+lamdalh*(v_h-v_l)+k*Ml*pow(z-ibtio,1-gamma)/(1-gamma)
eq_m = -(roum+k)*v_m+u_m+D_m+lamdaml*(v_l-v_m)+lamdamh*(v_h-v_m)+k*Mm*pow(z-ibtio,1-gamma)/(1-gamma)
eq_h = -(rouh+k)*v_h+u_h+D_h+lamdahm*(v_m-v_h)+lamdahl*(v_l-v_h)+k*Mh*pow(z-ibtio,1-gamma)/(1-gamma)

# 1.1 Initial values
# low
zl_optimal = 12.98
zl_down = 2.53
zl_up = 26.10
# medium
zm_optimal = 3.62
zm_down = 1.57
zm_up = 7.96
# high
zh_optimal = 2.18
zh_down = 0.45
zh_up = 4.83

# Define the start and end value of Runge Kutta method 
z_start = max([zh_up,zm_up,zl_up])
z_end = min([zh_down,zm_down,zl_down])

# Define value function involves transaction
vl_initial = Ml*((z-ibtio)**(1-gamma))/(1-gamma)
vm_initial = Mm*((z-ibtio)**(1-gamma))/(1-gamma)
vh_initial = Mh*((z-ibtio)**(1-gamma))/(1-gamma)
vl_initial_z = sympy.diff(vl_initial,z)
vm_initial_z = sympy.diff(vm_initial,z)
vh_initial_z = sympy.diff(vh_initial,z)
vl_initial_zz = sympy.diff(vl_initial_z,z)
vm_initial_zz = sympy.diff(vm_initial_z,z)
vh_initial_zz = sympy.diff(vh_initial_z,z)

# 1.2 Solving odes
vl_start = vl_initial.replace(z,z_start)
vm_start = vm_initial.replace(z,z_start)
vh_start = vh_initial.replace(z,z_start)
vl_start_z = vl_initial_z.replace(z,z_start)
vm_start_z = vm_initial_z.replace(z,z_start)
vh_start_z = vh_initial_z.replace(z,z_start)
vl_start_zz = vl_initial_zz.replace(z,z_start)
vm_start_zz = vm_initial_zz.replace(z,z_start)
vh_start_zz = vh_initial_zz.replace(z,z_start)
value_state = {}
value_state.update({'v_l':vl_start})
value_state.update({'v_m':vm_start})
value_state.update({'v_h':vh_start})
value_state.update({'v_z_l':vl_start_z})
value_state.update({'v_z_m':vm_start_z})
value_state.update({'v_z_h':vh_start_z})
value__inital_state = value_state.copy()

# 1.3 Solve Second Derivative
def solve_poly(poly_l):
    para_list = [items[1] for items in list(poly_l.as_dict().items())]
    delta = para_list[1]**2-4*para_list[0]*para_list[2]
    x_min = (-para_list[1]-pow(delta,0.5))/(2*para_list[2])
    x_max = (-para_list[1]+pow(delta,0.5))/(2*para_list[2])
    return (x_min,x_max)
    
def runge_kutta_solve(equations,value_state,z_input):
    logil = (z_input<=zl_down or z_input>=zl_up)
    logim = (z_input<=zm_down or z_input>=zm_up)  
    logih = (z_input<=zh_down or z_input>=zh_up)  
    
    if logil:
        vlzz = vl_initial_zz.replace(z,z_input)
    else:
        eq_l_tem = equations[0].copy()
        for keys in ['v_l','v_m','v_h','v_z_l','v_z_m','v_z_h']:
            eq_l_tem = eq_l_tem.replace(keys,value_state[keys])
        eq_l_tem = eq_l_tem.replace(z,z_input)  
        poly_l = sympy.poly(sympy.expand(eq_l_tem*v_zz_l))
        vlzz = solve_poly(poly_l)[0]

    if logim:
        vmzz = vm_initial_zz.replace(z,z_input)
    else:
        eq_m_tem = equations[1].copy()
        for keys in ['v_l','v_m','v_h','v_z_l','v_z_m','v_z_h']:
            eq_m_tem = eq_m_tem.replace(keys,value_state[keys])
        eq_m_tem = eq_m_tem.replace(z,z_input)  
        poly_m = sympy.poly(sympy.expand(eq_m_tem*v_zz_m))
        vmzz = solve_poly(poly_m)[0]

    if logih:
        vhzz = vh_initial_zz.replace(z,z_input)
    else:
        eq_h_tem = equations[2].copy()
        for keys in ['v_l','v_m','v_h','v_z_l','v_z_m','v_z_h']:
            eq_h_tem = eq_h_tem.replace(keys,value_state[keys])
        eq_h_tem = eq_h_tem.replace(z,z_input)  
        poly_h = sympy.poly(sympy.expand(eq_h_tem*v_zz_h))
        vhzz = solve_poly(poly_h)[0]
        
    return (value_state['v_z_l'],vlzz,value_state['v_z_m'],vmzz,value_state['v_z_h'],vhzz)

equations = [eq_l,eq_m,eq_h]
current_state = value_state.copy()
z_input = z_start
h = -0.001
y_result = []

print_num=0
while True:
    save_state = current_state.copy()
    k1 = np.array(runge_kutta_solve(equations,current_state,z_input))
    k1_update = k1*h/2    
    current_state['v_l']=save_state['v_l']+k1_update[0]
    current_state['v_z_l']=save_state['v_z_l']+k1_update[1]
    current_state['v_m']=save_state['v_m']+k1_update[2]
    current_state['v_z_m']=save_state['v_z_m']+k1_update[3]
    current_state['v_h']=save_state['v_h']+k1_update[4]
    current_state['v_z_h']=save_state['v_z_h']+k1_update[5]
    
    k2 = np.array(runge_kutta_solve(equations,current_state,z_input+(h/2)))
    k2_update = k2*h/2    
    current_state['v_l']=save_state['v_l']+k2_update[0]
    current_state['v_z_l']=save_state['v_z_l']+k2_update[1]
    current_state['v_m']=save_state['v_m']+k2_update[2]
    current_state['v_z_m']=save_state['v_z_m']+k2_update[3]
    current_state['v_h']=save_state['v_h']+k2_update[4]
    current_state['v_z_h']=save_state['v_z_h']+k2_update[5]
    
    k3 = np.array(runge_kutta_solve(equations,current_state,z_input+(h/2)))
    k3_update = k3*h    
    current_state['v_l']=save_state['v_l']+k3_update[0]
    current_state['v_z_l']=save_state['v_z_l']+k3_update[1]
    current_state['v_m']=save_state['v_m']+k3_update[2]
    current_state['v_z_m']=save_state['v_z_m']+k3_update[3]
    current_state['v_h']=save_state['v_h']+k3_update[4]
    current_state['v_z_h']=save_state['v_z_h']+k3_update[5]
    k4 = np.array(runge_kutta_solve(equations,current_state,z_input+h))    
    
    # Update the new state
    delta = ((k1+2*k2+2*k3+k4)*h)/6
    current_state['v_l']=save_state['v_l']+delta[0]
    current_state['v_z_l']=save_state['v_z_l']+delta[1]
    current_state['v_m']=save_state['v_m']+delta[2]
    current_state['v_z_m']=save_state['v_z_m']+delta[3]
    current_state['v_h']=save_state['v_h']+delta[4]
    current_state['v_z_h']=save_state['v_z_h']+delta[5]
    z_input+=h 

    # Adjust the new state
    logil = (z_input<=zl_down or z_input>=zl_up)
    logim = (z_input<=zm_down or z_input>=zm_up)  
    logih = (z_input<=zh_down or z_input>=zh_up)  

    if logil:
        current_state['v_l'] = vl_initial.replace(z,z_input)
        current_state['v_z_l'] = vl_initial_z.replace(z,z_input)
    if logim:
        current_state['v_m'] = vm_initial.replace(z,z_input)
        current_state['v_z_m'] = vm_initial_z.replace(z,z_input)
    if logih:
        current_state['v_h'] = vh_initial.replace(z,z_input)
        current_state['v_z_h'] = vh_initial_z.replace(z,z_input)
    
    standard_l = vl_initial.replace(z,z_input)
    standard_m = vm_initial.replace(z,z_input)
    standard_h = vh_initial.replace(z,z_input)
    y_result.append([current_state['v_l']-standard_l,current_state['v_m']-standard_m,current_state['v_h']-standard_h,z_input])
    if z_input<z_end:
        break
    print_num+=1
    if print_num%10==0:
        print([current_state['v_l']-standard_l,current_state['v_m']-standard_m,current_state['v_h']-standard_h,z_input])
    
tem = pd.DataFrame(y_result)
tem.columns = ['low','medium','high','z']
for column in tem.columns:
    tem[column] = np.array(tem[column]).astype(float)
tem.to_csv('original_data.csv')
   
if not os.path.exists('pictures'):
    os.mkdir('pictures')
sns.lineplot(x='z',y='low',data = tem,palette="ch:2.5,.25")
plt.savefig('pictures/low.jpg',dpi = 300)
sns.lineplot(x='z',y='medium',data = tem,palette="ch:2.5,.25")
plt.savefig('pictures/medium.jpg',dpi = 300)
sns.lineplot(x='z',y='high',data = tem,palette="ch:2.5,.25")
plt.savefig('pictures/high.jpg',dpi = 300)

