#Import packages
import sympy

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
# Read the essay and change the data
lamdall = 0.7466
lamdalm = 0.2533
lamdalh = 1-lamdall-lamdalm
lamdaml = 0.0592
lamdamm = 0.9249
lamdamh = 1 - lamdamm-lamdaml
lamdahl = 0
lamdahm = 0.0322
lamdahh = 1 - lamdahl - lamdahm 
sigmap= pow(sigmap1**2+sigmap2**2,0.5)
roups = sigmap1/sigmap
sigmap_fang= pow(sigmap,2)
# 0.2 derived parameters
roul = 0.5*(-2*rho+beta*(-1+gamma)*(-2*miul+(1+beta*(gamma-1))*sigmap_fang))
roum = 0.5*(-2*rho+beta*(-1+gamma)*(-2*mium+(1+beta*(gamma-1))*sigmap_fang))
rouh = 0.5*(-2*rho+beta*(-1+gamma)*(-2*miuh+(1+beta*(gamma-1))*sigmap_fang))
para_1_l = r-miul+sigmap_fang*(1+beta*(gamma-1))
para_1_m = r-mium+sigmap_fang*(1+beta*(gamma-1))
para_1_h = r-miuh+sigmap_fang*(1+beta*(gamma-1))
rousigmasigma = roups*sigma_s*sigmap
para_2 = alpha_s-r-(1+beta*(gamma-1))*rousigmasigma


"""
1 Equation Pattern 
"""
(v_l,c_l,h_l,seta_l,v_m,c_m,h_m,seta_m,v_h,c_h,h_h,seta_h)= sympy.symbols('v_l,c_l,h_l,seta_l,v_m,c_m,h_m,seta_m,v_h,c_h,h_h,seta_h')
(alpha_v_l,alpha_c_l,alpha_h_l,alpha_seta_l,alpha_v_m,alpha_c_m,alpha_h_m,alpha_seta_m,alpha_v_h,alpha_c_h,alpha_h_h,alpha_seta_h,x)= sympy.symbols('alpha_v_l,alpha_c_l,alpha_h_l,alpha_seta_l,alpha_v_m,alpha_c_m,alpha_h_m,alpha_seta_m,alpha_v_h,alpha_c_h,alpha_h_h,alpha_seta_h,x')
(u_l,u_m,u_h) = sympy.symbols('u_l,u_m,u_h')
(D_l,D_m,D_h) = sympy.symbols('D_l,D_m,D_h')
(eq_l,eq_m,eq_h) = sympy.symbols('eq_l,eq_m,eq_h')

# 1.1 value function
v_l = alpha_v_l*(pow(x,1-gamma)/(1-gamma))
v_m = alpha_v_m*(pow(x,1-gamma)/(1-gamma))
v_h = alpha_v_h*(pow(x,1-gamma)/(1-gamma))
# 1.2 optimal consumption
c_l = alpha_c_l*x
c_m = alpha_c_m*x
c_h = alpha_c_h*x
# 1.3 optimal housing
h_l = alpha_h_l*x
h_m = alpha_h_m*x
h_h = alpha_h_h*x
# 1.4 optimal risk investment
seta_l = alpha_seta_l*x
seta_m = alpha_seta_m*x
seta_h = alpha_seta_h*x
# 1.5 utility in three periods
u_l = (pow((pow(c_l,beta)*pow(h_l,1-beta)),1-gamma))/(1-gamma)
u_m = (pow((pow(c_m,beta)*pow(h_m,1-beta)),1-gamma))/(1-gamma)
u_h = (pow((pow(c_h,beta)*pow(h_h,1-beta)),1-gamma))/(1-gamma)
# 1.6 derivatives of value function in three periods
D_l = ((x-h_l)*para_1_l+seta_l*para_2-c_l)*sympy.diff(v_l,x)+0.5*(pow(((x-h_l)*sigmap),2)-2*(x-h_l)*seta_l*rousigmasigma+(seta_l**2)*(sigma_s**2))*sympy.diff(sympy.diff(v_l,x),x)
D_m = ((x-h_m)*para_1_m+seta_m*para_2-c_m)*sympy.diff(v_m,x)+0.5*(pow(((x-h_m)*sigmap),2)-2*(x-h_m)*seta_m*rousigmasigma+(seta_m**2)*(sigma_s**2))*sympy.diff(sympy.diff(v_m,x),x)
D_h = ((x-h_h)*para_1_h+seta_h*para_2-c_h)*sympy.diff(v_h,x)+0.5*(pow(((x-h_h)*sigmap),2)-2*(x-h_h)*seta_h*rousigmasigma+(seta_h**2)*(sigma_s**2))*sympy.diff(sympy.diff(v_h,x),x)
# 1.7 original equations
eq_l = -roul*v_l+u_l+D_l+lamdalm*(v_m-v_l)+lamdalh*(v_h-v_l)
eq_l = eq_l.replace('x',1)
eq_m = -roum*v_m+u_m+D_m+lamdaml*(v_l-v_m)+lamdamh*(v_h-v_m)
eq_m = eq_m.replace('x',1)
eq_h = -rouh*v_h+u_h+D_h+lamdahl*(v_l-v_h)+lamdahm*(v_m-v_h)
eq_h = eq_h.replace('x',1)
# 1.8 Derived equations
eq_l_c = sympy.diff(eq_l,alpha_c_l)
eq_l_seta = sympy.diff(eq_l,alpha_seta_l)
eq_l_h = sympy.diff(eq_l,alpha_h_l)

eq_m_c = sympy.diff(eq_m,alpha_c_m)
eq_m_seta = sympy.diff(eq_m,alpha_seta_m)
eq_m_h = sympy.diff(eq_m,alpha_h_m)

eq_h_c = sympy.diff(eq_h,alpha_c_h)
eq_h_seta = sympy.diff(eq_h,alpha_seta_h)
eq_h_h = sympy.diff(eq_h,alpha_h_h)

# 1.9 Add up total_eq
total_eq = eq_l**2+eq_l_c**2+eq_l_seta**2+eq_l_h**2+eq_m**2+eq_m_c**2+eq_m_seta**2+eq_m_h**2+eq_h**2+eq_h_c**2+eq_h_seta**2+eq_h_h**2

# 1.10 Solve the equation
paras_initial=[('alpha_c_l',0.032),('alpha_c_m',0.017),('alpha_c_h',0.023),('alpha_h_l',0.136425),('alpha_h_m',0.33783),('alpha_h_h',1.587301),('alpha_seta_l',0.242),('alpha_seta_m',0.212),('alpha_seta_h',0.182)]
initial_dic = {}
for items in paras_initial:
    initial_dic.update({items[0]:items[1]}) 

tem_eq = total_eq.copy()
for keys in initial_dic.keys():
    tem_eq = tem_eq.replace(keys,initial_dic[keys])
    
dif_l = sympy.diff(tem_eq,alpha_v_l)  
dif_m = sympy.diff(tem_eq,alpha_v_m)  
dif_h = sympy.diff(tem_eq,alpha_v_h)  

value_dic = sympy.solve([dif_l,dif_m,dif_h],[alpha_v_l, alpha_v_m,alpha_v_h])
print(value_dic)

"""
for key in value_dic.keys():
    tem_eq = tem_eq.replace(key,value_dic[key])
print(tem_eq)
"""