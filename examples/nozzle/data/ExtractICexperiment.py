#!/usr/bin/env python3
"""
Read data from TROVA experiment and extract initial conditions for CFD or nozzle
solver.

@author: Andrea Pinardi <andrea.pinardi@polimi.it>
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CoolProp import AbstractState
import CoolProp.CoolProp as CP
import CoolProp.Plots as CPplots
import scipy.interpolate as interp
import sys
import yaml
import shutil
TROVApkg = Path(r'C:\Andrea\Dottorato\TROVAvision')
if str(TROVApkg) not in sys.path:
    sys.path.append(str(TROVApkg))
import TROVA.postexp as tpost

plt.close('all')


#%% USER INPUT

exp_name = 'TET14'

# in experiment file:
#   - time in [s]
#   - temperatures in [°C]
#   - pressures in [bar]


#%% --------------- DO NOT TOUCH ANYTHING BELOW THIS LINE ----------------

C2K = 237.15    # [°C] -> [K]

# directory where TROVA files are stored (data, geometry, ecc)
TROVA_dir = Path(r'C:\Users\andre\OneDrive - Politecnico di Milano\Dottorato\Simulazioni_CFD\TROVA')
data_dir = TROVA_dir / 'data'
geom_dir = TROVA_dir / 'geometry'
out_dir = Path.cwd() / f'{exp_name}'

data_file = data_dir / f'{exp_name}/{exp_name}_MeanData.dat'
log_file = data_dir / f'{exp_name}/{exp_name}_log.xlsx'

# using data from the experiment
IC_exp_file = out_dir / 'IC_experiment.dat'
BC_exp_file = out_dir / 'BC_experiment.dat'
# interpolating/extrapolating experimental data
IC_extrap_file = out_dir / 'IC_extrapolated.dat'
BC_extrap_file = out_dir / 'BC_extrapolated.dat'

solve_input_file = out_dir / 'input.yaml'

out_dir.mkdir(exist_ok=True)
print(f'Data will be saved in \n\t{out_dir.resolve()}')


#%% READ TEST DATA

#%%% Data and log

print(f'Reading data from \n\t{data_file}')
# specify Python engine to avoid warning due to regex (??) used as separator
data = pd.read_csv(data_file, sep=', ', comment='#', skipinitialspace=True,
                   engine='python')
# P_4 is the pressure of F10 transducer (broken) => remove it
#del data['P_4']
log_data = tpost.read_log(log_file)

# one is for total pressure, the others for static pressure
N_probes = len(log_data['transducers_names']) - 1

solver_data = {'fluid': log_data['fluid'],
               'initial_conditions': str(IC_extrap_file.name),
               'boundary_conditions': str(BC_extrap_file.name)
               }


#%%% Nozzle profile

# name of the nozzle profile (2PH_nozzle_low_area_ratio or 2PH_nozzle_scaled)
nozzle_file = geom_dir / f"{log_data['test_section']}.dat"
nozzle = np.loadtxt(nozzle_file, delimiter=',', usecols=[0,1])
# position of probe holes
probes_file = geom_dir / f"probes_{log_data['test_section']}.dat"
probes_holes = np.loadtxt(probes_file, delimiter=',')

# get physical coordinates of mounted pressure probes
idx_used_holes = log_data['transducers_table']['transducer'].notna()
transducers = log_data['transducers_table'][idx_used_holes]
# channels connected to a probe (i.e. actually in use)
channels = np.array(transducers['channels'], dtype=int)
probes_coord = probes_holes[idx_used_holes,:]

shutil.copy(nozzle_file, out_dir/'nozzle.dat')


#%% POSTPROCESSING

FLUID = AbstractState('REFPROP', log_data['fluid'])
s_in_0 = np.zeros_like(data['TT_0'])
s_in_1 = np.zeros_like(data['TT_1'])
h_tot_0 = np.zeros_like(data['TT_0'])
h_tot_1 = np.zeros_like(data['TT_1'])
q_0 = np.zeros_like(data['TT_0'])
q_1 = np.zeros_like(data['TT_1'])
for i in range(s_in_0.shape[0]):
    FLUID.update(CP.PT_INPUTS, data['PT'][i]*1e5, data['TT_0'][i]+C2K)
    s_in_0[i] = FLUID.smass()
    q_0[i] = FLUID.Q()
    h_tot_0[i] = FLUID.hmass()
    FLUID.update(CP.PT_INPUTS, data['PT'][i]*1e5, data['TT_1'][i]+C2K)
    s_in_1[i] = FLUID.smass()
    q_1[i] = FLUID.Q()
    h_tot_1[i] = FLUID.hmass()
q_0 = np.where(q_0 < 0, -1, q_0)
q_0 = np.where(q_0 > 1, 2, q_0)
q_1 = np.where(q_1 < 0, -1, q_1)
q_1 = np.where(q_1 > 1, 2, q_1)

s_in_avg = (s_in_0 + s_in_1) / 2
h_tot_avg = (h_tot_0 + h_tot_1) / 2

idx_two_phase_0 = ((q_0 > 0) & (q_0 < 1))
idx_two_phase_1 = ((q_1 > 0) & (q_1 < 1))
if (idx_two_phase_0 | idx_two_phase_1).any():
    TWO_PHASE = True
else:
    TWO_PHASE = False


#%% PLOT TIME HISTORIES FOR i-TH TEST

plt.figure()
plt.plot(data['t'], data['PT'])
plt.xlabel('Time [s]')
plt.ylabel('Total pressure [bar]')
plt.autoscale(enable=True, axis='x', tight=True)
plt.title(f'{exp_name}')
plt.show()

plt.figure()
plt.plot(data['t'], data['TT_0'], label='TT_0')
plt.plot(data['t'], data['TT_1'], label='TT_1')
plt.xlabel('Time [s]')
plt.ylabel('Total temperature [°C]')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.title(f'{exp_name}')
plt.show()

plt.figure()
plt.plot(data['t'], s_in_0, label='s(PT, TT_0)')
plt.plot(data['t'], s_in_1, label='s(PT, TT_1)')
plt.xlabel('Time [s]')
plt.ylabel('Specific entropy [J/kgK]')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.title(f'{exp_name}')
plt.show()

plt.figure()
plt.plot(data['t'], q_0, '*', label='x(PT, TT_0)')
plt.plot(data['t'], q_1, '*', label='x(PT, TT_1)')
plt.xlabel('Time [s]')
plt.ylabel('Vapour quality')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.title(f'{exp_name}')
plt.show()


plt.figure()
# N pressure transducers = 1 for total pressure + (N-1) for static pressures
for i in range(1, len(log_data['transducers_names'])-1):
    name = f'P_{i}'
    plt.plot(data['t'], data[name], label=name)
plt.xlabel('Time [s]')
plt.ylabel('Total pressure [bar]')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.title(f'{exp_name}')
plt.show()


#%% PLOT INLET POINT EVOLUTION

idx_start = np.argmax(data['PT'])

plot_Ts = CPplots.PropertyPlot(log_data['fluid'], 'TS', unit_system='SI')
# draw the outer shape of the T-s curve, composed of the two iso-quality lines
# x = 0 (saturated liquid curve)
# x = 1 (saturated vapour curve)
plot_Ts.calc_isolines(CP.iQ, num=2, points=2000)
# and a few isobars
#plot_Ts.calc_isolines(CoolProp.iP)
#plot_Ts.props[CoolProp.iP]['color'] = 'red'
# NB: using just
#   plot_Ts.calc_isolines()
# adds too many lines
plot_Ts.draw()

plot_Ts.axis.plot(s_in_0[idx_start:], data['TT_0'][idx_start:]+C2K, 's-', markersize=3, label='TT_0')
plot_Ts.axis.plot(s_in_0[idx_start], data['TT_0'][idx_start]+C2K, 'r*', markersize=5)
plot_Ts.axis.plot(s_in_1[idx_start:], data['TT_1'][idx_start:]+C2K, 's-', markersize=3, label='TT_0')
plot_Ts.axis.plot(s_in_1[idx_start], data['TT_1'][idx_start]+C2K, 'g*', markersize=5)
plot_Ts.ylabel('Total temperature [K]')
plot_Ts.xlabel('Specific entropy [J/kgK]')
plot_Ts.set_axis_limits((np.min([s_in_0.min(), s_in_1.min()]), 
                         np.max([s_in_0.max(), s_in_1.max()]), 
                         C2K+np.min([data['TT_0'].min(), data['TT_1'].min()]), 
                         C2K+np.max([data['TT_0'].max(), data['TT_1'].max()])
                         ))
plot_Ts.axis.legend()
plot_Ts.title('Inlet point')


plot_Ts_mark = CPplots.PropertyPlot(log_data['fluid'], 'TS', unit_system='SI')
plot_Ts_mark.calc_isolines(CP.iQ, num=2, points=2000)
plot_Ts_mark.draw()
# last value is ignored => interval [20, 140)
time_inst = [20, 40, 60, 80, 100, 120]
markers = ['s', 'v', '*', 'o', '^', 'x']
plot_Ts_mark.axis.plot(s_in_1[idx_start:], data['TT_1'][idx_start:]+C2K, markersize=3, label='TT_1')
for t_i, mark_i in zip(time_inst, markers):
    idx_time = np.argmin(np.abs(data['t'] - t_i))
    s_i = s_in_1[idx_time]
    TT_i = data['TT_1'][idx_time] + C2K
    plot_Ts_mark.axis.plot(s_i, TT_i, color='r', marker=mark_i, markersize=5, label=f't = {t_i} s')
plot_Ts_mark.ylabel('Total temperature [K]')
plot_Ts_mark.xlabel('Specific entropy [J/kgK]')
plot_Ts_mark.set_axis_limits((s_in_1[idx_start:].min(), s_in_1[idx_start:].max(), 
                              C2K+data['TT_1'][idx_start:].min(), C2K+data['TT_1'][idx_start:].max()
                              ))
plot_Ts_mark.axis.legend()
plot_Ts_mark.title('Inlet point (using TT1 only)')


#%% IDENTIFY USEFUL EXPERIMENT DATA

data_exp_only = data.iloc[idx_start:,:]
t0 = data_exp_only['t'].iloc[0]
t_exp = np.array(data_exp_only['t'])
p_exp = []
for ch in channels:
    # channels are ordered from 1 to 12, but channel 1 is the Ptot channel and
    # static pressure signals are ordered from P_1 to P_11
    p_exp.append(np.array(data_exp_only[f'P_{ch-1}']))
p_exp = np.column_stack(p_exp)
data_exp = np.column_stack((t_exp, p_exp))


IC = {}
# ignore 1st column (time)
IC['p'] = data_exp[0,1:] * 1e5     # p(x, t=t0)
IC['t'] = np.array(data_exp_only['t'])

BC = {}
BC['p_tot'] = np.array(data_exp_only['PT']) * 1e5
TT0 = data_exp_only['TT_0'] + C2K
TT1 = data_exp_only['TT_1'] + C2K
TT_avg = (TT0 + TT1) / 2
BC['T_tot_0'] = np.array(TT0)
BC['T_tot_1'] = np.array(TT1)
BC['T_tot_avg'] = np.array(TT_avg)
BC['p_out'] = np.array(data_exp_only[f'P_{N_probes}']) * 1e5
BC['t'] = np.array(data_exp_only['t'])
BC['s'] = s_in_avg[idx_start:]

header_p = ', '.join([f'P{i}' for i in range(1, N_probes+1)])
header_IC = ('Initial conditions (at t = t0, when blowdown starts)\n'
             f'Experiment: {exp_name}\n'
             f't0 = {t0} s\n'
             'T_tot_0: total temperature TT0 at t = t0 [K]\n'
             'T_tot_1: total temperature TT1 at t = t0 [K]\n'
             'T_tot_avg: average of TT0 and TT1 at t = t0 [K]\n'
             'p_tot: total pressure PT at t = t0 [Pa]\n'
             f'Pn: static pressure at t = t0, from inlet (P1) to outlet (P{N_probes}) [Pa]\n'
             f'T_tot_0, T_tot_1, T_tot_avg, p_tot, {header_p}')
# shape: (M, )
data4file = np.concatenate((np.atleast_1d(BC['T_tot_0'][0]),
                            np.atleast_1d(BC['T_tot_1'][0]),
                            np.atleast_1d(BC['T_tot_avg'][0]),
                            np.atleast_1d(BC['p_tot'][0]),
                            IC['p']))
# convert to (1, M) to print it on a single line
data4file = np.reshape(data4file, (1, data4file.shape[0]))
np.savetxt(IC_exp_file, data4file, delimiter=', ', header=header_IC)


header_BC = ('Boundary conditions (at x = x0 and x = xL, from t = t0)\n' 
             f'Experiment: {exp_name}\n'
             f't: physical time [s]\n'
             'T_tot_0: total temperature TT0 [K]\n'
             'T_tot_1: total temperature TT1 [K]\n'
             'T_tot_avg: average of TT0 and TT1 [K]\n'
             'p_tot: total pressure PT [Pa]\n'
             f'p_out: static pressure at last probe location [Pa]\n'
             f't, T_tot_0, T_tot_1, T_tot_avg, p_tot, p_out')
data4file = np.column_stack((BC['t'], BC['T_tot_0'], BC['T_tot_1'], BC['T_tot_avg'],
                             BC['p_tot'], BC['p_out']))
np.savetxt(BC_exp_file, data4file, delimiter=', ', header=header_BC)


#%%% Initial conditions profiles: phi(x, 0)

# assumed linear trend in the last portion => using the same slope as the last
# two points led to an excessively low pressure (h(p,s*) crashed), so reduce
# the (assumed) slope by the same rate as the slope that you can calculate from
# the 3rd and 2nd-to-last data
dpdx = np.gradient(IC['p'], probes_coord[:,1])
K_out = dpdx[-1] / dpdx[-2]
p_last = IC['p'][-1] + K_out*dpdx[-1]*(nozzle[-1,0] - probes_coord[-1,1])
# and do the same for the inlet => excessive slope because tap 1, 2, 3 are 
# not instrumented => assume same slope => still excessive
#K_in = dpdx[0] / dpdx[1]
#K_in = K_out
#p_in = IC['p'][0] + K_in*dpdx[0]*(nozzle[0,0] - probes_coord[0,1])
# midpoint between p 1st probe and p_tot_in
p_in = 1e5 * np.array((data_exp_only['PT'] + data_exp[:,1]) / 2)
p_fun = interp.PchipInterpolator(np.concatenate((np.atleast_1d(nozzle[0,0]), 
                                                 probes_coord[:,1],
                                                 np.atleast_1d(nozzle[-1,0]))), 
                                  np.concatenate((np.atleast_1d(p_in[0]), 
                                                  IC['p'],
                                                  np.atleast_1d(p_last))))
p_interp = p_fun(nozzle[:,0])
s_interp = np.full_like(p_interp, s_in_avg[idx_start])
u_interp = np.zeros_like(p_interp)
a_interp = np.zeros_like(p_interp)
for i in range(p_interp.shape[0]):
    try:
        FLUID.update(CP.PSmass_INPUTS, p_interp[i], s_interp[i])
        h = FLUID.hmass()
        u_interp[i] = np.sqrt(2*(h_tot_avg[idx_start] - h))
        if TWO_PHASE:
            raise NotImplementedError('Not implemented yet for two-phase flows')
        a_interp[i] = FLUID.speed_sound()
    except Exception as e:
        print(f'Error at point x = {nozzle[i,0]*1e3} mm: {e.args[0]}')
M_interp = u_interp / a_interp

header_IC = ('Initial conditions (at t = t0, when blowdown starts)\n'
             f'Experiment: {exp_name}\n'
             f't0 = {t0} s\n'
             'x: axial coordinate [m]\n'
             'p: interpolated static pressure [Pa]\n'
             'u: velocity [m/s]\n'
             's: average entropy [J/kgK]\n'
             'a: sound speed [m/s]\n'
             'M: Mach number [-]\n'
             f'x, p, u, s, a, M')
data4file = np.column_stack((nozzle[:,0], p_interp, u_interp, s_interp, a_interp, M_interp))
np.savetxt(IC_extrap_file, data4file, delimiter=', ', header=header_IC)


#%%% Boundary conditions profiles: phi(0, t)

delta = 0.01
# p_out is not really known => calculate it using the spline as above for each
# time instant
p_out = np.zeros_like(t_exp)
check = np.full_like(t_exp, False, dtype=bool)
for j in range(t_exp.shape[0]):
    dpdx_j = np.gradient(p_exp[j,:], probes_coord[:,1]) # [bar/mm]
    K_j = K_out
    p0_j = data_exp_only['PT'].iloc[j]  # [bar]
    p_last_j = p_exp[j,-1] + K_j*dpdx_j[-1]*(nozzle[-1,0] - probes_coord[-1,1]) # [bar]
# =============================================================================
#     p_fun_j = interp.PchipInterpolator(np.concatenate((np.atleast_1d(nozzle[0,0]), 
#                                                        probes_coord[:,1],
#                                                        np.atleast_1d(nozzle[-1,0]))), 
#                                        np.concatenate((np.atleast_1d(p0_j), 
#                                                        p_exp[j,:]*1e5,
#                                                        np.atleast_1d(p_last_j))))
# =============================================================================
    FLUID.update(CP.PSmass_INPUTS, p_exp[j,-1]*1e5, s_in_avg[idx_start+j])
    h_j = FLUID.hmass()
    u_j = np.sqrt(2*(h_tot_avg[idx_start+j] - h_j))
    if TWO_PHASE:
        raise NotImplementedError('Not implemented yet for two-phase flows')
    a_j = FLUID.speed_sound()
    M_j = u_j / a_j
    if M_j > 1:
        # dA > 0 when M > 1: flow must accelerate => dp < 0
        if p_last_j > p_exp[j,-1]*(1+delta):
            p_out[j] = p_exp[j,-1] * 1e5
            check[j] = True
        else:
            p_out[j] = p_last_j * 1e5
    else:
        p_out[j] = p_last_j * 1e5
    
    # calculate inlet pressure from entropy too
    #FLUID.update(CP.PT_INPUTS, p_exp[j,-1]*1e5, s_in_avg[idx_start+j])


header_BC = ('Boundary conditions (at x = x0 and x = xL, from t = t0)\n' 
             f'Experiment: {exp_name}\n'
             't: physical time [s]\n'
             'T_tot: total temperature (average of TT0 and TT1) [K]\n'
             'p_tot: total pressure PT [Pa]\n'
             's: specific entropy (average of s(TT0,PT) and s(TT1,PT)) [J/kgK]'
             'p_in: approximated pressure at the inlet [Pa]\n'
             'p_out: extrapolated pressure at the outlet [Pa]\n'
             f't, T_tot, p_tot, s, p_in, p_out')
data4file = np.column_stack((BC['t'], BC['T_tot_avg'], BC['p_tot'], BC['s'], p_in, p_out))
np.savetxt(BC_extrap_file, data4file, delimiter=', ', header=header_BC)


#%% WRITE SOLVER INPUT FILE

print(f'Writing solver input file to \n\t{solve_input_file.resolve()}')
with open(solve_input_file, 'w') as file:
    yaml.safe_dump(solver_data, file)


#%% PLOT INITIAL AND BOUNDARY CONDITIONS 

idx_th = np.argmin(nozzle[:,1])
x_th, y_th = nozzle[idx_th]
M_th = M_interp[idx_th]

plt.figure()
plt.axvline(x_th*1e3, linestyle='--', color='k')
plt.axhline(data_exp_only['PT'].iloc[0], linestyle='--', color='m')
plt.plot(nozzle[:,0]*1e3, p_interp*1e-5, label='Spline')
plt.plot(probes_coord[:,1]*1e3, IC['p']*1e-5, linestyle='', marker='*', label='Data')
plt.plot(nozzle[0,0]*1e3, p_in[0]*1e-5, marker='s', label='$p_{in,guess}$')
plt.plot(nozzle[-1,0]*1e3, p_last*1e-5, marker='s', label='$p_{out,extrap}$')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('x [mm]')
plt.ylabel('p [bar]')
plt.title(f'Pressure at $t = t_0 = {t0}$ s')
plt.legend()

plt.figure()
plt.axvline(x_th*1e3, linestyle='--', color='k')
plt.plot(nozzle[:,0]*1e3, u_interp)
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('x [mm]')
plt.ylabel('u [m/s]')
plt.title(f'Velocity at $t = t_0 = {t0}$ s')

plt.figure()
plt.axvline(x_th*1e3, linestyle='--', color='k')
plt.plot(nozzle[:,0]*1e3, M_interp)
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('x [mm]')
plt.ylabel('M [-]')
plt.title(f'Mach number at $t = t_0 = {t0}$ s ($M_t = {M_th:.4f}$)')

plt.figure()
plt.plot(t_exp, p_out*1e-5, label='Extrapolated outlet')
plt.plot(t_exp, p_exp[:,-1], label='Last probe')
plt.plot(t_exp[check], p_out[check]*1e-5, 'r*', label='Patched')
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('t [s]')
plt.ylabel('$p_{out}$ [bar]')
plt.title(f'Extrapolated outlet pressure from $t = t_0 = {t0}$ s')
plt.legend()

plt.figure()
plt.plot(t_exp, 100*(p_exp[:,-1]-p_out*1e-5)/p_exp[:,-1])
plt.autoscale(enable=True, axis='x', tight=True)
plt.xlabel('t [s]')
plt.ylabel('$(p_{last} - p_{out,ext})/p_{last}$ [%]')
plt.title(f'Difference between last probe $p$ and extrapolated $p$ from $t = t_0 = {t0}$ s')

plt.show()
