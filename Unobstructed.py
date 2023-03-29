import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import tree_gen

(nu,nx,N,lf,lr,ts) = (2,4,38,1,1,0.1)       #nu = Number of inputs, nx = number of states
ltot = lf+lr
(Rcar, Robj) = (1,1)
Rmin = 0.5

(xref, yref, psiref, vref) = (1,1,0,0)   #Target
(q,qpsi,qbeta,r_v,r_beta,qN,qPsiN,qBetaN) = (10.0, 0.1, 0.1, 1,1, 200, 2,2)

p = np.array([[0.3,0.15,0.2,0.2,0.15],      #Tranisitional probabilitiy of tree
              [0.2,0.3,0.15,0,0.35],
              [0.25,0.2,0.3,0.25,0],
              [0.3,0,0.25,0.2,0.25],
              [0.2,0.25,0,0.3,0.25]])

v_tree = np.array([0.6, 0.1, 0.1, 0.1,0.1])       #initial probability of starting node
(N_tree, tau) = (N, 2)                   #N -> number of stages, tau -> stage at tree becomes stopped tree.
tree = tree_gen.MarkovChainScenarioTreeFactory(p, v_tree, N_tree, tau).create()

# tree.bulls_eye_plot()

u = cs.SX.sym('u', nu*tree.num_nonleaf_nodes)
z0 = cs.SX.sym('z0', nx)

(x,y,psi,v) = (z0[0], z0[1], z0[2], z0[3])

cost = tree.probability_of_node(0)*(q*((x-xref)**2+(y-yref)**2)+qpsi*(psi-psiref)**2+qbeta*(v-vref)**2)
cost += tree.probability_of_node(0)*(r_v*u[0]**2+r_beta*u[1]**2)
# cost += tree.probability_of_node(0)*(1*cs.dot(u, u))

z_sequence = [None]*tree.num_nonleaf_nodes
z_sequence[0] = z0

c=0
for i in range(1,tree.num_nonleaf_nodes):       #Looping through all non-leaf nodes
    idx_anc = tree.ancestor_of(i) 
    prob_i = tree.probability_of_node(i)

    x_anc = z_sequence[idx_anc][0]
    y_anc = z_sequence[idx_anc][1]
    v_anc = z_sequence[idx_anc][2]
    psi_anc = z_sequence[idx_anc][3]

    u_anc = u[idx_anc*nu:(idx_anc+1)*nu] 
    u_current = u[i*nu:(i+1)*nu] 

    x_current  = x_anc+ts*(u_anc[0]*cs.cos(psi_anc+u_anc[1]))
    y_current  = y_anc+ts*(u_anc[0]*cs.sin(psi_anc+u_anc[1]))
    v_current  = v_anc+ts*(u_anc[0])
    psi_current  = psi_anc+ts*(u_anc[0]*cs.sin(u_anc[1]))/lr

    cost += prob_i*(q*((x_current-xref)**2+(y_current-yref)**2)+qpsi*(psi_current-psiref)**2+qbeta*(v_current-vref)**2)
    cost += prob_i*(r_v*u[0]**2+r_beta*u[1]**2)
    # cost += prob_i*(1*cs.dot(u_current, u_current))

    z_sequence[i]=cs.vertcat(x_current,y_current,v_current,psi_current)

bounds = og.constraints.BallInf(radius = 1)

for i in range(tree.num_nodes):
    tree.set_data_at_node(i, {"pos": [1,1]})

# f2  =cs.vertcat(cs.fmax(0.0,Rmin**2-(0-z_sequence[:][0])**2-(2-z_sequence[:][1])**2))

problem = og.builder.Problem(u, z0,cost)\
        .with_constraints(bounds)
        # .with_penalty_constraints(f2)  
  
build_config = og.config.BuildConfiguration()\
    .with_build_directory("basic_optimizer")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()

meta = og.config.OptimizerMeta()\
    .with_optimizer_name("bicycle")

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-4)\
    .with_initial_tolerance(1e-3)\
    .with_penalty_weight_update_factor(1.4)\
    .with_max_inner_iterations(2000)\
    .with_max_outer_iterations(60)

builder = og.builder.OpEnOptimizerBuilder(problem,
                                            meta,
                                            build_config,
                                            solver_config)

builder.build()

# # Use TCP server
# # ------------------------------------
mng = og.tcp.OptimizerTcpManager('basic_optimizer/bicycle')
mng.start()

init_pos = [-1.0, 4.0, 0.0, 0.0]

mng.ping()
solution = mng.call(init_pos)
mng.kill()


# Plot solution
# ------------------------------------
time = np.arange(0, ts*N, ts)
u_star = solution['solution']
v = u_star[0:nu*N:2]
delta = u_star[1:nu*N:2]
print(u_star)


plt.subplot(2,1,1)
plt.plot(time, v, '-o')
plt.ylabel('v  (m/s)')
plt.subplot(2,1,2)
plt.plot(time, delta, '-o')
plt.ylabel('delta (rad)')
plt.xlabel('Time')
plt.show()

# # Plot trajectory
# # ------------------------------------
x_init = init_pos
x_states = [0.0] * (nx*(N+2))
x_states[0:nx+1] = x_init

#print(u_star)
for t in range(0, N):

    u_t = u_star[t*nu:(t+1)*nu]
    x = x_states[t * nx]
    y = x_states[t * nx + 1]
    psi = x_states[t* nx + 2]
    beta = x_states[t * nx + 3]

    beta_dot = cs.atan((lr/ltot)*cs.tan(u_t[1]))
    psi_dot = (u_t[0]/lr)*cs.sin(beta_dot)

    x_states[(t+1)*nx]  = x + ts*(u_t[0]*cs.cos(psi_dot+beta_dot))
    x_states[(t+1)*nx+1] = y + ts*(u_t[0]*cs.sin(psi_dot+beta_dot))
    x_states[(t+1)*nx+2] = psi + ts*psi_dot
    x_states[(t+1)*nx+3] = beta + ts*beta_dot
    #print(f"x_states -> {x_states[t:t+nx]}")

xx = x_states[0:nx*N:nx]
xy = x_states[1:nx*N:nx]
xpsi = x_states[3:nx*N:nx]
xbeta = x_states[4:nx*N:nx]

figure, axes = plt.subplots()
plt.plot(xx, xy, '-o')
axes.set_aspect(1)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()

print(f"Solution exit status "+str(solution["exit_status"]))
print(f"Solution penalty "+str(solution["penalty"]))
print(f"Solution time "+str(solution["solve_time_ms"]))
print(f"Inner iterations "+str(solution["num_inner_iterations"]))
print(f"Outer iterations "+str(solution["num_outer_iterations"]))
print(f"f2 norm:" +str(solution["f2_norm"]))
print(f"Cost "+str(solution["cost"]))

