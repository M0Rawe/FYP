import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import tree_gen

(nu,nx,N,lf,lr,ts) = (2,4,20,1,1,0.1)       #nu = Number of inputs, nx = number of states
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

tree.bulls_eye_plot()

u = cs.SX.sym('u', nu*tree.num_nonleaf_nodes)
z0 = cs.SX.sym('z0', nx)
(x,y,v,psi) = (z0[0], z0[1], z0[2], z0[3])

cost = tree.probability_of_node(0)*(q*(x-xref)**2+(y-yref)**2)+qpsi*(psi-psiref)**2+qbeta*(v-vref)**2
cost += r_v*u[0]**2+r_beta*u[1]**2
z_sequence = [None]*tree.num_nonleaf_nodes
z_sequence[0] = z0

c=0
for i in range(1,tree.num_nonleaf_nodes):
    idx_anc = tree.ancestor_of(i) 

    x_anc = z_sequence[idx_anc][0]
    y_anc = z_sequence[idx_anc][1]
    v_anc = z_sequence[idx_anc][2]
    psi_anc = z_sequence[idx_anc][3]

    u_anc = u[idx_anc*nu:(idx_anc+1)*nu] 
    u_current = u[i*nu:(i+1)*nu] 

    prob_i = tree.probability_of_node(i)
    x_current  = x_anc+ts*(u_anc[0]*cs.cos(psi_anc+u_anc[1]))
    y_current  = y_anc+ts*(u_anc[0]*cs.sin(psi_anc+u_anc[1]))
    v_current  = v_anc+ts*(u_anc[0])
    psi_current  = psi_anc+ts*(u_anc[0]*cs.sin(u_anc[1]))/lr

    cost += prob_i*((q*(x_current-xref)**2+(y_current-yref)**2)+qpsi*(psi_current-psiref)**2+qbeta*(v_current-vref)**2)
    cost += prob_i*(r_v*u[0]**2+r_beta*u[1]**2)

    z_sequence[i]=cs.vertcat(x_current,y_current,v_current,psi_current)

    c+= cs.fmax(0.0,Rmin**2-(0-x_current)**2-(0-y_current)**2)

bounds = og.constraints.BallInf(radius = 1)

f2  =cs.vertcat(cs.fmax(0.0,Rmin**2-((1-z_sequence[:][0])**2-(1-z_sequence[:][1])**2)))

problem = og.builder.Problem(u, z0,cost)\
        .with_constraints(bounds)\
        .with_penalty_constraints(f2)   
ros_config = og.config.RosConfiguration()\
    .with_package_name("FYP_controller")\
    .with_node_name("open_mpc_controller_node")\
    .with_rate((int)(1/ts))\
    .with_description("FYP Ros")

build_config = og.config.BuildConfiguration()\
    .with_build_directory("ros_branch")\
    .with_build_mode("release")\
    .with_build_c_bindings()\
    .with_ros(ros_config)

meta = og.config.OptimizerMeta()\
    .with_optimizer_name("mpc_controller")

solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-3)\
    .with_initial_tolerance(1e-3)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                            meta,
                                            build_config,
                                            solver_config)

builder.build()
