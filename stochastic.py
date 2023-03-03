import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import tree_gen

(nu,nx,N,lf,lr,ts) = (2,4,20,1,1,0.1)       #nu = Number of inputs, nx = number of states
ltot = lf+lr

(xref, yref, psiref, betaref) = (1,1,0,0)
(q,qpsi,qbeta,r,qN,qPsiN,qBetaN) = (10.0, 0.1, 0.1, 1, 200.0, 2,2)

u = cs.SX.sym('u', nu*N)
z0 = cs.SX.sym('z0', nx)

(x,y,psi,beta) = (z0[0], z0[1], z0[2], z0[3])

p = np.array([[0.3,0.15,0.2,0.2,0.15],      #Tranisitional probabilitiy of tree
                  [0.2,0.3,0.15,0,0.35],
                  [0.25,0.2,0.3,0.25,0],
                  [0.3,0,0.25,0.2,0.25],
                  [0.2,0.25,0,0.3,0.25]])
v_tree = np.array([0.4, 0.15, 0.15,0.15,0.15])       #initial probability of starting node
(N_tree, tau) = (3, 2)                   #N -> number of stages, tau -> stage at which branching stops
tree = tree_gen.MarkovChainScenarioTreeFactory(p, v_tree, N_tree, tau).create()

all_nodes = []
for i in range(tree.num_stages):
    all_nodes.append(tree.nodes_at_stage(i))

probabilities = []
for i in all_nodes:
    innerlist = []
    for j in i:
        innerlist.append(tree.probability_of_node(j))
    probabilities.append(innerlist)

all_nodes_list = []
for i in all_nodes:
    all_nodes_list.append(i.tolist())

combined = zip(all_nodes_list,probabilities)
nodes_flat = [item for sublist in all_nodes_list for item in sublist]
probabilities_flat = [item for sublist in probabilities for item in sublist]

cost = 0
for i in all_nodes_list[1:(tree.num_stages-1)]:
     for j in i:
        cost += tree.probability_of_node(j)*(qN*(x-xref)**2+(y-yref)**2+qPsiN*(psi-psiref)**2+qBetaN*(beta-betaref)**2)
        #print(f"At node {j} the probability is {tree.probability_of_node(j)}")
        #print(i,j)
        u_t = u[j:j+2]
        beta_dot = cs.atan((lr/ltot)*cs.tan(u_t[1]))
        cost += r*cs.dot(u_t,u_t)
        x += u_t[0]*cs.cos(psi+beta_dot)*ts
        y += u_t[0]*cs.sin(psi+beta_dot)*ts
        psi += (u_t[0]/ltot)*cs.cos(beta_dot)*cs.tan(u_t[1])*ts
        beta += ts*beta_dot


for i in all_nodes_list[tree.num_stages-1]:
        #print(f"At node {i} the probability is {tree.probability_of_node(i)}")
        cost += tree.probability_of_node(i)*(qN*(x-xref)**2+(y-yref)**2+qPsiN*(psi-psiref)**2+qBetaN*(beta-betaref)**2)

problem = og.builder.Problem(u, z0, cost)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("basic_optimizer")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("bicycle")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                            meta,
                                            build_config,
                                            solver_config)

#builder.build()

# Use TCP server
# ------------------------------------
mng = og.tcp.OptimizerTcpManager('basic_optimizer/bicycle')
mng.start()

mng.ping()
solution = mng.call([-1.0, 2.0, 0.0, 0.0], initial_guess=[1.0] * (nu*N))
mng.kill()


# Plot solution
# ------------------------------------
time = np.arange(0, ts*N, ts)
u_star = solution['solution']
ux = u_star[0:nu*N:2]
uy = u_star[1:nu*N:2]
upsi = u_star[2:nu*N:2]
ubeta = u_star[3:nu*N:2]

plt.subplot(4,1,1)
plt.plot(time, ux, '-o')
plt.ylabel('u_x')
plt.subplot(4,1,2)
plt.plot(time, uy, '-o')
plt.ylabel('u_y')
plt.subplot(4,1,3)
plt.plot(time[0:-1], upsi, '-o')
plt.ylabel('u_psi')
plt.subplot(4,1,4)
plt.plot(time[0:-1], ubeta, '-o')
plt.ylabel('u_beta')
plt.xlabel('Time')
plt.show()

# Plot trajectory
# ------------------------------------
x_init = [-1.0,2.0,0.0]
x_states = [0.0] * (nx*(N+2))
x_states[0:nx+1] = x_init
for t in range(0, N):
    u_t = u_star[t*nu:(t+1)*nu]

    x = x_states[t * nx]
    y = x_states[t * nx + 1]
    psi = x_states[t* nx + 2]
    beta = x_states[t * nx + 3]

    beta_dot = cs.atan((lr/ltot)*cs.tan(u_t[1]))

    x_states[(t+1)*nx]  = x + ts*(u_t[0]*cs.cos(psi+beta_dot))
    x_states[(t+1)*nx+1] = y + ts*(u_t[0]*cs.sin(psi+beta_dot))
    x_states[(t+1)*nx+2] = (u_t[0]/ltot)*cs.cos(beta_dot)*cs.tan(u_t[1])
    x_states[(t+1)*nx+3] = beta + ts*beta_dot

xx = x_states[0:nx*N:nx]
xy = x_states[1:nx*N:nx]

print(x_states)
print(xx)
plt.plot(xx, xy, '-o')
plt.show()