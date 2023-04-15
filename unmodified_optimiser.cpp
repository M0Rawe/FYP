/**
 * This is an auto-generated file by Optimization Engine (OpEn)
 * OpEn is a free open-source software - see doc.optimization-engine.xyz
 * dually licensed under the MIT and Apache v2 licences.
 *
 * Generated at 2023-04-14 15:26:22.231833.
 */
#include "ros/ros.h"
#include "FYP_controller/OptimizationResult.h"
#include "FYP_controller/OptimizationParameters.h"
#include "mpc_controller_bindings.hpp"
#include "open_optimizer.hpp"

namespace FYP_controller {
/**
 * Class FYP_controller::OptimizationEngineManager manages the
 * exchange of data between the input and output topics
 * of this node
 */
class OptimizationEngineManager {
/**
 * Private fields and methods
 */
private:
    /**
     * Optimization parameters announced on the corresponding
     * topic (FYP_controller/parameters)
     */
    FYP_controller::OptimizationParameters params;
    /**
     * Object containing the result (solution and solver
     * statistics), which will be announced on FYP_controller/results
     */
    FYP_controller::OptimizationResult results;
    /**
     * Vector of parameters (provided by the client on
     * FYP_controller/parameters)
     */
    double p[MPC_CONTROLLER_NUM_PARAMETERS] = { 0 };
    /**
     * Solution vector
     */
    double u[MPC_CONTROLLER_NUM_DECISION_VARIABLES] = { 0 };
    /**
     * Vector of Lagrange multipliers (if any)
     */
    double *y = NULL;
    /**
     * Workspace variable used by the solver - initialised once
     */
    mpc_controllerCache* cache;
    /**
     * Initial guess for the penalty parameter
     */
    double init_penalty = ROS_NODE_MPC_CONTROLLER_DEFAULT_INITIAL_PENALTY;

    /**
     * Publish obtained results to output topic
     */
    void publishToTopic(ros::Publisher& publisher)
    {
        publisher.publish(results);
    }

    /**
     * Updates the input data based on the data that are posted
     * on /mpc/open_parameters (copies value from topic data to
     * local variables). This method is responsible for parsing
     * the data announced on the input topic.
     */
    void updateInputData()
    {
        init_penalty = (params.initial_penalty > 1.0)
            ? params.initial_penalty
            : ROS_NODE_MPC_CONTROLLER_DEFAULT_INITIAL_PENALTY;

        if (params.parameter.size() > 0) {
            for (size_t i = 0; i < MPC_CONTROLLER_NUM_PARAMETERS; ++i)
                p[i] = params.parameter[i];
        }

        if (params.initial_guess.size() == MPC_CONTROLLER_NUM_DECISION_VARIABLES) {
            for (size_t i = 0; i < MPC_CONTROLLER_NUM_DECISION_VARIABLES; ++i)
                u[i] = params.initial_guess[i];
        }

		if (params.initial_y.size() == MPC_CONTROLLER_N1) {
            for (size_t i = 0; i < MPC_CONTROLLER_N1; ++i)
                y[i] = params.initial_y[i];
		}
    }

    /**
     * Call OpEn to solve the problem
     */
    mpc_controllerSolverStatus solve()
    {
        return mpc_controller_solve(cache, u, p, y, &init_penalty);
    }
/**
 * Public fields and methods
 */
public:
    /**
     * Constructor of OptimizationEngineManager
     */
    OptimizationEngineManager()
    {
	    y = new double[MPC_CONTROLLER_N1];
        cache = mpc_controller_new();
    }

    /**
     * Destructor of OptimizationEngineManager
     */
    ~OptimizationEngineManager()
    {
		if (y!=NULL) delete[] y;
        mpc_controller_free(cache);
    }

    /**
     * Copies results from `status` to the local field `results`
     */
    void updateResults(mpc_controllerSolverStatus& status)
    {
        std::vector<double> sol(u, u + MPC_CONTROLLER_NUM_DECISION_VARIABLES);
        results.solution = sol;
        std::vector<double> y(status.lagrange, status.lagrange + MPC_CONTROLLER_N1);
        results.lagrange_multipliers = y;
        results.inner_iterations = status.num_inner_iterations;
        results.outer_iterations = status.num_outer_iterations;
        results.norm_fpr = status.last_problem_norm_fpr;
        results.cost = status.cost;
        results.penalty = status.penalty;
        results.status = (int)status.exit_status;
        results.solve_time_ms = (double)status.solve_time_ns / 1000000.0;
        results.infeasibility_f2 = status.f2_norm;
        results.infeasibility_f1 = status.delta_y_norm_over_c;
    }

    /**
     * Callback that obtains data from topic `/FYP_controller/open_params`
     */
    void mpcReceiveRequestCallback(
        const FYP_controller::OptimizationParameters::ConstPtr& msg)
    {
        params = *msg;
    }

    void solveAndPublish(ros::Publisher& publisher)
    {
        updateInputData(); /* get input data */
        mpc_controllerSolverStatus status = solve(); /* solve!  */
        updateResults(status); /* pack results into `results` */
        publishToTopic(publisher);
    }
}; /* end of class OptimizationEngineManager */
} /* end of namespace FYP_controller */

/**
 * Main method
 *
 * This advertises a new (private) topic to which the optimizer
 * announces its solution and solution status and details. The
 * publisher topic is 'FYP_controller/result'.
 *
 * It obtains inputs from 'FYP_controller/parameters'.
 *
 */
int main(int argc, char** argv)
{
    std::string result_topic, params_topic;  /* parameter and result topics */
    double rate; /* rate of node (specified by parameter) */

    FYP_controller::OptimizationEngineManager mng;
    ros::init(argc, argv, ROS_NODE_MPC_CONTROLLER_NODE_NAME);
    ros::NodeHandle private_nh("~");

    /* obtain parameters from config/open_params.yaml file */
    private_nh.param("result_topic", result_topic,
                     std::string(ROS_NODE_MPC_CONTROLLER_RESULT_TOPIC));
    private_nh.param("params_topic", params_topic,
                     std::string(ROS_NODE_MPC_CONTROLLER_PARAMS_TOPIC));
    private_nh.param("rate", rate,
                     double(ROS_NODE_MPC_CONTROLLER_RATE));

    ros::Publisher mpc_pub
        = private_nh.advertise<FYP_controller::OptimizationResult>(
            ROS_NODE_MPC_CONTROLLER_RESULT_TOPIC,
            ROS_NODE_MPC_CONTROLLER_RESULT_TOPIC_QUEUE_SIZE);
    ros::Subscriber sub
        = private_nh.subscribe(
            ROS_NODE_MPC_CONTROLLER_PARAMS_TOPIC,
            ROS_NODE_MPC_CONTROLLER_PARAMS_TOPIC_QUEUE_SIZE,
            &FYP_controller::OptimizationEngineManager::mpcReceiveRequestCallback,
            &mng);
    ros::Rate loop_rate(ROS_NODE_MPC_CONTROLLER_RATE);

    while (ros::ok()) {
        mng.solveAndPublish(mpc_pub);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
