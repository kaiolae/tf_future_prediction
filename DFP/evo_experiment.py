from __future__ import print_function
import numpy as np
from .future_target_maker import FutureTargetMaker
from .multi_doom_simulator import MultiDoomSimulator
from .multi_experience_memory import MultiExperienceMemory
from .future_predictor_agent_basic import FuturePredictorAgentBasic
from .future_predictor_agent_advantage import FuturePredictorAgentAdvantage
from .future_predictor_agent_advantage_nonorm import FuturePredictorAgentAdvantageNoNorm
from . import defaults
import tensorflow as tf
import scipy.misc
from . import util as my_util
import shutil

### Experiment-setup for initializing the ANNs for the evolving objectives.
class EvoExperiment:

    def __init__(self, target_maker_args={}, 
                       simulator_args={}, 
                       train_experience_args={}, 
                       test_policy_experience_args={}, 
                       agent_args={},
                       experiment_args={}):
        

        target_maker_args = my_util.merge_two_dicts(defaults.target_maker_args, target_maker_args)
        if isinstance(simulator_args, dict):
            simulator_args = my_util.merge_two_dicts(defaults.simulator_args, simulator_args)
        else:
            for n in range(len(simulator_args)):
                simulator_args[n] = my_util.merge_two_dicts(defaults.simulator_args, simulator_args[n])
        train_experience_args = my_util.merge_two_dicts(defaults.train_experience_args, train_experience_args)
        test_policy_experience_args = my_util.merge_two_dicts(defaults.test_policy_experience_args, test_policy_experience_args)
        agent_args = my_util.merge_two_dicts(defaults.agent_args, agent_args)
        experiment_args = my_util.merge_two_dicts(defaults.experiment_args, experiment_args)
        
        if not (experiment_args['args_file'] is None):
            print(' ++ Reading arguments from ', experiment_args['args_file'])
            with open(experiment_args['args_file'], 'r') as f:
                input_args = my_util.json_load_byteified(f)
            
            for arg_name,arg_val in input_args.items():
                print(arg_name, arg_val)
                for k,v in arg_val.items():
                    locals()[arg_name][k] = v
 
        self.target_maker = FutureTargetMaker(target_maker_args)
        self.results_file = experiment_args['results_file']
        self.net_name = experiment_args['net_name']
        self.num_predictions_to_show = experiment_args['num_predictions_to_show']
        agent_args['target_dim'] = self.target_maker.target_dim
        agent_args['target_names'] = self.target_maker.target_names    

        if isinstance(simulator_args, list):
            # if we are given a bunch of different simulators
            self.multi_simulator =MultiDoomSimulator(simulator_args)
        else:
            # if we have to replicate a single simulator
            self.multi_simulator = MultiDoomSimulator([simulator_args] * simulator_args['num_simulators'])
        agent_args['discrete_controls'] = self.multi_simulator.discrete_controls
        agent_args['continuous_controls'] = self.multi_simulator.continuous_controls

        agent_args['objective_indices'], agent_args['objective_coeffs'] = my_util.make_objective_indices_and_coeffs(agent_args['objective_coeffs_temporal'],
                                                                                                                    agent_args['objective_coeffs_meas'])
        print("original obj indices: ", agent_args['objective_indices'])

        train_experience_args['obj_shape'] = (len(agent_args['objective_coeffs']),)
        test_policy_experience_args['obj_shape'] = (len(agent_args['objective_coeffs']),)
        self.train_experience = MultiExperienceMemory(train_experience_args, multi_simulator = self.multi_simulator, target_maker = self.target_maker)
        agent_args['state_imgs_shape'] = self.train_experience.state_imgs_shape
        agent_args['obj_shape'] = (len(agent_args['objective_coeffs']),)
        agent_args['num_simulators'] = self.multi_simulator.num_simulators

        if 'meas_for_net' in experiment_args:
            agent_args['meas_for_net'] = []
            for ns in range(self.train_experience.history_length):
                agent_args['meas_for_net'] += [i + self.multi_simulator.num_meas * ns for i in experiment_args['meas_for_net']] # we want these measurements from all timesteps
            agent_args['meas_for_net'] = np.array(agent_args['meas_for_net'])
        else:
            agent_args['meas_for_net'] = np.arange(self.train_experience.state_meas_shape[0])
        if len(experiment_args['meas_for_manual']) > 0:
            agent_args['meas_for_manual'] = np.array([i + self.multi_simulator.num_meas*(self.train_experience.history_length-1) for i in experiment_args['meas_for_manual']]) # current timestep is the last in the stack
        else:
            agent_args['meas_for_manual'] = []
        agent_args['state_meas_shape'] = [len(agent_args['meas_for_net'])]
        self.agent_type = agent_args['agent_type']
        
        if agent_args['random_objective_coeffs']:
            assert('fc_obj_params' in agent_args)
    
        self.test_policy_experience = MultiExperienceMemory(test_policy_experience_args, multi_simulator = self.multi_simulator, target_maker = self.target_maker)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)  # avoid using all gpu memory
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))

        if self.agent_type == 'basic':
            self.ag = FuturePredictorAgentBasic(self.sess, agent_args)
        elif self.agent_type == 'advantage':
            self.ag = FuturePredictorAgentAdvantage(self.sess, agent_args) # inital design: concat meas and img, then 2 branches for adv and val
        elif self.agent_type == 'advantage_nonorm':
            self.ag = FuturePredictorAgentAdvantageNoNorm(self.sess, agent_args) # no normalizatio in the advantage stream
        else:
            raise Exception('Unknown agent type', self.agent_type)
        
        self.num_train_iterations = experiment_args['num_train_iterations']

        # KOE: This needs to be set for each agent, since it initializes the (evolved) objective vectors. Almost everything else, I think could be common.
        #TODO Set dummy value here first, then initialize it for each agent.
        #_, self.test_objective_coeffs = my_util.make_objective_indices_and_coeffs(experiment_args['test_objective_coeffs_temporal'],
        #                                                                          experiment_args['test_objective_coeffs_meas'])

        self.test_checkpoint = experiment_args['test_checkpoint']
        self.test_policy_num_steps = experiment_args['test_policy_num_steps']

        #KOETODO: Testing loading the agent just once for all evals. BEWARE! Could have undesired consequences?
        if not self.ag.load(self.test_checkpoint):
            print('Could not load the checkpoint ', self.test_checkpoint)
            return

    #TODO: I think the arguments here should be experiment_args['test_objective_coeffs_temporal'], experiment_args['test_objective_coeffs_meas']
    #TODO Consider letting the evolved NN also decide the temporal weights.
    def test_new_individual(self, evolved_nn, store_to_video = False):
        #TODO: Take this into the decision-step. _, test_objective_coeffs = my_util.make_objective_indices_and_coeffs(temporal_objective_weights, measurement_objective_weights)
        # Since we run N different simulations at the same time, we want to offset their stored data (with N different "write heads") spaced apart by the length of the planned episode.
        self.train_experience.head_offset = self.test_policy_num_steps + 1
        self.train_experience.log_prefix = 'logs/log_test'
        # KOE: Tests the trained policy represented by stored weights, with given objectives. Results are remembered, and plotted by the method below.

        #Returns measurements and rewards for this agent, averaged across N simulations.
        #avg_measurements, avg_rewards = self.ag.test_policy(self.multi_simulator, self.train_experience, test_objective_coeffs,
        #                    self.test_policy_num_steps, write_summary=False,
        #                    write_predictions=False) #Don't want to write for every agent, but will need some external checkpointing later.
        #My own test policy method that allows adaptive objectives.

        self.ag.reset_evo_logs() #Resets the things we are logging for evolution, so we get fresh logs for next agent.
        avg_measurements, avg_rewards = self.ag.test_policy_with_evolved_NEAT_objectives(self.multi_simulator, self.train_experience, evolved_nn,
                            self.test_policy_num_steps, write_summary=False,
                            write_predictions=False)
        if store_to_video:
            self.train_experience.show(start_index=0,
                                       end_index=self.test_policy_num_steps * self.multi_simulator.num_simulators,
                                       display=False, write_imgs=False, write_video=True,
                                       preprocess_targets=self.ag.preprocess_input_targets,
                                       show_predictions=0,#KOEChange:self.num_predictions_to_show,
                                       net_discrete_actions=self.ag.net_discrete_actions)
        #KOETODO: Can I just skip the "show" here?
        print("Eval done. Avg meas: ", avg_measurements)
        print("Eval done. Avg reward: ", avg_rewards)

        #Storing agent history and objective values if requested.
        if self.ag.store_objectives_to_history:
            #TODO: Enable to store to any file later.
            obj_hist = np.array(self.ag.objectives_history) #1 row per timestep. Inside each there are 8(num_simulators) tuples of coeffs.
            print("obj hist shape is ", obj_hist.shape)
            for i in range(self.ag.num_simulators):
                obj_hist_file = "objectives_history" + str(i) +".csv"
                my_history = obj_hist[:,i,:]
                np.savetxt(obj_hist_file, my_history, delimiter=" ")

        self.multi_simulator.close_games()

        return avg_rewards #Fitness?

    '''
    def run(self, mode):
        #KOE: The main run-method, that can either train a new agent, or show the performance of a trained agent.
        shutil.copy('run_exp.py', 'run_exp.py.' + mode)
        if mode == 'show':
            #KOE: Shows performance of trained agent.
            if not self.ag.load(self.test_checkpoint):
                print('Could not load the checkpoint ', self.test_checkpoint)
                return
            #Since we run N different simulations at the same time, we want to offset their stored data (with N different "write heads") spaced apart by the length of the planned episode.
            self.train_experience.head_offset = self.test_policy_num_steps + 1
            self.train_experience.log_prefix = 'logs/log_test'
            #KOE: Tests the trained policy represented by stored weights, with given objectives. Results are remembered, and plotted by the method below.
            self.ag.test_policy(self.multi_simulator, self.train_experience, self.test_objective_coeffs, self.test_policy_num_steps, random_prob = self.test_random_prob, write_summary=False, write_predictions=True)
            # KoeChange: WriteVideo
            # Shows/Stores the experience of the tested agent above.
            #KOETODO What does num_simulators do?
            self.train_experience.show(start_index=0, end_index=self.test_policy_num_steps * self.multi_simulator.num_simulators, display=False, write_imgs=False,  write_video=True,
                                       preprocess_targets = self.ag.preprocess_input_targets, show_predictions=self.num_predictions_to_show, net_discrete_actions = self.ag.net_discrete_actions)
        elif mode == 'train':
            self.test_policy_experience.log_prefix = 'logs/log'
            self.ag.train(self.multi_simulator, self.train_experience, self.num_train_iterations, test_policy_experience=self.test_policy_experience)
        else:
            print('Unknown mode', mode)
        
        '''
        
        
