#For doing analysis on the best evolved network
import pickle
import neat
import sys
import numpy as np
import time

sys.path = ['../..'] + sys.path
from DFP.evo_experiment import EvoExperiment

def set_up_nn_experiment():
    ### Set all arguments

    ## Target maker
    target_maker_args = {}
    target_maker_args['future_steps'] = [1, 2, 4, 8, 16, 32]
    target_maker_args['meas_to_predict'] = [0, 1, 2]
    target_maker_args['min_num_targs'] = 3
    target_maker_args['rwrd_schedule_type'] = 'exp'
    target_maker_args['gammas'] = []
    target_maker_args['invalid_targets_replacement'] = 'nan'

    ## Simulator
    simulator_args = {}
    simulator_args['config'] = '../../maps/D3_battle.cfg'
    simulator_args['resolution'] = (84, 84)
    simulator_args['frame_skip'] = 4#1 #TODO4  # 4 #TODO Change back to 4 for experiements. 1 helps get nicer videos though.
    simulator_args['color_mode'] = 'GRAY'
    simulator_args['maps'] = ['MAP01']
    simulator_args['switch_maps'] = False
    # train
    simulator_args['num_simulators'] = 8#1 #TODO8  # TODO - Keep 8 to get a more informed fitness measure?

    ## Experience
    # Train experience
    train_experience_args = {}
    train_experience_args['memory_capacity'] = 20000
    train_experience_args['history_length'] = 1
    train_experience_args['history_step'] = 1
    train_experience_args['action_format'] = 'enumerate'
    train_experience_args['shared'] = False

    # Test prediction experience
    test_prediction_experience_args = train_experience_args.copy()
    test_prediction_experience_args['memory_capacity'] = 1

    # Test policy experience
    test_policy_experience_args = train_experience_args.copy()
    test_policy_experience_args['memory_capacity'] = 55000

    ## Agent
    agent_args = {}

    # agent type
    agent_args['agent_type'] = 'advantage'

    # preprocessing
    agent_args['preprocess_input_images'] = lambda x: x / 255. - 0.5
    agent_args['preprocess_input_measurements'] = lambda x: x / 100. - 0.5
    targ_scale_coeffs = np.expand_dims(
        (np.expand_dims(np.array([7.5, 30., 1.]), 1) * np.ones((1, len(target_maker_args['future_steps'])))).flatten(),
        0)
    agent_args['preprocess_input_targets'] = lambda x: x / targ_scale_coeffs
    agent_args['postprocess_predictions'] = lambda x: x * targ_scale_coeffs

    # agent properties
    agent_args['objective_coeffs_temporal'] = [0., 0., 0., 0.5, 0.5, 1.]
    agent_args['objective_coeffs_meas'] = [0.5, 0.5,
                                           1.]  # KOE: These are just dummy values - will be replaced with evolved ones.
    agent_args['random_exploration_schedule'] = lambda step: (0.02 + 145000. / (float(step) + 150000.))
    agent_args['new_memories_per_batch'] = 8
    agent_args['random_objective_coeffs'] = True
    agent_args['objective_coeffs_distribution'] = 'uniform_pos_neg'

    # net parameters
    agent_args['conv_params'] = np.array([(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                                         dtype=[('out_channels', int), ('kernel', int), ('stride', int)])
    agent_args['fc_img_params'] = np.array([(512,)], dtype=[('out_dims', int)])
    agent_args['fc_meas_params'] = np.array([(128,), (128,), (128,)], dtype=[('out_dims', int)])
    agent_args['fc_obj_params'] = np.array([(128,), (128,), (128,)], dtype=[('out_dims', int)])
    agent_args['fc_joint_params'] = np.array([(512,), (-1,)], dtype=[
        ('out_dims', int)])  # we put -1 here because it will be automatically replaced when creating the net
    agent_args['weight_decay'] = 0.00000

    # optimization parameters
    agent_args['batch_size'] = 64
    agent_args['init_learning_rate'] = 0.0001
    agent_args['lr_step_size'] = 250000
    agent_args['lr_decay_factor'] = 0.3
    agent_args['adam_beta1'] = 0.95
    agent_args['adam_epsilon'] = 1e-4
    agent_args['optimizer'] = 'Adam'
    agent_args['reset_iter_count'] = False

    # directories
    agent_args['checkpoint_dir'] = 'checkpoints'
    agent_args['log_dir'] = 'logs'
    agent_args['init_model'] = ''
    agent_args['model_name'] = "predictor.model"
    agent_args['model_dir'] = time.strftime("%Y_%m_%d_%H_%M_%S")

    # logging and testing
    agent_args['print_err_every'] = 50
    agent_args['detailed_summary_every'] = 1000
    agent_args['test_pred_every'] = 0
    agent_args['test_policy_every'] = 7812
    agent_args['num_batches_per_pred_test'] = 0
    agent_args['num_steps_per_policy_test'] = test_policy_experience_args['memory_capacity'] / simulator_args[
        'num_simulators']
    agent_args['checkpoint_every'] = 10000
    agent_args['save_param_histograms_every'] = 5000
    agent_args['test_policy_in_the_beginning'] = True

    # experiment arguments
    experiment_args = {}
    experiment_args['num_train_iterations'] = 820000
    # KOE: This defines the weights and temporal weights of objectives.
    # KOETODO Revert to old values above.

    # Health, ammo, frags
    # experiment_args['test_objective_coeffs_meas'] = np.array([-1,-1,-1]) KOE Opposite objectives, just for testing.
    experiment_args['test_random_prob'] = 0.
    experiment_args['test_checkpoint'] = 'checkpoints/2017_04_09_09_13_17'  # KOE: This defines the weights to load
    experiment_args['test_policy_num_steps'] = 2000  # KOE: How many steps to run the test agent.
    experiment_args['show_predictions'] = False
    experiment_args['multiplayer'] = False

    # Create and run the experiment

    experiment = EvoExperiment(target_maker_args=target_maker_args,
                               simulator_args=simulator_args,
                               train_experience_args=train_experience_args,
                               test_policy_experience_args=test_policy_experience_args,
                               agent_args=agent_args,
                               experiment_args=experiment_args)

    return experiment

def store_individual_fitness(genome):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    experiment_interface = set_up_nn_experiment()
    fitness = experiment_interface.test_new_individual(net)
    print("Fitness was ", fitness)
    avg_reward_vector = []
    avg_reward_vector.append(fitness)
    avg_reward_vector = np.array(avg_reward_vector)
    f1 = open('reward_stats_with_evolved_nn.csv', 'a')
    np.savetxt(f1, avg_reward_vector, delimiter=" ")

def store_individual_behavior(genome):

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    num_steps_per_dimension = 20
    measures_range = {"ammo":(0,50),"health" : (0,100), "frags":(0,50)}
    measures_to_objectives_matrix = [] #Vector of vector where each element has form [m1, m2, m3, o1, o2, o3]
    col_headers = ["m_ammo", "m_health", "m_frags", "o_ammo", "o_health", "o_frags"]

    for ammo in np.linspace(start=measures_range["ammo"][0], stop=measures_range["ammo"][1], num=num_steps_per_dimension):
        for health in np.linspace(start=measures_range["health"][0], stop=measures_range["health"][1], num=num_steps_per_dimension):
            for frags in np.linspace(start=measures_range["frags"][0], stop=measures_range["frags"][1],
                                      num=num_steps_per_dimension):
                print(ammo, ", ", health, ", ", frags)
                nn_output = net.activate([ammo, health, frags])
                print("Act output: ", nn_output)
                measures_to_objectives_matrix.append([ammo, health, frags,*nn_output])

    measures_to_objectives_matrix=np.array(measures_to_objectives_matrix)
    f1=open("nn_behavior_measures_to_objectives.csv", 'w+')
    for item in col_headers:
        f1.write(item+" ")
    f1.write("\n")
    np.savetxt(f1, measures_to_objectives_matrix, delimiter=" ")

if __name__ == '__main__':

    # Either: fitness - to store the re-evaluated fitness of this individual, or behavior - to store the learned measure-objective mappings
    # video -to store a video of agent behavior.
    analysis_mode = sys.argv[1]
    winner_filename = sys.argv[2] #Pickled winner indiv
    with open(winner_filename, 'rb') as pickle_file:
        winner_genome = pickle.load(pickle_file)

    config_file = "config"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    if analysis_mode=="fitness":
        store_individual_fitness(winner_genome)
    elif analysis_mode=="behavior":
        store_individual_behavior(winner_genome)
    elif analysis_mode == "video":
        net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        experiment_interface = set_up_nn_experiment()
        experiment_interface.test_new_individual(net, store_to_video=True)
    else:
        print("Analysis mode ", analysis_mode, " is not supported.")

