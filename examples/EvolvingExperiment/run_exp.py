from __future__ import print_function
import sys
sys.path = ['../..'] + sys.path
from DFP.evo_experiment import EvoExperiment
import numpy as np
import time
import neat
import visualize
import pickle

def main(main_args):
	
	### Set all arguments
	
	## Target maker
	target_maker_args = {}
	target_maker_args['future_steps'] = [1,2,4,8,16,32]
	target_maker_args['meas_to_predict'] = [0,1,2]
	target_maker_args['min_num_targs'] = 3	
	target_maker_args['rwrd_schedule_type'] = 'exp'
	target_maker_args['gammas'] = []
	target_maker_args['invalid_targets_replacement'] = 'nan'
	
	## Simulator
	simulator_args = {}
	simulator_args['config'] = '../../maps/D3_battle.cfg'
	simulator_args['resolution'] = (84,84)
	simulator_args['frame_skip'] = 4#4 #TODO Change back to 4 for experiements. 1 helps get nicer videos though.
	simulator_args['color_mode'] = 'GRAY'	
	simulator_args['maps'] = ['MAP01']
	simulator_args['switch_maps'] = False
	#train
	simulator_args['num_simulators'] = 8 #TODO - Keep 8 to get a more informed fitness measure?
	
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
	targ_scale_coeffs = np.expand_dims((np.expand_dims(np.array([7.5,30.,1.]),1) * np.ones((1,len(target_maker_args['future_steps'])))).flatten(),0)
	agent_args['preprocess_input_targets'] = lambda x: x / targ_scale_coeffs
	agent_args['postprocess_predictions'] = lambda x: x * targ_scale_coeffs
		
	# agent properties
	agent_args['objective_coeffs_temporal'] = [0., 0. ,0. ,0.5, 0.5, 1.]
	agent_args['objective_coeffs_meas'] = [0.5, 0.5, 1.] # KOE: These are just dummy values - will be replaced with evolved ones.
	agent_args['random_exploration_schedule'] = lambda step: (0.02 + 145000. / (float(step) + 150000.))
	agent_args['new_memories_per_batch'] = 8
	agent_args['random_objective_coeffs'] = True
	agent_args['objective_coeffs_distribution'] = 'uniform_pos_neg'
	
	# net parameters
	agent_args['conv_params']     = np.array([(32,8,4), (64,4,2), (64,3,1)],
									 dtype = [('out_channels',int), ('kernel',int), ('stride',int)])
	agent_args['fc_img_params']   = np.array([(512,)], dtype = [('out_dims',int)])
	agent_args['fc_meas_params']  = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)]) 
	agent_args['fc_obj_params']  = np.array([(128,), (128,), (128,)], dtype = [('out_dims',int)]) 
	agent_args['fc_joint_params'] = np.array([(512,), (-1,)], dtype = [('out_dims',int)]) # we put -1 here because it will be automatically replaced when creating the net
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
	agent_args['num_steps_per_policy_test'] = test_policy_experience_args['memory_capacity'] / simulator_args['num_simulators']
	agent_args['checkpoint_every'] = 10000
	agent_args['save_param_histograms_every'] = 5000
	agent_args['test_policy_in_the_beginning'] = True				
	
	# experiment arguments
	experiment_args = {}
	experiment_args['num_train_iterations'] = 820000
	# KOE: This defines the weights and temporal weights of objectives.
	#KOETODO Revert to old values above.

	#Health, ammo, frags
	#experiment_args['test_objective_coeffs_meas'] = np.array([-1,-1,-1]) KOE Opposite objectives, just for testing.
	experiment_args['test_random_prob'] = 0.
	experiment_args['test_checkpoint'] = 'checkpoints/2017_04_09_09_13_17' #KOE: This defines the weights to load
	experiment_args['test_policy_num_steps'] = 2000 #KOE: How many steps to run the test agent.
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


	#Define random coeffs vectors, and test individuals.
	'''temporal_objectives_weights = [0., 0., 0., 0.5, 0.5, 1.]
	meas_objective_weights1 = [1.0, 1.0, 1.0]
	meas_objective_weights2 = [0.01, 0.01, 0.01] #TODO: Beware, it may seem that 0,0,0 does not work. Could cause a crash?
	meas_objective_weights3 = [-0.5, -0.5, -1.]
	meas_objective_weights = [meas_objective_weights1, meas_objective_weights2, meas_objective_weights3]
	for i in range(3):
		experiment.test_new_individual(temporal_objectives_weights, meas_objective_weights[i])
		print("Evaluated weight vector ", meas_objective_weights[i])'''

	#problem: Reward depends only on frags - keeping a strong health doesn't matter. It will be hard to find solutions that do well when sacrificing frags for other objectives.


#EA code based on NEAT XOR example
def eval_genomes(genomes, config):
	global avg_outputs_storage
	for genome_id, genome in genomes:
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		genome.fitness = experiment_interface.test_new_individual(net)
		#We want to store the average outputs generated by the NN to see how the coeffs evolve over generations.
		#TODO Later, I may want to store all this rather than average. Maybe I can see interesting trends of how different objectives are prioritized at different points in agent's life?
		print("All nn output shape: ", np.array(experiment_interface.ag.objective_coeffs_log).shape)
		avg_nn_outputs = np.mean(np.array(experiment_interface.ag.objective_coeffs_log), axis=0)
		print("Avg nn outputs was: ", avg_nn_outputs)
		avg_outputs_storage.append(avg_nn_outputs)

		print("Finished evaluating genome: ", genome)

def run_neat(config_file):
	# Load configuration.
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 config_file)

	# Create the population, which is the top-level object for a NEAT run.
	p = neat.Population(config)

	# Add a stdout reporter to show progress in the terminal.
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	p.add_reporter(neat.Checkpointer(5))

	# 50 indivs seems to give 15 min per generation. 100 gen in 1 day, 250 in a weekend run.
	winner = p.run(eval_genomes, 100) #TODO#300) pop size 25

	# Display the winning genome.
	print('\nBest genome:\n{!s}'.format(winner))
	pickle.dump(winner, open("winner_network.pickle", "wb")) # Stored winner network. Load later and test.

	# Show output of the most fit genome against training data.
	print('\nOutput:')
	winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


	stats.save_genome_fitness()
	visualize.plot_stats(stats, ylog=False, view=False)
	#visualize.plot_species(stats, view=True)

if __name__ == '__main__':
	#Configuring the Deep NN experiment
	global experiment_interface #Couldn't find any other way to make this available to NEAT's fitness-function.
	experiment_interface = main(sys.argv[1:])

	#Stores average NN outputs. TODO Here we should also somehow store the generation number. Look into if NEAT allows that.
	global avg_outputs_storage
	avg_outputs_storage = []

	#Configuring NEAT
	run_neat("config")

	avg_outputs_storage=np.array(avg_outputs_storage)
	np.savetxt("evolving_nn_outputs.csv", avg_outputs_storage, delimiter=" ")


