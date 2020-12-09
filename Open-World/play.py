raw_training_data_queue = Queue() 
training_data_queue = Queue() 
data_generators = list() 
for i in range(args.n_generator): 
data_generators.append(Process(target=data_generator_func, args=( raw_training_data_queue, training_data_queue, model.train_hr_t, model.n_entity, args.neg_weight , model.edge_to_all_H,model.n_relation,model.H,model.L,model.entity_to_edge))) 
data_generators[-1].start() 
data_evaluators = list() 
evaluation_queue = JoinableQueue() 
result_queue = Queue() 
for i in range(args.n_worker): 
data_evaluators.append(Process(target=worker_func, args=(evaluation_queue, result_queue, model.hr_t)))
 #worker = data_evaluators[-1].start()

 print ('hit @1:',np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 2)) 
 print ('hit @3:',np.mean(np.asarray(accu_mean_rank_t, dtype=np.int32) < 4)) 
 for p in data_evaluators: 
 	p.terminate() 
 for p in data_generators: 
 		p.terminate()