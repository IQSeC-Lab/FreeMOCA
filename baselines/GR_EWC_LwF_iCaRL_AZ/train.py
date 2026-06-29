import torch
from torch import optim
from torch.utils.data import ConcatDataset
import numpy as np
import tqdm
import os
import copy
import utils
from data import malwareSubDatasetExemplars as SubDataset
from sklearn.preprocessing import MinMaxScaler
from data import ExemplarDataset
from continual_learner import ContinualLearner
import evaluate
import seaborn as sns
from sklearn import metrics #, StandardScaler
from sklearn import decomposition
from sklearn import manifold
from sklearn.manifold import TSNE
from evaluate import test, get_iter_test_dataset, oh

def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def custom_validate(model, dataset, batch_size=256, test_size=1024, verbose=True, allowed_classes=None,
             with_exemplars=False, no_task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    model.eval()

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda())
    total_tested = total_correct = 0
    
    correct_labels = []
    predicted_labels = []
    y_predicts_scores = []
    normalized_scores = []
    
    all_scores = []
    all_labels = []
    
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(model._device()), labels.to(model._device())
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        #print(labels)
        with torch.no_grad():
            scores = model(data) if (allowed_classes is None) else model(data)[:, allowed_classes]
            
            for sc in scores:
                all_scores.append(sc.cpu().numpy())
            #print(scores)
            #_, predicted = torch.max(scores, 1)

            #y_predicts_scores += list(predicted.detach().cpu().numpy())
                      
                
        # -update statistics
        #total_correct += (predicted == labels).sum().item()
        #total_tested += len(data)
        
        #correct_labels += list(labels.cpu().numpy())
        #predicted_labels += list(predicted.cpu().numpy())
        all_labels += list(labels.cpu().numpy())
    #precision = total_correct / total_tested
    #correct_labels = np.array(correct_labels)

    #if verbose:
    #     print('=>Precision {:.3f}'.format(precision))
    #print("all_scores, all_labels", all_scores, ', ', all_labels)
    return all_scores, all_labels

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, path, epoch, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(path, epoch, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(path, epoch, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, path, epoch, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.delete_previous_saved_model(path)
        path = path + 'best_model_epoch_' + str(epoch) + '.pt'
        print(f'{path}')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        
    def delete_previous_saved_model(self, path):
        saved_models = os.listdir(path)
        for prev_model in saved_models:
            prev_model = path + prev_model
            #print(prev_model)
            if os.path.isfile(prev_model):
                os.remove(prev_model)
            else: pass


def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        

#MARK: train_epoch_cl
def train_epoch_cl(model, model_save_path, train_datasets, test_datasets, args, replay_mode="none", scenario="class",classes_per_task=None, iters=2000, batch_size=32,
             generator=None, gen_iters=0, gen_loss_cbs=list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             use_exemplars=True, add_exemplars=False, metric_cbs=list(), test_set=None):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''
    
    #XY_test = np.load(data_dir + '/XY_test.npz')
    #X_te, Y_te = XY_test['X_test'], XY_test['Y_test']
    
    #X_te, Y_te = test_set[0], test_set[1]
    
    #print(model)
    #model_save_path = './ember_saved_models/1_first_class.pt'
    #model.load_state_dict(torch.load(model_save_path))
    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
    
    
    all_task_average_f1scores = []
    
    all_task_scores = []
    all_task_labels = []
    
    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        n_class = 50 + (task-1)*5
        XY_test = test_set[task-1]
        X_test, Y_test = XY_test[0], XY_test[1]
        
        # EMBER
        # padded_features = np.zeros((len(X_test), 2401 - 2381), dtype=np.float32)
        # X_test = np.concatenate((X_test, padded_features), axis=1)

        # AZ
        if hasattr(args, 'data_set'):
            if args.data_set == 'EMBER':
                orig_feats, target_feats = 2381, 2401
            elif args.data_set == 'ANDROZOO':
                orig_feats, target_feats = 2439, 2500
            else:
                orig_feats = target_feats = X_test.shape[1]
        else:
            orig_feats = target_feats = X_test.shape[1]

        padding_size = target_feats - orig_feats
        if padding_size > 0:
            padded_features = np.zeros((len(X_test), padding_size), dtype=np.float32)
            X_test = np.concatenate((X_test, padded_features), axis=1)

        x_ = torch.FloatTensor(X_test) 

        # One-hot Encoding
        y_oh = oh(Y_test, num_classes=n_class)
        y_oh = torch.tensor(y_oh, dtype=torch.float32)
        data_tensored = torch.utils.data.TensorDataset(x_, y_oh)
        test_loader = torch.utils.data.DataLoader(data_tensored, batch_size=256, num_workers=5, drop_last=False)
        #test_loader, _ = get_dataloader(X_test, Y_test, batchsize=256, n_class=n_class, scaler = scaler, train=False)

        # If offline replay-setting, create large database of all tasks so far
        if replay_mode=="offline" and (not scenario=="task"):
            train_dataset = ConcatDataset(train_datasets[:task])
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode=="offline" and scenario == "task":
            Exact = True
            previous_datasets = train_datasets

        # Add exemplars (if available) to current dataset (if requested)
        if add_exemplars and task>1:
            target_transform = (lambda y, x=classes_per_task: y%x) if scenario=="domain" else None
            exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=target_transform)
            training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        else:
            training_dataset = train_dataset

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            #MARK: changed
            init_classes = args.init_classes
            if task == 1:
                active_classes = list(range(init_classes))
            else:
                active_classes = list(range(init_classes + classes_per_task * (task -1)))
 
        print(task, active_classes)
        
        
        # Reset state of optimizer(s) for every task (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        # iters_left = iters_left_previous = 1
        # if scenario=="task":
        #     up_to_task = task if replay_mode=="offline" else task-1
        #     iters_left_previous = [1]*up_to_task
        #     data_loader_previous = [None]*up_to_task

        # Define tqdm progress bar(s)
        # progress = tqdm.tqdm(range(1, iters+1))
        # if generator is not None:
        #     progress_gen = tqdm.tqdm(range(1, gen_iters+1))
        
        #early_stopping = EarlyStopping(model, patience=5, verbose=True)
        
        # Loop over all iterations
        # iters_to_use = iters if (generator is None) else max(iters, gen_iters)
        # iters_to_use = iters if (generator is None) else max(iters, gen_iters)

        # for batch_index in range(1, iters_to_use+1):

        for epoch in range(1, 50 + 1):
            data_loader = utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=False)
            progress = tqdm.tqdm(range(1, len(data_loader)+1))

            if generator is not None:
                progress_gen = tqdm.tqdm(range(1, len(data_loader)+1))

            if Exact:
                data_loader_previous = utils.get_data_loader(ConcatDataset(previous_datasets),
                                                    batch_size=min(batch_size, len(ConcatDataset(previous_datasets))),
                                                    cuda=cuda, drop_last=False)
                iters_left_previous = iter(data_loader_previous)
                
            for batch_index, (x, y) in enumerate(data_loader):
                # -----------------Collect data------------------#

                #####-----CURRENT BATCH-----#####
                if replay_mode=="offline" and scenario=="task":
                    x = y = scores = None
                else:
                    #MARK: changed
                    # x, y = next(data_loader)                                    #--> sample training data of current task
                    #x = x.double()
                    y = y-classes_per_task*(task-1) if scenario=="task" else y  #--> ITL: adjust y-targets to 'active range'
                    x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                    # If --bce, --bce-distill & scenario=="class", calculate scores of current batch with previous model
                    binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                    if binary_distillation and scenario=="class" and (previous_model is not None):
                        with torch.no_grad():
                            scores = previous_model(x)[:, :(classes_per_task * (task - 1))]
                    else:
                        scores = None


                #####-----REPLAYED BATCH-----#####
                if not Exact and not Generative and not Current:
                    x_ = y_ = scores_ = None   #-> if no replay

                ##-->> Exact Replay <<--##
                if Exact:
                    scores_ = None
                    if scenario in ("domain", "class"):
                        # Sample replayed training data, move to correct device
                        #MARK: changed
                        # x_, y_ = next(data_loader_previous)
                        try:
                            x_, y_ = next(data_loader_previous)
                        except StopIteration:
                            iters_left_previous = iter(data_loader_previous)
                            x_, y_ = next(iters_left_previous)

                        x_ = x_.to(device)
                        y_ = y_.to(device) if (model.replay_targets=="hard") else None
                        # If required, get target scores (i.e, [scores_]         -- using previous model, with no_grad()
                        if (model.replay_targets=="soft"):
                            with torch.no_grad():
                                scores_ = previous_model(x_)
                            scores_ = scores_[:, :(classes_per_task*(task-1))] if scenario=="class" else scores_
                            #-> when scenario=="class", zero probabilities will be added in the [utils.loss_fn_kd]-function
                    elif scenario=="task":
                        # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                        x_ = list()
                        y_ = list()
                        up_to_task = task if replay_mode=="offline" else task-1
                        for task_id in range(up_to_task):
                            x_temp, y_temp = next(data_loader_previous[task_id])
                            x_.append(x_temp.to(device))
                            # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                            if model.replay_targets=="hard":
                                y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                                y_.append(y_temp.to(device))
                            else:
                                y_.append(None)
                        # If required, get target scores (i.e, [scores_]         -- using previous model
                        if (model.replay_targets=="soft") and (previous_model is not None):
                            scores_ = list()
                            for task_id in range(up_to_task):
                                with torch.no_grad():
                                    scores_temp = previous_model(x_[task_id])
                                scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                                scores_.append(scores_temp)

                ##-->> Generative / Current Replay <<--##
                if Generative or Current:
                    # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                    x_ = x if Current else previous_generator.sample(batch_size)

                    # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                    # -if there are no task-specific mask, obtain all predicted scores at once
                    if (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None):
                        with torch.no_grad():
                            all_scores_ = previous_model(x_)
                    # -depending on chosen scenario, collect relevant predicted scores (per task, if required)
                    if scenario in ("domain", "class") and (
                            (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None)
                    ):
                        scores_ = all_scores_[:,:(classes_per_task * (task - 1))] if scenario == "class" else all_scores_
                        _, y_ = torch.max(scores_, dim=1)
                    else:
                        # NOTE: it's possible to have scenario=domain with task-mask (so actually it's the Task-IL scenario)
                        # -[x_] needs to be evaluated according to each previous task, so make list with entry per task
                        scores_ = list()
                        y_ = list()
                        for task_id in range(task - 1):
                            # -if there is a task-mask (i.e., XdG is used), obtain predicted scores for each task separately
                            if hasattr(previous_model, "mask_dict") and previous_model.mask_dict is not None:
                                previous_model.apply_XdGmask(task=task_id + 1)
                                with torch.no_grad():
                                    all_scores_ = previous_model(x_)
                            if scenario=="domain":
                                temp_scores_ = all_scores_
                            else:
                                temp_scores_ = all_scores_[:,
                                            (classes_per_task * task_id):(classes_per_task * (task_id + 1))]
                            _, temp_y_ = torch.max(temp_scores_, dim=1)
                            scores_.append(temp_scores_)
                            y_.append(temp_y_)

                    # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                    y_ = y_ if (model.replay_targets == "hard") else None
                    scores_ = scores_ if (model.replay_targets == "soft") else None
                
                #print(active_classes, task)

                #---> Train MAIN MODEL
                # if batch_index <= iters:
                #print(f'active classes in train.py {active_classes}')
                # Train the main model with this batch
                #print(active_classes, task)
                #print(y, y_, scores, scores_)

                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt = 1./task)

                # Update running parameter importance estimates in W
                if isinstance(model, ContinualLearner) and (model.si_c>0):
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad*(p.detach()-p_old[n]))
                            p_old[n] = p.detach().clone()

                # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, task=task)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, task=task)
                    

                #---> Train GENERATOR
                # if generator is not None and batch_index <= gen_iters:
                if generator is not None:

                    # Train the generator with this batch
                    loss_dict = generator.train_a_batch(x, y, x_=x_, y_=y_, scores_=scores_, active_classes=active_classes,
                                                        task=task, rnt=1./task)

                    # Fire callbacks on each iteration
                    for loss_cb in gen_loss_cbs:
                        if loss_cb is not None:
                            loss_cb(progress_gen, batch_index, loss_dict, task=task)
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(generator, batch_index, task=task)


        ##----------> UPON FINISHING EACH TASK...
        # Close progres-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()
        
        if task ==1:
            copied_base_model = copy.deepcopy(model)
            copied_base_model.classifier = Identity()

            #task_scores = []
            #task_labels = []
            #for tl in range(task):
            task_current_scores, labels_current = custom_validate(
                    copied_base_model, test_datasets[0], verbose=True, test_size=None, task=1, with_exemplars=False,
                    allowed_classes= None
                )
            #task_scores += list(task_current_scores)
            #task_labels += list(labels_current)

            task_scores, task_labels = np.array(task_current_scores), np.array(labels_current) 

            #print(task_scores.shape, task_labels.shape)
            #tsne = TSNE(n_components=2, verbose=1, random_state=123)
            #X_embedded = tsne.fit_transform(task_scores)

            all_task_scores.append(task_scores), all_task_labels.append(task_labels)
        
        
        
        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, ContinualLearner) and (model.ewc_lambda>0):
            # -find allowed classes
            allowed_classes = list(
                range(classes_per_task*(task-1), classes_per_task*task)
            ) if scenario=="task" else (list(range(classes_per_task*task)) if scenario=="class" else None)
            # -if needed, apply correct task-specific mask
            if model.mask_dict is not None:
                model.apply_XdGmask(task=task)
            # -estimate FI-matrix
            model.estimate_fisher(training_dataset, allowed_classes=allowed_classes)

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            model.update_omega(W, model.epsilon)

        # EXEMPLARS: update exemplar sets
        if (add_exemplars or use_exemplars) or replay_mode=="exemplars":
            #MARK: 클래스 범위 수정
            exemplars_per_class = int(np.floor(model.memory_budget / (args.init_classes+classes_per_task*(task-1))))
            # reduce examplar-sets
            model.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            # new_classes = list(range(classes_per_task)) if scenario=="domain" else list(range(classes_per_task*(task-1),
            #                                                                                   classes_per_task*task))
            if scenario=="domain":
                new_classes = list(range(args.init_classes+classes_per_task*(task-1)))
            else:
                if task == 1:
                    new_classes = list(range(args.init_classes))
                else:
                    new_classes = list(range(
                        args.init_classes+classes_per_task*(task-2),
                        args.init_classes+classes_per_task*(task-1)
                    ))

            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, 
                                           orig_length_features=2492,\
                                           target_length_features=2500,\
                                           sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            model.compute_means = True

        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                #print(f'metric_cb task {task}')
                metric_cb(args, model, iters, task=task)

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_mode == 'generative':
            Generative = True
            previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
        elif replay_mode == 'current':
            Current = True
        elif replay_mode in ('exemplars', 'exact'):
            Exact = True
            if replay_mode == "exact":
                previous_datasets = train_datasets[:task]
            else:
                if scenario == "task":
                    previous_datasets = []
                    for task_id in range(task):
                        previous_datasets.append(
                            ExemplarDataset(
                                model.exemplar_sets[
                                (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                target_transform=lambda y, x=classes_per_task * task_id: y + x)
                        )
                else:
                    target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                    previous_datasets = [
                        ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]
               
        #print(f'Saving Model after task {task}')   
        #model_save_path = './ember_saved_models/GR/'
        #create_parent_folder(model_save_path)
        #save_path = model_save_path + str(task) + '_task_ember_debug.pt'
        #torch.save(model.state_dict(), save_path)
        
        if task != 1:
            copied_base_model = copy.deepcopy(model)
            copied_base_model.classifier = Identity()

            task_scores = []
            task_labels = []
            for tl in range(task):
                task_current_scores, labels_current = custom_validate(
                        copied_base_model, test_datasets[tl], verbose=True, test_size=None, task=tl, with_exemplars=False,
                        allowed_classes= None
                    )
                task_scores += list(task_current_scores)
                task_labels += list(labels_current)

            task_scores, task_labels = np.array(task_scores), np.array(task_labels) 

            #print(task_scores.shape, task_labels.shape)
            #tsne = TSNE(n_components=2, verbose=1, random_state=123)
            #X_embedded = tsne.fit_transform(task_scores)

            all_task_scores.append(task_scores), all_task_labels.append(task_labels)
        
        test("cuda", model, test_loader)


#MARK: train_iter_cl
def train_iter_cl(model, model_save_path, train_datasets, test_datasets, args, replay_mode="none", scenario="class",classes_per_task=None,iters=2000,batch_size=32,
             generator=None, gen_iters=0, gen_loss_cbs=list(), loss_cbs=list(), eval_cbs=list(), sample_cbs=list(),
             use_exemplars=True, add_exemplars=False, metric_cbs=list(), test_set=None):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''
    
    #print(model)
    #model_save_path = './ember_saved_models/1_first_class.pt'
    #model.load_state_dict(torch.load(model_save_path))
    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, ContinualLearner) and (model.si_c>0):
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer('{}_SI_prev_task'.format(n), p.data.clone())
    
    
    all_task_average_f1scores = []
    
    all_task_scores = []
    all_task_labels = []
    
    # Loop over all tasks.
    for task, train_dataset in enumerate(train_datasets, 1):
        n_class = 50 + (task-1)*5
        XY_test = test_set[task-1]
        X_test, Y_test = XY_test[0], XY_test[1]
        
        # EMBER
        # padded_features = np.zeros((len(X_test), 2401 - 2381), dtype=np.float32)
        # X_test = np.concatenate((X_test, padded_features), axis=1)
        # x_ = torch.FloatTensor(X_test) 

        #AZ
        if hasattr(args, 'data_set'):
            if args.data_set == 'EMBER':
                orig_feats, target_feats = 2381, 2401
            elif args.data_set == 'ANDROZOO':
                orig_feats, target_feats = 2439, 2500
            else:
                orig_feats = target_feats = X_test.shape[1]
        else:
            orig_feats = target_feats = X_test.shape[1]

        padding_size = target_feats - orig_feats
        if padding_size > 0:
            padded_features = np.zeros((len(X_test), padding_size), dtype=np.float32)
            X_test = np.concatenate((X_test, padded_features), axis=1)
        
        x_ = torch.FloatTensor(X_test)

        # One-hot Encoding
        y_oh = oh(Y_test, num_classes=n_class)
        y_oh = torch.tensor(y_oh, dtype=torch.float32)
        data_tensored = torch.utils.data.TensorDataset(x_, y_oh)
        test_loader = torch.utils.data.DataLoader(data_tensored, batch_size=256, num_workers=5, drop_last=False)
        #test_loader, _ = get_dataloader(X_test, Y_test, batchsize=256, n_class=n_class, scaler = scaler, train=False)
        
        # If offline replay-setting, create large database of all tasks so far
        if replay_mode=="offline" and (not scenario=="task"):
            train_dataset = ConcatDataset(train_datasets[:task])
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode=="offline" and scenario == "task":
            Exact = True
            previous_datasets = train_datasets

        # Add exemplars (if available) to current dataset (if requested)
        if add_exemplars and task>1: # icarl
            target_transform = (lambda y, x=classes_per_task: y%x) if scenario=="domain" else None
            exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=target_transform)
            training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        else:
            training_dataset = train_dataset

        # Prepare <dicts> to store running importance estimates and param-values before update ("Synaptic Intelligence")
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            
            init_classes = args.init_classes
            if task == 1:
                active_classes = list(range(init_classes))
            else:
                active_classes = list(range(init_classes + classes_per_task * (task -1)))
 
        print(task, active_classes)
        
        
        # Reset state of optimizer(s) for every task (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
        if (generator is not None) and generator.optim_type=="adam_reset":
            generator.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if scenario=="task":
            up_to_task = task if replay_mode=="offline" else task-1
            iters_left_previous = [1]*up_to_task
            data_loader_previous = [None]*up_to_task

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, gen_iters+1))
        
        #early_stopping = EarlyStopping(model, patience=5, verbose=True)
        
        # Loop over all iterations
        iters_to_use = iters if (generator is None) else max(iters, gen_iters)

        for batch_index in range(1, iters_to_use+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
                # NOTE:  [train_dataset]  is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(data_loader)
            if Exact:
                if scenario=="task":
                    up_to_task = task if replay_mode=="offline" else task-1
                    batch_size_replay = int(np.ceil(batch_size/up_to_task)) if (up_to_task>1) else batch_size
                    # -in Task-IL scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[task_id]))
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id]==0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                previous_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])
                else:
                    iters_left_previous -= 1
                    if iters_left_previous==0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)


            # -----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            if replay_mode=="offline" and scenario=="task":
                x = y = scores = None
            else:
                x, y = next(data_loader)                                    #--> sample training data of current task
                #x = x.double()
                y = y-classes_per_task*(task-1) if scenario=="task" else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                # If --bce, --bce-distill & scenario=="class", calculate scores of current batch with previous model
                binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                if binary_distillation and scenario=="class" and (previous_model is not None):
                    with torch.no_grad():
                        scores = previous_model(x)[:, :(classes_per_task * (task - 1))]
                else:
                    scores = None


            #####-----REPLAYED BATCH-----#####
            if not Exact and not Generative and not Current:
                x_ = y_ = scores_ = None   #-> if no replay

            ##-->> Exact Replay <<--##
            if Exact:
                scores_ = None
                if scenario in ("domain", "class"):
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (model.replay_targets=="hard") else None
                    # If required, get target scores (i.e, [scores_]         -- using previous model, with no_grad()
                    if (model.replay_targets=="soft"):
                        with torch.no_grad():
                            scores_ = previous_model(x_)
                        scores_ = scores_[:, :(classes_per_task*(task-1))] if scenario=="class" else scores_
                        #-> when scenario=="class", zero probabilities will be added in the [utils.loss_fn_kd]-function
                elif scenario=="task":
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode=="offline" else task-1
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        x_.append(x_temp.to(device))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if model.replay_targets=="hard":
                            y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                    # If required, get target scores (i.e, [scores_]         -- using previous model
                    if (model.replay_targets=="soft") and (previous_model is not None):
                        scores_ = list()
                        for task_id in range(up_to_task):
                            with torch.no_grad():
                                scores_temp = previous_model(x_[task_id])
                            scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                            scores_.append(scores_temp)

            ##-->> Generative / Current Replay <<--##
            if Generative or Current:
                # Get replayed data (i.e., [x_]) -- either current data or use previous generator
                x_ = x if Current else previous_generator.sample(batch_size)

                # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                # -if there are no task-specific mask, obtain all predicted scores at once
                if (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None):
                    with torch.no_grad():
                        all_scores_ = previous_model(x_)
                # -depending on chosen scenario, collect relevant predicted scores (per task, if required)
                if scenario in ("domain", "class") and (
                        (not hasattr(previous_model, "mask_dict")) or (previous_model.mask_dict is None)
                ):
                    scores_ = all_scores_[:,:(classes_per_task * (task - 1))] if scenario == "class" else all_scores_
                    _, y_ = torch.max(scores_, dim=1)
                else:
                    # NOTE: it's possible to have scenario=domain with task-mask (so actually it's the Task-IL scenario)
                    # -[x_] needs to be evaluated according to each previous task, so make list with entry per task
                    scores_ = list()
                    y_ = list()
                    for task_id in range(task - 1):
                        # -if there is a task-mask (i.e., XdG is used), obtain predicted scores for each task separately
                        if hasattr(previous_model, "mask_dict") and previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(task=task_id + 1)
                            with torch.no_grad():
                                all_scores_ = previous_model(x_)
                        if scenario=="domain":
                            temp_scores_ = all_scores_
                        else:
                            temp_scores_ = all_scores_[:,
                                           (classes_per_task * task_id):(classes_per_task * (task_id + 1))]
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        scores_.append(temp_scores_)
                        y_.append(temp_y_)

                # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                y_ = y_ if (model.replay_targets == "hard") else None
                scores_ = scores_ if (model.replay_targets == "soft") else None
            
            #print(active_classes, task)

            #---> Train MAIN MODEL
            if batch_index <= iters:
                #print(f'active classes in train.py {active_classes}')
                # Train the main model with this batch
                #print(active_classes, task)
                #print(y, y_, scores, scores_)
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt = 1./task)

                # Update running parameter importance estimates in W
                if isinstance(model, ContinualLearner) and (model.si_c>0):
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad*(p.detach()-p_old[n]))
                            p_old[n] = p.detach().clone()

                # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)

                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, task=task)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, task=task)
                

            #---> Train GENERATOR
            if generator is not None and batch_index <= gen_iters:

                # Train the generator with this batch
                loss_dict = generator.train_a_batch(x, y, x_=x_, y_=y_, scores_=scores_, active_classes=active_classes,
                                                    task=task, rnt=1./task)

                # Fire callbacks on each iteration
                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict, task=task)
                for sample_cb in sample_cbs:
                    if sample_cb is not None:
                        sample_cb(generator, batch_index, task=task)


        ##----------> UPON FINISHING EACH TASK...

        # Close progres-bar(s)
        progress.close()
        if generator is not None:
            progress_gen.close()
        
        if task ==1:
            copied_base_model = copy.deepcopy(model)
            copied_base_model.classifier = Identity()

            #task_scores = []
            #task_labels = []
            #for tl in range(task):
            task_current_scores, labels_current = custom_validate(
                    copied_base_model, test_datasets[0], verbose=True, test_size=None, task=1, with_exemplars=False,
                    allowed_classes= None
                )
            #task_scores += list(task_current_scores)
            #task_labels += list(labels_current)

            task_scores, task_labels = np.array(task_current_scores), np.array(labels_current) 

            #print(task_scores.shape, task_labels.shape)
            #tsne = TSNE(n_components=2, verbose=1, random_state=123)
            #X_embedded = tsne.fit_transform(task_scores)

            all_task_scores.append(task_scores), all_task_labels.append(task_labels)
        
        
        
        # EWC: estimate Fisher Information matrix (FIM) and update term for quadratic penalty
        if isinstance(model, ContinualLearner) and (model.ewc_lambda>0):
            # -find allowed classes
            allowed_classes = list(
                range(classes_per_task*(task-1), classes_per_task*task)
            ) if scenario=="task" else (list(range(classes_per_task*task)) if scenario=="class" else None)
            # -if needed, apply correct task-specific mask
            if model.mask_dict is not None:
                model.apply_XdGmask(task=task)
            # -estimate FI-matrix
            model.estimate_fisher(training_dataset, allowed_classes=allowed_classes)

        # SI: calculate and update the normalized path integral
        if isinstance(model, ContinualLearner) and (model.si_c>0):
            model.update_omega(W, model.epsilon)

        # EXEMPLARS: update exemplar sets
        if (add_exemplars or use_exemplars) or replay_mode=="exemplars":
            #MARK: 클래스 범위 수정
            exemplars_per_class = int(np.floor(model.memory_budget / (args.init_classes+classes_per_task*(task-1) )))
            # reduce examplar-sets
            model.reduce_exemplar_sets(exemplars_per_class)
            # for each new class trained on, construct examplar-set
            # new_classes = list(range(classes_per_task)) if scenario=="domain" else list(range(classes_per_task*(task-1),
            #                                                                                   classes_per_task*task))
            if scenario=="domain":
                new_classes = list(range(args.init_classes+classes_per_task*(task-1)))
            else:
                if task == 1:
                    new_classes = list(range(args.init_classes))
                else:
                    new_classes = list(range(
                        args.init_classes+classes_per_task*(task-2),
                        args.init_classes+classes_per_task*(task-1)
                    ))

            for class_id in new_classes:
                # create new dataset containing only all examples of this class
                class_dataset = SubDataset(original_dataset=train_dataset, 
                                           orig_length_features=2492,\
                                           target_length_features=2500,\
                                           sub_labels=[class_id])
                # based on this dataset, construct new exemplar-set for this class
                model.construct_exemplar_set(dataset=class_dataset, n=exemplars_per_class)
            model.compute_means = True

        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                #print(f'metric_cb task {task}')
                metric_cb(args, model, iters, task=task)

        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_mode == 'generative':
            Generative = True
            previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model
        elif replay_mode == 'current':
            Current = True
        elif replay_mode in ('exemplars', 'exact'):
            Exact = True
            if replay_mode == "exact":
                previous_datasets = train_datasets[:task]
            else:
                if scenario == "task":
                    previous_datasets = []
                    for task_id in range(task):
                        previous_datasets.append(
                            ExemplarDataset(
                                model.exemplar_sets[
                                (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                target_transform=lambda y, x=classes_per_task * task_id: y + x)
                        )
                else:
                    target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                    previous_datasets = [
                        ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]
               
        #print(f'Saving Model after task {task}')   
        #model_save_path = './ember_saved_models/GR/'
        #create_parent_folder(model_save_path)
        #save_path = model_save_path + str(task) + '_task_ember_debug.pt'
        #torch.save(model.state_dict(), save_path)
        
        if task != 1:
            copied_base_model = copy.deepcopy(model)
            copied_base_model.classifier = Identity()

            task_scores = []
            task_labels = []
            for tl in range(task):
                task_current_scores, labels_current = custom_validate(
                        copied_base_model, test_datasets[tl], verbose=True, test_size=None, task=tl, with_exemplars=False,
                        allowed_classes= None
                    )
                task_scores += list(task_current_scores)
                task_labels += list(labels_current)

            task_scores, task_labels = np.array(task_scores), np.array(task_labels) 

            #print(task_scores.shape, task_labels.shape)
            #tsne = TSNE(n_components=2, verbose=1, random_state=123)
            #X_embedded = tsne.fit_transform(task_scores)

            all_task_scores.append(task_scores), all_task_labels.append(task_labels)
        
        test("cuda", model, test_loader)