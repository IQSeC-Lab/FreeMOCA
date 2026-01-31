import torch
from function import get_iter_train_dataset, get_iter_train_dataset_joint, get_iter_test_dataset, get_dataloader
from torch.autograd import Variable
import torch.nn.functional as F



################################################################################################
# This function sets up the essential variables used in either adversarial training (e.g., GANs) or general batch training scenarios, ensuring compatibility with both CPU and GPU environments.
def vars_batch_train(config, x_, y_):

      if config.dataset == "EMBER":
          feats_length = 2381
      elif config.dataset == "AZ":
            # AZ Class-IL vs Domain-IL differ in preprocessing
          feats_length = 2439 if config.scenario == "class" else 1789
      else:
            raise ValueError(f"Unknown dataset {config.dataset}")
      x_ = x_.view([-1, config.feats_length])
      #me added
      x_ = Variable(x_)
      if config.use_cuda:
        x_ = x_.to(config.device)
        y_ = y_.to(config.device)
      return x_, y_

################################################################################################

def update_Classifier(config, C_optimizer, C, criterion, x_, y_):
      C_optimizer.zero_grad()
      output = C(x_)
      if config.use_cuda:
         output = output.to(config.device)
      C_loss = criterion(output, y_)
      C_loss.backward()
      C_optimizer.step()
      

#####################################################################################################

def run_batch_BCE(config, C_optimizer, C, criterion, x_, y_):
  x_, y_ = vars_batch_train(config, x_, y_)
  update_Classifier(config, C_optimizer, C, criterion, x_, y_)
  
####################################################################################################

def data_task_joint(config, X_train, Y_train, X_test, Y_test, scaler):
    X_train_t, Y_train_t = get_iter_train_dataset_joint(X_train,  Y_train, n_class=config.n_class, n_inc=config.n_inc, task=config.task)
    train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler)
    X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class=config.n_class)
    test_loader, _ = get_dataloader(X_test_t, Y_test_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler, train=False)
    return X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler

def data_task(config, X_train, Y_train, X_test, Y_test, scaler):
    X_train_t, Y_train_t = get_iter_train_dataset(X_train,  Y_train, n_class=config.n_class, n_inc=config.n_inc, task=config.task)
    train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler)
    X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class=config.n_class)
    test_loader, _ = get_dataloader(X_test_t, Y_test_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler, train=False)
    return X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler

def data_task_domain(config, X_train_all, Y_train_all, X_test_all, Y_test_all, scaler):
    """
    Domain-IL:
    Uses preloaded per-domain data from data_.dataset(config)
    """

    domain_id = config.task

    X_train = X_train_all[domain_id]
    Y_train = Y_train_all[domain_id]
    X_test  = X_test_all[domain_id]
    Y_test  = Y_test_all[domain_id]

    train_loader, scaler = get_dataloader(
        X_train, Y_train,
        batchsize=config.batchsize,
        n_class=config.init_classes,
        scaler=scaler,
        train=True
    )

    test_loader, _ = get_dataloader(
        X_test, Y_test,
        batchsize=config.batchsize,
        n_class=config.init_classes,
        scaler=scaler,
        train=False
    )

    return X_train, Y_train, train_loader, X_test, Y_test, test_loader, scaler



def report_result(config, ls_a):
    # print(config.Generator_loss, ",", config.sample_select, ", seed: ", config.seed_, ",", config.epochs, "epochs")
    print("The Accuracy for each task:", ls_a)
    print("The Global Average:", sum(ls_a)/len(ls_a))
    file_name =  "main"+ "_seed"+ str(config.seed_)+ "_"+ str(config.epochs)+ "epochs"
    with open(file_name+'.txt', 'w') as file:
      file.write("The Accuracy for each task: " + str(ls_a))
      file.write("The Global Average: " + str(sum(ls_a)/len(ls_a)))

