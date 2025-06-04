import torch
from function import get_iter_train_dataset, get_iter_test_dataset, get_dataloader
from torch.autograd import Variable
import torch.nn.functional as F




################################################################################################
# This function sets up the essential variables used in either adversarial training (e.g., GANs) or general batch training scenarios, ensuring compatibility with both CPU and GPU environments.
def vars_batch_train(config, x_, y_):
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
      

# def compute_fisher(config, model, dataloader, criterion, device):
#     """
#     Compute diagonal Fisher Information Matrix (FIM) for model parameters.
#     """
#     model.eval()
#     fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}

#     for n, (inputs, labels) in enumerate(dataloader):

#         inputs = inputs.float().to(device)
#         labels = labels.float().to(device)
#         inputs, labels = vars_batch_train(config, inputs, labels)
#         model.zero_grad()
#         outputs = model(inputs)

#         loss = criterion(outputs, labels)  # assuming classification task
#         loss.backward()

#         for name, param in model.named_parameters():
#             if param.grad is not None and param.requires_grad:
#                 fisher[name] += param.grad.data.pow(2)

#     # Average over number of samples
#     for name in fisher:
#         fisher[name] /= len(dataloader)

#     return fisher

#####################################################################################################

def run_batch_BCE(config, C, C_optimizer, criterion, BCELoss, x_, y_):
  x_, y_ = vars_batch_train(config, x_, y_)
  update_Classifier(config, C_optimizer, C, criterion, x_, y_)
  
####################################################################################################

# def data_task_joint(config, X_train, Y_train, X_test, Y_test, scaler):
#     X_train_t, Y_train_t = get_iter_train_dataset_joint(X_train,  Y_train, n_class=config.n_class, n_inc=config.n_inc, task=config.task)
#     train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler)
#     X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class=config.n_class)
#     test_loader, _ = get_dataloader(X_test_t, Y_test_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler, train=False)
#     return X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler

def data_task(config, X_train, Y_train, X_test, Y_test, scaler):
    X_train_t, Y_train_t = get_iter_train_dataset(X_train,  Y_train, n_class=config.n_class, n_inc=config.n_inc, task=config.task)
    train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler)
    X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class=config.n_class)
    test_loader, _ = get_dataloader(X_test_t, Y_test_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler, train=False)
    return X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler



def report_result(config, ls_a):
    # print(config.Generator_loss, ",", config.sample_select, ", seed: ", config.seed_, ",", config.epochs, "epochs")
    print("The Accuracy for each task:", ls_a)
    print("The Global Average:", sum(ls_a)/len(ls_a))
    file_name =  "None"+ "_seed"+ str(config.seed_)+ "_"+ str(config.epochs)+ "epochs"
    with open(file_name+'.txt', 'w') as file:
      file.write("The Accuracy for each task: " + str(ls_a))
      file.write("The Global Average: " + str(sum(ls_a)/len(ls_a)))

