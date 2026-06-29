import torch
from function import get_iter_train_dataset, get_dataloader, get_iter_acc_test_dataset, get_iter_bwt_test_dataset
from torch.autograd import Variable



################################################################################################

def vars_batch_train(config, x_, y_):
      x_ = x_.view([-1, config.feats_length])
      # y_real and y_fake mean fake or true in Discriminator
      y_real_ = Variable(torch.ones(x_.size(0), 1))
      y_fake_ = Variable(torch.zeros(x_.size(0), 1))
      if config.use_cuda:
        y_real_, y_fake_ = y_real_.to(config.device), y_fake_.to(config.device)
      z_ = torch.rand((x_.size(0), config.z_dim))
      x_, z_ = Variable(x_), Variable(z_)
      if config.use_cuda:
        x_, z_, y_ = x_.to(config.device), z_.to(config.device), y_.to(config.device)
      return x_, y_, z_, y_real_, y_fake_

################################################################################################

def update_Discriminator(D, G, D_optimizer, BCELoss, x_, z_, y_real_, y_fake_):
      D_optimizer.zero_grad()
      D_real, _ = D(x_)
      D_real_loss = BCELoss(D_real, y_real_[:x_.size(0)])
      G_ = G(z_)
      D_fake, _ = D(G_)
      D_fake_loss = BCELoss(D_fake, y_fake_[:x_.size(0)])
      D_loss = D_real_loss + D_fake_loss
      D_loss.backward()
      D_optimizer.step()



def update_Generator_BCE(D, G, G_optimizer, BCELoss, x_, z_, y_real_):
      G_optimizer.zero_grad()
      G_ = G(z_)
      D_fake, _ = D(G_)
      G_loss = BCELoss(D_fake, y_real_[:x_.size(0)])
      G_loss.backward()
      G_optimizer.step()



def update_Generator_FML(D, G, G_optimizer, x_, z_):
      G_optimizer.zero_grad()
      fake_data = G(z_)
      _, features_fake = D(fake_data)
      _, features_real_unl = D(x_)
      feature_mean_real = torch.mean(features_real_unl, dim=0)
      feature_mean_fake = torch.mean(features_fake, dim=0)
      G_loss = torch.mean(torch.abs(feature_mean_real - feature_mean_fake))
      G_loss.backward()
      G_optimizer.step()



def update_Classifier(config, C_optimizer, C, criterion, x_, y_):
      C_optimizer.zero_grad()
      output = C(x_)
      if config.use_cuda:
         output = output.to(config.device)
      C_loss = criterion(output, y_)
      C_loss.backward()
      C_optimizer.step()


#####################################################################################################


def run_batch_BCE(config, G, D, C, G_optimizer, D_optimizer, C_optimizer, criterion, BCELoss, x_, y_):
  x_, y_, z_, y_real_, y_fake_ = vars_batch_train(config, x_, y_)
  update_Generator_BCE(D, G, G_optimizer, BCELoss, x_, z_, y_real_)
  z_ = Variable(torch.rand((x_.size(0), config.z_dim))).to(config.device)
  update_Discriminator(D, G, D_optimizer, BCELoss, x_, z_, y_real_, y_fake_)
  update_Classifier(config, C_optimizer, C, criterion, x_, y_)


def run_batch_FML(config, G, D, C, G_optimizer, D_optimizer, C_optimizer, criterion, BCELoss, x_, y_):
  x_, y_, z_, y_real_, y_fake_ = vars_batch_train(config, x_, y_)
  update_Generator_FML(D, G, G_optimizer, x_, z_)
  z_ = Variable(torch.rand((x_.size(0), config.z_dim))).to(config.device)
  update_Discriminator(D, G, D_optimizer, BCELoss, x_, z_, y_real_, y_fake_)
  update_Classifier(config, C_optimizer, C, criterion, x_, y_)


################################################################################################

def col_arr(config, X_train_t):
  logits_collect = []
  if config.sample_select == 'L1_C_Mean':
    logits_collect = [[] for k in range(config.n_class)]
  return logits_collect 



def collect_logits(config, C, logits_collect, inputs, labels, batch):
  with torch.no_grad():
    C.eval()
    if config.sample_select == 'L1_B_Mean': logits_collect.append(C.get_logits(inputs).to("cpu"))
    elif config.sample_select == 'L1_C_Mean':
      temp_vec = C.get_logits(inputs).to("cpu")
      for ind, (inp, lab) in enumerate(zip(inputs, labels)):
        logits_collect[int(torch.max(lab, dim = 0)[1])].append(temp_vec[int(ind)])
  return logits_collect


####################################################################################################

def data_task(config, X_train, Y_train, X_test, Y_test):
  X_train_t, Y_train_t = get_iter_train_dataset(X_train, Y_train, 
                                                n_class=config.n_class, n_inc=config.n_inc, task=config.task)
  train_loader = get_dataloader(X_train_t, Y_train_t, batchsize=config.batchsize, n_class=config.n_class)
  #MARK: 일부만 가져오도록 변경
  X_test_acc_t, Y_test_acc_t = get_iter_acc_test_dataset(X_test, Y_test, n_class=config.n_class)
  X_test_bwt_t, Y_test_bwt_t = get_iter_bwt_test_dataset(X_test, Y_test, 
                                                n_class=config.n_class, n_inc=config.n_inc, task=config.task)
  acc_test_loader = get_dataloader(X_test_acc_t, Y_test_acc_t, batchsize=config.batchsize, n_class=config.n_class, train=False)
  bwt_test_loader = get_dataloader(X_test_bwt_t, Y_test_bwt_t, batchsize=config.batchsize, n_class=config.n_class, train=False)
  return X_train_t, Y_train_t, X_test_acc_t, Y_test_acc_t, X_test_bwt_t, Y_test_bwt_t, train_loader, acc_test_loader, bwt_test_loader


def mean_logits(config, logits_collect):
    if config.sample_select == 'L1_B_Mean':
      logits_real = []
      for i, row in enumerate(logits_collect):
        if row == []: 
           print(i)
           continue
        logits_real.append(torch.mean(row, dim = 0).float())
      logits_real = torch.stack(logits_real)
    elif config.sample_select == 'L1_C_Mean':
      logits_real = []
      for i, row in enumerate(logits_collect):
        if row == []: 
           print(i)
           continue
        logits_real.append(torch.mean(torch.stack(row).float(), dim=0))
      logits_real = torch.stack(logits_real)
    else: logits_real = None
    
    return logits_real


def report_result(config, ls_a):
  print(config.Generator_loss, ",", config.sample_select, ", seed: ", config.seed_, ",", config.epochs, "epochs")
  print("The Accuracy for each task:", ls_a)
  print("The Global Average:", sum(ls_a)/len(ls_a))
  file_name = config.Generator_loss+ "_"+ config.sample_select+ "_seed"+ str(config.seed_)+ "_"+ str(config.epochs)+ "epochs"
  with open(file_name+'.txt', 'w') as file:
    file.write("The Accuracy for each task: " + str(ls_a))
    file.write("The Global Average: " + str(sum(ls_a)/len(ls_a)))

