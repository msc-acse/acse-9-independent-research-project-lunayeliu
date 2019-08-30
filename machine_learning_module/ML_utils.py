
def set_seed(seed):
    """
    Function which could fix all seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  -
    torch.backends.cudnn.enabled   = False
    return True


def Normlization(X, mean, std):
    """Normlize the image channel by channel
    using (x-mean)/std, to make the train data 0-mean, 1-std

    input
    X: the data to normlize
    mean: mean value of training data
    std: standard deviation   of training data

    return
    normlized data
    """

    X -=  mean
    X /=  std
    print('mean', mean)
    print('std', std)
    return X


def train(model, optimizer, criterion, data_loader):
    """The training funciton

    Input
    model: training model
    optimizer: the optimizer algorithm
    criterion: error function
    data_loader: torch data load used for feeding training data

    return
    Current step training loss and accuracy
    """
    model.train()
    train_loss, train_accuracy = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        a2 = model(X.view(-1, size))

        loss = criterion(a2, y)
        loss.backward()
        train_loss += loss*X.size(0)
        y_pred = F.log_softmax(a2, dim=1).max(1)[1]
        train_accuracy += accuracy_score(y.cpu().numpy(), y_pred.detach().cpu().numpy())*X.size(0)
        optimizer.step()

    return train_loss/len(data_loader.dataset), train_accuracy/len(data_loader.dataset)

def validate(model, criterion, data_loader):
    """The validation funciton

    Input
    model: validation model
    criterion: error function
    data_loader: torch data load used for feeding validation data

    return
    Current step validation loss and accuracy
    """
    model.eval()
    validation_loss, validation_accuracy = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, size))
            loss = criterion(a2, y)
            validation_loss += loss*X.size(0)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            validation_accuracy += accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())*X.size(0)

    return validation_loss/len(data_loader.dataset), validation_accuracy/len(data_loader.dataset)

def evaluate(model, data_loader):
    """The evaluation funciton

    Input
    model: evaluation model
    data_loader: torch data load used for feeding test data

    return
    Current step validation loss and accuracy
    """
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X.view(-1, size))
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    return np.concatenate(y_preds, 0),  np.concatenate(ys, 0)

def train_model(model):
  """The train model function, which controls the whole training process

  input
  model: model to train

  return
  trained model

  """
  set_seed(seed)

  #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay )
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay )
  criterion = nn.CrossEntropyLoss()

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
  validation_loader = DataLoader(validate_data, batch_size=test_batch_size, shuffle=False, num_workers=4)

  liveloss = PlotLosses()
  for epoch in range(n_epochs):
      logs = {}
      train_loss, train_accuracy = train(model, optimizer, criterion, train_loader)

      logs['' + 'log loss'] = train_loss.item()
      logs['' + 'accuracy'] = train_accuracy.item()

      validation_loss, validation_accuracy = validate(model, criterion, validation_loader)
      logs['val_' + 'log loss'] = validation_loss.item()
      logs['val_' + 'accuracy'] = validation_accuracy.item()
      liveloss.update(logs)
      liveloss.draw()

  ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
  model_save_name = 'DLCM' + st + '.pt'
  PATH = F"/content/gdrive/My Drive/{model_save_name}"

  # Save the model and all parameters
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, PATH)

  return model
