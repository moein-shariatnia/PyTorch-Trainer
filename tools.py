class AvgMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
        
class AvgMeterVector:
    def __init__(self, num_enteries=3):
        self.num_enteries = num_enteries
        self.meters = [AvgMeter() for i in range(num_enteries)]
    
    def update(self, values, counts):
        for i in range(self.num_enteries):
            meter = self.meters[i]
            meter.update(values[i], counts.get(i, 0))
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    
def one_epoch(model, dl, loss_func, opt=None, lr_schedule=None):
    running_loss = 0.
    running_acc = np.zeros((6,))
    
    for xb, yb in tqdm(dl):
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_func(preds, yb)
        
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
            if lr_schedule is not None:
                lr_schedule.step()

        running_acc += accuracy(preds, yb).cpu().numpy()
        running_loss += loss.item()
        
    return running_loss / len(dl), running_acc / len(dl)

def lr_finder(model, train_dl, start_lr=1e-6, end_lr=10,
              smoothing=0., epochs=1):
    opt = Adam(model.parameters(), lr=start_lr)
    ratio_log = math.log(end_lr / start_lr)
    n_iter = epochs * len(train_dl)
    b = ratio_log / n_iter
    schedule = lambda x: math.exp(x * b)
    lr_sch = lr_scheduler.LambdaLR(opt, schedule)
    
    losses, lrs = [], []
    best_loss = float('inf')
    
    model.train()
    i = 0
    done = False
    for e in range(epochs):
        print(f"Epoch {e + 1} / {epochs}")
        for xb, yb in tqdm(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if loss < best_loss:
                best_loss = loss
            
            if loss > 3 * best_loss:
                done = True
                break

            lr_sch.step()
            new_lr = get_lr(opt)
            lrs.append(new_lr)

            i += 1

            if i == 1:
                losses.append(loss.item())
            else:
                loss = losses[-1] * smoothing + (1 - smoothing) * loss.item()
                losses.append(loss)
        if done: break
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlim(1e-6, 1.)
    plt.ylim(min(losses), max(losses))
    plt.xlabel("Learning Rate [log]")
    plt.ylabel("Loss")
    plt.show()

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.0, num_classes=206, weight=None, reduction='mean',):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.num_classes = num_classes


    def forward(self, preds, yb):
        with torch.no_grad():
            yb = yb * (1 - self.smoothing) + torch.ones_like(yb) * self.smoothing / self.num_classes

        if self.weight is not None:
            preds = preds * self.weight

        loss = F.binary_cross_entropy_with_logits(preds, yb)

        return loss
    
    
def train_val(model, params):
   
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    one_cycle = params["one_cycle"]
    
    loss_history = {
        "train": [],
        "val": [],
    }
   
    metric_history = {
        "train": [],
        "val": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    best_loss=float('inf')
    
    for epoch in range(num_epochs):
        
        current_lr = get_lr(opt)
        print(f'Epoch {epoch + 1}/{num_epochs}, current lr = {current_lr}')
      
        model.train()
        train_loss, train_metric = one_epoch(model, train_dl, loss_func, opt, lr_scheduler if one_cycle else None)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
  
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = one_epoch(model, val_dl, loss_func, opt=None)
        
       
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
    
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        if not one_cycle:
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts) 
        
        print(f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}\n"
              f"Train Acc: Any: {train_metric[0]:.3f}, Epidural: {train_metric[1]:.3f}, Intraparenchymal: {train_metric[2]:.3f}, Intraventricular: {train_metric[3]:.3f}, Subarachnoid: {train_metric[4]:.3f}, Subdural: {train_metric[5]:.3f}\n" 
              f"Val Acc: Any: {val_metric[0]:.3f}, Epidural: {val_metric[1]:.3f}, Intraparenchymal: {val_metric[2]:.3f}, Intraventricular: {val_metric[3]:.3f}, Subarachnoid: {val_metric[4]:.3f}, Subdural: {val_metric[5]:.3f}\n")
        
        print("-"*10) 

    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

params_train = {
 "num_epochs": 5,
 "optimizer": opt,
 "loss_func": criterion,
 "train_dl": train_dl,
 "val_dl": val_dl,
 "lr_scheduler": lr_sch,
 "path2weights": "/kaggle/working/weights_freezed.pt",
 "one_cycle": True
}