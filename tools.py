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