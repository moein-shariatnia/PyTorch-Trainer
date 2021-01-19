class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
        
class AvgMeterVector:
    def __init__(self, num_enteries=3, names=["Metric 1", "Metric 2", "Metric 3"]):
        self.num_enteries = num_enteries
        self.meters = [AvgMeter(names[i]) for i in range(num_enteries)]
    
    def update(self, values, counts):
        for i in range(self.num_enteries):
            meter = self.meters[i]
            meter.update(values[i], counts.get(i, 0))

    def __repr__(self):
        text = [meter for meter in self.meters]
        return str(text)