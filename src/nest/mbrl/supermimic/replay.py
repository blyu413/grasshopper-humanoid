"""V3 GPU FIFO with robot-sized tensors and exogenous-event boundaries."""
import torch

class GPUReplay:
    """Time-major replay; reconstruct history on demand instead of storing it 5x."""
    def __init__(self, steps, envs, context_dim, device, layout):
        self.capacity, self.envs, self.device = steps, envs, device
        self.size = self.ptr = 0
        schema = dict(s=(layout.size, torch.float32), s1=(layout.size, torch.float32),
                      a=(layout.actions, torch.float32), ap=(layout.actions, torch.float32), c=(context_dim, torch.float32),
                      g=(8, torch.float32), r=(1, torch.float32),
                      ep=(1, torch.int32), done=(1, torch.bool), term=(1, torch.bool),
                      landing=(2, torch.bool), lags=(layout.decimation, torch.float32), cut=(1, torch.bool))
        self.data = {k:torch.empty((steps, envs, n), dtype=dt, device=device) for k,(n,dt) in schema.items()}

    @property
    def bytes_used(self):
        return sum(x.numel()*x.element_size() for x in self.data.values())

    def add(self, data):
        for k, storage in self.data.items():
            storage[self.ptr].copy_(data[k].reshape_as(storage[self.ptr]))
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def row(self, age):
        return (self.ptr-self.size+age) % self.capacity

    def windows(self, count, window):
        t = torch.randint(self.size-window+1, (count,), device=self.device)
        e = torch.randint(self.envs, (count,), device=self.device)
        rows = self.row(t[:,None]+torch.arange(window, device=self.device))
        out = {k:self.data[k][rows, e[:,None]] for k in ('s','s1','a','c','done','landing','lags','cut')}
        done = (out['done'] | out['cut']).squeeze(-1).float()
        out['mask'] = torch.cumprod(torch.cat((torch.ones_like(done[:,:1]), 1-done[:,:-1]), -1), -1)
        return out

    def starts(self, count, history):
        age = torch.randint(self.size, (count,), device=self.device)
        e = torch.randint(self.envs, (count,), device=self.device)
        row = self.row(age)
        ep = self.data['ep'][row,e,0].long()
        back = torch.arange(history-1, -1, -1, device=self.device)[None,:]
        back = torch.minimum(torch.minimum(back, ep[:,None]), age[:,None])
        hrows = self.row(age[:,None]-back)
        start = {k:self.data[k][row,e] for k in ('s','ap','c','g')}
        hist = {k:self.data[k][hrows,e[:,None]] for k in ('s','ap','c','g','ep')}
        start['_age'],start['_env'] = age,e
        return start, hist
