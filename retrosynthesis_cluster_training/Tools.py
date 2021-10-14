from modules import *
from constants import *
from Transformer import *


from torch.optim.lr_scheduler import _LRScheduler

def tokens_to_smiles(tokens):
    return "".join([ix_to_char[e] for e in tokens]).split(".")

def smiles_matching(tgt_smiles, out_smiles):
    out_mols = [Chem.MolFromSmiles(e) for e in out_smiles]
    tgt_mols = [Chem.MolFromSmiles(e) for e in tgt_smiles]
    if None in out_mols:
        return False
    else:
        out_smiles_canonical = [Chem.MolToSmiles(e) for e in out_mols]
        tgt_smiles_canonical = [Chem.MolToSmiles(e) for e in tgt_mols]

        if set(out_smiles_canonical) == set(tgt_smiles_canonical): return True
        else: return False

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            

class NoamLR(_LRScheduler):
    """The LR scheduler proposed by Noam

    Ref:
        "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf

    FIXME(kamo): PyTorch doesn't provide _LRScheduler as public class,
     thus the behaviour isn't guaranteed at forward PyTorch version.

    NOTE(kamo): The "model_size" in original implementation is derived from
     the model, but in this implementation, this parameter is a constant value.
     You need to change it if the model is changed.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int = 256,
        warmup_steps: int = 30,
        last_epoch: int = -1,
    ):
        self.model_size = model_size
        self.warmup_steps = warmup_steps

        lr = list(optimizer.param_groups)[0]["lr"]
        new_lr = self.lr_for_WarmupLR(lr)


        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def lr_for_WarmupLR(self, lr: float) -> float:
        return lr / self.model_size ** 0.5 / self.warmup_steps ** 0.5


    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model_size={self.model_size}, "
            f"warmup_steps={self.warmup_steps})"
        )

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.model_size ** -0.5
            * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
            for lr in self.base_lrs
        ]