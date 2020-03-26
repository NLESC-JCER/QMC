import torch
import numpy as np
from deepqmc.solver.solver_base import SolverBase

class VMC(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None, scheduler=None):
        """Serial solver

        Keyword Arguments:
            wf {WaveFunction} -- WaveFuntion object (default: {None})
            sampler {SamplerBase} -- Samppler (default: {None})
            optimizer {torch.optim} -- Optimizer (default: {None})
            scheduler (torch.schedul) -- Scheduler (default: {None})
        """

        SolverBase.__init__(self, wf, sampler, optimizer)

    def 
