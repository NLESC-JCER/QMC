import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from pyCHAMP.wavefunction.neural_wf_base import NEURAL_WF_BASE

from pyCHAMP.wavefunction.rbf import RBF_Slater_NELEC as RBF
from pyCHAMP.solver.deepqmc import DeepQMC
from pyCHAMP.sampler.metropolis import METROPOLIS_TORCH as METROPOLIS

from pyCHAMP.wavefunction.wave_modules import SlaterPooling, TwoBodyJastrowFactor, ElectronDistance

from pyCHAMP.solver.mesh import regular_mesh_3d

from pyCHAMP.solver.plot import plot_wf_3d
from pyCHAMP.solver.plot import plot_results_3d as plot_results

import matplotlib.pyplot as plt

import numpy as np




class RBF_H2(NEURAL_WF_BASE):

    def __init__(self,centers,sigma):
        super(RBF_H2,self).__init__(2,3)

        # basis function
        self.centers = centers
        self.ncenters = len(centers)

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, 
                       self.ncenters, 
                       centers=centers, 
                       sigma=sigma,
                       nelec=self.nelec)
        
        # define the mo layer
        self.mo = nn.Linear(self.ncenters, self.ncenters, bias=False)
        mo_coeff =  torch.sqrt(torch.tensor([2.]))  * torch.tensor([[1.,1.],[1.,-1.]])
        self.mo.weight = nn.Parameter(mo_coeff.transpose(0,1))

        # jastrow
        self.jastrow = TwoBodyJastrowFactor(1,1)
        # defin the SD pooling layer
        #self.pool = SlaterPooling([[[0]],[[0]]],1,1)

        
        

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''
        
        #edist  = ElectronDistance.apply(x)
        #J = self.jastrow(edist)

        x = self.rbf(x)
        x = self.mo(x)
        x = (x[:,0,0]*x[:,1,0]).view(-1,1)

        return x

        #return (x[:,0,0] * x[:,1,1] + x[:,0,1]*x[:,1,0]).view(-1,1)
        #return (x[:,0,0] + x[:,1,1]).view(-1,1)

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi

        TODO : vecorize that !! The solution below doesn't really wirk :(def plot_observable(obs_dict,e0=None,ax=None):)
        '''
        
        p = torch.zeros(pos.shape[0])
        for ielec in range(self.nelec):
            pelec = pos[:,(ielec*self.ndim):(ielec+1)*self.ndim]
            for iatom in range(len(self.centers)):
                patom = self.centers[iatom,:]

                r = torch.sqrt(   ((pelec-patom)**2).sum(1)  ) + 1E-6
                p += (-1./r)

        return p.view(-1,1)

    def electronic_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of Vee * psi
        '''

        pot = torch.zeros(pos.shape[0])
        
        for ielec1 in range(self.nelec-1):
            epos1 = pos[:,ielec1*self.ndim:(ielec1+1)*self.ndim]
            
            for ielec2 in range(ielec1+1,self.nelec):
                epos2 = pos[:,ielec2*self.ndim:(ielec2+1)*self.ndim]
                
                r = torch.sqrt( ((epos1-epos2)**2).sum(1) ) + 1E-6
                pot = (1./r) 

        return pot.view(-1,1)

    def nuclear_repulsion(self):
        c0 = self.centers[0,:]
        c1 = self.centers[1,:]
        rnn = torch.sqrt(   ((c0-c1)**2).sum()  )
        return (1./rnn).view(-1,1)

    def atomic_distance(self,pos=None):
        c0 = self.centers[0,:]
        c1 = self.centers[1,:]
        return torch.sqrt(   ((c0-c1)**2).sum()  )

    def get_sigma(self,pos=None):
        return self.rbf.sigma.data[0]
    

# wavefunction 
# bond distance : 0.74 A -> 1.38 a
# ground state energy : -31.688 eV -> -1.16 hartree
# bond dissociation energy 4.478 eV -> 0.16 hartree

X = 0.69 # <- opt ditance +0.69 and -0.69
S = 1.20 # <- roughly ideal zeta parameter


# define the RBF WF
centers = torch.tensor([[0.,0.,-X],[0.,0.,X]])
sigma = torch.tensor([S,S])
wf = RBF_H2(centers=centers,sigma=sigma)
#f.kinetic = 'fd'

#sampler
sampler = METROPOLIS(nwalkers=1000, nstep=1000,
                     step_size = 0.5, nelec = wf.nelec, move = 'one',
                     ndim = wf.ndim, domain = {'min':-5,'max':5})

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.005)
#opt = optim.SGD(wf.parameters(),lr=0.1)

# domain for the RBF Network
boundary = 5.
domain = {'xmin':-boundary,'xmax':boundary,
          'ymin':-boundary,'ymax':boundary,
          'zmin':-boundary,'zmax':boundary}
ncenter = [11,11,11]

# network
net = DeepQMC(wf=wf,sampler=sampler,optimizer=opt)
obs_dict = {'local_energy':[],
            'atomic_distance':[],
            'get_sigma':[]}

if 0:
    net.sampler.nstep = 5
    pos = net.sample()
    net.plot_density(pos.detach().numpy())

if 0:
    X = torch.zeros(100,6)
    X[:,2] = 0
    X[:,-1] = torch.linspace(-5,5,100)
    V = net.wf(X)

    plt.plot(X[:,-1].detach().numpy(),V.detach().numpy())
    plt.show()

if 0:
    X = np.linspace(0.5,2,11)
    energy, var = [], []
    for x in X:

        net.wf.rbf.sigma.data[:] = x
        pos = Variable(net.sample())
        pos.requires_grad = True
        e = net.wf.energy(pos)
        s = net.wf.variance(pos)

        energy.append(e)
        var.append(s)

    plt.plot(X,energy)
    plt.show()
    # plt.plot(X,var)
    # plt.show()


if 1:
    X = np.linspace(0.1,2,21)
    energy, var = [], []
    K,Vnn,Ven,Vee = [],[],[],[]
    for x in X:

        net.wf.rbf.centers.data[0,2] = -x
        net.wf.rbf.centers.data[1,2] = x
        pos = Variable(net.sample())
        pos.requires_grad = True

        K.append(net.wf.kinetic_energy(pos).mean())
        Vnn.append(net.wf.nuclear_repulsion())
        Ven.append(net.wf.nuclear_potential(pos).mean())
        Vee.append(net.wf.electronic_potential(pos).mean())

        e = net.wf.energy(pos)
        s = net.wf.variance(pos)

        energy.append(e)
        var.append(s)

    plt.plot(X,energy,linewidth=4,c='black',label='Energy')
    plt.plot(X,K,label='K')
    plt.plot(X,Vee,label='Vee')
    plt.plot(X,Ven,label='Ven')
    plt.plot(X,Vnn,label='Vnn')
    plt.legend()
    plt.grid()
    plt.show()
    plt.plot(X,var)
    plt.show()
    

if 0:
    pos = Variable(net.sample())
    pos.requires_grad = True
    e = net.wf.energy(pos)
    s = net.wf.variance(pos)

    print('Energy   :', e)
    print('Variance :', s)


if 0:

    x = 0.5
    net.wf.rbf.centers.data[0,2] = -x
    net.wf.rbf.centers.data[0,2] = x

    s = 1.20
    net.wf.rbf.sigma.data[:] = s 

    # do not optimize the weights of fc
    net.wf.fc.weight.requires_grad = False

    # optimize the position of the centers
    # do not optimize the std of the gaussian
    net.wf.rbf.centers.requires_grad = True
    net.wf.rbf.sigma.requires_grad = False

    # train
    pos,obs_dict = net.train(500,
             batchsize=500,
             pos = None,
             obs_dict = obs_dict,
             resample=1000,
             resample_every=25,
             ntherm=-1,
             loss = 'energy')

    # optimize the position of the centers
    # do not optimize the std of the gaussian
    net.wf.rbf.centers.requires_grad = False
    net.wf.rbf.sigma.requires_grad = True

    opt = optim.Adam(wf.parameters(),lr=0.0001)

    # train
    pos,obs_dict = net.train(250,
             batchsize=500,
             pos = None,
             obs_dict = obs_dict,
             resample=1000,
             resample_every=25,
             ntherm=-1,
             loss = 'energy')


    plot_results(net,obs_dict,domain,ncenter,isoval=0.02,hist=True)





