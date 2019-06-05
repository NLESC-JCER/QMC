import numpy as np 
import torch
from torch.autograd import Variable, grad
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
from pyCHAMP.solver.solver_base import SOLVER_BASE

import matplotlib.pyplot as plt

from tqdm import tqdm
import time



class QMC_DataSet(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,index):
        return self.data[index,:]

class QMCLoss(nn.Module):

    def __init__(self,wf,method='energy'):

        super(QMCLoss,self).__init__()
        self.wf = wf
        self.method = method

    def forward(self,vals,pos):

        if self.method == 'variance':
            loss = self.wf.variance(pos)

        elif self.method == 'energy':
            loss = self.wf.energy(pos)

        elif self.method == 'supervised':
            sol = torch.exp(-0.5*pos**2).view(-1,1,1)
            loss = torch.nn.MSELoss()
            return loss(vals,sol)

        return loss
            
class NN(SOLVER_BASE):

    def __init__(self, wf=None, sampler=None, optimizer=None):

        SOLVER_BASE.__init__(self,wf,sampler,None)
        #self.opt = optim.SGD(self.wf.parameters(),lr=0.05, momentum=0.9, weight_decay=0.001)
        self.opt = optim.Adam(self.wf.parameters(),lr=0.01)
        self.batchsize = 100

    def sample(self,ntherm=10,with_tqdm=True,pos=None):

        t0 = time.time()
        pos = self.sampler.generate(self.wf.pdf,ntherm=ntherm,with_tqdm=with_tqdm,pos=pos)
        pos = torch.tensor(pos)
        pos = pos.view(-1,self.sampler.ndim*self.sampler.nelec)
        pos.requires_grad = True
        return pos.float()

    def observalbe(self,func,pos):
        obs = []
        for p in tqdm(pos):
            obs.append( func(p).data.numpy().tolist() )
        return obs


    def get_wf(self,x):
        vals = self.wf(x)
        return vals.detach().numpy().flatten()

    def plot_wf(self,grad=False,hist=False):

        X = Variable(torch.linspace(-5,5,100).view(100,1,1))
        X.requires_grad = True

        vals = self.wf(X)
        vn = vals.detach().numpy().flatten()
        vn /= np.linalg.norm(vn)
        xn = X.detach().numpy().flatten()
        plt.plot(xn,vn**2)

        if grad:
            kin = self.wf.kinetic_autograd(X)
            g = np.gradient(vn,xn)
            h = np.gradient(g,xn)
            plt.plot(xn,kin.detach().numpy())
            plt.plot(xn,h)

        if hist:
            pos = self.sample(ntherm=-1)
            plt.hist(pos.detach().numpy(),density=False)
        
        sol = np.exp(-0.5*xn**2)
        sol /= np.linalg.norm(sol)
        plt.plot(xn,sol**2)
        
        plt.show()

    def train(self,nepoch,pos=None,ntherm=10,loss='variance'):

        if pos is None:
            pos = self.sample(ntherm=ntherm)

        dataset = QMC_DataSet(pos)
        dataloader = DataLoader(dataset,batch_size=self.batchsize)
        qmc_loss = QMCLoss(self.wf,method=loss)
        
        XPLOT = Variable(torch.linspace(-5,5,100).view(100,1,1))
        xp = XPLOT.detach().numpy().flatten()
        vp = self.get_wf(XPLOT)
        sol = np.exp(-0.5*xp**2)

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(xp,vp,color='red')
        line2, = ax.plot(xp,sol,color='blue')


        cumulative_loss = []
        for n in range(nepoch):

            cumulative_loss.append(0) 
            for data in dataloader:
                
                lpos = Variable(data).float()
                lpos.requires_grad = True
                vals = self.wf(lpos)

                loss = qmc_loss(vals,lpos)
                cumulative_loss[n] += loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.wf.fc.weight.data /= self.wf.fc.weight.data.norm() 
            vp = self.get_wf(XPLOT)
            line1.set_ydata(vp)
            fig.canvas.draw()            

            print('epoch %d loss %f' %(n,cumulative_loss[n]))
            print('variance : %f' %self.wf.variance(pos))
            print('energy : %f' %self.wf.energy(pos))

            self.sampler.nstep=100
            pos = self.sample(pos=pos.detach().numpy(),ntherm=ntherm,with_tqdm=False)
            dataloader.dataset.data = pos

        plt.plot(cumulative_loss)
        plt.show()


class NN4PYSCF(SOLVER_BASE):

    def __init__(self, wf=None, sampler=None, optimizer=None):

        SOLVER_BASE.__init__(self,wf,sampler,None)
        self.opt = optim.SGD(self.wf.parameters(),lr=0.005, momentum=0.9, weight_decay=0.001)
        self.batchsize = 100


    def sample(self,ntherm=10):

        t0 = time.time()
        pos = self.sampler.generate(self.wf.pdf,ntherm=ntherm)
        pos = torch.tensor(pos)
        pos = pos.view(-1,self.sampler.ndim*self.sampler.nelec)
        pos.requires_grad = True
        return pos.float()

    def observalbe(self,func,pos):
        obs = []
        for p in tqdm(pos):
            obs.append( func(p).data.numpy().tolist() )
        return obs

    def train(self,nepoch,pos=None,ntherm=0):

        if pos is None:
            pos = self.sample(ntherm=ntherm)

        dataset = QMC_DataSet(pos)
        dataloader = DataLoader(dataset,batch_size=self.batchsize)
        qmc_loss = QMCLoss(self.wf,method='variance')
        
        cumulative_loss = []
        for n in range(nepoch):
            print('\n === epoch %d' %n)

            cumulative_loss.append(0) 
            for data in dataloader:
                
                print("\n data ", data.shape)

                data = Variable(data).float()
                data.requires_grad = True
                t0 = time.time()
                out = self.wf(data)
                print("\t WF done in %f" %(time.time()-t0))

                t0 = time.time()
                loss = qmc_loss(data)
                cumulative_loss[n] += loss
                print("\t Loss (%f) done in %f" %(loss,time.time()-t0))
                self.wf = self.wf.train()

                self.opt.zero_grad()

                t0 = time.time()
                loss.backward()
                print("\t Backward done in %f" %(time.time()-t0))

                t0 = time.time()
                self.opt.step()
                print("\t opt done in %f" %(time.time()-t0))

                q,r = torch.qr(self.wf.layer_mo.weight.transpose(0,1))
                self.wf.layer_mo.weight.data = q.transpose(0,1)
                print(self.wf.layer_mo.weight)
                print(self.wf.layer_ci.weight)

            print('=== epoch %d loss %f \n' %(n,cumulative_loss[n]))

            if 1:
                pos = self.sample(ntherm=ntherm)
                dataloader.dataset.data = pos

        plt.plot(cumulative_loss)
        plt.show()



