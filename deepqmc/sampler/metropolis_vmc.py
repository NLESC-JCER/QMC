from deepqmc.sampler.sampler_base import SamplerBase
from deepqmc.wavefunction.kinetic_pooling import btrace
from tqdm import tqdm
import torch
from torch.distributions import MultivariateNormal
from time import time


class Metropolis(SamplerBase):

    def __init__(self, nwalkers=100, nstep=1000, step_size=3,
                 nelec=1, ndim=1,
                 init={'min': -5, 'max': 5},
                 move={'type': 'one-elec-iter', 'proba': 'uniform'},
                 wf=None):
        """Metropolis Hasting generator

        Args:
            nwalkers (int, optional): Number of walkers. Defaults to 100.
            nstep (int, optional): Number of steps. Defaults to 1000.
            step_size (int, optional): length of the step. Defaults to 3.
            nelec (int, optional): total number of electrons. Defaults to 1.
            ndim (int, optional): total number of dimension. Defaults to 1.
            init (dict, optional): method to init the positions of the walkers.
                                   uniform : {'min': min_val, 'max': max_val}
                                   normal : {'mean' : [x,y,z],
                                             'sigma':3x3 matrix}
                                   See Molecule.domain
                                   Defaults to {'min': -5, 'max': 5}.
            move (dict, optional): method to move the electrons.
                                   'type' :
                                        'one-elec': move a single electron
                                                    per iteration
                                        'all-elec': move all electrons at
                                                    the same time
                                        'all-elec-iter': move all electrons
                                                        by iterating
                                                        through single elec
                                                        moves
                                    'proba' : 'uniform', 'normal'
                                    Defaults to {'type': 'one-elec',
                                                 'proba': 'uniform'}.
        """

        SamplerBase.__init__(self, nwalkers, nstep,
                             step_size, nelec, ndim, init, move)
        self.wf = wf

        if 'type' not in self.movedict.keys():
            print('Metroplis : Set 1 electron move by default')
            self.movedict['type'] = 'one-elec'

        if 'proba' not in self.movedict.keys():
            print('Metroplis : Set uniform trial move probability')
            self.movedict['proba'] = 'uniform'

        if self.movedict['proba'] == 'normal':
            _sigma = self.step_size / \
                (2 * torch.sqrt(2 * torch.log(torch.tensor(2.))))
            self.multiVariate = MultivariateNormal(
                torch.zeros(self.ndim), _sigma * torch.eye(self.ndim))

        self._move_per_iter = 1
        if self.movedict['type'] not in ['all-elec-iter']:
            raise ValueError(
                " Only 1 all-elec-iter move allowed with kalos")

        if self.movedict['type'] == 'all-elec-iter':
            self.fixed_id_elec_list = range(self.nelec)
            self._move_per_iter = self.nelec
        else:
            self.fixed_id_elec_list = [None]

    def generate(self, pdf, ntherm=10, ndecor=100, pos=None,
                 with_tqdm=True):
        """Generate a series of point using MC sampling

        Args:
            pdf (callable): probability distribution function to be sampled
            ntherm (int, optional): number of step before thermalization.
                                    Defaults to 10.
            ndecor (int, optional): number of steps for decorrelation.
                                    Defaults to 50.
            pos (torch.tensor, optional): position to start with.
                                          Defaults to None.
            with_tqdm (bool, optional): tqdm progress bar. Defaults to True.

        Returns:
            torch.tensor: positions of the walkers
        """

        _type_ = torch.get_default_dtype()
        if _type_ == torch.float32:
            eps = 1E-7
        elif _type_ == torch.float64:
            eps = 1E-16

        if self.cuda:
            self.walkers.cuda = True
            self.device = torch.device('cuda')

        if ntherm >= self.nstep:
            raise ValueError('Thermalisation longer than trajectory')

        with torch.no_grad():

            if ntherm < 0:
                ntherm = self.nstep + ntherm

            # init walkers
            self.walkers.initialize(pos=pos)

            # first calculations of ao, mo and slater matrices
            self.init_matrices(self.walkers.pos)

            pos, rate, idecor = [], 0, 0

            if with_tqdm:
                rng = tqdm(range(self.nstep))
            else:
                rng = range(self.nstep)

            nup = self.wf.mol.nup
            for istep in rng:

                for id_elec in self.fixed_id_elec_list:

                    t0 = time()

                    # new positions
                    Xn = self.move(id_elec)

                    # update the matrices
                    ao_new, dao_new, d2ao_new = self.update_ao_matrices(Xn, id_elec)

                    # form the slater matrices of the new AOs
                    sup_new, sdown_new = self.get_slater_matrix(ao_new)

                    # get transition proba
                    R, new_wf_val = self.get_transition_probability(sup_new, sdown_new, Xn, id_elec)

                    # accept the moves
                    index = self._accept(R**2)

                    # acceptance rate
                    rate += index.byte().sum().float().to('cpu') / \
                        (self.nwalkers * self._move_per_iter)

                    # update position/function value
                    self.walkers.pos[index, :] = Xn[index, :]
                    self.ao[index] = ao_new[index]
                    self.dao[index] = dao_new[index]
                    self.d2ao[index] = d2ao_new[index]
                    self.wf_values[index] = new_wf_val[index]

                    # update the inverse slater matrix
                    self.update_inverse_slater_matrices(index, R, id_elec)

                    # update the Bkin operator
                    self.Bup[:,index,:,:], self.Bdown[:,index,:,:] = self.get_Bkin_matrices(self.ao[index],
                                                                                self.dao[index], 
                                                                                self.d2ao[index], 
                                                                                self.walkers.pos[index])
                    
                    # update the local energies
                    self.local_energies[index] = self.get_local_energies(index=index)
                    
                if (istep >= ntherm):
                    if (idecor % ndecor == 0):
                        pos.append(self.walkers.pos.to('cpu').clone())
                    idecor += 1

            if with_tqdm:
                print(
                    "Acceptance rate %1.3f %%" %
                    (rate / self.nstep * 100))

        return torch.cat(pos)

    def init_matrices(self, pos):
        """Compute the AO, Slater and inverse Slater matrices
        
        Arguments:
            pos {torch.tensor} -- positions of the walkers
        """

        # atomic orbitals / first / second derivatives of the AO
        self.ao = self.wf.ao(self.walkers.pos)
        self.dao = self.wf.ao(self.walkers.pos, derivative=1, jacobian=False)
        self.d2ao = self.wf.ao(self.walkers.pos, derivative=2)

        # Bkin operators
        self.Bup, self.Bdown = self.get_Bkin_matrices(self.ao, self.dao, self.d2ao, self.walkers.pos)
        
        # slater matrices
        self.sup, self.sdown = self.get_slater_matrix(self.ao)

        # determinant of the slater matrices
        self.det_up, self.det_down = torch.det(self.sup), torch.det(self.sdown)
        
        # wave function values
        self.wf_values = self.wf.fc(self.det_up * self.det_down)
        if self.wf.use_jastrow:
            self.wf_values *= self.wf.jastrow(self.walkers.pos)

        # inverse of the slater matrices
        self.isup, self.isdown = torch.inverse(self.sup), torch.inverse(self.sdown)

        # local energies
        self.local_energies = self.get_local_energies()

    def update_ao_matrices(self, pos_new, id_elec):
        """Update the AO, and its derivative matrices when 1 electrons has moved
        
        Arguments:
            pos_new {[type]} -- [description]
            id_elec {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        ao_new = self.wf.ao.update(self.ao, pos_new, id_elec)
        dao_new = self.wf.ao.update(self.dao, pos_new, id_elec, derivative=1, jacobian=False)
        d2ao_new = self.wf.ao.update(self.d2ao, pos_new, id_elec, derivative=2)

        return ao_new, dao_new, d2ao_new

    def get_Bkin_matrices(self, ao, dao, d2ao, pos):
        """Computes the B kin matrices from the ao matrices
        
        Arguments:
            ao {[type]} -- AO matrices
            dao {[type]} -- first derivative of AOs
            d2ao {[type]} -- second derivative of the AOs
            pos {[type]} -- pos to compute the jastrow
        """

        mo = self.wf._ao2mo(ao)
        dmo = self.wf._ao2mo(dao.transpose(2,3)).transpose(2,3)
        d2mo = self.wf._ao2mo(d2ao)

        if self.wf.use_jastrow:

            jast = self.wf.jastrow(pos)
            djast = self.wf.jastrow(pos, derivative=1, jacobian=False)
            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)
            d2jast = self.wf.jastrow(pos, derivative=2) / jast

            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)
            d2jast_mo = d2jast.unsqueeze(-1) * mo

        return self.wf.kinpool.get_Bkin_matrices(d2mo, djast_dmo, d2jast_mo)        

    def get_slater_matrix(self, ao):
        """Returns the slater matrices from the AO matrix

        Arguments:
            ao {torch.tensor} -- AO matrix

        Returns:
            torch.tensor, torch.tensor -- the slater matrices
        """
        mo = self.wf._ao2mo(ao)
        sup, sdown = self.wf.pool(mo, return_matrix=True)
        
        return sup, sdown

    
    def update_slater_determiant(self, sup, sdown, id_elec):
        """Update the determinant of the slater matrices
        
        Arguments:
            sup {[type]} -- spin up slater matrices
            sdown {[type]} -- spin down slater matrices
            id_elec {[type]} -- index of the moving elec
        """ 
        
        if id_elec < self.wf.mol.nup:
            new_det_up = self._update_det(sup, self.isup, self.det_up, id_elec)
            new_det_down = self.det_down.clone()
        else:
            new_det_up = self.det_up.clone()
            new_det_down = self._update_det(sdown, self.isdown, 
                                            self.det_down, id_elec-self.wf.mol.nup)
        return new_det_up, new_det_down                        

    @staticmethod
    def _update_det(new_slater_matrix, old_inv_slat_mat, old_det, id_elec):
        """update the values of the salter determinant
        
        Arguments:
            new_slater_matrix {torch.tensor} -- new slater matrices
            old_inv_slat_mat {torch.tensor} -- old inverse slater matrices
            old_det {[type]} -- old determinant of the slater matrices
            id_elec {[type]} -- index of the eletron

        Returns:
            torch.tensor -- determinant of the new slater matrices
        """

        nbatch, ndet, dim, _ = new_slater_matrix.shape
        ratio = torch.bmm( 
                new_slater_matrix[:,:,id_elec,:].unsqueeze(2).reshape(-1,dim,dim),
                old_inv_slat_mat[:,:,:,id_elec].unsqueeze(-1).reshape(-1,dim,dim)).view(nbatch,ndet)

        return ratio * old_det
    
    def get_transition_probability(self, new_sup, new_sdown, pos, id_elec):
        """Get the ratio between new/old wave function values
        
        Arguments:
            new_sup {[type]} -- new spin up slater matrices
            new_sdown {[type]} -- new spin down slater matrices
            pos {[type]} -- new positions of the electrons
            id_elec {[type]} -- index of the electroncs
        
        Returns:
            [type] -- ratio new/old, new_wf_values
        """

        new_det_up, new_det_down = self.update_slater_determiant(new_sup, new_sdown, id_elec)
        new_wf_values = self.wf.fc(new_det_up * new_det_down)
        if self.wf.use_jastrow:
            new_wf_values *= self.wf.jastrow(pos)
        return new_wf_values/self.wf_values,  new_wf_values 

    def update_inverse_slater_matrices(self, index, R, id_elec):
        """Update the inverse slater matrix
        
        Arguments:
            index {[type]} -- index of the accepted move
            R {[type]} --  transition probability
            id_elec {[type]} -- index of the electron
        """

        nup = self.wf.mol.nup
        if id_elec < nup:
            self.isup[index, :, :, id_elec] /= R[index].unsqueeze(-1)
        else:
            self.isdown[index, :, :, id_elec - nup] /= R[index].unsqueeze(-1)

    def get_local_energies(self,index=None):
        """Get the local energy values of the conf in index
        
        Arguments:
            index {[type]} -- index of the confs
        
        Returns:
            [type] -- local energy values
        """

        if index is None:
            index = range(self.isup.shape[0])

        kin = -0.5* (btrace(self.isup[index].transpose(0,1) @ self.Bup[:,index,:,:]) + btrace(self.isdown[index].transpose(0,1) @ self.Bdown[:,index,:,:])).transpose(0,1)
        det_prod = torch.det(self.sup[index]) * torch.det(self.sdown[index])
        Eloc = self.wf.fc(kin) / self.wf.fc(det_prod)

        Eloc += self.wf.nuclear_potential(self.walkers.pos[index])
        Eloc += self.wf.electronic_potential(self.walkers.pos[index])
        Eloc += self.wf.nuclear_repulsion()

        return Eloc

    def move(self, id_elec):
        """Move electron one at a time in a vectorized way.

        Args:
            idelec (int): index of the electron(s) to move

        Returns:
            torch.tensor: new positions of the walkers
        """
        if self.nelec == 1 or self.movedict['type'] == 'all-elec':
            return self.walkers.pos + self._move(self.nelec)

        else:

            # clone and reshape data : Nwlaker, Nelec, Ndim
            new_pos = self.walkers.pos.clone()
            new_pos = new_pos.view(self.nwalkers,
                                   self.nelec, self.ndim)

            # get indexes
            if id_elec is None:
                index = torch.LongTensor(self.nwalkers).random_(
                    0, self.nelec)
            else:
                index = torch.LongTensor(self.nwalkers).fill_(id_elec)

            # change selected data
            new_pos[range(self.nwalkers), index,
                    :] += self._move(1)

            return new_pos.view(self.nwalkers, self.nelec * self.ndim)

    def _move(self, num_elec):
        """Return a random array of length size between
        [-step_size,step_size]

        Args:
            step_size (float): boundary of the array
            size (int): number of points in the array

        Returns:
            torch.tensor: random array
        """
        if self.movedict['proba'] == 'uniform':
            d = torch.rand(
                (self.nwalkers, self.ndim), device=self.device)
            return self.step_size * (2. * d - 1.)

        elif self.movedict['proba'] == 'normal':
            displacement = self.multiVariate.sample(
                (self.nwalkers, num_elec)).to(self.device)
            return displacement.view(
                self.nwalkers, num_elec * self.ndim)

    def _accept(self, P):
        """accept the move or not

        Args:
            P (torch.tensor): probability of each move

        Returns:
            t0rch.tensor: the indx of the accepted moves
        """

        tau = torch.rand_like(P)
        index = (P >= tau).reshape(-1)
        return index.type(torch.bool)
