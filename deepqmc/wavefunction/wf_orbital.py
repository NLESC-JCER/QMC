import torch
from torch import nn
import numpy as np
from time import time

from deepqmc.wavefunction.atomic_orbitals import AtomicOrbitals
from deepqmc.wavefunction.slater_pooling import SlaterPooling
from deepqmc.wavefunction.kinetic_pooling import KineticPooling
from deepqmc.wavefunction.orbital_configurations import OrbitalConfigurations
from deepqmc.wavefunction.wf_base import WaveFunction
from deepqmc.wavefunction.jastrow import TwoBodyJastrowFactor


class Orbital(WaveFunction):

    def __init__(self, mol, configs='ground_state',
                 kinetic='jacobi', use_jastrow=True, cuda=False):
        """Network to compute a wave function

        Arguments:
            mol {Molecule} -- Instance of a molecule object

        Keyword Arguments:
            configs {str} -- configuration in the active space (default: {'ground_state'})
                            'ground state',
                            'cas(nelec,norb)'
                            'single(nelec,norb)'
                            'single_double(nelec,norb)'

            kinetic {str} -- method to compute the kinetic energy (jacobi, auto, fd) (default: {'jacobi'})
            use_jastrow {bool} -- use a jastrow factor (default: {True})
            cuda {bool} -- use cuda (default: {False})

        Raises:
            ValueError: if cuda requested and not available
        """

        super(Orbital, self).__init__(mol.nelec, 3, kinetic, cuda)

        # check for cuda
        if not torch.cuda.is_available and self.wf.cuda:
            raise ValueError('Cuda not available, use cuda=False')

        # number of atoms
        self.mol = mol
        self.atoms = mol.atoms
        self.natom = mol.natom

        # define the atomic orbital layer
        self.ao = AtomicOrbitals(mol, cuda)

        # define the mo layer
        self.mo_scf = nn.Linear(
            mol.basis.nao, mol.basis.nmo, bias=False)
        self.mo_scf.weight = self.get_mo_coeffs()
        self.mo_scf.weight.requires_grad = False
        if self.cuda:
            self.mo_scf.to(self.device)

        # define the mo mixing layer
        self.mo = nn.Linear(mol.basis.nmo, mol.basis.nmo, bias=False)
        self.mo.weight = nn.Parameter(torch.eye(mol.basis.nmo))
        if self.cuda:
            self.mo.to(self.device)

        # jastrow
        self.use_jastrow = use_jastrow
        self.jastrow = TwoBodyJastrowFactor(mol.nup, mol.ndown,
                                            w=1., cuda=cuda)

        # define the SD we want
        self.orb_confs = OrbitalConfigurations(mol)
        self.configs_method = configs
        self.configs = self.orb_confs.get_configs(configs)
        self.nci = len(self.configs[0])

        #  define the SD pooling layer
        self.pool = SlaterPooling(
            self.configs, mol, cuda)

        # pooling operation to directly compute
        # the kinetic energies via Jacobi formula
        self.kinpool = KineticPooling(
            self.configs, mol, cuda)

        # define the linear layer
        self.fc = nn.Linear(self.nci, 1, bias=False)
        self.fc.weight.data.fill_(1.)
        if self.nci > 1:
            self.fc.weight.data.fill_(0.)
            self.fc.weight.data[0][0] = 1.
        if self.cuda:
            self.fc = self.fc.to(self.device)
        self.fc.clip = False

        if kinetic == 'jacobi':
            self.local_energy = self.local_energy_jacobi

        if self.cuda:
            self.device = torch.device('cuda')
            self.to(self.device)

    def get_mo_coeffs(self):
        """get the molecular orbital coefficient

        Returns:
            nn.Parameters -- MO matrix as a parameter
        """
        mo_coeff = torch.tensor(
            self.mol.calculator.get_mo_coeffs()).type(
                torch.get_default_dtype())
        # return nn.Parameter(mo_coeff)
        return nn.Parameter(mo_coeff.transpose(0, 1).contiguous())

    def update_mo_coeffs(self):
        """Update the MO matrix for example in a geo opt run."""
        self.mol.atom_coords = self.ao.atom_coords.detach().numpy().tolist()
        self.mo.weight = self.get_mo_coeffs()

    def forward(self, x, ao=None):
        """Compute the value of the wave function for a multiple conformation of the electrons

        Arguments:
            x {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Keyword Arguments:
            ao {torch.tensor} -- AO matrix [nbatch, nelec,nao]
                                if present used as input of the MO.
                                usefull when updating the waeve function after a 1 elec move
                                 (default: {None})

        Returns:
            torch.tensor -- value of the wave function for the configurations
        """

        if self.use_jastrow:
            J = self.jastrow(x)

        # atomic orbital
        if ao is None:
            x = self.ao(x)

        else:
            x = ao

        # molecular orbitals
        x = self.mo_scf(x)

        # mix the mos
        x = self.mo(x)

        # pool the mos
        x = self.pool(x)

        if self.use_jastrow:
            return J * self.fc(x)

        else:
            return self.fc(x)

    def _get_mo_vals(self, x, derivative=0):
        """Get the values of MOs

        Arguments:
            x {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Keyword Arguments:
            derivative {int} -- order of the derivative (default: {0})

        Returns:
            torch.tensor -- MO matrix [nbatch, nelec, nmo]
        """
        return self.mo(self.mo_scf(self.ao(x, derivative=derivative)))

    def local_energy_jacobi(self, pos):
        """Computes the local energy using the jacobi formula (trace trick)
        for the kinetic energy

        Arguments:
            pos {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Returns:
            torch.tensor -- value of the local energy [nbatch]
        """

        ke = self.kinetic_energy_jacobi(pos)

        return ke \
            + self.nuclear_potential(pos) \
            + self.electronic_potential(pos) \
            + self.nuclear_repulsion()

    def kinetic_energy_jacobi(self, x, **kwargs):
        """Compute the value of the kinetic enery using
        the Jacobi formula for derivative of determinant.

        Arguments:
            x {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Returns:
            torch.tensor -- value of the kinetic energy [nbatch]
        """

        mo = self._get_mo_vals(x)
        d2mo = self._get_mo_vals(x, derivative=2)
        djast_dmo, d2jast_mo = None, None

        if self.use_jastrow:

            jast = self.jastrow(x)
            djast = self.jastrow(x, derivative=1, jacobian=False)
            djast = djast.transpose(1, 2) / jast.unsqueeze(-1)

            dao = self.ao(
                x,
                derivative=1,
                jacobian=False).transpose(
                2,
                3)
            dmo = self.mo(self.mo_scf(dao)).transpose(2, 3)
            djast_dmo = (djast.unsqueeze(2) * dmo).sum(-1)

            d2jast = self.jastrow(x, derivative=2) / jast
            d2jast_mo = d2jast.unsqueeze(-1) * mo

        kin, psi = self.kinpool(mo, d2mo, djast_dmo, d2jast_mo)

        return self.fc(kin) / self.fc(psi)

    def nuclear_potential(self, pos):
        """Computes the electron-nuclear term

        Arguments:
            pos {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Returns:
            torch.tensor -- value of the electron-nuclear term [nbatch]

        TODO : vectorize that !!
        """

        p = torch.zeros(pos.shape[0], device=self.device)
        for ielec in range(self.nelec):
            pelec = pos[:, (ielec * self.ndim):(ielec + 1) * self.ndim]
            for iatom in range(self.natom):
                patom = self.ao.atom_coords[iatom, :]
                Z = self.ao.atomic_number[iatom]
                r = torch.sqrt(((pelec - patom)**2).sum(1))  # + 1E-12
                p += -Z / r
        return p.view(-1, 1)

    def electronic_potential(self, pos):
        """Computes the electron-electron term

        Arguments:
            pos {torch.tensor} -- positions of the electrons [nbatch, nelec*ndim]

        Returns:
            torch.tensor -- value of the el-el repulsion [nbatch]

        TODO : vectorize that !!
        """

        pot = torch.zeros(pos.shape[0], device=self.device)

        for ielec1 in range(self.nelec - 1):
            epos1 = pos[:, ielec1 *
                        self.ndim:(ielec1 + 1) * self.ndim]
            for ielec2 in range(ielec1 + 1, self.nelec):
                epos2 = pos[:, ielec2 *
                            self.ndim:(ielec2 + 1) * self.ndim]
                r = torch.sqrt(((epos1 - epos2)**2).sum(1))  # + 1E-12
                pot += (1. / r)
        return pot.view(-1, 1)

    def nuclear_repulsion(self):
        """Computes the nuclear-nuclear repulsion term

        Returns:
            torch.tensor -- value of the nuclear repulsion

        TODO : vectorize that !!
        """

        vnn = 0.
        for at1 in range(self.natom - 1):
            c0 = self.ao.atom_coords[at1, :]
            Z0 = self.ao.atomic_number[at1]
            for at2 in range(at1 + 1, self.natom):
                c1 = self.ao.atom_coords[at2, :]
                Z1 = self.ao.atomic_number[at2]
                rnn = torch.sqrt(((c0 - c1)**2).sum())
                vnn += Z0 * Z1 / rnn
        return vnn

    def geometry(self, pos):
        """Return the current geometry of the molecule

        Arguments:
            pos {[type]} -- dummy argument

        Returns:
            list -- atomic types and positions
        """
        d = []
        for iat in range(self.natom):
            at = self.atoms[iat]
            xyz = self.ao.atom_coords[iat,
                                      :].detach().numpy().tolist()
            d.append((at, xyz))
        return d


if __name__ == "__main__":

    from deepqmc.wavefunction.molecule import Molecule

    mol = Molecule(atom='Li 0 0 0; H 0 0 3.015', basis='sz')

    # define the wave function
    wf_jacobi = Orbital(mol, kinetic='jacobi',
                        configs='cas(2,2)',
                        use_jastrow=True,
                        cuda=False)

    wf_auto = Orbital(mol, kinetic='auto',
                      configs='cas(2,2)',
                      use_jastrow=True,
                      cuda=False)

    pos = torch.rand(20, wf_auto.ao.nelec * 3)
    pos.requires_grad = True

    ej = wf_jacobi.energy(pos)
    ej.backward()

    ea = wf_auto.energy(pos)
    ea.backward()

    for p1, p2 in zip(wf_auto.parameters(), wf_jacobi.parameters()):
        if p1.requires_grad:
            print('')
            print(p1.grad)
            print(p2.grad)

    if torch.cuda.is_available():
        pos_gpu = pos.to('cuda')
        wf_gpu = Orbital(mol, kinetic='jacobi',
                         configs='singlet(1,1)',
                         use_jastrow=True, cuda=True)
