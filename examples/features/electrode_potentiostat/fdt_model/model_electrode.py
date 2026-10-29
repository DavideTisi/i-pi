"""Analytic constant-charge 'electrode' PES for testing the FDT potentiostat.

N charged particles sit in anharmonic wells in front of an electrode carrying
n excess electrons. The canonical energy at fixed charge is (eV, Angstrom, e)

    U(x, n) = sum_i [ k/2 |x_i - x0_i|^2 + g/4 |x_i - x0_i|^4 ]
              + n kappa P_z - Phi_pzc n + n^2 / (2 C_true),

with P_z = sum_i q_i z_i the dipole of the particles along the surface normal.
The electrode potential is Phi = -dU/dn = Phi_pzc - n/C_true - kappa P_z, so
the electrode charge creates a uniform field acting on the particles and the
particle dipole shifts the electrode potential, as in a solvated slab.

The FDT potentiostat samples exp[-(U + Phi0 n)/kT], for which <Phi> = Phi0 and
Var(Phi) = kT/C_true exactly (Phi - Phi0 = -(n - <n>_x)/C_true at any x).

mode='electrode'  constant charge, n set by i-PI through update_parameters()
mode='nofield'    same particles without electrode terms (n = 0): a stand-in
                  for a machine-learned potential that does not see the field
"""

import json

import numpy as np

from ipi.pes.dummy import Dummy_driver
from ipi.utils.units import unit_to_internal, unit_to_user

__DRIVER_NAME__ = "model_electrode"
__DRIVER_CLASS__ = "ModelElectrode"


class ModelElectrode(Dummy_driver):
    def __init__(
        self,
        mode="electrode",
        k=2.0,
        g=0.5,
        kappa=0.15,
        phi_pzc=4.6,
        C_true=0.3,
        excess_electrons=0.0,
        ref="",
        *args,
        **kwargs,
    ):
        self.mode = str(mode)
        self.k, self.g, self.kappa = float(k), float(g), float(kappa)
        self.phi_pzc, self.C_true = float(phi_pzc), float(C_true)
        self.n = float(excess_electrons)
        self.x0 = None
        self.ref = ref
        super().__init__(*args, **kwargs)

    def update_parameters(self, **pars):
        if "excess_electrons" in pars:
            self.n = float(pars["excess_electrons"])

    def charges(self, nat):
        return np.where(np.arange(nat) % 2 == 0, -0.8, 0.8)

    def evaluate(self, pos, n):
        """Energy (eV), forces (eV/A) and Phi (V) at positions (A) and charge n."""
        if self.x0 is None:
            self.x0 = reference_sites(len(pos))
        d = pos - self.x0
        r2 = (d**2).sum(axis=1)
        e = (0.5 * self.k * r2 + 0.25 * self.g * r2**2).sum()
        f = -(self.k + self.g * r2)[:, None] * d
        q = self.charges(len(pos))
        pz = float(q @ pos[:, 2])
        e += n * self.kappa * pz - self.phi_pzc * n + n**2 / (2 * self.C_true)
        f[:, 2] -= n * self.kappa * q
        phi = self.phi_pzc - n / self.C_true - self.kappa * pz
        return e, f, phi

    def compute_structure(self, cell, pos):
        pos_A = unit_to_user("length", "angstrom", pos)
        n = self.n if self.mode == "electrode" else 0.0
        e, f, phi = self.evaluate(pos_A, n)
        extras = {}
        if self.mode == "electrode":
            extras = {"electrode_potential": phi, "excess_electrons": n}
        return (
            unit_to_internal("energy", "electronvolt", e),
            unit_to_internal("force", "ev/ang", f),
            np.zeros((3, 3)),
            json.dumps(extras),
        )


def reference_sites(nat):
    """Well centres: a 2-row grid of sites in front of the electrode (A)."""
    i = np.arange(nat)
    return np.stack([1.5 * (i // 2), 1.0 * (i % 2), 3.0 + 0.4 * (i % 2)], axis=1).astype(float)
