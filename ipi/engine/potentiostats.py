"""Electrode potentiostats: dynamics of the electrode charge.

The electrode charge is treated as an extended (classical) variable of the
simulation. i-PI propagates it and passes its current value to the electronic
structure client, which computes energy and forces at fixed charge and returns
the instantaneous electrode potential in its ``extras``.

At present one scheme is implemented: the fluctuation-dissipation-theorem
(FDT) thermopotentiostat of Deissenbeck et al., Phys. Rev. Lett. 126, 136803
(2021), as used with the solvated jellium method (SJM) by Sechi et al.,
J. Chem. Theory Comput. 21, 5279 (2025). With n the number of excess electrons
on the electrode, Phi the electrode potential, Phi0 the target potential, C0
the potentiostat capacitance and tau = R C0 its relaxation time, the charge is
updated once per (outer) time step dt with the exact Ornstein-Uhlenbeck step

    n(t+dt) = n(t) + C0 [Phi(t) - Phi0] (1 - exp(-dt/tau))
                   + xi sqrt(kB T C0 (1 - exp(-2 dt/tau))),   xi ~ N(0, 1).

Since dPhi/dn < 0, the update drives Phi towards Phi0, and the noise term
makes the potential fluctuate with the canonical variance at temperature T.

Ordering with respect to the integrator
---------------------------------------
The update is applied at the beginning of each (outer) step, before the
integrator moves the atoms. At that point the force of the electrode
forcefield at the current positions x(t), computed with n(t), has already been
evaluated (it is the force used in the closing kick of the previous step), so
Phi(t) is read from its extras without any additional electronic-structure
call. The new charge n(t+dt) is then handed to the forcefield, and is used the
next time the forcefield is evaluated, i.e. at x(t+dt). This is the same
scheme as an FDT loop wrapped around a constant-charge calculator, and with
multiple time stepping dt is the outer time step, i.e. the interval between
two evaluations of the electrode forcefield.

Passing the charge to the client
--------------------------------
The charge is stored in the forcefield parameters (``ForceField.pars``) under
the name given by ``parameter``. Socket and MPI forcefields send these
parameters in the INIT string, so a client that wants the current charge has
to answer NEEDINIT after every force evaluation. FFDirect forcefields call
``driver.update_parameters(**{parameter: value})`` if the PES defines it. If
the client echoes the charge it used (``charge_key`` in the extras), the
value is checked against the potentiostat charge.
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2026 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.utils.depend import depend_value, dproperties, dpipe, dstrip
from ipi.utils.units import Constants, unit_to_internal
from ipi.utils.messages import warning, info, verbosity

__all__ = ["Potentiostat", "PotentiostatFDT"]


class Potentiostat:
    """Dummy potentiostat: the electrode charge is not propagated.

    Depend objects:
        ework: Energy exchanged with the electrode-charge reservoir. It enters
            the conserved quantity like the thermostat energy.
    """

    def __init__(self, ework=0.0):
        self._ework = depend_value(name="ework", value=ework)

    def bind(self, motion):
        """Nothing to bind for the dummy potentiostat."""
        pass

    def step(self):
        """Dummy step: does nothing."""
        pass


dproperties(Potentiostat, ["ework"])


class PotentiostatFDT(Potentiostat):
    """FDT thermopotentiostat for the electrode charge (see module docstring).

    All quantities are stored in i-PI internal (atomic) units: the charge in
    electrons, potentials in Hartree/e, the capacitance in e^2/Hartree, times
    in atomic time units and the temperature in Hartree (kB = 1).

    Attributes:
        forcefield: Name of the forcefield that sees the electrode charge.
        parameter: Name under which the charge is passed to that forcefield.
        potential_key: Extras key holding the electrode potential.
        charge_key: Extras key holding the charge used by the client (checked
            if present, ignored otherwise).
        work: How the energy exchanged with the reservoir is estimated:
            'none' (not accounted), 'canonical' (dU/dn = -Phi, valid when the
            client returns the canonical energy E(n)), or 'extras' (dU/dn is
            read from the extras key ``work_key``). The work is integrated
            with the trapezoidal rule over each charge increment.
        extras_units: Units of the potential (and dU/dn) in the extras.

    Depend objects:
        charge: Number of excess electrons on the electrode.
        phi0: Target electrode potential.
        capacitance: Potentiostat capacitance C0.
        tau: Relaxation time tau = R C0.
        temp: Temperature of the potentiostat. Taken from the ensemble if
            not given.
        dt: Time step of the charge update (the outer MD time step).
        ework: Energy exchanged with the reservoir.
    """

    def __init__(
        self,
        phi0,
        capacitance,
        tau,
        charge=0.0,
        temp=-1.0,
        forcefield="",
        parameter="excess_electrons",
        potential_key="electrode_potential",
        charge_key="excess_electrons",
        work="none",
        work_key="dudn",
        extras_units="volt",
        charge_tol=1e-6,
        ework=0.0,
        dcharge=0.0,
        dudn=0.0,
    ):
        super().__init__(ework=ework)
        if capacitance <= 0.0:
            raise ValueError("The potentiostat capacitance must be positive.")
        if tau <= 0.0:
            raise ValueError("The potentiostat relaxation time tau must be positive.")
        if work not in ("none", "canonical", "extras"):
            raise ValueError("Potentiostat work must be 'none', 'canonical' or 'extras'.")
        if forcefield == "":
            raise ValueError(
                "The potentiostat needs the name of the forcefield that sees the electrode charge."
            )
        self._charge = depend_value(name="charge", value=float(charge))
        self._phi0 = depend_value(name="phi0", value=float(phi0))
        self._capacitance = depend_value(name="capacitance", value=float(capacitance))
        self._tau = depend_value(name="tau", value=float(tau))
        self._temp = depend_value(name="temp", value=float(temp))
        self._dt = depend_value(name="dt", value=0.0)
        self.forcefield = forcefield
        self.parameter = parameter
        self.potential_key = potential_key
        self.charge_key = charge_key
        self.work = work
        self.work_key = work_key
        self.extras_units = extras_units
        self.charge_tol = charge_tol
        self._extras_conv = unit_to_internal("electric-potential", extras_units, 1.0)
        self._warned_nocharge = False
        self.ncalls = 0
        # last charge increment and ring-polymer dU/dn before it: needed to
        # complete the trapezoidal estimate of the work at the next step
        self.dcharge = float(dcharge)
        self.dudn = float(dudn)

    def bind(self, motion):
        """Connects the potentiostat to the dynamics, forces and random numbers.

        Args:
            motion: The Dynamics object that owns the potentiostat.
        """

        self.motion = motion
        self.prng = motion.prng
        self.nbeads = motion.beads.nbeads
        dpipe(motion._dt, self._dt)
        if self.temp < 0.0:
            # the electrode charge is a classical variable coupled to the
            # bead-averaged potential: physical (not ring-polymer) temperature
            dpipe(motion.ensemble._temp, self._temp)

        comps = [
            fc for fc in motion.forces.mforces if fc.ffield == self.forcefield
        ]
        if len(comps) == 0:
            raise ValueError(
                "The potentiostat forcefield '%s' is not used by any force component."
                % self.forcefield
            )
        if len(comps) > 1:
            raise ValueError(
                "The potentiostat forcefield '%s' appears in more than one force component."
                % self.forcefield
            )
        self.component = comps[0]
        if self.component.nbeads != self.nbeads:
            raise ValueError(
                "Ring-polymer contraction of the potentiostat forcefield is not supported."
            )
        self.ff = self.component.ff
        # the client must see the charge from the very first evaluation
        self.ff.set_parameter(self.parameter, self.charge)
        info(
            " @ POTENTIOSTAT: FDT, forcefield '%s', parameter '%s', n0 = %.6f e"
            % (self.forcefield, self.parameter, self.charge),
            verbosity.low,
        )

    def _extras_array(self, key):
        """Returns the bead-resolved values of an extras key, in internal units."""

        ex = self.component.extras
        if key not in ex:
            raise KeyError(
                "The potentiostat needs the key '%s' in the extras returned by forcefield '%s', found: %s"
                % (key, self.forcefield, list(ex.keys()))
            )
        return np.asarray(ex[key], float).reshape(-1) * self._extras_conv

    def get_potential(self):
        """Bead-averaged electrode potential (internal units) at the current
        positions and charge. Evaluates the forcefield if needed."""

        return self._extras_array(self.potential_key).mean()

    def get_coefficients(self):
        """Returns the drift factor (1 - exp(-dt/tau)) and the noise amplitude."""

        x = self.dt / self.tau
        a = -np.expm1(-x)
        b = np.sqrt(Constants.kb * self.temp * self.capacitance * (-np.expm1(-2.0 * x)))
        return a, b

    def _check_charge(self):
        ex = self.component.extras
        if self.charge_key in ex:
            nused = np.asarray(ex[self.charge_key], float).reshape(-1)
            if np.any(np.abs(nused - self.charge) > self.charge_tol):
                raise ValueError(
                    "The forcefield '%s' was evaluated with charge %s, but the potentiostat charge is %.10f. "
                    "The client must read the charge from the INIT string before every evaluation "
                    "(i.e. answer NEEDINIT after each force calculation)."
                    % (self.forcefield, nused, self.charge)
                )
        elif not self._warned_nocharge:
            warning(
                "The forcefield '%s' does not return '%s' in its extras: "
                "cannot check that the client uses the potentiostat charge."
                % (self.forcefield, self.charge_key),
                verbosity.low,
            )
            self._warned_nocharge = True

    def step(self):
        """Advances the electrode charge by one time step dt."""

        phi_b = self._extras_array(self.potential_key)
        self._check_charge()
        phi = phi_b.mean()

        a, b = self.get_coefficients()
        xi = self.prng.g
        dn = self.capacitance * (phi - self.phi0) * a + b * xi

        # Energy exchanged with the reservoir. The potential energy of the
        # ring polymer (sum over beads) changes by ~ sum_b dU_b/dn * dn. A
        # first-order estimate g(t) dn leaves out the always-positive term
        # (1/2) d2U/dn2 dn^2, i.e. a systematic drift of the conserved
        # quantity, so the trapezoidal rule (1/2)[g(t) + g(t+dt)] dn is used.
        # g(t+dt) is only known at the next step: g(t) dn is booked now (so
        # that the conserved quantity has no jump), and the correction
        # (1/2)[g(t+dt) - g(t)] dn is added at the next step.
        if self.work != "none":
            if self.work == "canonical":
                g = -phi_b.sum()
            else:
                g = self._extras_array(self.work_key).sum()
            self.ework -= 0.5 * (g - self.dudn) * self.dcharge
            self.ework -= g * dn
            self.dudn = g
            self.dcharge = dn

        self.charge = self.charge + dn
        self.ff.set_parameter(self.parameter, self.charge)
        self.ncalls += 1


dproperties(
    PotentiostatFDT, ["charge", "phi0", "capacitance", "tau", "temp", "dt"]
)
