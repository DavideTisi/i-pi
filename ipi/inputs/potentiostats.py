"""Input class for the electrode potentiostats."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2026 i-PI developers
# See the "licenses" directory for full license information.

from ipi.engine.potentiostats import Potentiostat, PotentiostatFDT
from ipi.utils.inputvalue import Input, InputAttribute, InputValue

__all__ = ["InputPotentiostat"]


class InputPotentiostat(Input):
    """Potentiostat input class.

    Handles generating the potentiostat object from the xml input, and
    storing its state (in particular the current electrode charge and the
    accumulated reservoir energy) in checkpoints.
    """

    attribs = {
        "mode": (
            InputAttribute,
            {
                "dtype": str,
                "default": "dummy",
                "options": ["dummy", "fdt"],
                "help": """The type of potentiostat. 'fdt' propagates the number of excess
                electrons on an electrode with the fluctuation-dissipation-theorem
                thermopotentiostat [Deissenbeck et al., PRL 126, 136803 (2021)], i.e. the exact
                Ornstein-Uhlenbeck update n <- n + C0 (Phi - Phi0)(1 - exp(-dt/tau)) +
                xi sqrt(kB T C0 (1 - exp(-2 dt/tau))), once per (outer) time step. The charge
                is passed to the forcefield as a parameter (in the INIT string for socket
                clients) and the electrode potential Phi is read from the extras returned by
                the forcefield. 'dummy' does nothing.""",
            },
        ),
        "forcefield": (
            InputAttribute,
            {
                "dtype": str,
                "default": "",
                "help": "Name of the forcefield that sees the electrode charge (e.g. the constant-charge DFT client).",
            },
        ),
    }

    fields = {
        "phi0": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "electric-potential",
                "help": "The target electrode potential, on the same scale as the potential returned by the client.",
            },
        ),
        "capacitance": (
            InputValue,
            {
                "dtype": float,
                "default": 1.0,
                "dimension": "capacitance",
                "help": "The potentiostat capacitance C0 (e.g. in units='e/v').",
            },
        ),
        "tau": (
            InputValue,
            {
                "dtype": float,
                "default": 1.0,
                "dimension": "time",
                "help": "The potentiostat relaxation time tau = R C0.",
            },
        ),
        "temperature": (
            InputValue,
            {
                "dtype": float,
                "default": -1.0,
                "dimension": "temperature",
                "help": "The potentiostat temperature. If negative, the ensemble temperature is used.",
            },
        ),
        "charge": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "help": "The current number of excess electrons on the electrode (the initial value, or the restart value).",
            },
        ),
        "ework": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "dimension": "energy",
                "help": "The energy exchanged with the electrode-charge reservoir, used to restart the conserved quantity.",
            },
        ),
        "dcharge": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "help": "Restart information: the last charge increment (used to complete the trapezoidal estimate of the work).",
            },
        ),
        "dudn": (
            InputValue,
            {
                "dtype": float,
                "default": 0.0,
                "help": "Restart information: the ring-polymer dU/dn before the last charge increment, in atomic units.",
            },
        ),
        "parameter": (
            InputValue,
            {
                "dtype": str,
                "default": "excess_electrons",
                "help": "Name of the forcefield parameter that carries the charge.",
            },
        ),
        "potential_key": (
            InputValue,
            {
                "dtype": str,
                "default": "electrode_potential",
                "help": "Extras key holding the electrode potential returned by the client.",
            },
        ),
        "charge_key": (
            InputValue,
            {
                "dtype": str,
                "default": "excess_electrons",
                "help": "Extras key holding the charge actually used by the client. If present, it is checked against the potentiostat charge.",
            },
        ),
        "work": (
            InputValue,
            {
                "dtype": str,
                "default": "none",
                "options": ["none", "canonical", "extras"],
                "help": """How the energy exchanged with the reservoir enters the conserved quantity.
                'none': not accounted (the conserved quantity jumps at every charge update);
                'canonical': dU/dn = -Phi, appropriate if the client returns the canonical energy E(n);
                'extras': dU/dn is returned by the client under work_key.
                The work is integrated with the trapezoidal rule over each charge increment.""",
            },
        ),
        "work_key": (
            InputValue,
            {
                "dtype": str,
                "default": "dudn",
                "help": "Extras key holding dU/dn (same units as the potential) when work='extras'.",
            },
        ),
        "extras_units": (
            InputValue,
            {
                "dtype": str,
                "default": "volt",
                "help": "Units of the electrode potential (and of dU/dn) in the extras.",
            },
        ),
    }

    default_help = "Propagates the charge of an electrode to keep its potential at a target value."
    default_label = "POTENTIOSTAT"

    def store(self, pstat):
        """Stores a minimal representation of a potentiostat."""

        super().store(pstat)
        self.ework.store(pstat.ework)
        if type(pstat) is PotentiostatFDT:
            self.mode.store("fdt")
            self.forcefield.store(pstat.forcefield)
            self.phi0.store(pstat.phi0)
            self.capacitance.store(pstat.capacitance)
            self.tau.store(pstat.tau)
            self.temperature.store(pstat.temp)
            self.charge.store(pstat.charge)
            self.parameter.store(pstat.parameter)
            self.potential_key.store(pstat.potential_key)
            self.charge_key.store(pstat.charge_key)
            self.work.store(pstat.work)
            self.work_key.store(pstat.work_key)
            self.extras_units.store(pstat.extras_units)
            self.dcharge.store(pstat.dcharge)
            self.dudn.store(pstat.dudn)
        elif type(pstat) is Potentiostat:
            self.mode.store("dummy")
        else:
            raise TypeError(
                "The type " + type(pstat).__name__ + " is not a valid potentiostat type"
            )

    def fetch(self):
        """Creates a potentiostat object."""

        super().fetch()
        mode = self.mode.fetch()
        if mode == "fdt":
            return PotentiostatFDT(
                phi0=self.phi0.fetch(),
                capacitance=self.capacitance.fetch(),
                tau=self.tau.fetch(),
                charge=self.charge.fetch(),
                temp=self.temperature.fetch(),
                forcefield=self.forcefield.fetch(),
                parameter=self.parameter.fetch(),
                potential_key=self.potential_key.fetch(),
                charge_key=self.charge_key.fetch(),
                work=self.work.fetch(),
                work_key=self.work_key.fetch(),
                extras_units=self.extras_units.fetch(),
                ework=self.ework.fetch(),
                dcharge=self.dcharge.fetch(),
                dudn=self.dudn.fetch(),
            )
        elif mode == "dummy":
            return Potentiostat(ework=self.ework.fetch())
        raise ValueError("Invalid potentiostat mode " + mode)
