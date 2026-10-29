"""Unit tests for the FDT electrode potentiostat."""

import numpy as np
import pytest

from ipi.engine.potentiostats import Potentiostat, PotentiostatFDT
from ipi.inputs.potentiostats import InputPotentiostat
from ipi.utils.depend import depend_value
from ipi.utils.units import unit_to_internal, Constants
from ipi.utils.prng import Random

VOLT = unit_to_internal("electric-potential", "volt", 1.0)
EV_ = unit_to_internal("capacitance", "e/v", 1.0)
FS = unit_to_internal("time", "femtosecond", 1.0)
KELVIN = unit_to_internal("temperature", "kelvin", 1.0)


class FakeFF:
    """Constant-charge 'client' with Phi = Phi_pzc - n/C_true (V), fixed atoms."""

    def __init__(self, phi_pzc=4.6, c_true=0.3, echo=True):
        self.pars = {}
        self.phi_pzc, self.c_true, self.echo = phi_pzc, c_true, echo

    def set_parameter(self, name, value):
        self.pars[name] = value

    def energy(self, n):  # canonical U(n) in eV:  -dU/dn = Phi
        return -self.phi_pzc * n + n**2 / (2 * self.c_true)


class FakeComponent:
    def __init__(self, ff, nbeads=1):
        self.ffield, self.ff, self.nbeads = "dft", ff, nbeads

    @property
    def extras(self):
        n = self.ff.pars["excess_electrons"]
        ex = {"electrode_potential": [self.ff.phi_pzc - n / self.ff.c_true] * self.nbeads}
        if self.ff.echo:
            ex["excess_electrons"] = [n] * self.nbeads
        return ex


class Obj:
    pass


def make_motion(ff, dt_fs=2.0, temp_K=300.0, seed=12345):
    m = Obj()
    m.prng = Random(seed=seed)
    m.beads = Obj()
    m.beads.nbeads = 1
    m._dt = depend_value(name="dt", value=dt_fs * FS)
    m.ensemble = Obj()
    m.ensemble._temp = depend_value(name="temp", value=temp_K * KELVIN)
    m.forces = Obj()
    m.forces.mforces = [FakeComponent(ff)]
    return m


def make_pstat(**kw):
    args = dict(phi0=4.0 * VOLT, capacitance=0.3 * EV_, tau=20.0 * FS, forcefield="dft")
    args.update(kw)
    return PotentiostatFDT(**args)


def test_units():
    assert np.isclose(VOLT, unit_to_internal("energy", "electronvolt", 1.0), rtol=1e-6)
    assert np.isclose(VOLT * EV_, 1.0)  # C0 * Phi is a number of electrons


def test_step_matches_ou_formula():
    ff = FakeFF()
    p = make_pstat(charge=0.1)
    p.bind(make_motion(ff))
    assert ff.pars["excess_electrons"] == 0.1  # charge sent at bind
    ref = Random(seed=12345)
    n = 0.1
    kT = Constants.kb * 300.0 * KELVIN
    a, b = -np.expm1(-0.1), np.sqrt(kT * 0.3 * EV_ * -np.expm1(-0.2))
    for _ in range(5):
        phi = (4.6 - n / 0.3) * VOLT
        n = n + 0.3 * EV_ * (phi - 4.0 * VOLT) * a + b * ref.g
        p.step()
        assert np.isclose(p.charge, n, rtol=0, atol=1e-13)
        assert ff.pars["excess_electrons"] == p.charge


def test_deterministic_relaxation():
    """T = 0: Phi relaxes to Phi0 (exactly, for C0 = C_true, as exp(-t/tau))."""
    ff = FakeFF()
    p = make_pstat(temp=0.0)
    p.bind(make_motion(ff))
    for _ in range(200):
        p.step()
    assert np.isclose(p.get_potential() / VOLT, 4.0, atol=1e-6)


def test_trapezoidal_work_exact_for_quadratic_energy():
    """Fixed atoms, U quadratic in n: U + ework is conserved to round-off."""
    ff = FakeFF()
    p = make_pstat(work="canonical", charge=0.05)
    p.bind(make_motion(ff))
    e0 = ff.energy(p.charge) * VOLT
    for _ in range(100):
        p.step()
    p.step()  # completes the trapezoid of the last increment ...
    # ... and books g*dn of a new one: undo that for the check
    e = ff.energy(p.charge - p.dcharge) * VOLT + p.ework + p.dudn * p.dcharge
    assert abs(e - e0) < 1e-12


def test_charge_check():
    ff = FakeFF()
    p = make_pstat()
    p.bind(make_motion(ff))
    ff.pars["excess_electrons"] = 0.5  # client 'used' a different charge
    p.charge = 0.0
    with pytest.raises(ValueError):
        p.step()


def test_missing_forcefield():
    with pytest.raises(ValueError):
        make_pstat(forcefield="other").bind(make_motion(FakeFF()))


def test_input_roundtrip():
    ff = FakeFF()
    p = make_pstat(work="canonical", charge=0.2, temp=310.0 * KELVIN)
    p.bind(make_motion(ff))
    p.step()
    ip = InputPotentiostat()
    ip.store(p)
    q = ip.fetch()
    for k in ("charge", "phi0", "capacitance", "tau", "temp", "ework", "dcharge", "dudn"):
        assert getattr(q, k) == getattr(p, k), k
    assert q.forcefield == "dft" and q.work == "canonical"
    ip2 = InputPotentiostat()
    ip2.store(Potentiostat())
    assert type(ip2.fetch()) is Potentiostat
