FDT electrode thermopotentiostat with multiple time stepping
=============================================================

The `<potentiostat mode='fdt'>` block (inside `<dynamics>`) propagates the number of
excess electrons n on an electrode with the fluctuation-dissipation-theorem
thermopotentiostat [Deissenbeck et al., PRL 126, 136803 (2021)], as used with the
solvated jellium method in Sechi et al., JCTC 21, 5279 (2025):

    n <- n + C0 (Phi - Phi0)(1 - exp(-dt/tau)) + xi sqrt(kB T C0 (1 - exp(-2 dt/tau)))

once per (outer) time step dt. The forcefield named in `forcefield=` must compute
energy and forces at fixed charge and return the electrode potential in its extras
(`electrode_potential`, in volt by default) and, optionally, the charge it used
(`excess_electrons`, checked by i-PI).

How the charge reaches the client
---------------------------------
* `ffsocket` / `ffmpi`: the charge is a forcefield parameter, sent in the INIT
  string (` excess_electrons : <value> , `). The client must answer NEEDINIT after
  every force evaluation to receive the current value before each POSDATA.
* `ffdirect`: i-PI calls `update_parameters(excess_electrons=<value>)` on the PES.
  The PES constructor must also accept `excess_electrons=` (it is stored with the
  forcefield parameters in checkpoints).

Ordering: the charge update uses Phi at the current positions (already computed for
the closing kick of the previous step), so no extra electronic-structure call is made;
the new charge is used at the next evaluation of the forcefield.

Conserved quantity: `<work> canonical </work>` (dU/dn = -Phi, for clients returning
the canonical energy E(n)) or `<work> extras </work>` (dU/dn from the extras key
`work_key`) adds the energy exchanged with the charge reservoir to the conserved
quantity (trapezoidal rule over each charge increment). The default `none` leaves it
out, and the conserved quantity then jumps at every charge update.

Output properties: `electrode_charge`, `electrode_potential`, `potentiostat_work`.

This example
------------
`model_electrode.py` is an analytic constant-charge electrode: charged particles in
anharmonic wells, U(x, n) = U0(x) + n kappa P_z - Phi_pzc n + n^2/(2 C_true), so that
Phi = -dU/dn = Phi_pzc - n/C_true - kappa P_z. The FDT potentiostat then gives
<Phi> = Phi0 and Var(Phi) = kB T / C_true. The `electrode` forcefield (stand-in for a
constant-charge DFT/SJM client) is evaluated every outer step, while `nofield` (stand-in
for a machine-learned potential that does not see the electrode charge) propagates the
inner MTS steps.

    i-pi input.xml
