"""Study controls; kept outside the immutable simulation kernel."""
from chemistry import CASES as CORE_CASES, Chemistry, replace
CASES=dict(CORE_CASES,neutral_binding=replace(CORE_CASES['binding'],binding_delta_g_eV=0.))
