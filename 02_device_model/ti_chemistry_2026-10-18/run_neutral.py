from chemistry import *
from scenarios import CASES

if __name__=='__main__':
    m=Device(CASES['neutral_binding'])
    solve(m,et.protocol(10),m.initial,'neutral_binding-C10-L256-M256-R2e-09')
