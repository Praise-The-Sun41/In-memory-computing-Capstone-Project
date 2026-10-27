from concurrent.futures import ThreadPoolExecutor
from validate_levels import worker,analyse_level

# Independent future levels; the primary coordinator is still at RESET21.
# Two extra native workers limit memory pressure. Existing validated exports
# are reused by the primary coordinator when it later reaches these levels.
for k in [30,31]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda h:worker(32,10,k,h),range(4)))
    analyse_level(32,10,k)
print('FAR_LEVELS_COMPLETE',flush=True)
