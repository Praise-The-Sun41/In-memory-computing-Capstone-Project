import json,time
from native_io import R,dump
from screen_protocol import execute
from validate_levels import validate,refine

def main():
    start=time.monotonic()
    while not (R/'screen_8.json').exists():
        if 'Traceback (most recent call last)' in (R/'screen8.log').read_text(errors='replace'):raise RuntimeError('Initial screen failed')
        if time.monotonic()-start>7200:raise TimeoutError('Initial screen did not complete')
        time.sleep(5)
    screens=[];selection=None;validation_attempts=[]
    for n in [8,32,128]:
        screen=json.loads((R/'screen_8.json').read_text(encoding='utf-8')) if n==8 else execute(n)
        screens.append(screen)
        dump('campaign_progress.json',dict(status='screened',set_count=n,selection=None))
        for candidate in screen['candidates']:
            if candidate['wait_tau']==0 or not candidate['stationary_set_response'] or not candidate['separated_groups']:continue
            proposed=validate(candidate)
            validation_attempts.append(dict(set_count=n,wait_tau=candidate['wait_tau'],stable_selection=proposed))
            if proposed:
                numerical=refine(proposed)
                if numerical['passed']:
                    selection=dict(**proposed,numerical=numerical)
                    break
        if selection:break
    result=dict(status='completed',accepted=selection is not None,selection=selection,screened_set_counts=[x['set_count'] for x in screens],validation_attempts=validation_attempts)
    dump('campaign_result.json',result)
    dump('campaign_progress.json',result)
    print('EXPERIMENT_CAMPAIGN_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
