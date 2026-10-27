"""Check the required Addendum C-before-results commit ordering."""
import subprocess

def main():
    # The protocol commit must be identifiable by its subject and precede every commit
    # that changes registered Tier 2 result paths.
    log=subprocess.check_output(['git','log','--reverse','--format=%H%x09%s'],text=True).splitlines()
    protocol=None; results=[]
    for row in log:
        sha,subject=row.split('\t',1)
        paths=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',sha],text=True).splitlines()
        if subject.startswith('protocol: Addendum C'): protocol=sha
        # The plan explicitly requires the planning power simulation to be committed
        # before Addendum C. Only empirical Tier 2 outputs must follow the protocol.
        if any(p.startswith(('results/stage4b','results/stage6b','results/stage8v5')) for p in paths): results.append(sha)
    if results and (protocol is None or any(log.index(next(x for x in log if x.startswith(h))) <= log.index(next(x for x in log if x.startswith(protocol))) for h in results)):
        raise SystemExit('Tier 2 results commit precedes Addendum C protocol commit')
    print('commit order verified')
if __name__=='__main__': main()
