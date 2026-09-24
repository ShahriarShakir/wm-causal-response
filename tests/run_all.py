"""Run every suite; results are used only after this exits 0.

Properties and mutants run first because they check the tests themselves: a passing unit suite
only shows that the tests agree with the code, while the mutation suite shows that the tests
detect the known faults.
"""
import subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
SUITES = ["test_properties.py",   # invariants swept over dim / sparsity / scale
          "test_mutation.py",     # every reproduced fault must be detected
          "test_joint.py",        # the measurement
          "test_validate.py"]     # the analysis gates
# tests/test_registration.py {diamond|iris|storm} ALE/Pong-v5 needs the checkpoints and the emulator; run it separately (about 2 min each on CPU).
fails = []
for s in SUITES:
    r = subprocess.run([sys.executable, str(HERE / s)], capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:] or ["(no output)"]
    print(f"{'PASS' if r.returncode == 0 else 'FAIL'}  {s:<22} {tail[0][:60]}")
    if r.returncode != 0:
        fails.append(s); print(r.stdout[-600:], r.stderr[-600:])
print()
if fails:
    print(f"FAILED: {fails}"); sys.exit(1)
print("ALL SUITES GREEN: measurement, analysis gates, properties, and mutants")
