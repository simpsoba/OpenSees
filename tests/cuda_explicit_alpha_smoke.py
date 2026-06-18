"""Smoke test for CudaExplicitAlpha + CuDSS SOE on a 2-DOF truss-like model."""
import opensees as ops

ops.wipe()
ops.model("basic", "-ndm", 1, "-ndf", 1)
ops.node(1, 0.0)
ops.node(2, 1.0)
ops.mass(2, 1.0)
ops.fix(1, 1)
ops.uniaxialMaterial("Elastic", 1, 100.0)
ops.element("Truss", 1, 1, 2, 1.0, 1)

ops.system("CuDSS")
ops.numberer("RCM")
ops.constraints("Plain")
ops.algorithm("Linear")
ops.integrator("CudaExplicitAlpha", 0.5, 0.5, 0.5, 0.25)
ops.analysis("Transient")

dt = 0.01
nsteps = 20
ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
ops.load(2, 1.0)

disp = []
for _ in range(nsteps):
    ok = ops.analyze(1, dt)
    if ok != 0:
        raise RuntimeError(f"analyze failed at step {_}")
    d = ops.nodeDisp(2)
    disp.append(d[0] if isinstance(d, list) else d)

if not any(abs(x) > 1e-12 for x in disp):
    raise RuntimeError("response remained zero")

print("CudaExplicitAlpha smoke test passed.")
print(f"final disp = {disp[-1]:.6e}")
