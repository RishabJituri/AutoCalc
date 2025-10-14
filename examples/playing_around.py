import ag
import numpy as np

print("playing_around: ag module attributes:\n", dir(ag))

A_arr = np.random.randn(2, 4).astype(np.float32)
B_arr = np.random.randn(4, 3).astype(np.float32)
A_var = ag.Variable(A_arr.ravel().tolist(), [2, 4], False)
B_var = ag.Variable(B_arr.ravel().tolist(), [4, 3], False)
print("A_var.shape", tuple(A_var.shape()), "B_var.shape", tuple(B_var.shape()))

C_np = A_arr @ B_arr
print("numpy matmul result shape", C_np.shape)

def to_var(arr, requires_grad=False):
    a = np.asarray(arr, dtype=np.float32)
    flat = a.ravel().tolist()
    shape = [int(s) for s in a.shape]
    return ag.Variable(flat, shape, requires_grad)

def var_to_numpy(v):
    vals = np.array(v.value(), dtype=np.float32)
    shp = tuple(v.shape())
    return vals.reshape(shp)

if hasattr(ag, 'ops') and hasattr(ag.ops, 'matmul'):
    try:
        C_var = ag.ops.matmul(A_var, B_var)
        C_ag = var_to_numpy(C_var)
        print("ag.ops.matmul -> numpy shape:", C_ag.shape)
        print("ag vs numpy close:", np.allclose(C_ag, C_np, atol=1e-5))
    except Exception as e:
        print("ag.ops.matmul invocation failed:\n", repr(e))
else:
    print("ag.ops.matmul not available; falling back to numpy")

# elementwise multiply test using Variables
x_arr = np.random.randn(5).astype(np.float32)
y_arr = np.random.randn(5).astype(np.float32)
x_var = ag.Variable(x_arr.ravel().tolist(), [5], False)
y_var = ag.Variable(y_arr.ravel().tolist(), [5], False)
print("x_var[:3]", var_to_numpy(x_var)[:3], "y_var[:3]", var_to_numpy(y_var)[:3])
if hasattr(ag, 'ops') and hasattr(ag.ops, 'mul'):
    try:
        r_var = ag.ops.mul(x_var, y_var)
        r_np = var_to_numpy(r_var).ravel()
        print("ag.ops.mul first 3:", r_np[:3])
    except Exception as e:
        print("ag.ops.mul failed:", e)
else:
    print("ag.ops.mul not available; numpy fallback:", (x_arr * y_arr)[:3])

print("done")
