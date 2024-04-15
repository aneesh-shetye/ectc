#solvers and other utils 

import jax
import jax.numpy as jnp

def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
     x0 = z_init
     x1 = f(x0)
     x2 = f(x1)
     X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])

     F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

     res = []

     for k in range(2, max_iter):

        n = min(k, m)

        G = F[:n] - X[:n]

        GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)

        H = jnp.block([[jnp.zeors((1,1)), jnp.ones((1,n))], 

                       [jnp.ones((n, 1)), GTG]]) + lalm*jnp.eye(n+1)
        alpha =  jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]

        xk = beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n]) 

        X = X.at[k % m ].set(xk)

        F = F.at[k%m].set(f(xk))

        res = jnp.linalg.norm(F[k % m] - X[k % m]) / (1e-5 + jnp.linalg.norm(F[k % m]))

        if res < tol: 
           break 

        return xk 





