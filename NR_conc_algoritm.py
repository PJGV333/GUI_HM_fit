def concentraciones(K, max_iter=1000, tol=1e-10, args = (C_T, modelo, nas)):    
    ctot = np.array(C_T)
    n_reacciones, n_componentes = ctot.shape
    pre_ko = np.zeros(n_componentes)
    K = np.concatenate((pre_ko, K))
    K = np.cumsum(K)
    K = 10**K
    
    nspec = len(K)

    def calcular_concentraciones(ctot_i, c_guess):
        c_spec = np.prod(np.power(np.tile(c_guess, (nspec, 1)).T, modelo), axis=0) * K
        c_tot_cal = np.sum(modelo * np.tile(c_spec, (n_componentes, 1)), axis=1)
        d = ctot_i - c_tot_cal

        J = np.empty((n_componentes, n_componentes))
        for j in range(n_componentes):
            for h in range(n_componentes):
                J[j, h] = np.sum(modelo.T[:, j] * modelo.T[:, h] * c_spec) 
        
        delta_c = d @ np.linalg.pinv(J) @ np.diagflat(c_guess)

        c_guess += delta_c
        return c_guess, np.linalg.norm(d), c_spec
    
    c_calculada = np.zeros((n_reacciones, nspec))
    for i in range(n_reacciones):
        c_guess = np.ones(n_componentes) * 1e-10
        c_guess[0], c_guess[1] = ctot[i, 0], ctot[i, 1]
        dif = tol + 1
        it = 0
        while dif > tol and it < max_iter:
            c_guess, delta_c_norm_sq, c_spec = calcular_concentraciones(ctot[i], c_guess)
            dif = delta_c_norm_sq
            it += 1
        c_calculada[i] = c_spec
    
    C = np.delete(c_calculada, nas, axis = 1)
    return C, c_calculada