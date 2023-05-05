import numpy as np


def ForwardBackward_BernoulliJump(obs, pJ, pgrid, Alpha0, Pass='Forward'):
    # Forward-Backward algorithm to solve a hidden markov model, in which the
    # hidden state is a Bernoulli parameter x controlling the observed outcome s.
    #
    # Usage
    # [rGamma, rAlpha, rBeta, JumpPost] = ...
    #       ForwardBackward_BernoulliJump(s, pJ, pgrid, Alpha0, Pass)
    #
    # Input:
    #      s: sequence of numeric values (coded as 1s and 2s for convenience)
    #     pJ: prior on jump occurrence (change in Bernoulli parameter) at
    #         any given moment (this is a scalar value)
    #  pgrid: grid for numeric estimation of the Bernoulli parameter.
    #         pgrid[i] is p(s=1|x[i])
    # Alpha0: prior on states (i.e. prior on the Bernoulli parameter, provided
    #         the grid pgrid).
    #   Pass: if 'Forward' (default), computes only the forward estimation. If
    #         'Backward', also computes the backward estimation and the
    #         combination of forward and backward estimates.
    #
    # Output:
    #   rGamma: rGamma[:,t] is the posterior distribution for states x(t)
    #           (i.e. hidden Bernoulli parameters) given observation s[1:N]
    # 	    => the Forward-Backward estimation
    #   rAlpha: rAlpha[:,t] is the posterior distribution for states x(t)
    #           (i.e. hidden Bernoulli parameters) given observation s[1:t]
    # 	    => the Forward estimation
    #    rBeta: rBeta[:,t] is the posterior distribution for states x(t)
    #           (i.e. hidden Bernoulli parameters) given observation s[t:N]
    # 	    => the Backward estimation
    # JumpPost: Posterior on jump probability given s[1:N].
    #
    # Checks and initialization
    # =========================
    N = len(obs)
    n_noise = len(pgrid)

    if pJ > 1 or pJ < 0:
        raise ValueError(f'pJ must be in the [0 1] interval (current value: {pJ})')
    if Pass not in ['Forward', 'Backward']:
        raise ValueError(f'Pass must be either "Forward" or "Backward" (current value: {Pass})')

    # FORWARD PASS: p(x(i,t) | s[1:t])
    # ==============================

    # Alpha[i, 0, t] = p(x(i,t), J=0 | s[1:t])
    # Alpha[i, 1, t] = p(x(i,t), J=1 | s[1:t])
    # Initialize alpha (forward algorithm)
    Alpha = np.zeros((n_noise, 2, N))

    # get the matrix of non-diagonal elements
    NonDiag = np.ones((n_noise, n_noise))
    np.fill_diagonal(NonDiag, 0)
    # Compute the transition matrix (jump = non diagonal transitions).
    # NB: the prior on jump occurrence pJ is NOT included here.
    # Trans(i,j) is the probability to jump FROM i TO j
    # Hence, sum(Trans(i,:)) = 1
    # The likelihood of new values after a jump correspond the prior on states.
    
    Trans = NonDiag * Alpha0.reshape(1, -1)
    Trans = Trans / Trans.sum(axis=1).reshape(-1, 1)  # normalize values
    assert np.all(np.isclose(Trans.sum(axis=1), np.ones(Trans.shape[0])))  # check normalization

    # Compute alpha iteratively (forward pass)
    for t in range(N):

        # Specify likelihood of current observation
        if obs[t] == 1:  # 1 is inconsistent
            LL = pgrid.copy()
        elif obs[t] == 2:
            LL = (1 - pgrid)
        elif np.isnan(obs[t]):  # likelihood is flat in the absence of observation ('NaN')
            LL = np.ones(n_noise) / n_noise
        else:
            raise ValueError

        # Compute the new alpha, based on the former alpha, the prior on transition between states and the likelihood.
        # See for instance Wikipedia entry for 'Forward algorithm'.
        if t == 0:
            prior = Alpha0
        else:
            prior = Alpha[:, 0, t - 1] + Alpha[:, 1, t - 1]

        # No Jump at t:
        # - take the prior on 'no jump': (1-pJ)
        # - take the current observation likelihood under x_i (LL)
        # - take the posterior on x_i(t-1) (summed over whether there was a jump of not at t-1)
        Alpha[:, 0, t] = (1 - pJ) * LL * prior
        # Jump at t:
        # - take the prior on 'jump': (pJ)
        # - take the current observation likelihood under x_i (LL)
        # - take the posterior on all the other states, excluding x_i(t-1) (summed over whether there was a jump or not at i-1)
        # - sum over the likelihood of the ORIGINS of such state (hence the transpose on Trans, to sum over the ORIGIN)
        Alpha[:, 1, t] = pJ * LL * (Trans.T @ prior)

        # scale alpha as a posterior (which we will do eventually) to avoid numeric overflow
        NormalizationCst = np.sum(np.sum(Alpha[:, :, t], axis=1), axis=0)
        Alpha[:, 0, t] = Alpha[:, 0, t] / NormalizationCst
        Alpha[:, 1, t] = Alpha[:, 1, t] / NormalizationCst
  

    # BACKWARD PASS: p(y(t+1:N) | x(i,t)
    # ================================
    if Pass.lower() == 'backward':

        # Beta(i,t) = p(s(t+1:N) | x(i,t))
        # Since we want the to distinguish jump vs. no jump, we split the values of Beta in 2 columns, corresponding to jump or no jump.
        # Beta(i,0,t) = p(s(t+1:N), J=0 | x(i,t))
        # Beta(i,1,t) = p(s(t+1:N), J=1 | x(i,t))
        #
        # In addition, we normalize Beta(i,t) so that it sums to 1 over i. This is only for convenience (to make the interpretation of numeric values easier)
        # since in the end the backward probability is normalized, we can apply any scaling factor to Beta(i,t)

        # Initialize beta (backward estimation)
        Beta = np.zeros((n_noise, 2, N))

        # Compute beta iteratively (backward pass)
        for t in range(N - 1, -1, -1):

            # Specify likelihood of current observation
            if obs[t] == 1:  # 1 is inconsistent
                LL = pgrid.copy()
            elif obs[t] == 2:
                LL = (1 - pgrid)
            elif np.isnan(obs[t]):  # likelihood is flat in the absence of observation ('NaN')
                LL = np.ones(n_noise) / n_noise
            else: raise ValueError

            if t == N - 1:
                next_beta = np.ones(n_noise) / (2 * n_noise)
            else:
                next_beta = (Beta[:, 0, t + 1] + Beta[:, 1, t + 1])

            # No Jump from t to t+1
            # take only diagonal elements
            Beta[:, 0, t] = (1 - pJ) * (LL * next_beta)

            # Jump from t to t+1
            # sum over non diagonal elements
            # NB: there is no transpose here on Trans because we sum over
            # TARGET location (not ORIGIN)
            Beta[:, 1, t] = pJ * LL * (Trans @ next_beta)

            # scale beta to sum = 1. This normalization is only for convenience,
            # since we don't need this scaling factor in the end.
            NormalizationCst = np.sum(np.sum(Beta[:, :, t], axis=1), axis=0)
            Beta[:, 0, t] = Beta[:, 0, t] / NormalizationCst
            Beta[:, 1, t] = Beta[:, 1, t] / NormalizationCst

        # Shift Beta so that Beta[:,:,t] is the posterior given s(t+1:N)
        # newBeta = np.zeros_like(Beta)
        # # newBeta[:, :, 0] = 1 / (n * n)  # TODO: what is this?
        # newBeta[:, :, 0] = 1 / (2 * n_noise)
        # newBeta[:, :, 1:] = Beta[:, :, :-1]
        # Beta = newBeta.copy()

    # COMBINE FORWARD AND BACKWARD PASS
    # =================================
    if Pass.lower() == 'backward':
        # Compute the forward & backward posterior
        rAlpha = np.sum(Alpha, axis=1)
        rBeta = np.sum(Beta, axis=1)

        # p(x(t)|y(1:N)) ~ p(y(t+1:N)|x(t)) p(x(t)|y(1:t))
        # NB: the sum over dimension 1 is to average out the J=0 or 1
        rGamma = rAlpha * rBeta

        # Scale gamma as a posterior on observations
        cst = np.tile(np.sum(rGamma, axis=0), (n_noise, 1))
        rGamma = rGamma / cst

        # Compute the posterior on jump, summed over the states
        # GammaJ = p(x(t),J=1|y(1:N)) ~ p(y(t+1:N) | x(t),J=1) p(x(t),J=1|y(1:t))
        #                             ~ [1/p(J=1) * p(y(t+1:N),J=1|x(t))] p(x(t),J=1|y(1:t))
        #                             ~ [1/p(J=1) * Beta2] Alpha2
        GammaJ = Alpha[:, 1, :] * ((1 / pJ) * Beta[:, 1, :])
        GammaJ = GammaJ / cst

        # GammaJ0 = Alpha[:, 0, :] * ((1 / (1 - pJ)) * Beta[:, 0, :])
        # GammaJ0 = GammaJ0 / cst

        JumpPost = np.sum(GammaJ, axis=0)
        if np.any(JumpPost > 1):
            print('JumpPost problem')
        # JumpPost = [0.1]
    else:
        rGamma = np.array([])
        rBeta = np.array([])
        JumpPost = np.array([])
        rAlpha = np.sum(Alpha, axis=1)

    return Alpha, rGamma, rAlpha, rBeta, JumpPost, Trans


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    probabilities = []
    outcomes = []
    p_change = 0.1
    p = np.random.uniform(0, 1)
    for j in range(100):
        change = np.random.rand() < p_change
        change=j==50
        if change:
            p = np.random.uniform(0, 1)
        probabilities.append(p)
        outcomes.append(np.random.rand() > p)
    outcomes = np.array(outcomes) + 1
    pgrid = np.linspace(0, 1, 20)
    Alpha0 = np.ones(pgrid.size) / pgrid.size
    pJ = 0.05
    # ForwardBackward_BernoulliJump(s=[1], pJ=pJ, Alpha0=Alpha0, pgrid=pgrid, Pass='Backward')
    Alpha, rGamma, rAlpha, rBeta, JumpPost, Trans = ForwardBackward_BernoulliJump(obs=outcomes, pJ=pJ, Alpha0=Alpha0, pgrid=pgrid, Pass='Backward')
    predicted_probas = pgrid @ rGamma
    plt.figure()
    plt.plot(probabilities, label='p')
    plt.plot(predicted_probas, label='predicted p')
    plt.plot(JumpPost, label='p_change')
    plt.legend()
    plt.ylim([0, 1])
    for i in range(1, len(probabilities)):
        if probabilities[i-1] != probabilities[i]:
            plt.axvline(x=i, ymax=1, ymin=0, color='black')
    plt.show()
    stop = 1