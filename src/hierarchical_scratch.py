import numpy as np

def ForwardBackward_BernoulliJump(s, pJ, pgrid, Alpha0, Pass='Forward'):
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
    seqL   = len(s)
    n      = len(pgrid)
    pgrid  = pgrid.reshape(1,-1) # make sure that pgrid is a row vector
    Alpha0 = Alpha0.reshape(-1,1) # make sure that Alpha0 is a column vector

    if pJ > 1 or pJ < 0:
        raise ValueError(f'pJ must be in the [0 1] interval (current value: {pJ})')
    if Pass not in ['Forward', 'Backward']:
        raise ValueError(f'Pass must be either "Forward" or "Backward" (current value: {Pass})')

    # FORWARD PASS: p(x(i,t) | s[1:t])
    # ==============================

    # Alpha[i, 0, t] = p(x(i,t), J=0 | s[1:t])
    # Alpha[i, 1, t] = p(x(i,t), J=1 | s[1:t])
    # Initialize alpha (forward algorithm)
    Alpha = np.zeros((n, 2, seqL))

    # get the matrix of non diagonal elements
    NonDiag = np.ones((n, n))
    np.fill_diagonal(NonDiag, 0)
    # Compute the transition matrix (jump = non diagonal transitions).
    # NB: the prior on jump occurrence pJ is NOT included here.
    # Trans(i,j) is the probability to jump FROM i TO J
    # Hence, sum(Trans(i,:)) = 1
    # The likelihood of new values after a jump correspond the prior on states.
    Trans = (1 / (NonDiag * Alpha0)) * Alpha0.T
    Trans = Trans * NonDiag

    # Compute alpha iteratively (forward pass)
    for k in range(seqL):

        # Specify likelihood of current observation
        LL = np.eye(n)
        if s[k] == 1:
            LL[LL == 1] = pgrid
        elif s[k] == 2:
            LL[LL == 1] = 1 - pgrid
        elif np.isnan(s[k]):
            # likelihood is flat in the absence of observation ('NaN')
            LL[LL == 1] = 1 / n

        # Compute the new alpha, based on the former alpha, the prior on
        # transition between states and the likelihood. See for instance
        # Wikipedia entry for 'Forward algorithm'.
        if k == 0:
            Alpha[:, 0, k] = (1 - pJ) * LL @ Alpha0
            Alpha[:, 1, k] = pJ / (n - 1) * LL @ Alpha0
        else:

            # No Jump at k:
            # - take the prior on 'no jump': (1-pJ)
            # - take the current observation likelihood under x_i (LL)
            # - take the posterior on x_i(t-1) (summed over whether there was a
            # jump of not at t-1)
            Alpha[:, 0, k] = (1 - pJ) * LL @ (Alpha[:, 0, k - 1] + Alpha[:, 1, k - 1])

            # Jump at k:
            # - take the prior on 'jump': (1-pJ)
            # - take the current observation likelihood under x_i (LL)
            # - take the posterior on all the other states, excluding x_i(t-1)
            # (summed over whether there was a jump or not at i-1)
            # - sum over the likelihood of the ORIGINS of such state
            # (hence the transpose on Trans, to sum over the ORIGIN)
            Alpha[:, 1, k] = pJ * LL @ (Trans.T @ (Alpha[:, 0, k - 1] + Alpha[:, 1, k - 1]))

        # scale alpha as a posterior (which we will do eventually) to avoid
        # numeric overflow
        NormalizationCst = np.sum(np.sum(Alpha[:, :, k], axis=1), axis=0)
        Alpha[:, 0, k] = Alpha[:, 0, k] / NormalizationCst
        Alpha[:, 1, k] = Alpha[:, 1, k] / NormalizationCst

    # BACKWARD PASS: p(y(t+1:N | x(i,t))
    # ================================
    if Pass.lower() == 'backward':

        # Beta(i,t) = p(s(t+1:N) | x(i,t))
        # Since we want the to distinguish jump vs. no jump, we split
        # the values of Beta in 2 columns, corresponding to jump or no jump.
        # Beta(i,1,t) = p(s(t+1:N), J=0 | x(i,t))
        # Beta(i,1,t) = p(s(t+1:N), J=1 | x(i,t))
        #
        # In addition, we normalize Beta(i,t) so that it sums to 1 over i. This is
        # only for convenience (to make the interpretation of numeric values easier)
        # since in the end the backward probability is normalized, we can
        # apply any scaling factor to Beta(i,t)

        # Initialize beta (backward estimation)
        Beta = np.zeros((n, 2, seqL))

        # Compute beta iteratively (backward pass)

        for k in range(seqL - 1, -1, -1):

            # Specify likelihood of current observation
            LL = np.eye(n)
            if s[k] == 1:
                LL = LL * pgrid
            elif s[k] == 2:
                LL = LL * (1 - pgrid)
            elif np.isnan(s[k]):
                # likelihood is flat in the absence of observation ('NaN')
                LL = LL * (1 / n)

            if k == seqL - 1:
                Beta[:, 0, k] = 1
                Beta[:, 1, k] = 1
            else:
                # No Jump from k to k+1
                # take only diagonal elements
                Beta[:, 0, k] = (1 - pJ) * (LL @ (Beta[:, 0, k + 1] + Beta[:, 1, k + 1]))

                # Jump from k to k+1
                # sum over non diagonal elements
                # NB: there is no transpose here on Trans because we sum over
                # TARGET location (not ORIGIN)
                Beta[:, 1, k] = (pJ * Trans) @ (LL @ (Beta[:, 0, k + 1] + Beta[:, 1, k + 1]))

            # scale beta to sum = 1. This normalization is only for convenience,
            # since we don't need this scaling factor in the end.
            NormalizationCst = np.sum(np.sum(Beta[:, :, k], axis=1), axis=0)
            Beta[:, 0, k] = Beta[:, 0, k] / NormalizationCst
            Beta[:, 1, k] = Beta[:, 1, k] / NormalizationCst

        # Shift Beta so that Beta[:,:,k] is the posterior given s(k+1:N)
        newBeta = np.zeros_like(Beta)
        newBeta[:, :, 0] = 1 / (n * n)
        newBeta[:, :, 1:] = Beta[:, :, 0:-1]
        Beta = newBeta

    # COMBINE FORWARD AND BACKWARD PASS
    # =================================
    if Pass.lower() == 'backward':
        # p(x(t)|y(1:N)) ~ p(y(t+1:N)|x(t)) p(x(t)|y(1:t))
        # NB: the sum over the second dimension is to average out the J=0 or 1
        rGamma = np.sum(Alpha, axis=2) * np.sum(Beta, axis=2)

        # Scale gamma as a posterior on observations
        cst = np.tile(np.sum(rGamma, axis=1), (n, 1))
        rGamma = rGamma / cst

        # Compute the forward & backward posterior
        rAlpha = np.sum(Alpha, axis=2)
        rBeta = np.sum(Beta, axis=2)

        # Compute the posterior on jump, summed over the states
        # GammaJ = p(x(t),J=1|y(1:N)) ~ p(y(t+1:N) | x(t),J=1) p(x(t),J=1|y(1:t))
        #                             ~ [1/p(J=1) * p(y(t+1:N),J=1|x(t))] p(x(t),J=1|y(1:t))
        #                             ~ [1/p(J=1) * Beta2] Alpha2
        GammaJ = Alpha[:, 1, :] * ((1 / pJ) * Beta[:, 1, :])
        GammaJ = GammaJ / cst
        JumpPost = np.sum(GammaJ, axis=0)
    else:
        rGamma = np.array([]);
        rBeta = np.array([]);
        JumpPost = np.array([])
        rAlpha = np.sum(Alpha, axis=2)

    return rGamma, rAlpha, rBeta, JumpPost, Trans