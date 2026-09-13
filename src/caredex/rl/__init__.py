"""Priors as policy action spaces.

Everything before this package treats a prior as a generator: sample a window,
replay it, score it. Here the prior becomes an *interface*: a policy emits one
latent per control step, the frozen decoder turns it into a hand pose, and the
environment scores what the pose does to an object. That is the experiment the
paper names as not yet run, and the only one that can say whether modularity
buys a policy anything.
"""
