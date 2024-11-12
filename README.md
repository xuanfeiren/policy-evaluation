# policy-evaluation
In factor.py, I try to randomly generate environments and save one seed (seed=2153) under which BRM has smaller approximation factor
In main.py, I compare sample-based losses using BRM or LSTD. In this environment, BRM performs better (see the attached graph)
Since I just generate transition matrix randomly so it is not a deterministic environment, so that may not be BRM, in fact it is LSTD not using projection, so called "MSBE loss"
seems in most randomly generated environments, LSTD performs better