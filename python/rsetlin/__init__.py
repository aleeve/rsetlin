from .rsetlin import TsetlinMachine


def trainingloop(data, targets):
    tm = TsetlinMachine(4)
    for d, t in zip(data, targets):
        tm.fit(d, t)
    return tm
