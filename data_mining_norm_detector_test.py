from data_mining_norm_detector import data_mining_norm_detector
from planlib import Action, Goal

goal = Goal('a','d')
planlib = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
observation1 = ['a','c','!','e','d']
observation2 = ['a','b','d','!']
observation3 = ['a','b','e','d','!']
dmnt = data_mining_norm_detector(planlib, Goal('a','d'))
dmnt.reinitialise()
dmnt.update_with_observations(observation1)
dmnt.update_with_observations(observation2)
dmnt.update_with_observations(observation3)
numTopNorms = 3
norms = dmnt.get_inferred_norms(numTopNorms)
print "Inferred norms: ", norms
n = len(norms)
if n != numTopNorms:
    print "Asked for", numTopNorms, "norms"
    print n, "norms returned due to ties or zero probabilities"
    
