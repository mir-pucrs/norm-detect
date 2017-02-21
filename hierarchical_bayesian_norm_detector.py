from bayesian_norm_detector import bayesian_norm_detector
from hierarchical_norm_identification_logodds import HierarchicalNormSuite, log, defaultdict
import copy
from norm_behaviour import NormBehaviour

class hierarchical_bayesian_norm_detector(bayesian_norm_detector):
    """Implementation of the norm_detector interface to wrap around the HierarchicalNormSuite class from Stephen to facilitate testing"""
    def __init__(self, planlib, goal, prob_non_compliance=0.1, prob_viol_detection=0.99,
                 prob_sanctioning=0.99, prob_random_punishment=0.01, prior=log(0.5), prior_none = log(1)):
        """ Assigns variables for this class, I'm assuming here that planlib is a set of Actions (from the planlib module)"""
        super(hierarchical_bayesian_norm_detector,self).__init__(planlib, goal, prob_non_compliance,
                    prob_viol_detection, prob_sanctioning, prob_random_punishment, prior, prior_none)

    def build_norm_suite(self, planlib, goal, prob_non_compliance=0.1, prob_viol_detection=0.99,
                         prob_sanctioning=0.99, prob_random_punishment=0.01, prior=log(0.5), prior_none=log(1)):
        nodes = { node for action in planlib for node in action.path }
        successors = defaultdict(set)
        for action in planlib:
            for i, node in enumerate(action.path[0:-1]):
                successors[node].add(action.path[i+1])
        conditional_norms = [ (context, modality, node) for context in nodes for node in nodes \
                                                      for modality in 'eventually', 'never' ]
        conditional_norms += [ (context, modality, node) for context in nodes for node in nodes \
                                                       for modality in 'next', 'not next' \
                                                       if node in successors[context] ]
        
        unconditional_norms = [ (modality, node) for node in nodes for modality in 'eventually', 'never' ]
        poss_norms = unconditional_norms + conditional_norms
        
        # hypotheses = dict.fromkeys(poss_norms, 0.05)
        # hypotheses[None] = 1 # Set prior odds ration for hypothesis None
        # hypotheses = dict.fromkeys(poss_norms, log(0.05))
        # hypotheses[None] = log(1) # Set prior odds ration for hypothesis None
        # hypotheses = dict.fromkeys(poss_norms, log(1))
        # hypotheses[None] = log(1) # Set prior odds ration for hypothesis None
        hypotheses = dict.fromkeys(poss_norms, prior)
        hypotheses[None] = prior_none # Set prior odds ration for hypothesis None
        
        newsuite = HierarchicalNormSuite(goal, hypotheses, planlib, prob_non_compliance, prob_viol_detection, prob_sanctioning, prob_random_punishment)
        return newsuite
