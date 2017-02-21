from norm_detector import norm_detector
from norm_identification_logodds import NormSuite, log, defaultdict
import copy
from norm_behaviour import NormBehaviour

class bayesian_norm_detector(norm_detector):
    """Implementation of the norm_detector interface to wrap around the NormSuite class from Stephen to facilitate testing"""
    def __init__(self, planlib, goal, prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01, prior=log(0.5), prior_none = log(1)):
        """ Assigns variables for this class, I'm assuming here that planlib is a set of Actions (from the planlib module)"""
        super(bayesian_norm_detector,self).__init__(planlib)
        self.past_observations = []
        self.suite = self.build_norm_suite(planlib, goal, prob_non_compliance, prob_viol_detection, prob_sanctioning, prob_random_punishment, prior, prior_none)
     
    def build_norm_suite(self, planlib, goal, prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01, prior=log(0.5), prior_none = log(1)):
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
        
        newsuite = NormSuite(goal, hypotheses, planlib, prob_non_compliance, prob_viol_detection, prob_sanctioning, prob_random_punishment)
        return newsuite
        
    
    def reinitialise(self):
        """Reinitialises the norm detector to a state in which no observations 
           have been made. Implementations may have parameters"""
        prior_none = self.suite.initial_hypotheses[None]
        prior = None
        for hyp_key in self.suite.initial_hypotheses.iterkeys():
            if (hyp_key != None):
                prior = self.suite.initial_hypotheses[hyp_key]
                break
         
        assert(prior != None)
        self.suite = self.build_norm_suite(self.suite.actions, self.suite.inferred_goal, self.suite.prob_non_compliance, self.suite.prob_viol_detection, self.suite.prob_sanctioning, self.suite.prob_random_punishment, prior, prior_none)
    
    def update_with_observations(self,observation):
        """Updates the norm detector with the sequence of actions in /observations/"""
        self.past_observations+=observation
        self.suite.UpdateOddsRatioVsNoNorm(observation)
    
    def set_goal(self, goal):
        """Updates the goal assumed by the norm detector"""
        self.suite.SetGoal(goal)
    
    def get_goal(self):
        return self.suite.inferred_goal
    
    def get_inferred_norms(self, topNorms=1):
        return self.suite.most_probable_norms(topNorms)[0]
    
    def get_norm_hypotheses(self):
        """Returns a set of the possible norms considered by the detector (mostly for debugging)"""
        nodes = { node for action in self.planlib for node in action.path }
        successors = defaultdict(set)
        for action in self.planlib:
            for i, node in enumerate(action.path[0:-1]):
                successors[node].add(action.path[i+1])
        conditional_norms = [ (context, modality, node) for context in nodes for node in nodes \
                                                      for modality in 'eventually', 'never' ]
        conditional_norms += [ (context, modality, node) for context in nodes for node in nodes \
                                                       for modality in 'next', 'not next' \
                                                       if node in successors[context] ]
        
        unconditional_norms = [ (modality, node) for node in nodes for modality in 'eventually', 'never' ]
        hypotheses = unconditional_norms + conditional_norms
        return hypotheses
    
    def separate_norms(self,norms):
        """ Separate /norms/ into each individual type of norm"""
        nxt = set([])
        not_next = set([])
        eventually = set([])
        never = set([])
        ## Get rid of the None norm in the set if there is one
        nnorms = set(norms)
        nnorms.discard(None)
        for (context,modality,node) in nnorms:
            if(modality == 'next'):
                nxt.add((context,modality,node))
            elif(modality == 'not next'):
                not_next.add((context,modality,node))
            elif(modality == 'eventually'):
                eventually.add((context,modality,node))
            elif(modality == 'never'):
                never.add((context,modality,node))
            else:
                print "!! Invalid norm %s !!" % (modality)
        return (nxt,not_next,eventually,never)

    def count_violations(self, plan,norms):
        """Counts the number of violations of /norms/ in /plan/ and returns a list with the indexes of the violations"""
        nb = NormBehaviour()
        return nb.count_violations(plan, norms)
            

    
