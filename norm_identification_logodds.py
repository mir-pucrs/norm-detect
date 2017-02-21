from thinkbayes import Suite
from operator import concat
from collections import defaultdict
from priorityqueue import PriorityQueue
from math import log
from planlib import Action, Goal, planned, flattened_plan_tree


# def flattened_plan_tree(goal_or_plan, prefix=[]):
#   if isinstance(goal_or_plan, Goal):
#     plan_list = goal_or_plan.plans
#     return reduce(concat, (flattened_plan_tree(plan, prefix) for plan in plan_list), [])
#   elif isinstance(goal_or_plan, Action):
#     return [prefix + [goal_or_plan]]
#   elif isinstance(goal_or_plan, list): # A plan (list of goals)
#     if len(goal_or_plan) == 1:
#       return flattened_plan_tree(goal_or_plan[0], prefix)
#     elif len(goal_or_plan) == 2:
#       # First goal must be satisfied by an atomic plan, and so results in a singleton flattened plan list
#       [ new_prefix ] = flattened_plan_tree(goal_or_plan[0], prefix)
#       return flattened_plan_tree(goal_or_plan[1], new_prefix)
#     else:
#       raise ValueError("flattened_plan_tree called with incorrect plan length (not 1 or 2): %s" % goal_or_plan)
#   else:
#     raise TypeError("flattened_plan_tree called with argument of wrong type: %s" % goal_or_plan)

# compute path (list of nodes) for a plan (list of Action terms)
def plan_path(plan):
  paths = [ plan[0].path[0] ]
  for action in plan:
    paths.extend(action.path[1:])
  return paths

def is_sub_list(list1, list2):
  return any(list1 == list2[i:i+len(list1)] for i in range(len(list2)-len(list1)+1))

# NormSuite represents a set of mutually exclusive norm hypotheses, where norms can be
# ('forbidden', NodeName), ('obliged', NodeName), or None.
# Given a goal and a set of actions, it produces and store a list of plans (each being
# a list of actions that achieve the goal.
# The hypotheses can be provided as a list (if the prior probability distribution is
# uniform, or as a dict mapping hypotheses to their probabilities.
# The Update method (inherited from thinkbayes.Suite) updates the probability distribution
# given some observed data (a path of travel in a graph), and that calls the LogLikelihood
# method below (for each hypothesis) to compute the log likelihood of the data given the
# hypothesis.

class NormSuite(Suite):
  def __init__(self, inferred_goal, hypotheses, actions, \
               prob_non_compliance=0.1, prob_viol_detection=0.5, \
               prob_sanctioning=0.2, prob_random_punishment=0.01, \
               name=''):
    Suite.__init__(self, hypotheses, name)
    self.initial_hypotheses = hypotheses # Keep the initial hypotheses to allow later reinitilization
    self.inferred_goal = inferred_goal
    self.actions = actions
    self.nodes = { node for action in actions for node in action.path }
    self.prob_non_compliance = prob_non_compliance
    self.prob_viol_detection = prob_viol_detection
    self.prob_sanctioning = prob_sanctioning
    self.prob_random_punishment = prob_random_punishment
    self.SetGoal(inferred_goal)

  # Override Normalize method to do nothing, as we are working with odds ratios
  def Normalize(self, fraction=1.0):
    pass

  def SetGoal(self, goal):
    self.inferred_goal = goal
    if planned(goal, self.actions, set([])):
      self.plans = flattened_plan_tree(goal)
      self.plan_paths = [ plan_path(plan) for plan in self.plans ]
    else:
      raise ValueError("Error: Failed to find any plans for goal %s given actions %s" % (goal, self.actions))

  # Override UpdateSet to apply both sanction-based and plan recognition odds updates
  # to a collection of observations
  def UpdateSet(self, dataset):
    """Updates each hypothesis based on the dataset.

    dataset: a sequence of data
    """
    for data in dataset:
      self.UpdateOddsRatioVsNoNorm(data)
      
  def UpdateOddsRatioVsNoNorm(self, obs):
    """Updates each hypothesis based on the data.

    obs: the observed path

    """
    sanction_indices = set([])
    obs_without_sanctions = []
    for index, item in enumerate(obs):
      if item == "!":
        sanction_indices.add(index - len(sanction_indices))
      else:
        obs_without_sanctions += item
    no_norm_log_likelihood_using_sanctions = self.LogLikelihoodUsingSanctions(obs_without_sanctions, sanction_indices, None)
    try:
      no_norm_log_likelihood_using_plans = self.LogLikelihoodUsingPlans(obs_without_sanctions, None)
    except ValueError as ve:
      print "Skipping odds update using plans: %s" % ve
    for hypo in self.Values():
      if hypo != None:
        hypo_log_likelihood = self.LogLikelihoodUsingSanctions(obs_without_sanctions, sanction_indices, hypo)
        self.Incr(hypo, hypo_log_likelihood - no_norm_log_likelihood_using_sanctions)
        try:
          hypo_log_likelihood = self.LogLikelihoodUsingPlans(obs_without_sanctions, hypo)
          self.Incr(hypo, hypo_log_likelihood - no_norm_log_likelihood_using_plans)
        except ValueError:
          pass

  def LogLikelihoodUsingSanctions(self, obs_path, sanction_indices, hypothesis):
    # sanction_indices is a list of indices in obs_path for actions that were followed by a sanction or random punishment
    # The sanctions/punishments have been removed from obs_path
    #print "\nHypothesis: ", hypothesis
    if hypothesis == None:
      log_likelihood = ( (len(obs_path)-len(sanction_indices))*log(1-self.prob_random_punishment)
                         + len(sanction_indices)*log(self.prob_random_punishment) )
      #print "Returning log likelihood: ", log_likelihood
      return log_likelihood
    # Norms have form (context_node, modality, node) or (modality, node)
    elif isinstance(hypothesis, tuple) and len(hypothesis) in [2,3] and all(isinstance(x, str) or x==True for x in hypothesis):

      # Define function violation_indices, dependent on norm
      # TODO: Remove seemingly unused parameter 'path'
      def violation_indices(path):
        pass # dummy definition is overwritten below
      if len(hypothesis) == 2:
        (modality, node) = hypothesis
        if modality == "eventually": # Interpreted as meaning eventually <node> within the plan"
          violation_indices = lambda path: {len(path)-1} if all(item!=node for item in path) else set([])
        elif modality == "never":
	  # Assumption: A "never" norm may be sanctioned multiple times on separate breaches (an agent may miss earlier breaches)
          violation_indices = lambda path: {index for index,item in enumerate(path) if item==node}
        else:
          raise ValueError("Invalid modality in hypothesis %s" % hypothesis)
      elif len(hypothesis) == 3:
        (context_node, modality, node) = hypothesis
        if modality == "next":  # Interpreted as only applying if there *is* a next node after the context node
          violation_indices = \
            lambda path: {i+1 for i in range(len(path)-1) if path[i]==context_node and path[i+1]!=node}
        elif modality == "not next":  # Interpreted as only applying if there *is* a next node after the context node
          violation_indices = \
            lambda path: {i+1 for i in range(len(path)-1) if path[i]==context_node and path[i+1]==node}
        elif modality == "eventually":
          # Interpreted as meaning "after the context state (if there is a next state), eventually <node> within the plan".
          # The "after the current state" is for the case when context_node == node
          def pred(path):
            last_context_index = len(path) # off end of list
            # Loop code adapted from http://stackoverflow.com/a/9836681
            for index, item in enumerate(reversed(path[:-1])): # ignore last element of path: context not relevant there
              if item == context_node:
                last_context_index = len(path)-index-2
                break
            if last_context_index < len(path)-1 and all(item!=node for item in path[last_context_index+1:]):
              #print "violation_indices = %s" % {len(path)-1}
              return {len(path)-1}
            else:
              #print "violation_indices = %s" % set([])
              return set([])
          violation_indices = pred
        elif modality == "never": # If we allow context_node == node then this means "next never"
          def pred(path):
            first_context_index = len(path) # off end of list
            for index, item in enumerate(path[:-1]): # ignore last element of path: context not relevant therex
              if item == context_node:
                first_context_index = index
                break
            #print "First context index is %s" % first_context_index
            #print "violation_indices = %s" % {index for index,item in enumerate(path[first_context_index+1:]) if item==node}
            return {index+first_context_index+1 for index,item in enumerate(path[first_context_index+1:]) if item==node}
          violation_indices = pred
        else:
          raise ValueError("Invalid modality in hypothesis %s" % hypothesis)

      # Calculate log likelihood
      # Assumption: punishment is immediate (relaxing this in the presence of multiple violations
      # gives us a bi-partite mathing problem)
      # Assumption: there is only one sanction possible (or recorded) after each action
      viol_indices = violation_indices(obs_path)
      log_likelihood = 0  # initial log value for a product of probabilities
      for i in range(len(obs_path)): # compute 'and' of log likelihoods for each node in obs_path
        #print "Index %s" % i
        if i in viol_indices:
          #print "Violation!"
          if i+1 in sanction_indices:
            #print "Punished!"
            log_likelihood += log(self.prob_random_punishment +  # Either a random punishment ... 
                                  # or it's not explained by random punishment, so it's a sanction
                                  (1-self.prob_random_punishment) * self.prob_viol_detection * self.prob_sanctioning)
          else:
            #print "Not punished"
            log_likelihood += ( log(1-self.prob_viol_detection*self.prob_sanctioning) +  # Not sanctioned, and ...
                                log(1-self.prob_random_punishment) ) # no random punishment

        else:
          #print "No violation"
          if i+1 in sanction_indices:
            #print "Punished"
            # Random punishment
            log_likelihood += log(self.prob_random_punishment)
          else:
            #print "Not punished"
            # No random punishment
            log_likelihood += log(1-self.prob_random_punishment)
      #print "Returning log likelihood %s" % log_likelihood
      return log_likelihood
    else:
      raise ValueError("Invalid hypothesis passed to LogLikelihood function: %s" % (hypothesis))

  def LogLikelihoodUsingPlans(self, obs_path, hypothesis):
    # Assumption: The *number* of times that a plan breaches a (conditional) norm is not relevant
    #print "\nHypothesis: ", hypothesis
    num_plans = float(len(self.plans)) # make it a float to ensure non-truncating division later
    assert num_plans > 0  # An exception is raised by NormSuite constructor in this case
    plan_paths_containing_obs = filter(lambda path: is_sub_list(obs_path, path), self.plan_paths)
    num_plans_containing_obs = len(plan_paths_containing_obs)
    if num_plans_containing_obs == 0:
      raise ValueError("Assumption violated: path %s cannot be generated for %s given plans %s" \
                       % (obs_path, self.inferred_goal, self.plans))
    # Calculate probability of seeing obs_path if there is no norm (assume all plans are equally likely)
    prob_obs_if_no_norm = num_plans_containing_obs / num_plans
    #print "Prob. obs if no norm: ", prob_obs_if_no_norm
    # consider hypothesised norm
    if hypothesis == None:
      #print "Returning log likelihood: ", log(prob_obs_if_no_norm)
      return log(prob_obs_if_no_norm)
    # Norms have form (context_node, modality, node) or (modality, node)
    elif isinstance(hypothesis, tuple) and len(hypothesis) in [2,3] and all(isinstance(x, str) or x==True for x in hypothesis):

      # Define function violation_indices, dependent on norm
      def is_norm_breaching(path):
        pass # dummy definition is overwritten below
      if len(hypothesis) == 2:
        (modality, node) = hypothesis
        if modality == "eventually": # Interpreted as meaning eventually <node> within the plan"
          is_norm_breaching = lambda path: all(item!=node for item in path)
        elif modality == "never":
          is_norm_breaching = lambda path: any(item==node for item in path)
        else:
          raise ValueError("Invalid modality in hypothesis %s" % hypothesis)
      elif len(hypothesis) == 3:
        (context_node, modality, node) = hypothesis
        if modality == "next":  # Interpreted as only applying if there *is* a next node after the context node
          is_norm_breaching = \
            lambda path: any((i!=len(path)-1 and path[i+1]!=node) \
                              for i in range(len(path)) if path[i]==context_node)
        elif modality == "not next":  # Interpreted as only applying if there *is* a next node after the context node
          is_norm_breaching = \
            lambda path: any((i!=len(path)-1 and path[i+1]==node) \
                              for i in range(len(path)) if path[i]==context_node)
        elif modality == "eventually": # Interpreted as meaning "after the context state (if there is a next state), eventually <node> within the plan". The "after the current state" is for the case when context_node == node
          def pred(path):
            last_context_index = len(path) # off end of list
            # Loop code adapted from http://stackoverflow.com/a/9836681
            for index, item in enumerate(reversed(path[:-1])): # ignore last element of path: context not relevant there
              if item == context_node:
                last_context_index = len(path)-index-2
                break
            #print "Last context index for path %s is %s" % (path, last_context_index)
            return last_context_index < len(path)-1 and all(item!=node for item in path[last_context_index+1:])
          is_norm_breaching = pred
        elif modality == "never": # If we allow context_node == node then this means "next never"
          def pred(path):
            first_context_index = len(path) # off end of list
            for index, item in enumerate(path[:-1]): # ignore last element of path: context not relevant therex
              if item == context_node:
                first_context_index = index
                break
            #print "First context index for path %s is %s" % (path, first_context_index)
            #print "is_norm_breaching = %s" % any(item==node for item in path[first_context_index+1:])
            return any(item==node for item in path[first_context_index+1:])
          is_norm_breaching = pred
        else:
          raise ValueError("Invalid modality in hypothesis %s" % hypothesis)

      # Calculate likelihood
      norm_breaching_plans = filter(is_norm_breaching, self.plan_paths)
      num_non_norm_breaching_plans = num_plans - len(norm_breaching_plans)
      if num_non_norm_breaching_plans == 0:
        log_likelihood = log(self.prob_non_compliance) + log(prob_obs_if_no_norm)
      else:
        norm_breaching_plans_containing_obs = filter(is_norm_breaching, plan_paths_containing_obs)
        num_non_norm_breaching_plans_containing_obs = num_plans_containing_obs - len(norm_breaching_plans_containing_obs)
        log_likelihood = log((1-self.prob_non_compliance) * num_non_norm_breaching_plans_containing_obs/num_non_norm_breaching_plans
                             + self.prob_non_compliance * prob_obs_if_no_norm)
      #print "Returning log likelihood ", log_likelihood
      return log_likelihood
    else:
      raise ValueError("Invalid hypothesis passed to LogLikelihood function: %s" % (hypothesis))

  def most_probable_norms(self, topN):
    """Computes the topN most probable norms within a suite, returning either a single norm 
       (if there is a unique most likely norm) or all norms tied at the top"""
    pq = PriorityQueue()
    for n in self.d:
        prob = self.d[n]
        pq.add_task(n, prob)
    #endfor
    sorted_norms = pq.sorted_queue()
    norms = [x for x in reversed(sorted_norms)]
    
    # Check that we select either the topN, or the first ones with the same odds
    for (i,n) in enumerate(norms):
        if(i+1 == topN):
            tied = len(norms) > i+1 and (self.d[n] == self.d[norms[i+1]])
            if (tied): 
                topN += 1
#                 print "Tie between norms %s and %s with prob=%d, topN is now %d" % (n,norms[i-1],self.d[n],topN)                         else: 
                break
    #endfor
    return (norms[0:topN],topN)

  def print_ordered(self):
    pq = PriorityQueue()
    for n in self.d:
        prob = self.d[n]
        pq.add_task(n, prob)
    #endfor
    sorted = pq.sorted_queue()
    norms = [x for x in reversed(sorted)]
    for n in norms:
        print n, self.d[n]

def test():
  goal = Goal('a','d')
  actions = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
  observation1 = ['a','c','e','d']
  observation2 = ['a','b','d','!']
  nodes = { node for action in actions for node in action.path }
  successors = defaultdict(set)
  for action in actions:
    for i, node in enumerate(action.path[0:-1]):
      successors[node].add(action.path[i+1])
  conditional_norms = [ (context, modality, node) for context in nodes for node in nodes \
                                                  for modality in 'eventually', 'never' ]
  conditional_norms += [ (context, modality, node) for context in nodes for node in nodes \
                                                   for modality in 'next', 'not next' \
                                                   if node in successors[context] ]
  unconditional_norms = [ (modality, node) for node in nodes for modality in 'eventually', 'never' ]
  poss_norms = unconditional_norms + conditional_norms

  hypotheses = dict.fromkeys(poss_norms, log(0.05)) # Prior for log likelihood of odds vs. no norm
  hypotheses[None] = 0 # Set prior log odds ratio for hypothesis None

  print "Goal: ", goal
  print "Actions: ", actions
  print "Norm hypotheses (with prior odds ratios): ", hypotheses

  suite = NormSuite(goal, hypotheses, actions)
  print "Plans: ", suite.plan_paths
  
  print "Updating odds ratios after observing ", observation1
  suite.UpdateOddsRatioVsNoNorm(observation1)

  print "The posterior odds ratios are:"
  suite.Print()

  print "Updating odds ratios after observing ", observation2
  suite.UpdateOddsRatioVsNoNorm(observation2)

  print "The posterior log odds are:"
  suite.Print()

  f = open('graph.dot','w')
  f.write('digraph {\n')
  for a in suite.actions:
      for i in range(len(a.path)-1):
          f.write(a.path[i]+' -> '+a.path[i+1]+'\n')
  f.write("}\n")
  f.close()
