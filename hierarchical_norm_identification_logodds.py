# pylint: disable=line-too-long, bad-indentation, bad-whitespace, bad-builtin, invalid-name
from norm_identification_logodds import NormSuite, plan_path, is_sub_list
from operator import concat
from collections import defaultdict
from priorityqueue import PriorityQueue
from math import log, exp, sqrt
from planlib import Action, Goal, planned, flattened_plan_tree
import numpy as np

# Used in online EM algorithm
def gamma(n):
    return n**(-0.6)

class HierarchicalNormSuite(NormSuite):
  def __init__(self, inferred_goal, hypotheses, actions,
               prob_non_compliance=0.1, prob_viol_detection=0.5, prob_sanctioning=0.2, prob_random_punishment=0.01,
               name='',
               num_agents=1, minibatch_size=1, num_M_steps_to_skip=5):
    NormSuite.__init__(self, inferred_goal, hypotheses, actions,
               prob_non_compliance, prob_viol_detection,
               prob_sanctioning, prob_random_punishment,
               name)
    self.num_agents = num_agents
    self.minibatch_size = minibatch_size
    self.signal_data = []
    self.p_viol_opt_data = []
    self.num_M_steps_to_skip = num_M_steps_to_skip
    self.num_M_steps_skipped = 0
    self.z1 = 0  # Not used initially (multiplied by gamma=0), then overwritten
    self.z2 = 0  # Not used initially (multiplied by gamma=0), then overwritten
    self.num_observations_seen = 0
    self.num_norm_breaching_plans = None

  # Overridden
  def SetGoal(self, goal):
    super(HierarchicalNormSuite, self).SetGoal(goal)
    self.num_norm_breaching_plans = {}
    for hypothesis in self.Values():
      norm_breaching_pred = self.norm_breaching_pred(hypothesis)
      norm_breaching_plans = filter(norm_breaching_pred, self.plan_paths)
      self.num_norm_breaching_plans[hypothesis] = len(norm_breaching_plans)
    print "Goal: %s" % goal
    # print "Non-zero Violating ratios: %s" % { hypo:ratio for hypo,ratio in self.violating_plan_ratios.iteritems() if ratio > 0 }

  def update_p_non_comp(self, any_signals, p_viol_opt):
    # Online EM algorithm based on https://hal.archives-ouvertes.fr/hal-00532968/document
    self.signal_data.append(any_signals)
    self.p_viol_opt_data.append(p_viol_opt)
    if len(self.signal_data) < self.minibatch_size:
        return
    p_act_obs = self.prob_viol_detection # Are these quantities the same?
    p_sanc = self.prob_sanctioning
    p_pun = self.prob_random_punishment
    p_non_comp = self.prob_non_compliance
    chunk = np.array(self.signal_data)
    self.signal_data = []
    p_viol_opt_chunk = np.array(self.p_viol_opt_data)
    self.p_viol_opt_data = []

    mu1 = p_viol_opt_chunk * p_act_obs * p_sanc # array
    mu2 = p_act_obs * p_pun # singleton
    pos_factor1 = mu1**chunk
    neg_factor1 = (1.0-mu1)**(1-chunk)
    pos_factor2 = mu2**chunk
    neg_factor2 = (1.0-mu2)**(1-chunk)
    num1 = p_non_comp * pos_factor1 * neg_factor1
    num2 = (1.0 - p_non_comp) * pos_factor2 * neg_factor2
    denom = num1 + num2
    z1_new = num1/denom
    z2_new = num2/denom
    # print z1_new, z2_new
    gamma_val = gamma(self.num_observations_seen)
    self.z1 = (1 - gamma_val) * self.z1 + gamma_val * (np.sum(z1_new)/self.minibatch_size)
    self.z2 = (1 - gamma_val) * self.z2 + gamma_val * (np.sum(z2_new)/self.minibatch_size)
    # print z1,z2
    # np.mean(z1, dtype=np.float64)
    # p2 = (1 - gamma_val) * num2 + gamma_val * np.mean(z2)
    # print 'z1: {}'.format(z1)
    # print 'z2: {}'.format(z2)
    # M step
    if self.num_M_steps_skipped < self.num_M_steps_to_skip:
        self.num_M_steps_skipped += 1
    else:
        self.prob_non_compliance = self.z1
        assert self.prob_non_compliance < 1

    print 'New prob_non_compliance: {}'.format(self.prob_non_compliance)

  def violation_indices_func(self, hypothesis):
    """Return a function that takes an observation list as an argument
       and returns a generator for the indices in the observation at
       which hypothesis (a possible norm) is violated"""
    if hypothesis is None:
      return lambda(path): iter([])  # No norm so no violations
    elif len(hypothesis) == 2:
      (modality, node) = hypothesis
      if modality == "eventually":  # Interpreted as meaning eventually <node> within the plan"
        return lambda path: {len(path)-1} if all(item != node for item in path) else set([])
      elif modality == "never":
          # Assumption: A "never" norm may be sanctioned multiple times on separate breaches (an agent may miss earlier breaches)
        return lambda path: {index for index, item in enumerate(path) if item == node}
      else:
        raise ValueError("Invalid modality in hypothesis %s" % hypothesis)
    elif len(hypothesis) == 3:
      (context_node, modality, node) = hypothesis
      if modality == "next":  # Interpreted as only applying if there *is* a next node after the context node
        return \
          lambda path: {i+1 for i in range(len(path)-1) if path[i]==context_node and path[i+1]!=node}
      elif modality == "not next":  # Interpreted as only applying if there *is* a next node after the context node
        return \
          lambda path: {i+1 for i in range(len(path)-1) if path[i]==context_node and path[i+1]==node}
      elif modality == "eventually":
        # Interpreted as meaning "after the context state (if there is a next state), eventually <node> within the plan".
        # The "after the current state" is for the case when context_node == node
        def pred(path):
          last_context_index = len(path)  # off end of list
          # Loop code adapted from http://stackoverflow.com/a/9836681
          for index, item in enumerate(reversed(path[:-1])):  # ignore last element of path: context not relevant there
            if item == context_node:
              last_context_index = len(path)-index-2
              break
          if last_context_index < len(path) - 1 and all(item != node for item in path[last_context_index + 1 : ]):
            # print "violation_indices = %s" % {len(path)-1}
            return {len(path)-1}
          else:
            # print "violation_indices = %s" % set([])
            return set([])
        return pred
      elif modality == "never":  # If we allow context_node == node then this means "next never"
        def pred(path):
          first_context_index = len(path)  # off end of list
          for index, item in enumerate(path[:-1]):  # ignore last element of path: context not relevant therex
            if item == context_node:
              first_context_index = index
              break
          #print "First context index is %s" % first_context_index
          #print "violation_indices = %s" % {index for index,item in enumerate(path[first_context_index+1:]) if item==node}
          return {index+first_context_index+1 for index,item in enumerate(path[first_context_index+1:]) if item==node}
        return pred
      else:
        raise ValueError("Invalid modality in hypothesis %s" % hypothesis)

  def norm_breaching_pred(self, hypothesis):
    """Returns a Boolean-valued function that can be applied to an observation
       to determine whether it breaches the given norm hypothesis. This function
       depends on the form of the hypothesis"""
    if hypothesis is None:
      return lambda path: False # No norm so it can never be breached
    elif len(hypothesis) == 2:
      (modality, node) = hypothesis
      if modality == "eventually": # Interpreted as meaning eventually <node> within the plan"
        return lambda path: all(item!=node for item in path)
      elif modality == "never":
        return lambda path: any(item==node for item in path)
      else:
        raise ValueError("Invalid modality in hypothesis %s" % hypothesis)
    elif len(hypothesis) == 3:
      (context_node, modality, node) = hypothesis
      if modality == "next":  # Interpreted as only applying if there *is* a next node after the context node
        return \
          lambda path: any((i!=len(path)-1 and path[i+1]!=node)
                            for i in range(len(path)) if path[i]==context_node)
      elif modality == "not next":  # Interpreted as only applying if there *is* a next node after the context node
        return \
          lambda path: any((i!=len(path)-1 and path[i+1]==node)
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
        return pred
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
        return pred
      else:
        raise ValueError("Invalid modality in hypothesis %s" % hypothesis)

  def UpdateOddsRatioVsNoNorm(self, obs):
    """Updates each hypothesis based on the data.

    obs: the observed path

    """
    self.num_observations_seen += 1 # Used in EM algorithm
    sanction_indices = set([])
    obs_without_sanctions = []
    for index, item in enumerate(obs):
      if item == "!":
        sanction_indices.add(index - len(sanction_indices))
      else:
        obs_without_sanctions += item

    if self.num_agents != 1:
      raise ValueError("UpdateOddsRatioVsNoNorm only support num_agents=1")
    # Note: the next line will need an array if there is more than one observation
    # observed_agent_data = 0 # Agent ID of our (assumed one and only) agent is 0
    num_hypotheses = len(self.d)
    p_norms = np.empty([num_hypotheses], dtype=np.dtype(np.float64))
    p_viol_opt_data = np.empty([num_hypotheses], dtype=np.dtype(np.float64))

    print "Plan paths: %s" % self.plan_paths
    print "Observation: %s" % obs_without_sanctions

    for i,(hypothesis,logodds) in enumerate(self.d.iteritems()):
      p_norms[i] = exp(logodds) # Odds of hypothesis, transformed from log space
      # Ratio of plans that violate hypothesized norm for inferred goal:
      plan_paths_containing_obs = filter(lambda path: is_sub_list(obs_without_sanctions, path), self.plan_paths)
      num_plans_containing_obs = len(plan_paths_containing_obs)
      norm_breaching_pred = self.norm_breaching_pred(hypothesis)
      norm_breaching_plans_containing_obs = filter(norm_breaching_pred, plan_paths_containing_obs)
      num_norm_breaching_plans_containing_obs = len(norm_breaching_plans_containing_obs)
      p_viol_opt_data[i] = num_norm_breaching_plans_containing_obs / num_plans_containing_obs

    # Calculate p_viol_opt
    p_norms = np.array(p_norms)
    p_norms /= p_norms.sum() # Vectorized division by a scalar
    p_viol_opt = p_norms.dot(p_viol_opt_data)

    # Compute overall observation signal value
    any_signals = bool(sanction_indices) # True if sanction_indices not empty

    self.update_p_non_comp(any_signals, p_viol_opt)

    # Update norm odds using self.prob_non_compliance and other parameters
    no_norm_log_likelihood_using_sanctions = self.LogLikelihoodUsingSanctions(obs_without_sanctions, sanction_indices, None)
    try:
      no_norm_log_likelihood_using_plans = self.LogLikelihoodUsingPlans(obs_without_sanctions, None)
    except ValueError as ve:
      print "Skipping odds update using plan recognition: %s" % ve
    for hypo in self.Values():
      if hypo != None:
        hypo_log_likelihood = self.LogLikelihoodUsingSanctions(obs_without_sanctions, sanction_indices, hypo)
        self.Incr(hypo, hypo_log_likelihood - no_norm_log_likelihood_using_sanctions)
        try:
          hypo_log_likelihood = self.LogLikelihoodUsingPlans(obs_without_sanctions, hypo)
          self.Incr(hypo, hypo_log_likelihood - no_norm_log_likelihood_using_plans)
        except ValueError:
          pass

  # TO DO: Consider whether to use self.prob_non_compliance here (estimated using plans)
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
      # Calculate log likelihood
      # Assumption: punishment is immediate (relaxing this in the presence of multiple violations
      # gives us a bi-partite mathing problem)
      # Assumption: there is only one sanction possible (or recorded) after each action
      violation_indices_func = self.violation_indices_func(hypothesis)
      viol_indices_iter = violation_indices_func(obs_path)
      log_likelihood = 0  # initial log value for a product of probabilities
      for i in range(len(obs_path)): # compute 'and' of log likelihoods for each node in obs_path
        #print "Index %s" % i
        if i in viol_indices_iter:
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
      # Calculate likelihood
      num_norm_breaching_plans = self.num_norm_breaching_plans[hypothesis]
      num_non_norm_breaching_plans = num_plans - num_norm_breaching_plans
      if num_non_norm_breaching_plans == 0:
        log_likelihood = log(self.prob_non_compliance) + log(prob_obs_if_no_norm)
      else:
        norm_breaching_pred = self.norm_breaching_pred(hypothesis)
        norm_breaching_plans_containing_obs = filter(norm_breaching_pred, plan_paths_containing_obs)
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
        if i+1 == topN:
            tied = len(norms) > i+1 and (self.d[n] == self.d[norms[i+1]])
            if tied:
                topN += 1
#                 print "Tie between norms %s and %s with prob=%d, topN is now %d" % (n,norms[i-1],self.d[n],topN) else:
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
