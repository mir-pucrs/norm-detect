from thinkbayes import Suite
from operator import concat
from planlib import Action, Goal, planned, flattened_plan_tree

# compute path (list of nodes) for a plan (list of Action terms)
def plan_path(plan):
  return reduce(lambda prev_action_path, action: \
                  concat(prev_action_path[:-1], action.path), plan, [])

def is_sub_list(list1, list2):
  return any(list1 == list2[i:i+len(list1)] for i in range(len(list2)-len(list1)+1))

# NormSuite represents a set of mutually exclusive norm hypotheses, where norms can be
# ('forbidden', NodeName), ('obliged', NodeName), or None.
# Given a goal and a set of actions, it produces and store a list of plans (each being
# a list of actions that achieve the goal.
# The hypotheses can be provided as a list (if the prior probability distribution is
# uniform, or as a dict mapping hypotheses to their probabilities.
# The Update method (inherited from thinkbayes.Suite) updates the probability distribution
# given some observed data (a path of travel in a graph), and that calls the Likelihood
# method below (for each hypothesis) to compute the likelihood of the data given the
# hypothesis.

class NormSuite(Suite):
  def __init__(self, inferred_goal, hypotheses, actions, prob_non_compliance=0.1, name=''):
    Suite.__init__(self, hypotheses, name)
    self.inferred_goal = inferred_goal
    self.actions = actions
    self.prob_non_compliance = prob_non_compliance
    if planned(inferred_goal, actions):
      self.plans = flattened_plan_tree(inferred_goal)
    else:
      print "Error: Failed to find any plans for %s given actions %s" % (inferred_goal, actions)

  # Override Normalize method to do nothing, as we are working with odds ratios
  def Normalize(self, fraction=1.0):
    pass

  def UpdateOddsRatioVsNoNorm(self, data):
    """Updates each hypothesis based on the data.

    data: any representation of the data

    returns: the normalizing constant
    """
    no_norm_likelihood = self.Likelihood(data, None)
    for hypo in self.Values():
      if hypo != None:
        hypo_likelihood = self.Likelihood(data, hypo)
        self.Mult(hypo, hypo_likelihood/no_norm_likelihood)

  def Likelihood(self, path, hypothesis):
    n_plans = float(len(self.plans)) # make a float to ensure non-truncating division later
    plans_containing_path = filter(lambda plan: is_sub_list(path, plan_path(plan)), self.plans)
    n_matching_plans = len(plans_containing_path)
    if n_matching_plans == 0:
      raise ValueError("Assumption violated: path %s cannot be generated for %s given plans %s" \
                       % (path, self.inferred_goal, self.plans))
    # Calculate probability of seeing path if there is no norm (assume all plans are equally likely)
    prob_path_if_no_norm = n_matching_plans / n_plans
    # consider hypothesised norm
    if hypothesis == None:
      return prob_path_if_no_norm
    elif isinstance(hypothesis, tuple) and len(hypothesis) == 2:
      (norm_type, node) = hypothesis
      if norm_type == "forbidden":
        norm_abiding_plans = \
          filter(lambda plan: not any(node in action.path for action in plan), self.plans)
        norm_abiding_plans_containing_path = \
          filter(lambda plan: not any(node in action.path for action in plan), plans_containing_path)
      elif norm_type == "obliged":
        norm_abiding_plans = \
          filter(lambda plan: any(node in action.path for action in plan), self.plans)
        norm_abiding_plans_containing_path = \
          filter(lambda plan: any(node in action.path for action in plan), plans_containing_path)
      else:
        raise ValueError("Likelihood only handles norm types 'forbidden', 'obliged' and None (not %s)" \
                         % norm_type)
      if len(norm_abiding_plans) == 0:
        prob_compliance_and_norm_abiding_path = 0
      else:
        prob_path_via_norm_abiding_plans = len(norm_abiding_plans_containing_path) / len(norm_abiding_plans)
        prob_compliance_and_norm_abiding_path = (1-self.prob_non_compliance)*prob_path_via_norm_abiding_plans
      return self.prob_non_compliance*prob_path_if_no_norm + prob_compliance_and_norm_abiding_path
    else:
      raise ValueError("Invalid hypothesis passed to Likelihood function: %s" % hypothesis)
    
goal = Goal('a','d')
actions = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
observation = ['a','c','e','d']

nodes = { node for action in actions for path in action.path for node in path }
poss_norms = [norm for node in nodes for norm in ("forbidden", node), ("obliged", node)]
hypotheses = dict.fromkeys(poss_norms, 0.1)
hypotheses[None] = 1 # Set prior odds ration for hypothesis None

print "Goal: ", goal
print "Actions: ", actions
print "Norm hypotheses (with prior odds ratios): ", hypotheses

suite = NormSuite(goal, hypotheses, actions)

print "Updating odds rations after observing ", observation

suite.UpdateOddsRatioVsNoNorm(observation)

print "The posterior odds ratios are:"

suite.Print()

f = open('graph.dot','w')
f.write('digraph {\n')
for a in suite.actions:
    for i in range(len(a.path)-1):
        f.write(a.path[i]+' -> '+a.path[i+1]+'\n')
f.write("}\n")
f.close()
