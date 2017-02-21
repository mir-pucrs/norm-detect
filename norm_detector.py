from planlib import plan_library_to_dot
# TODO Finish this interface and create implementations for all possible algorithms
class norm_detector(object):
    """An interface for a generic norm detection algorithm"""
    
    def __init__(self,planlib):
#         raise NotImplementedError( "Should have implemented this" )
        self.planlib = planlib
    
    def reinitialise(self):
        """Reinitialises the norm detector to a state in which no observations 
           have been made. Implementations may have parameters"""
        raise NotImplementedError( "Should have implemented this" )
        pass
    
    def update_with_observations(self,observation):
        """Updates the norm detector with the sequence of actions in /observations/"""
        raise NotImplementedError( "Should have implemented this" )
        pass
    
    def set_goal(self, goal):
        """Updates the goal assumed by the norm detector"""
        raise NotImplementedError( "Should have implemented this" )
        pass
    
    def get_goal(self):
        """Returns the goal assumed by the norm detector"""
        raise NotImplementedError( "Should have implemented this" )
        pass
    
    
    def get_inferred_norms(self,topNorms=1):
        """Returns the top norms"""
        raise NotImplementedError( "Should have implemented this" )
        pass
    
    def get_norm_hypotheses(self):
        """Returns a set of the possible norms considered by the detector (mostly for debugging)"""
        raise NotImplementedError( "Should have implemented this" )
        pass
    
    def all_possible_states(self,planlib):
        """Helper method to enumerate all states in a plan library """
        nodes = { node for action in planlib for path in action.path for node in path }
        return nodes;
    
    # TODO Decide whether to move this to norm_behaviour
    def alternative_plans(self,plan,start,goal,planlib):
        """Returns all alternative plans to /plan/ that go from /start/ to /goal/ using /planlib/"""
        Pi = []
        for pn in self.generate_all_plans(planlib, goal, start):
            if(pn != plan):
                Pi.append(pn)
         
        return Pi

    def separate_norms(self,norms):
        """ Separate /norms/ into each individual type of norm"""
        raise NotImplementedError( "Should have implemented this" )
        pass
    
    def count_violations(self, plan, norms):
        """Counts the number of violations of /norms/ in /plan/ and returns a list with the indexes of the violations"""
        raise NotImplementedError( "Should have implemented this" )
        pass

    def generate_all_plans(self, planlib, goal, node, prefix=[]):
        """Generates all plans for a plan library - check replication in planlib"""
        if(prefix == []):
            prefix = [node]
        if(node == goal):
            yield prefix
        else:
            for a in self.next_actions(planlib,node):
                plan = prefix[:]
                plan = plan + a.path[1:]
                subnode = a.path[-1]
                for p in self.generate_all_plans(planlib, goal, subnode, plan):
                    yield p

    def next_actions(self,planlib,node):
        """Returns a list of next actions given a current node"""
        next_actions =[]
        for a in planlib:
            if(a.path[0] == node):
                next_actions[len(next_actions):] = [a]
            
        return next_actions
    
    def print_plan_library(self,filename="graph.dot"):
        plan_library_to_dot(self.planlib,filename)