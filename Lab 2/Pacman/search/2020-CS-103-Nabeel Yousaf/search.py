# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from math import sqrt
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    
    if not problem.isGoalState(problem.getStartState()):
        startNode = problem.getStartState()     # starting poit of the node
        explored = []       # list of the nodes, that have been explored 
        priorityQueue = util.PriorityQueue()
        actualNode = startNode,[],0     # creating a special node called actual node
        priorityQueue.push(actualNode,0)

        while not priorityQueue.isEmpty():

            newNode = priorityQueue.pop()
            currentNode, pathArray , p_cost = newNode 
            if currentNode in explored:
                continue
            else:
                explored.append(currentNode)
                if not problem.isGoalState(currentNode):
                      
                    for childNode in problem.getSuccessors(currentNode):           #add succsessors in P QUeue 

                        successorNode, path, cost = childNode
                        newCost = p_cost + cost
                        # print(newCost)
                        newPath = pathArray + [path]
                        # print(newPath)
                        
                        newChildNode = successorNode,newPath,newCost

                        priorityQueue.push(newChildNode,newCost)
                else:
                    return pathArray # return path if current node is goal

        util.raiseNotDefined()
    else:
        return [] # if start state is also goal state then we will return empty list

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    Goal = problem.goal
    dx = abs(state[0] - Goal[0])
    dy = abs(state[1] - Goal[1])

    D = 1
    D2 = sqrt(2)

    return (D * (dx+dy) + (D2 -2 * D ) * min(dx,dy))

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    if not problem.isGoalState(problem.getStartState()):

        startingNode = problem.getStartState()
        visitedNodes = []
        priorityQueue = util.PriorityQueue()
        priorityQueue.push((startingNode, [], 0), 0)

        while not priorityQueue.isEmpty():

            currentNode, actions, oldCost = priorityQueue.pop()

            if currentNode not in visitedNodes:
                visitedNodes.append(currentNode)
                if problem.isGoalState(currentNode):

                    return actions

                for nextNode, action, cost in problem.getSuccessors(currentNode):
                    nextAction = actions + [action]
                    newCostToNode = oldCost + cost
                    heuristicCost = newCostToNode + heuristic(nextNode, problem)
                    priorityQueue.push((nextNode, nextAction, newCostToNode), heuristicCost)
    else:
        return []

    util.raiseNotDefined()
    
def aStarSearchV2(problem: SearchProblem, heuristic=nullHeuristic):

    startingNode = problem.getStartState()
    if not problem.isGoalState(startingNode):
        visitedNodes = []
        priorityQueue = util.PriorityQueue()
        priorityQueue.push((startingNode, [], 0), 0)

        while not priorityQueue.isEmpty():
            currentNode, actions, oldCost = priorityQueue.pop()
            if currentNode not in visitedNodes:
                visitedNodes.append(currentNode)
                if problem.isGoalState(currentNode):
                    return actions
                for nextNode, action, cost in problem.getSuccessors(currentNode):
                    nextAction = actions + [action]
                    newCostToNode = oldCost + cost
                    heuristicCost = newCostToNode + heuristic(nextNode, problem)
                    priorityQueue.push((nextNode, nextAction, newCostToNode), heuristicCost)
    else:
        return []

    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
