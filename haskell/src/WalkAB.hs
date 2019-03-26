module WalkAB
  (
    EnvState,
    Action,
    Observation,
    legalActs,
    initialState,
    isGoalState,
    observe,
    nextState
  ) where

data EnvState = S Int Bool
data Action = ActLeft | ActRight
data Observation = ObsA | ObsB | ObsMiddle

legalActs = [ActLeft, ActRight]

initialState :: EnvState
initialState = S 0 False

isGoalState :: EnvState -> Bool
isGoalState (S 0 True) = True
isGoalState _ = False

observe :: EnvState -> Observation
observe (S 0 _) = ObsA
observe (S 3 _) = ObsB
observe (S _ _) = ObsNeither

nextState :: EnvState -> Action -> EnvState -- [(EnvState, Double)]
nextState (S 0 b) ActLeft = S 0 b
nextState (S 3 _) ActRight = S 3 True
nextState (S 2 _) ActRight = S 3 True
nextState (S n b) ActLeft = S (n-1) b
nextState (S n b) ActRight = S (n+1) b
