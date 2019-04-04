module DetEnvAbstract
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

data EnvState = S deriving (Eq, Show)
data Action = A deriving (Show)
data Observation = O deriving (Eq, Show)

legalActs :: [Action]
legalActs = [A]

initialState :: EnvState
initialState = S

isGoalState :: EnvState -> Bool
isGoalState S = True

observe :: EnvState -> Observation
observe S = O

nextState :: EnvState -> Action -> EnvState
nextState S A = S
