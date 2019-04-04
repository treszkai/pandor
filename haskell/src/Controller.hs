module Controller
  (
    Controller,
    ContTransition,
    ContState(..),
    emptyController,
    numContStates
  ) where

import DetEnvAbstract

newtype ContState = Q Int deriving (Eq, Show)
-- data Controller = C [((ContState, Observation), (ContState, Action))]
type ContTransition = ((ContState, Observation), (ContState, Action))
type Controller = [ContTransition]

emptyController :: Controller
emptyController = []

contStateId :: ContState -> Int
contStateId (Q n) = n

numContStates :: Controller -> Int
numContStates = maximum . (map (contStateId . fst . snd))
