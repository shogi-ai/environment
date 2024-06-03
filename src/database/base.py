import json
import numpy as np
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Boolean, Float, Text, String
from src.game.ai_action import ActionTaken

Base = declarative_base()


class ShogiGameAction(Base):
    """
    SQLAlchemy model for storing actions taken in a Shogi game.

    Attributes:
        id (int): Primary key, auto-incremented.
        priority (int): Priority of the action.
        action (int): Action taken by the agent.
        reward (float): Reward received after taking the action.
        terminated (bool): Indicates if the game terminated after the action.
        truncated (bool): Indicates if the game was truncated after the action.
        current_state (str): Serialized numpy array representing the current state.
        current_moves (str): Serialized numpy array representing the current valid moves.
        next_state (str): Serialized numpy array representing the next state.
        next_moves (str): Serialized numpy array representing the next valid moves.
    """

    __tablename__ = "shogi_game_actions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    priority = Column(Integer)
    action = Column(Integer)
    reward = Column(Float)
    terminated = Column(Boolean)
    truncated = Column(Boolean)
    current_state = Column(Text)
    current_moves = Column(Text)
    next_state = Column(Text)
    next_moves = Column(Text)
    version = Column(String(255))

    def __init__(self, action_taken: ActionTaken, version: str = ""):
        """
        Initializes a ShogiGameAction instance.

        Args:
            action_taken (ActionTaken): An instance of ActionTaken containing the details of the action.
        """
        self.priority = action_taken.priority
        self.action = action_taken.action
        self.reward = action_taken.reward
        self.terminated = action_taken.terminated
        self.truncated = action_taken.truncated
        self.current_state = self.serialize_np_array(action_taken.current_state)
        self.current_moves = self.serialize_np_array(action_taken.current_moves)
        self.next_state = self.serialize_np_array(action_taken.next_state)
        self.next_moves = self.serialize_np_array(action_taken.next_moves)
        self.version = version

    @staticmethod
    def serialize_np_array(array: np.array) -> str:
        return json.dumps(array.tolist())

    @staticmethod
    def deserialize_np_array(array_str: str) -> np.array:
        return np.array(json.loads(array_str))

    def to_action(self) -> ActionTaken:
        action = ActionTaken(
            priority=self.priority,
            action=self.action,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            current_state=self.deserialize_np_array(self.current_state),
            current_moves=self.deserialize_np_array(self.current_moves),
            next_state=self.deserialize_np_array(self.next_state),
            next_moves=self.deserialize_np_array(self.next_moves),
        )
        return action
