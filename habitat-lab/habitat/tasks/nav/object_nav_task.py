# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

try:
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass

task_cat2mpcat40 = [
    3,  # ('chair', 2, 0)
    5,  # ('table', 4, 1)
    6,  # ('picture', 5, 2)
    7,  # ('cabinet', 6, 3)
    8,  # ('cushion', 7, 4)
    10,  # ('sofa', 9, 5),
    11,  # ('bed', 10, 6)
    13,  # ('chest_of_drawers', 12, 7),
    14,  # ('plant', 13, 8)
    15,  # ('sink', 14, 9)
    18,  # ('toilet', 17, 10),
    19,  # ('stool', 18, 11),
    20,  # ('towel', 19, 12)
    22,  # ('tv_monitor', 21, 13)
    23,  # ('shower', 22, 14)
    25,  # ('bathtub', 24, 15)
    26,  # ('counter', 25, 16),
    27,  # ('fireplace', 26, 17),
    33,  # ('gym_equipment', 32, 18),
    34,  # ('seating', 33, 19),
    38,  # ('clothes', 37, 20),
    43,  # ('foodstuff', 42, 21),
    44,  # ('stationery', 43, 22),
    45,  # ('fruit', 44, 23),
    46,  # ('plaything', 45, 24),
    47,  # ('hand_tool', 46, 25),
    48,  # ('game_equipment', 47, 26),
    49,  # ('kitchenware', 48, 27)
]

mapping_mpcat40_to_goal21 = {
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    14: 9,
    15: 10,
    18: 11,
    19: 12,
    20: 13,
    22: 14,
    23: 15,
    25: 16,
    26: 17,
    27: 18,
    33: 19,
    34: 20,
    38: 21,
    43: 22,  #  ('foodstuff', 42, task_cat: 21)
    44: 28,  #  ('stationery', 43, task_cat: 22)
    45: 26,  #  ('fruit', 44, task_cat: 23)
    46: 25,  #  ('plaything', 45, task_cat: 24)
    47: 24,  # ('hand_tool', 46, task_cat: 25)
    48: 23,  # ('game_equipment', 47, task_cat: 26)
    49: 27,  # ('kitchenware', 48, task_cat: 27)
}

@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_name_id: Optional[int] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.GOAL_SPEC_MAX_VAL - 1
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )


@registry.register_task(name="ObjectNav-v1")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
