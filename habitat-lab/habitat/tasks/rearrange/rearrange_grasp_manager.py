#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import magnum as mn
import numpy as np

from habitat.config.default import Config
from habitat.tasks.rearrange.utils import get_aabb
from habitat_sim.physics import (
    CollisionGroupHelper,
    CollisionGroups,
    ManagedRigidObject,
    RigidConstraintSettings,
)


class RearrangeGraspManager:
    """
    Manages the agent grasping onto rigid objects and the links of articulated objects.
    """

    def __init__(
        self,
        sim,
        config: Config,
    ) -> None:
        """Initialize a grasp manager for the simulator instance provided.

        :param config: The task's "SIMULATOR" subconfig node. Defines grasping parameters.
        """
        self._sim = sim
        self._snapped_obj_id: Optional[int] = None
        self._snapped_marker_id: Optional[str] = None
        self._snap_constraints: List[int] = []
        self._leave_info: Optional[Tuple[mn.Vector3, float]] = None
        self._config = config

    def reconfigure(self) -> None:
        """Removes any existing constraints managed by this structure.
        Called from _sim.reconfigure().
        """
        self._snap_constraints.clear()

    def reset(self) -> None:
        """Reset the grasp manager by re-defining the collision group and dropping any grasped object."""
        # Setup the collision groups. UserGroup7 is the held object group, it
        # can interact with anything except for the robot.
        CollisionGroupHelper.set_mask_for_group(
            CollisionGroups.UserGroup7, ~CollisionGroups.Robot
        )

        self.desnap(True)
        self._leave_info = None

    def is_violating_hold_constraint(self) -> bool:
        """
        Returns true if the object is too far away from the gripper, meaning
        the agent violated the hold constraint.
        """
        ee_pos = self._sim.robot.ee_transform.translation
        if self._snapped_obj_id is not None and (
            np.linalg.norm(ee_pos - self.snap_rigid_obj.translation)
            >= self._config.HOLD_THRESH
        ):
            return True
        if self._snapped_marker_id is not None:
            marker = self._sim.get_marker(self._snapped_marker_id)
            if (
                np.linalg.norm(ee_pos - marker.get_current_position())
                >= self._config.HOLD_THRESH
            ):
                return True

        return False

    @property
    def is_grasped(self) -> bool:
        """Returns whether or not an object or marker is currently grasped."""
        return (
            self._snapped_obj_id is not None
            or self._snapped_marker_id is not None
        )

    def update(self) -> None:
        """Reset the collision group of the grasped object if its distance to the end effector exceeds a threshold.

        Used to wait for a dropped object to clear the end effector's proximity before re-activating collisions between them.
        """
        if self._leave_info is not None:
            ee_pos = self._sim.robot.ee_transform.translation
            rigid_obj = self._leave_info[0]
            dist = np.linalg.norm(ee_pos - rigid_obj.translation)
            if dist >= self._leave_info[1]:
                rigid_obj.override_collision_group(CollisionGroups.Default)
                self._leave_info = None

    def desnap(self, force=False) -> None:
        """Removes any hold constraints currently active. Removes hold constraints for regular and articulated objects.

        :param force: If True, reset the collision group of the now released object immediately instead of waiting for its distance from the end effector to reach a threshold.
        """
        if len(self._snap_constraints) == 0:
            # No constraints to unsnap
            self._snapped_obj_id = None
            self._snapped_marker_id = None
            return

        if self._snapped_obj_id is not None:
            obj_bb = get_aabb(self.snap_idx, self._sim)
            if obj_bb is not None:
                if force:
                    self.snap_rigid_obj.override_collision_group(
                        CollisionGroups.Default
                    )
                else:
                    self._leave_info = (
                        self.snap_rigid_obj,
                        max(obj_bb.size_x(), obj_bb.size_y(), obj_bb.size_z()),
                    )

        for constraint_id in self._snap_constraints:
            self._sim.remove_rigid_constraint(constraint_id)
        self._snap_constraints = []

        self._snapped_obj_id = None
        self._snapped_marker_id = None
        self._sim.robot.close_gripper()

    @property
    def snap_idx(self) -> Optional[int]:
        """
        The index of the grasped RigidObject. None if nothing is being grasped.
        """
        return self._snapped_obj_id

    @property
    def snapped_marker_id(self) -> Optional[str]:
        """
        The name of the marker for the grasp. None if nothing is being grasped.
        """
        return self._snapped_marker_id

    @property
    def snap_rigid_obj(self) -> ManagedRigidObject:
        """The grasped object instance."""
        ret_obj = self._sim.get_rigid_object_manager().get_object_by_id(
            self._snapped_obj_id
        )
        if ret_obj is None:
            raise ValueError(
                f"Tried to get non-existence object from ID {self._snapped_obj_id}"
            )
        return ret_obj

    def snap_to_marker(self, marker_name: str) -> None:
        """
        Create a constraint between the end-effector and the marker on the
        articulated object to be grasped.

        :param marker_name: The name/id of the marker.
        """
        if marker_name == self._snapped_marker_id:
            return

        if len(self._snap_constraints) != 0:
            # We were already grabbing something else.
            raise ValueError(
                f"Tried snapping to {marker_name} when already snapped"
            )

        marker = self._sim.get_marker(marker_name)
        self._snap_constraints = [
            self.create_hold_constraint(
                mn.Vector3(0.0, 0.0, 0.0),
                mn.Vector3(*marker.offset_position),
                marker.ao_parent.object_id,
                marker.link_id,
            ),
        ]
        self._snapped_marker_id = marker_name
        self._sim.robot.open_gripper()

    def create_hold_constraint(
        self,
        pivot_in_link: mn.Vector3,
        pivot_in_obj: mn.Vector3,
        obj_id_b: int,
        link_id_b: Optional[int] = None,
    ) -> int:
        """Create a new rigid point-to-point (ball joint) constraint between the robot and an object.

        :param pivot_in_link: The origin of the constraint in end effector local space.
        :param pivot_in_obj: The origin of the constraint in object local space.
        :param obj_id_b: The id of the object to be constrained to the end effector.
        :param link_id_b: If the object is articulated, provide the link index for the constraint.

        :return: The id of the newly created constraint or -1 if failed.
        """
        c = RigidConstraintSettings()
        c.object_id_a = self._sim.robot.get_robot_sim_id()
        c.link_id_a = self._sim.robot.ee_link_id
        c.object_id_b = obj_id_b
        if link_id_b is not None:
            c.link_id_b = link_id_b
        c.pivot_a = pivot_in_link
        c.pivot_b = pivot_in_obj
        c.max_impulse = self._config.GRASP_IMPULSE
        return self._sim.create_rigid_constraint(c)

    def update_object_to_grasp(self) -> None:
        """
        Kinematically update held object to be within robot's grasp.
        """
        self.snap_rigid_obj.transformation = self._sim.robot.ee_transform

    def snap_to_obj(self, snap_obj_id: int, force: bool = True) -> None:
        """Attempt to grasp an object, snapping/constraining it to the robot's end effector with 3 ball-joint constraints forming a fixed frame.

        :param snap_obj_id: The id of the object to be constrained to the end effector.
        :param force: Will kinematically snap the object to the robot's end-effector, even if
            the object is already in the grasped state.
        """
        if snap_obj_id == self._snapped_obj_id:
            # Already grasping this object.
            return

        if len(self._snap_constraints) != 0:
            # We were already grabbing something else.
            raise ValueError(
                f"Tried snapping to {snap_obj_id} when already snapped to {self._snapped_obj_id}"
            )

        self._snapped_obj_id = snap_obj_id
        if force:
            # Set the transformation to be in the robot's hand already.
            self.update_object_to_grasp()

        # Set collision group to GraspedObject so that it doesn't collide
        # with the links of the robot.
        self.snap_rigid_obj.override_collision_group(
            CollisionGroups.UserGroup7
        )

        self._snap_constraints = [
            self.create_hold_constraint(
                mn.Vector3(0.1, 0, 0),
                mn.Vector3(0, 0, 0),
                self._snapped_obj_id,
            ),
            self.create_hold_constraint(
                mn.Vector3(0.0, 0, 0),
                mn.Vector3(-0.1, 0, 0),
                self._snapped_obj_id,
            ),
            self.create_hold_constraint(
                mn.Vector3(0.1, 0.0, 0.1),
                mn.Vector3(0.0, 0.0, 0.1),
                self._snapped_obj_id,
            ),
        ]

        self._sim.robot.open_gripper()

        if any((x == -1 for x in self._snap_constraints)):
            raise ValueError("Created bad constraint")
