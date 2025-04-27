import math
import pdb
import sys
import os
import numpy as np
import json
from envs.flood.utils import ObjectState as FloodObjectState
from envs.fire.fire_utils import ObjectState as FireObjectState
from tdw.add_ons.occupancy_map import OccupancyMap

# locate HAZARD root
PATH = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(PATH) != "HAZARD":
    PATH = os.path.dirname(PATH)
sys.path.append(PATH)
sys.path.append(os.path.join(PATH, "ppo"))

class OracleAgent:
    def __init__(self, task):
        self.task = task
        self.agent_speed = 1.0 / 62
        self.agent_type = "oracle"
        self.goal_objects = None
        self.objects_info = None
        self.controller = None
        self.map_list = []
        self.target_damaged_status_sequence = []
        self.target_position_sequence = []
        self.first_save = True
        self.agent_position = []
        self.step_limit = 0
        self.frame_bias = 0
        # pruning parameters
        self.global_best_value = 0
        self.max_unit_value = 5

    def reset(self, goal_objects, objects_info, controller, step_limit):
        self.goal_objects = goal_objects
        self.objects_info = objects_info
        self.controller = controller
        self.first_save = True
        self.step_limit = step_limit
        self.target_damaged_status_sequence = []
        self.target_position_sequence = []
        self.map_list = []
        self.step_limit = step_limit
        self.frame_bias = 0
        # reset pruning
        self.global_best_value = 0

    def find_path(self, agent_pos, target, start_step):
        meet = False
        additional_steps = 0
        while not meet:
            additional_steps += 1
            if start_step + additional_steps >= self.step_limit:
                return agent_pos, self.step_limit - start_step, 0
            target_id = self.controller.target_ids.index(target)
            cur_step = max(0, start_step + additional_steps - self.frame_bias)
            try:
                if target_id not in self.target_position_sequence[cur_step]:
                    return agent_pos, 0, 0
                target_position = self.target_position_sequence[cur_step][target_id]
            except Exception:
                pdb.set_trace()
            dx = agent_pos[0] - target_position[0]
            dz = agent_pos[2] - target_position[2]
            distance = math.sqrt(dx*dx + dz*dz)
            if additional_steps * self.agent_speed >= distance:
                meet = True
                agent_pos = target_position
                val_dict = json.load(open("data/meta_data/value.json"))
                name = self.controller.target_id2name[target]
                if name in val_dict and val_dict[name] == 1:
                    value = 5
                else:
                    value = 1
                if self.target_damaged_status_sequence[cur_step][target_id]:
                    value /= 2
        return agent_pos, additional_steps, value

    def search_step(self, search_order, agent_pos, step, value):
        # base: all done or reached limit
        if len(search_order) == len(self.controller.target_ids) or step >= self.step_limit:
            return step, search_order, value
        # pruning by optimistic bound
        remaining = len(self.controller.target_ids) - len(search_order)
        if value + remaining * self.max_unit_value <= self.global_best_value:
            return step, search_order, value
        min_step = float('inf')
        max_value = -1
        best_order = None
        # explore next target
        for idx in self.controller.target_ids:
            if idx in search_order:
                continue
            new_pos, d_step, d_value = self.find_path(agent_pos, idx, step)
            # if stepping beyond limit: record partial solution
            if step + d_step > self.step_limit:
                sub_order = search_order + [idx]
                sub_value = value + d_value
                sub_step = self.step_limit
            else:
                sub_step, sub_order, sub_value = self.search_step(
                    search_order + [idx], new_pos, step + d_step, value + d_value)
            # update global best
            if sub_value > self.global_best_value:
                self.global_best_value = sub_value
            # select best
            if sub_value > max_value or (sub_value == max_value and sub_step < min_step):
                max_value = sub_value
                min_step = sub_step
                best_order = sub_order
        # return best found
        if best_order is None:
            return step, search_order, value
        return min_step, best_order, max_value

    def search_plan(self):
        print(len(self.controller.target_ids))
        if len(self.controller.target_ids) > 11:
            return []
        self.frame_bias = self.step_limit - len(self.target_position_sequence)
        self.global_best_value = 0
        min_step, best_order, best_value = self.search_step([], self.agent_position, 0, 0)
        print("End search", min_step, best_order, best_value)
        if not best_order:
            return [("walk_to", idx) for idx in search_order]  # fallback to partial
        return [("walk_to", idx) for idx in best_order]

    def save_info(self):
        if self.first_save:
            self.agent_position = self.controller.agents[0].dynamic.transform.position
            self.first_save = False
        position = {}
        damaged = []
        for idx in self.controller.target_ids:
            obj = self.controller.manager.objects[idx]
            position[idx] = obj.position
            if self.task == "fire":
                damaged.append(obj.state in (FloodObjectState.FLOODED, FloodObjectState.FLOODED_FLOATING))
            elif self.task == "flood":
                damaged.append(obj.state in (FireObjectState.BURNING, FireObjectState.BURNT))
            else:
                damaged.append(False)
        self.target_damaged_status_sequence.append(damaged)
        self.target_position_sequence.append(position)

    def choose_target(self, state, processed_input):
        return "explore", None

if __name__ == "__main__":
    agent = OracleAgent("fire")
