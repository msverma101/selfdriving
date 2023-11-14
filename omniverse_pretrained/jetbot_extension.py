# Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from omni.isaac.examples.base_sample import BaseSampleExtension
from omni.isaac.examples.user_examples import Jetbot
import omni.ui as ui
import asyncio
import omni
from omni.isaac.ui.ui_utils import btn_builder

import omni.ext
from omni.isaac.ui.ui_utils import setup_ui_headers, get_style, btn_builder, scrolling_frame_builder




class JetbotExtenstion(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="Jetbot",
            title="Jetbot",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_core_hello_world.html",
            overview="This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
            sample = Jetbot(model_path=rf"C:\users\ai_admin\appdata\local\ov\pkg\isaac_sim-2022.2.1\exts\omni.isaac.examples\omni\isaac\examples\user_examples\best_steering_model_xy.pth",
                lego_asset_path = "C:\Users\AI_Admin\Downloads\jetbot\Jebot\usd\legoblue.usd"),
            file_path=os.path.abspath(__file__),
            number_of_extra_frames=1
        )
        self.task_ui_elements = {}
        frame = self.get_frame(index=0)
        self.build_task_controls_ui(frame)
        
        return

    def _on_start_party_button_event(self):
        asyncio.ensure_future(self.sample._on_start_party_event_async())
        self.task_ui_elements["lane follow"].enabled = False
        return

    def post_reset_button_event(self):
        self.task_ui_elements["lane follow"].enabled = True
        return

    def post_load_button_event(self):
        self.task_ui_elements["lane follow"].enabled = True
        return

    def post_clear_button_event(self):
        self.task_ui_elements["lane follow"].enabled = False
        return
    
    def _on_follow_target_button_event(self, val):
        asyncio.ensure_future(self.sample._on_follow_target_event_async(val))
        return
        
    def build_task_controls_ui(self, frame):
        with frame:
            with ui.VStack(spacing=5):
                # Update the Frame Title
                frame.title = "Task_to_do"
                frame.visible = True
                dict = {
                    "label": "lane follow",
                    "type": "button",
                    "text": "lane follow",
                    "tooltip": "lane follow",
                    "on_clicked_fn": self._on_start_party_button_event,
                }

                self.task_ui_elements["lane follow"] = btn_builder(**dict)
                self.task_ui_elements["lane follow"].enabled = False

                dict = {
                            "label": "Reset",
                            "type": "button",
                            "text": "Reset",
                            "tooltip": "Reset robot and environment",
                            "on_clicked_fn": self._on_reset,
                        }
                self.task_ui_elements["Reset"] = btn_builder(**dict)
                self.task_ui_elements["Reset"].enabled = False
    
                # dict =  {
                #     "label": "Move Robot",
                #     "type": "button",
                #     "text": "Move",
                #     "tooltip": "Move the robot Forward",
                #     "on_clicked_fn": lambda: self.move_robot(self._rc, 3, 3),
                # }
