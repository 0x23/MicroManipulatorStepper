// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

//*** INCLUDE ***************************************************************************

#include <cstdint>

#include "path_segment.h"

//*** CLASS *****************************************************************************

class PathPlanner;

//--- MotionController ------------------------------------------------------------------

class MotionController {
  public:
    MotionController(PathPlanner* path_planner);

    // updates the motion controller and computes new joint positions and velocities
    // after dt has passed. Ouput array must hav space for 'NUM_JOINTS' entries.
    bool update(float dt, float* joint_positions, float* joint_velocities, float* tool_outputs);

    // debug: number of joint space buffer underruns during motion. incremented from
    // the 2kHz ISR, read from the main loop (never do serial i/o in the ISR).
    uint32_t get_underrun_count() const { return underrun_count; }

  private:
    PathPlanner* path_planner;

    float current_time;
    JointSpacePathSegment current_path_segment;
    bool current_segment_finished;
    volatile uint32_t underrun_count = 0;
};
