// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

//*** INCLUDE ***************************************************************************

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

  private:
    PathPlanner* path_planner;

    float current_time;
    JointSpacePathSegment current_path_segment;
    bool current_segment_finished;
};
