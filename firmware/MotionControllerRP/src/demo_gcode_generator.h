// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

#include <string>
#include "utilities/math3d.h"

//*** CLASS *****************************************************************************

class Robot;


class DemoGcodeGenerator {
  public:
    DemoGcodeGenerator(Robot* robot);  
    void run();
    
  private:
    void robot_update();
    void run_command(const std::string& cmd);

    void move_to(Vec3F p, float feedrate);
    void move_to(float x, float y, float z, float feedrate);
    void set_acceleration(int accel);
    void dwell(int time_ms);
  
  private:
    void random_motion(int count, float range, float feedrade, int dwell_ms=-1, int acceleration=-1);
    void circle_motion(int count, float radius, float feedrade);
    void grid_motion(int w, int h, float size, float feedrate, int dwell_ms=-1, int acceleration=-1);

  private:
    int cmd_counter = 0;
    QuaternionF wst;
    Robot* robot;
};
