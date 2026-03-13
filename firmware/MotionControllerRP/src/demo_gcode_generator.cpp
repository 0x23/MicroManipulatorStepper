// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include <string>
#include "demo_gcode_generator.h"
#include "robot.h"

//*** FUCNTION **************************************************************************

Vec3F rand_vec3d(float range) {
  auto r = [](float s) {
      return ((float)rand() / RAND_MAX) * 2*s - s;
  };

  return Vec3F(r(range), r(range), r(range));
}

//*** CLASS *****************************************************************************

DemoGcodeGenerator::DemoGcodeGenerator(Robot* robot) {
  DemoGcodeGenerator::cmd_counter = 0;
  DemoGcodeGenerator::robot = robot;

  wst = QuaternionF::from_axis_angle(Vec3F(-0.707106781f, 0.707106781f, 0.0f), 0.955316618);
}

void DemoGcodeGenerator::run() {
  run_command("G28");
  run_command("G0 X0 Y0 Z0 F100");
  run_command("M204 L200");

  while(true) {
    move_to(0.5, 0, 0, 10); dwell(1000);

    circle_motion(10, 0.5, 50);
    // grid_motion(5, 5, 3, 2000, 150, 5000);
    grid_motion(6, 6, 4, 2000, 150, 5000);

    random_motion(20, 7, 1000, 100, 200);

    // grid_motion(2, 2, 8, 1000, 150, 2000);

    move_to(0, 0, 0, 100); dwell(1000);
    circle_motion(2, 10, 20);
    circle_motion(3, 10, 500);
    set_acceleration(500);
    move_to(0, 0, 0, 100); dwell(1000);

    grid_motion(5, 5, 16, 1000, 150, 2000);
    random_motion(20, 7, 1000, 200, 2000);


    dwell(1000);
    for(int i=0; i<4; i++)
      grid_motion(2, 2, 16, 1000, 100, 1000);

    set_acceleration(2000);
    for(int i=0; i<3; i++) {
      move_to(0, 0, -12, 500);
      move_to(0, 0, 12, 500);
    }

    // end
    move_to(0, 0, -14, 10);
    run_command("G4 S5");
  }
};

void DemoGcodeGenerator::robot_update() {
  // run robot update loop until command parse can accept a new command
  do { 
    robot->update_command_parser();
    robot->update_path_planner();
  } while(robot->get_command_parser()->is_command_ready());
}

void DemoGcodeGenerator::run_command(const std::string& cmd) {
  // perform robot update, this will block until command parser can accept next command
  robot_update();
  robot->get_command_parser()->parse_line(cmd.c_str());
  LOG_INFO("DEMO: %s", cmd.c_str());
}

void DemoGcodeGenerator::move_to(Vec3F p, float feedrate) {
  char b[64];
  p = wst.rotate(p);
  std::snprintf(b, sizeof(b), "G0 X%.3f Y%.3f Z%.3f F%.3f", p.x, p.y, p.z, feedrate);
  run_command(b);
}

void DemoGcodeGenerator::move_to(float x, float y, float z, float feedrate) {
  move_to(Vec3F(x, y, z), feedrate);
}

void DemoGcodeGenerator::set_acceleration(int accel) {
  if(accel>0)
    run_command(std::string("M204 L")+std::to_string(accel));
}

void DemoGcodeGenerator::dwell(int time_ms) {
  if(time_ms>0)
    run_command(std::string("G4 P")+std::to_string(time_ms));
}

void DemoGcodeGenerator::random_motion(int count, float range, float feedrade, int dwell_ms, int acceleration) {
  set_acceleration(acceleration);

  for(int i=0; i<count; i++) {
    move_to(rand_vec3d(range), 1000);
    dwell(dwell_ms);
  }
} 

void DemoGcodeGenerator::circle_motion(int count, float radius, float feedrade) {
  Vec3F eu(radius,0,0);
  Vec3F ev(0,radius,0);
  
  move_to(eu, feedrade);
  set_acceleration(4000);

  for(int i=0; i<count; i++) {
    for(float t=0; t<Constants::TWO_PI_F; t+=0.03) {
      move_to(eu*cos(t) + ev*sin(t), feedrade);
    }
  }
}

void DemoGcodeGenerator::grid_motion(int w, int h, float size, float feedrate, int dwell_ms, int acceleration) {
  Vec3F eu(1,0,0);
  Vec3F ev(0,1,0);

  set_acceleration(acceleration);

  float uo = -size*0.5f;
  float vo = -size*0.5f;
  float us = size/(w-1);
  float vs = size/(h-1);

  for(int v=0; v<h; v++) {
    for(int u=0; u<w; u++) {
      float d = v%2 == 0 ? -1.0f : 1.0f;
      move_to(eu*((float(u)*us+uo)*d) + ev*(float(v)*vs+vo) + Vec3F(0, 0, 0), feedrate);
      dwell(dwell_ms);
    }
  }
}

