// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

#include "utilities/logging.h"
#include "utilities/frequency_counter.h"
#include "hardware/MT6701_encoder.h"
#include "hardware/MT6835_encoder.h"
#include "hardware/TB6612_motor_driver.h"
#include "servo_control/servo_controller.h"
#include "utilities/lookup_table.h"
#include "utilities/math_constants.h"

#include "motion_control/path_planner.h"
#include "motion_control/motion_controller.h"
#include "command_parser/command_parser.h"
#include "robot_joint_interface.h"

//*** CLASS *****************************************************************************

//--- RobotJoint ------------------------------------------------------------------------

// TODO: create abstract interface for RobotJoint (to support remote Joints)
//       hide direct HW access 

class RobotJoint {
  public:
    RobotJoint(MT6835Encoder* encoder, TB6612MotorDriver* motor_driver, int pole_pairs);
    ~RobotJoint();

    void init(int joint_idx);
    bool calibrate(bool print_measurements);
    void update(float dt, float one_over_dt);
    void update_target(float p, float v);
    bool load_calibration();
    bool store_calibration();


  private:
    std::string calib_data_filename(std::string data_name) const;

  public:
    int joint_idx = 0;
    bool is_homed = false;
    bool is_calibrated = false;

    float position;
    float velocity;

    MT6835Encoder* encoder;
    ServoController* servo_controller;

  private:
    TB6612MotorDriver* motor_driver;
};
