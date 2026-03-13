// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include <LittleFS.h> 
#include "robot_joint.h"
#include "hw_config.h"
#include "utilities/logging.h"
#include "utilities/utilities.h"
#include "servo_control/homing_controller.h"
#include "servo_control/actuator_calibration.h"

//*** FUNCTION **************************************************************************

//*** CLASS *****************************************************************************

//--- RobotJoint ------------------------------------------------------------------------

RobotJoint::RobotJoint(MT6835Encoder* encoder, 
                       TB6612MotorDriver* motor_driver,
                       int pole_pairs) 
{
  RobotJoint::encoder = encoder;
  RobotJoint::motor_driver = motor_driver;
  servo_controller = new ServoController(*motor_driver, *encoder, pole_pairs);
  position = 0.0f;
  velocity = 0.0f;
}

RobotJoint::~RobotJoint() {
  delete servo_controller;
  delete motor_driver;
  delete encoder;
  servo_controller = nullptr;
  motor_driver = nullptr;
  encoder = nullptr;
}

void RobotJoint::init(int joint_idx) {
  RobotJoint::joint_idx = joint_idx;

  if(encoder->init(0x5, 0x4) == false) {
    LOG_ERROR("Failed to initialize encoder for joint %i", joint_idx);
  }

  servo_controller->init(MOTOR_MAX_CURRENT_FACTOR);
  servo_controller->set_motor_enabled(false, false);
}

bool RobotJoint::calibrate(bool print_measurements) {
  LOG_INFO("Joint-%i: calibrating joint...", joint_idx);

  HomingController homing_controller;
  bool homing_ok = homing_controller.run_blocking(servo_controller, -HOMING_VELOCITY, 
                                                  360.0f*DEG_TO_RAD, HOMING_CURRENT,
                                                  ENCODER_ANGLE_TO_ROTOR_ANGLE,
                                                  0.0f);
  if(homing_ok == false) {
    LOG_ERROR("Joint-%i: Calibration failed due to unsuccessful homing sequence", joint_idx);
    return false;
  }

  // measure lookup tables
  LookupTable encoder_raw_to_motor_pos_lut;
  LookupTable motor_pos_to_field_angle_lut;
  bool ok = measure_calibration_data(encoder_raw_to_motor_pos_lut, 
                                     motor_pos_to_field_angle_lut, 
                                     *servo_controller, 
                                     CALIBRATION_RANGE*DEG_TO_RAD,
                                     CALIBRATION_FIELD_VELOCITY, 
                                     256,
                                     print_measurements);
  if(!ok) {
    LOG_ERROR("Joint-%i: calibrating failed", joint_idx);
    return false;
  }

  servo_controller->set_enc_to_pos_lut(encoder_raw_to_motor_pos_lut);
  servo_controller->set_pos_to_field_lut(motor_pos_to_field_angle_lut);
  is_calibrated = true;
  is_homed = true;

  LOG_INFO("Joint-%i: calibrating joint successful.", joint_idx);

  return true;
}

void RobotJoint::update(float dt, float one_over_dt) {
  servo_controller->update(position, dt, one_over_dt);
}

void RobotJoint::update_target(float p, float v) {
  position = p;
  velocity = v;
}

bool RobotJoint::load_calibration() {
  std::string fn1 = calib_data_filename("enc_to_pos_lut").c_str();
  std::string fn2 = calib_data_filename("pos_to_field_lut").c_str();
  if(!LittleFS.exists(fn1.c_str()) || !LittleFS.exists(fn2.c_str())) {
    LOG_WARNING("Joint-%i: Not all calibration files found. Run joint calibration with M56.", joint_idx);
    return false;
  }

  LookupTable enc_to_pos_lut;
  LookupTable pos_to_field_lut;
  bool res = true; 
  res &= load_lut_from_file(enc_to_pos_lut, fn1.c_str());
  res &= load_lut_from_file(pos_to_field_lut, fn2.c_str());
  if(res == false)
    return false;

  servo_controller->set_enc_to_pos_lut(enc_to_pos_lut);
  servo_controller->set_pos_to_field_lut(pos_to_field_lut);

  is_calibrated = true;
  LOG_INFO("Joint-%i: Encoder lookup tables loaded (size=%i,%i)", 
           joint_idx, enc_to_pos_lut.size(), pos_to_field_lut.size());

  return true;
}

bool RobotJoint::store_calibration() {
  bool res = true;

  res &= save_lut_to_file(servo_controller->get_enc_to_pos_lut(), 
                          calib_data_filename("enc_to_pos_lut").c_str());
  res &= save_lut_to_file(servo_controller->get_pos_to_field_lut(), 
                          calib_data_filename("pos_to_field_lut").c_str());

  return res;
}

std::string RobotJoint::calib_data_filename(std::string data_name) const {
  return std::string("joint")+std::to_string(joint_idx)+"_"+data_name+".dat";
}
