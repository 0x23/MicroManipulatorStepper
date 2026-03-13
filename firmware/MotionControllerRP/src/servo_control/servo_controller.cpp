// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include "hardware/timer.h"
#include "pico/stdlib.h"

#include "servo_controller.h"
#include "utilities/logging.h"
#include "utilities/math_constants.h"

#include "hw_config.h"

#include <algorithm>

ServoController::ServoController(
  MOTOR_DRIVER_TYPE& motor_driver, 
  ENCODER_TYPE& encoder, 
  int32_t motor_pole_pair_count) :
    motor_driver(motor_driver),
    encoder(encoder)
{
  ServoController::motor_pole_pair_count = motor_pole_pair_count;
  ServoController::motor_update_enabled = false;
  ServoController::encoder_update_enabled = true;
  ServoController::motor_pos = 0.0f;
  ServoController::pos_error = 0.0f;

  // set default encoder lut
  using namespace Constants;
  float magnet_array_radius = 30.0f; // mm
  float magnet_pitch = 3.0f;         // mm
  float g = float(encoder.get_rawcounts_per_rev())*(TWO_PI_F*magnet_array_radius/magnet_pitch)*0.5f;
  build_linear_lut(encoder_raw_to_motor_pos_lut, -g, g, -TWO_PI_F, TWO_PI_F);
}

void ServoController::init(float max_motor_amplitude) {
  ServoController::motor_current_amplitude = max_motor_amplitude;

  // setup motor driver
  motor_driver.begin();
  motor_driver.set_amplitude(0.0f, true); // correct amplitude will be set by 'set_motor_enabled()' 
  motor_driver.enable();
  motor_driver.set_field_angle(0.0f);

  velocity_lowpass.set_time_constant(VEL_LOWPASS_TC);

  pos_controller.set_parameter(POS_KP, POS_KI, 0.0f, Constants::PI_F*2.0F, Constants::PI_F*0.5F);
  velocity_controller.set_parameter(VEL_KP, VEL_KI, 0.0f, Constants::PI_F*0.45f, Constants::PI_F*0.45f);

  // large 0.9° steppers
//  pos_controller.set_parameter(75.0f, 50000.0f, 0.0f, Constants::PI_F*2.0F, Constants::PI_F*0.5F);
//  velocity_controller.set_parameter(0.2f, 150.0f, 0.0f, Constants::PI_F*0.45f, Constants::PI_F*0.45f);

 // pos_controller.set_parameter(75.0f, 2000.0f, 0.0f, Constants::PI_F*2.0F, Constants::PI_F*0.5F);
 // velocity_controller.set_parameter(0.2f, 0.0f, 0.0f, Constants::PI_F*0.45f, Constants::PI_F*0.45f);
}

void ServoController::set_enc_to_pos_lut(LookupTable& lut) {
  ServoController::encoder_raw_to_motor_pos_lut = lut;
}

// get the motor position to field angle lookup table
const LookupTable& ServoController::get_enc_to_pos_lut() const {
  return encoder_raw_to_motor_pos_lut;
}

void ServoController::set_pos_to_field_lut(LookupTable& lut) {
  ServoController::motor_pos_to_field_angle_lut = lut;
}

// get the motor position to field angle lookup table
const LookupTable& ServoController::get_pos_to_field_lut() const {
  return motor_pos_to_field_angle_lut;
}


void ServoController::update(float target_motor_pos, float dt, float one_over_dt) {
  if(encoder_update_enabled == false)
    return;

  // read encoder
  int32_t encoder_angle_raw = encoder.read_abs_angle_raw();

  // convert encoder angle to motor pos using LUT and compute field angle
  motor_pos = encoder_angle_to_motor_pos(encoder_angle_raw);
  float field_angle = motor_pos_to_field_angle(motor_pos);

  // position controll loop
  pos_error = target_motor_pos-motor_pos;
  float velocity_target = pos_controller.compute(pos_error, dt, one_over_dt);

  // velocity controll loop
  float velocity_unfiltered = (motor_pos - motor_pos_prev)*one_over_dt;
  velocity = velocity_lowpass.update(velocity_unfiltered, dt);
  float torque_target = velocity_controller.compute(velocity_target-velocity, dt, one_over_dt);

  // torque controll loop
  output = torque_target;

  // set new field direction
  // motor_driver.set_amplitude(std::clamp(abs(output*10.0f), 0.1f, 0.5f), false);
  if(motor_update_enabled) {
    motor_driver.set_field_angle(field_angle + output);
  } 

  // store values for next update
  motor_pos_prev = motor_pos;
}

bool ServoController::at_position(float motor_pos_eps) {
  return fabs(pos_error) < motor_pos_eps;
}

float ServoController::read_position() {
  return encoder_angle_to_motor_pos(encoder.read_abs_angle_raw());
}

float ServoController::get_position() {
  return motor_pos;
}

float ServoController::get_position_error() {
  return pos_error;
}

bool ServoController::move_to(float target_motor_pos, float at_pos_eps, float settle_time_s, float timeout_s) {
  uint64_t start_time_us = time_us_64();
  uint64_t time_us = start_time_us;
  uint64_t pos_reached_time_us = 0;
  uint64_t last_time = time_us;
  uint32_t settle_time_us = settle_time_s*1e6f;
  uint32_t timeout_us = timeout_s*1e6f;

  do {
    // get time and detla time
    time_us = time_us_64();
    float dt = float(time_us - last_time)*1e-6f;
    last_time = time_us;

    float pos_error;
    update(target_motor_pos, dt, 1.0f/dt);

    // check if traget position reached
    if(pos_reached_time_us == 0) {
      if(at_position(at_pos_eps))
        pos_reached_time_us = time_us;
    } else {
      if(time_us-pos_reached_time_us > settle_time_us)
        return true;
    }

  } while(time_us-start_time_us < timeout_us);

  return false;
}

void ServoController::move_to_open_loop(float delta_motor_pos, float motor_angular_velocity) {
  // Determine direction of movement at the start
  const bool moving_forward = delta_motor_pos > 0.0f;

  uint64_t last_time = time_us_64();
  float pos = 0.0f;
  while (fabs(pos) < delta_motor_pos)
  {
    uint64_t time_us = time_us_64();
    float dt = float(time_us - last_time) * 1e-6f;
    last_time = time_us;

    // update encoder regularly
    if(encoder_update_enabled)
      encoder.read_abs_angle_raw();

    // update motor position
    pos += moving_forward ? motor_angular_velocity * dt : -motor_angular_velocity * dt;

    // set field ange to new position
    float clamped_motor_pos = moving_forward ? std::min(pos, delta_motor_pos) : 
                                               std::max(pos, -delta_motor_pos);
    motor_driver.set_field_angle(clamped_motor_pos*motor_pole_pair_count);
    sleep_us(100);
  }

  motor_pos += delta_motor_pos;
}

ServoController::ENCODER_TYPE& ServoController::get_encoder() {
  return encoder;
}

ServoController::MOTOR_DRIVER_TYPE& ServoController::get_motor_driver() {
  return motor_driver;
}

float ServoController::get_pole_pair_count() {
  return motor_pole_pair_count;
}

void ServoController::set_motor_enabled(bool enable, bool synchronize_field_angle) {
  if(enable) {
    // synchronize field angle to motor_pos
    if(synchronize_field_angle) {
      float start_field_angle = motor_pos_to_field_angle(motor_pos);
      motor_driver.set_field_angle(start_field_angle);
    }

    motor_driver.set_amplitude_smooth(motor_current_amplitude, 100);
    pos_controller.reset();
    velocity_controller.reset();
    velocity_lowpass.reset(0.0f);
    motor_pos_prev = motor_pos;

  } else {
    motor_driver.set_amplitude_smooth(0.0f, 100);
  }
}

// enable or disable servo loop update and encoder reads
void ServoController::set_motor_update_enabled(bool enable) {
  pos_controller.reset();
  velocity_controller.reset();
  velocity_lowpass.reset(0.0f);
  motor_pos_prev = motor_pos;
  motor_update_enabled = enable;
}

void ServoController::set_encoder_update_enabled(bool enable) {
  pos_controller.reset();
  velocity_controller.reset();
  motor_pos_prev = motor_pos;
  encoder_update_enabled = enable;
}

float ServoController::encoder_angle_to_motor_pos(int32_t encoder_angle_raw) {
  return encoder_raw_to_motor_pos_lut.evaluate(encoder_angle_raw);
}

float ServoController::motor_pos_to_field_angle(float motor_pos) {
  return motor_pos_to_field_angle_lut.evaluate(motor_pos);
}