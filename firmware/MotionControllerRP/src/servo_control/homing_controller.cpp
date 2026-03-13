#include "homing_controller.h"
#include "utilities/math_constants.h"
#include "utilities/logging.h"
#include "pico/time.h"

HomingController::HomingController() {
  retract_field_velocity = 100.0f; // rad per second
  retract_field_angle = Constants::TWO_PI_F*0.25f;
}

bool HomingController::run_blocking(ServoController* servo_controller, 
                                    float motor_velocity, 
                                    float search_range_angle, 
                                    float current, 
                                    float encoder_angle_to_motor_angle,
                                    float retract_angle_rad)
{
  start(servo_controller, motor_velocity, search_range_angle, current, 
        encoder_angle_to_motor_angle, retract_angle_rad);

  while(is_finished() == false) {
    update();
  }

  finalize();
  return is_successful();
}

void HomingController::start(ServoController* servo_controller, 
                            float velocity, 
                            float search_range_angle, 
                            float current, 
                            float encoder_angle_to_motor_angle,
                            float retract_angle_rad)
{
  float pole_pair_count = servo_controller->get_pole_pair_count();
  float field_angle_to_encoder_angle = 1.0f / pole_pair_count / encoder_angle_to_motor_angle;

  if(retract_angle_rad > 0.0f)
    HomingController::retract_field_angle = retract_angle_rad * pole_pair_count;

  // field_angle_to_rotor_angle = 1.0 / pole_pair_count
  // encoder_angle_to_rotor_angle = encoder_period_pitch/encoder_radius
  // field_angle_to_encoder_angle = field_angle_to_rotor_angle/encoder_angle_to_rotor_angle
  
  eval_field_angle_delta = Constants::TWO_PI_F*0.1f;
  expected_encoder_delta = eval_field_angle_delta * field_angle_to_encoder_angle;
  
  servo_ctrl = servo_controller;
  field_velocity = velocity * pole_pair_count;
  field_angle_search_range = search_range_angle * pole_pair_count;
  homing_current = current;

  auto& motor_driver = servo_ctrl->get_motor_driver();
  auto& encoder = servo_ctrl->get_encoder();

  // perform 'soft start'
  servo_ctrl->set_motor_enabled(true, false);

  initial_current = servo_ctrl->get_motor_driver().get_amplitude();
  servo_ctrl->get_motor_driver().set_amplitude_smooth(homing_current, 100);
  search_failed = false;

  // motor_driver.rotate_field(Constants::TWO_PI_F*0.5f * (field_velocity>0.0f ? -1.0f : 1.0f), 12.0f);
  start_field_angle = fmodf(motor_driver.get_field_angle(), Constants::TWO_PI_F);

  last_eval_encoder_angle = encoder.read_abs_angle();
  last_time = 0;

  state = State::Homing;
}

void HomingController::update() {
  if (state != State::Homing)
    return;

  auto& motor_driver = servo_ctrl->get_motor_driver();
  auto& encoder = servo_ctrl->get_encoder();

  uint64_t time_us = time_us_64();
  if(last_time == 0) last_time = time_us;
  float dt = float(time_us - last_time) * 1e-6f;
  last_time = time_us;

  // Move motor and read encoder
  field_angle_offset += field_velocity * dt;
  motor_driver.set_field_angle(start_field_angle+field_angle_offset);

  float encoder_angle = encoder.read_abs_angle();

  if (fabs(last_eval_field_angle_offset - field_angle_offset) > eval_field_angle_delta) {
    float encoder_delta = encoder_angle - last_eval_encoder_angle;
    float encoder_velocity_ratio = encoder_delta / expected_encoder_delta;

    // LOG_DEBUG("encoder_delta=%f/ %f", encoder_delta, expected_encoder_delta);
    // LOG_DEBUG("encoder_velocity_ratio=%f", encoder_velocity_ratio);

    if (fabsf(encoder_velocity_ratio) < 0.05f) {
      LOG_DEBUG("End stop detected");
      on_endstop_detected();
      return;
    }

    last_eval_encoder_angle = encoder_angle;
    last_eval_field_angle_offset = field_angle_offset;
  }

  if (fabs(field_angle_offset) > field_angle_search_range) {
    LOG_INFO("End stop not detected");
    search_failed = true;
    finalize();
  }
}

void HomingController::on_endstop_detected() {
  state = State::Done;
  auto& motor_driver = servo_ctrl->get_motor_driver();
  auto& encoder = servo_ctrl->get_encoder();

  if(search_failed) {
    servo_ctrl->set_motor_enabled(false, false);
    return;
  }

  // reset encoder period, the remainder will provide a very repeatable position reference
  encoder.read_abs_angle();
  servo_ctrl->get_encoder().reset_abs_angle_period();

  // the motor is currently held against the end stop by the field, defining a geometric reference
  home_encoder_angle = encoder.read_abs_angle();
  LOG_DEBUG("home_encoder_angle=%f deg", home_encoder_angle*Constants::RAD2DEG);
  if(home_encoder_angle < Constants::TWO_PI_F*0.01 || home_encoder_angle > Constants::TWO_PI_F*0.99)
    LOG_WARNING("encoder angle at home position close to wrap around point !");
}

void HomingController::finalize() {
   auto& motor_driver = servo_ctrl->get_motor_driver();
   float pole_pair_count = servo_ctrl->get_pole_pair_count();
 
  // back off from home position
  motor_driver.rotate_field(retract_field_angle * (field_velocity>0.0f ? -1.0f : 1.0f), 
                            retract_field_velocity, [this](){
                              // update encoder so it doesnt miss a period
                              servo_ctrl->get_encoder().read_abs_angle();
                            });

  // restore previous motor current
  motor_driver.set_amplitude_smooth(initial_current, 100);
}

bool HomingController::is_finished() const {
  return state == State::Done;
}

bool HomingController::is_successful() const {
  return state == State::Done && !search_failed;
}

float HomingController::get_home_encoder_angle() const {
  return home_encoder_angle;
}


