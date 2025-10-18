// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include <algorithm>

#include "servo_controller.h"
#include "utilities/logging.h"
#include "utilities/math_constants.h"

#include "actuator_calibration.h"


//*** FUNCTION ***********************************************************************************/

bool measure_calibration_data(
            LookupTable& encoder_raw_to_motor_pos_lut,
            LookupTable& motor_pos_to_field_angle_lut,
            ServoController& servo_controller,
            float calibration_range,
            float field_velocity,
            size_t table_size,
            bool print_measurements)
{
  LOG_INFO("Measuring motor to encoder angle lookup table...");
  int sample_count = table_size*4;

  // get required values
  std::vector<std::pair<float, float>> motor_pos_and_field_angle;
  std::vector<std::pair<float, float>> encoder_angle_and_motor_pos;

  auto& motor_driver = servo_controller.get_motor_driver();
  float pole_pair_count = servo_controller.get_pole_pair_count();
  float start_field_angle = motor_driver.get_field_angle();
  float field_angle_step = calibration_range*pole_pair_count/(sample_count-1);

  auto run_measurement = [&](int sample_count, float field_angle_step) {
    // Measure in increasing direction
    for (size_t i = 0; i < sample_count; ++i) {
      if(i>0)
        motor_driver.rotate_field(field_angle_step, field_velocity, nullptr);

      float encoder_angle_raw = servo_controller.get_encoder().read_abs_angle_raw();
      float field_angle = motor_driver.get_field_angle();
      float motor_pos = (field_angle-start_field_angle)/pole_pair_count;
      // TODO: read motor_pos from precise reference encoder

      if(print_measurements)
        LOG_RAW("%15.10f, %15.10f, %f", motor_pos, field_angle, encoder_angle_raw);

      encoder_angle_and_motor_pos.push_back({encoder_angle_raw, motor_pos});
      motor_pos_and_field_angle.push_back({motor_pos, field_angle});
    }
  };

  // Measure in increasing direction
  LOG_DEBUG("Running foreward pass...");
  run_measurement(sample_count, field_angle_step);
  LOG_DEBUG("Running backward pass...");
  run_measurement(sample_count, -field_angle_step);

  // rotate back to start position
  motor_driver.rotate_field(start_field_angle-motor_driver.get_field_angle(), 
    Constants::TWO_PI_F*40.0f, [&servo_controller]() {
      servo_controller.get_encoder().read_abs_angle_raw();
  });

  // build lookup tables
  bool ok = encoder_raw_to_motor_pos_lut.init_interpolating(encoder_angle_and_motor_pos, table_size, true);
  encoder_raw_to_motor_pos_lut.optimize_lut(encoder_angle_and_motor_pos);
  if(ok == false) {
    LOG_ERROR("Creating lookup table encoder_raw_angle -> motor_pos failed.");
    return false;
  }

  ok = motor_pos_to_field_angle_lut.init_interpolating(motor_pos_and_field_angle, table_size/2, true);
  motor_pos_to_field_angle_lut.optimize_lut(motor_pos_and_field_angle);
  if(ok == false) {
    LOG_ERROR("Creating lookup table motor_pos -> field_angle failed.");
    return false;
  }

  LOG_INFO("finished");
  return true;
}
