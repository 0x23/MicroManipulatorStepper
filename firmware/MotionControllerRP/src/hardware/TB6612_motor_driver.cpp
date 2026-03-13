// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include "TB6612_motor_driver.h"
#include "utilities/logging.h"

#include <math.h>
#include <algorithm>
#include "hardware/pwm.h"
#include "hardware/gpio.h"
#include "hardware/clocks.h"
#include "pico/time.h"

// Helper rounding function
int32_t round_int32(float val) {
  return (val >= 0.0f) ? (int32_t)(val + 0.5f) : (int32_t)(val - 0.5f);
}

TB6612MotorDriver::TB6612MotorDriver(
  uint8_t pin_en_a, uint8_t pin_pos_a, uint8_t pin_neg_a, uint8_t pin_pwm_a,
  uint8_t pin_en_b, uint8_t pin_pos_b, uint8_t pin_neg_b, uint8_t pin_pwm_b,
  uint8_t ch_pos_a, uint8_t ch_neg_a,
  uint8_t ch_pos_b, uint8_t ch_neg_b,
  uint16_t pwm_freq,
  uint8_t pwm_resolution
)
: pin_en_a(pin_en_a), pin_pos_a(pin_pos_a), pin_neg_a(pin_neg_a), pin_pwm_a(pin_pwm_a),
  pin_en_b(pin_en_b), pin_pos_b(pin_pos_b), pin_neg_b(pin_neg_b), pin_pwm_b(pin_pwm_b),
  ch_pos_a(ch_pos_a), ch_neg_a(ch_neg_a),
  ch_pos_b(ch_pos_b), ch_neg_b(ch_neg_b),
  pwm_freq(pwm_freq), pwm_resolution(pwm_resolution)
{
  max_pwm = (1 << pwm_resolution) - 1;
  set_amplitude(0.0f, false);
  field_angle = 0.0f;
}

void init_output_pin(uint8_t pin, bool value) {
  gpio_init(pin);
  gpio_set_dir(pin, GPIO_OUT);
  gpio_put(pin, value);
}

void TB6612MotorDriver::begin() {
  init_output_pin(pin_en_a, false);
  init_output_pin(pin_en_b, false);

  // Enable pwm pins, see TB6612 documentation for how the PWM pins work,
  // for slow decay mode they are constantly enabled.
  init_output_pin(pin_pwm_a, true);
  init_output_pin(pin_pwm_b, true);

  // Read system clock dynamically
  uint32_t sys_clk = clock_get_hz(clk_sys);
  float clkdiv = (float)sys_clk / (pwm_freq * max_pwm);

  // Setup helper
  auto setup_pwm_pin = [&](uint8_t pin) {
    gpio_set_function(pin, GPIO_FUNC_PWM);
    uint slice = pwm_gpio_to_slice_num(pin);
    uint chan  = pwm_gpio_to_channel(pin);  // 0 for A, 1 for B
    pwm_config config = pwm_get_default_config();
    pwm_config_set_clkdiv(&config, clkdiv);
    pwm_config_set_wrap(&config, max_pwm);
    pwm_init(slice, &config, true);
  };

  setup_pwm_pin(pin_pos_a);
  setup_pwm_pin(pin_neg_a);
  setup_pwm_pin(pin_pos_b);
  setup_pwm_pin(pin_neg_b);

  disable();
}

void TB6612MotorDriver::enable() {
  gpio_put(pin_en_a, 1);
  gpio_put(pin_en_b, 1);
}

void TB6612MotorDriver::disable() {
  gpio_put(pin_en_a, 0);
  gpio_put(pin_en_b, 0);
}

void TB6612MotorDriver::set_amplitude(float amplitude, bool immediate_update) {
  amplitude = std::clamp(amplitude, 0.0f, 1.0f);
  TB6612MotorDriver::amplitude = amplitude;
  TB6612MotorDriver::amplitude_raw = amplitude * max_pwm;
  if(immediate_update)
    set_field_angle(field_angle);
}


void TB6612MotorDriver::set_amplitude_smooth(float amplitude, int ramp_time_ms) {
  amplitude = std::clamp(amplitude, 0.0f, 1.0f);
  float start = TB6612MotorDriver::amplitude;
  int step_time = 10;
  int steps = std::max(1, ramp_time_ms / step_time);

  for (int i = 1; i <= steps; ++i) {
    float t = float(i) / steps;
    set_amplitude(start + t * (amplitude - start), true);
    sleep_ms(step_time);
  }
}

float TB6612MotorDriver::get_amplitude() const {
  return amplitude;
}

void TB6612MotorDriver::rotate_field(float delta_angle, float rad_per_s, const std::function<void()>& on_step) {
  if(delta_angle == 0.0f)
    return;
    
  float start = field_angle;
  float rad_per_µs = (delta_angle >= 0.0f) ? rad_per_s*1e-6f : -rad_per_s*1e-6f;

  // Determine how long the movement should take (in microseconds)
  uint64_t duration_us = (uint64_t)(fabs(delta_angle / rad_per_s) * 1e6f);
  uint64_t start_time_us = time_us_64();
  while (true) {
      uint64_t elapsed_us = time_us_64() - start_time_us;
      if (elapsed_us >= duration_us)
          break;

      // Update the field
      set_field_angle(start + rad_per_µs * elapsed_us);
      
      if(on_step)
        on_step();

      sleep_us(500);
  }

  // Ensure final angle is reached
  set_field_angle(start + delta_angle);
}


void TB6612MotorDriver::set_field_angle(float angle_rad) {
  field_angle = angle_rad;
  float sin_a = sin(angle_rad);
  float cos_a = cos(angle_rad);

  set_pwm(pin_pos_a, pin_neg_a, round_int32(sin_a * amplitude_raw));
  set_pwm(pin_pos_b, pin_neg_b, round_int32(cos_a * amplitude_raw));
}

float TB6612MotorDriver::get_field_angle() {
  return field_angle;
}

void TB6612MotorDriver::set_pwm(uint8_t pin_pos, uint8_t pin_neg, int32_t value) {
  value = std::min(std::max(value, -(int32_t)max_pwm), (int32_t)max_pwm);

  if (value >= 0) {
    pwm_set_gpio_level(pin_pos, max_pwm - value);
    pwm_set_gpio_level(pin_neg, max_pwm);
  } else {
    pwm_set_gpio_level(pin_pos, max_pwm);
    pwm_set_gpio_level(pin_neg, max_pwm + value);
  }
}