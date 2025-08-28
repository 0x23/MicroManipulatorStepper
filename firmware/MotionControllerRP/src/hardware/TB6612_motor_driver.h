// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

#include <stdint.h>
// #include <Arduino.h>

class TB6612MotorDriver {
  public:
    TB6612MotorDriver(
      uint8_t pin_en_a, uint8_t pin_pos_a, uint8_t pin_neg_a, uint8_t pin_pwm_a,
      uint8_t pin_en_b, uint8_t pin_pos_b, uint8_t pin_neg_b, uint8_t pin_pwm_b,
      uint8_t ch_pos_a=0, uint8_t ch_neg_a=1,
      uint8_t ch_pos_b=2, uint8_t ch_neg_b=3,
      uint16_t pwm_freq = 20000,    // might not be hit exactly and may be rounded to nearby freq.
      uint8_t pwm_resolution = 12
    );

    void  begin();
    void  enable();
    void  disable();
    void  set_field_angle(float angle_rad);
    float get_field_angle();
    void  set_amplitude(float amplitude, bool immediate_update);  // Input in range 0.0–1.0

  private:
    void  set_pwm(uint8_t ch_pos, uint8_t ch_neg, int32_t value);

    uint8_t pin_en_a, pin_pos_a, pin_neg_a, pin_pwm_a;
    uint8_t pin_en_b, pin_pos_b, pin_neg_b, pin_pwm_b;
    uint8_t ch_pos_a, ch_neg_a;
    uint8_t ch_pos_b, ch_neg_b;

    uint16_t pwm_freq;
    uint8_t pwm_resolution;
    uint16_t max_pwm;
    float amplitude;  // scaled to 0–max_pwm
    float field_angle;
};
