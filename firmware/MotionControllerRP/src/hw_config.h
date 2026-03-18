#pragma once
#include "utilities/math_constants.h"

// #define DEMO_MODE

//--- MOTORS ------------------------------------------------------------------

// motor pole pair count
//  * 100 for 0.9deg stepper motors
//  * 50  for 1.8deg stepper motors
constexpr float MOTOR1_POLE_PAIRS = 50;
constexpr float MOTOR2_POLE_PAIRS = 50;
constexpr float MOTOR3_POLE_PAIRS = 50;

// max current factor in range [0..1]. Lower values reduce pwm resolution so a
// value above 0.4 is recommended.
constexpr float MOTOR_MAX_CURRENT_FACTOR = 0.6f;

//--- ENCODERS ----------------------------------------------------------------

// Conversion factor from encoder angle (one 2pi period every two magnets) to rotor angle.
// Used when the system can not rely on calibration data being present (e.g. during homing)
constexpr float ENCODER_MAGNET_PITCH = 3.0f;    // [mm]
constexpr float ENCODER_MAGNET_RADIUS = 30.0f;  // [mm]
constexpr float ENCODER_ANGLE_TO_ROTOR_ANGLE = (ENCODER_MAGNET_PITCH*2.0f) / 
                                               (ENCODER_MAGNET_RADIUS * Constants::TWO_PI_F);
											   
// enables error checking for encoders (slow) - useful for debugging
// Note: some chips seem to return always a crc of 0 producing massiv false errors       
constexpr bool ENABLE_ENCODER_CRC = false;

//--- HOMING ------------------------------------------------------------------

constexpr float HOMING_VELOCITY   = 1.0f;        // rad per s
constexpr float HOMING_CURRENT    = 0.15f;       // range 0..1
// NOT IMPLEMENTED YET: constexpr float HOMING_FINISH_POS = 0.5f;        // in rad

//--- CALIBRATION -------------------------------------------------------------

// degrees from home position
constexpr float CALIBRATION_RANGE = 83; 

// velocity of the magnetic field during calibration (lower is more accurate)
constexpr float CALIBRATION_FIELD_VELOCITY = 20.0f; 

// size of the calibration lookup table
constexpr int ENCODER_LUT_SIZE = 256;

//--- CLOSED LOOP CONTROL -----------------------------------------------------

// position controller 
constexpr float POS_KP = 60.0f;
constexpr float POS_KI = 30000.0f;

// velocity controller
constexpr float VEL_LOWPASS_TC = 0.004f;
constexpr float VEL_KP = 0.2f;
constexpr float VEL_KI = 90.0f;

//--- KINEMATIC ---------------------------------------------------------------

// Kinematic Parameters are defined kinematic_modes/kinematic_model_delta3d.cpp

// NUM_JOINTS and NUM_TOOLS are defined in 'path_segment.h'. Note that changing
// the number of joints requires changing the kinematic model accordingly and
// also requires the initialization of the correct number of 'RobotJoint' objects
// in the Robtos init method.

//--- PINS --------------------------------------------------------------------

#define JOINT_READY_OVERRIDE

// #define SINGLE_AXIS_BOARD
#ifndef SINGLE_AXIS_BOARD
  // Pins for 3Axis Board
  // #define PIN_BUILTIN_LED 23 // RP2040 pico clone
  // #define PIN_USER_BUTTON 24 // RP2040 pico clone
  #define PIN_BUILTIN_LED 25

  #define PIN_M1_PWM_A_POS  13
  #define PIN_M1_PWM_A_NEG  12
  #define PIN_M1_PWM_B_POS  14
  #define PIN_M1_PWM_B_NEG  15

  #define PIN_M2_PWM_A_POS  9
  #define PIN_M2_PWM_A_NEG  8
  #define PIN_M2_PWM_B_POS  10
  #define PIN_M2_PWM_B_NEG  11

  #define PIN_M3_PWM_A_POS  5
  #define PIN_M3_PWM_A_NEG  4
  #define PIN_M3_PWM_B_POS  6
  #define PIN_M3_PWM_B_NEG  7


  #define PIN_MOTOR_EN      18
  #define PIN_MOTOR_PWMAB   19

  #define PIN_ENCODER1_CS 20
  #define PIN_ENCODER2_CS 21
  #define PIN_ENCODER3_CS 22
  #define PIN_ENCODER_SCK 2
  #define PIN_ENCODER_MISO 0
  #define PIN_ENCODER_MOSI 3

  #define PIN_TOOL1 16
  #define PIN_TOOL2 17

#else
  // Single Axis Board
  #define PIN_BUILTIN_LED 16
  #define PIN_USER_BUTTON 24

  #define PIN_M1_PWM_A_POS  13
  #define PIN_M1_PWM_A_NEG  12
  #define PIN_M1_PWM_B_POS  14
  #define PIN_M1_PWM_B_NEG  15

  #define PIN_M2_PWM_A_POS  9
  #define PIN_M2_PWM_A_NEG  8
  #define PIN_M2_PWM_B_POS  10
  #define PIN_M2_PWM_B_NEG  11

  #define PIN_M3_PWM_A_POS  5
  #define PIN_M3_PWM_A_NEG  4
  #define PIN_M3_PWM_B_POS  6
  #define PIN_M3_PWM_B_NEG  7


  #define PIN_MOTOR_EN      18
  #define PIN_MOTOR_PWMAB   19

  #define PIN_ENCODER1_CS 20
  #define PIN_ENCODER2_CS 21
  #define PIN_ENCODER3_CS 22
  #define PIN_ENCODER_SCK 2
  #define PIN_ENCODER_MISO 0
  #define PIN_ENCODER_MOSI 3
#endif