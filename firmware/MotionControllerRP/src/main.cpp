// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include "main.h"

#include "hardware/clocks.h"
#include "hardware/pll.h"
#include "hardware/vreg.h"

#include <Wire.h>
#include <algorithm>

#include "robot.h"
#include "utilities/logging.h"
#include "utilities/frequency_counter.h"
#include "kinematic_models/kinematic_model_delta3d.h"
#include "version.h"
#include "hw_config.h"
#include "LittleFS.h"

#include "demo_gcode_generator.h"

//*** GLOBALS ***************************************************************************

// NeoPixelConnect strip(PIN_BUILTIN_LED, 1);
Robot robot(0.01f);

/*
MT6835Encoder encoder1(spi0, PIN_ENCODER1_CS);
MT6835Encoder encoder2(spi0, PIN_ENCODER2_CS);
MT6835Encoder encoder3(spi0, PIN_ENCODER3_CS);

TB6612MotorDriver motor_driver1(
  PIN_MOTOR_EN, PIN_M1_PWM_A_POS, PIN_M1_PWM_A_NEG, PIN_MOTOR_PWMAB,
  PIN_MOTOR_EN, PIN_M1_PWM_B_POS, PIN_M1_PWM_B_NEG, PIN_MOTOR_PWMAB
);
TB6612MotorDriver motor_driver2(
  PIN_MOTOR_EN, PIN_M2_PWM_A_POS, PIN_M2_PWM_A_NEG, PIN_MOTOR_PWMAB,
  PIN_MOTOR_EN, PIN_M2_PWM_B_POS, PIN_M2_PWM_B_NEG, PIN_MOTOR_PWMAB
);
TB6612MotorDriver motor_driver3(
  PIN_MOTOR_EN, PIN_M3_PWM_A_POS, PIN_M3_PWM_A_NEG, PIN_MOTOR_PWMAB,
  PIN_MOTOR_EN, PIN_M3_PWM_B_POS, PIN_M3_PWM_B_NEG, PIN_MOTOR_PWMAB
);

ServoController servo_controller1(motor_driver1, encoder1, 400/4);
ServoController servo_controller2(motor_driver2, encoder2, 400/4);
ServoController servo_controller3(motor_driver3, encoder3, 400/4);
FrequencyCounter loop_freq_counter(1000);

PathPlanner planner(0.01f);
MotionController motion_controller(&planner);
Pose6DF current_pose;

CommandParser command_parser; */

//*** FUNCTIONS *************************************************************************

// Run before setup()
//__attribute__((constructor))
void overclock() {
  vreg_set_voltage(VREG_VOLTAGE_1_20);         // For >133 MHz
  busy_wait_us(10 * 1000);  // 10 ms delay
  set_sys_clock_khz(250000, true);             // Set to 250 MHz
}

void set_led_color(uint8_t r, uint8_t g, uint8_t b) {
 /* strip.neoPixelSetValue(0, r, g, b, false);
  delayMicroseconds(2000);
  strip.neoPixelShow(); */
}

void led_blink(uint8_t r, uint8_t g, uint8_t b, int count, int period_time_ms) {
  for(int i=0; i<count; i++) {
    gpio_put(PIN_BUILTIN_LED, 1);
    // set_led_color(r, g, b);
    sleep_ms(period_time_ms/2);
    // set_led_color(0, 0, 0);
    gpio_put(PIN_BUILTIN_LED, 0);
    sleep_ms(period_time_ms/2);
  }
}

void main_core0() {
  uint64_t last_time = time_us_64();

  #ifdef DEMO_MODE
    auto* demo_gcode_generator = new DemoGcodeGenerator(&robot);
    demo_gcode_generator->run();  // blocks forever
  #endif

  while(true) {
    // update motion controller
    robot.update_command_parser();
    robot.update_path_planner();
  }
}

void main_core1() {
  LOG_INFO("Starting servo controll loops on core 1...");

  uint64_t last_time = time_us_64();
  while(true) {
    // get time and detla time
    uint64_t time_us = time_us_64();
    float dt = float(time_us - last_time)*1e-6f;
    last_time = time_us;

    // limit time delta
    dt = std::min(dt, 0.0001f);

    // update servo loops
    robot.update_servo_controllers(dt);
  }
}

void setup() {
  gpio_init(PIN_BUILTIN_LED);
  gpio_set_dir(PIN_BUILTIN_LED, GPIO_OUT);
  
  led_blink(0, 0, 30, 3, 100);
  // stdio_init_all();  // Initializes USB or UART stdio
  overclock();
  // Serial.begin(921600);
  Logger::instance().begin(921600, false);
  #ifndef DEMO_MODE
  while(!Serial);
  #endif

  set_led_color(50, 10, 0);
  // auto* test = new KinematicModel_Delta3D(); test->test(); delete test;

  delay(100);  // Allow time for serial monitor to connect
  
  LOG_INFO("Open Micro Stage Firmware: %s", FIRMWARE_VERSION);
  LOG_INFO("System clock: %i Mhz", int32_t(clock_get_hz(clk_sys))/1000/1000);

  // LittleFS.format();
  if (!LittleFS.begin()) {
    LOG_ERROR("Mounting filesystem failed");
  } else {
    FSInfo fs_info;
    LittleFS.info(fs_info);
    LOG_INFO("Mounting filesystem successfully [%i/%i bytes used]", 
             (int)fs_info.usedBytes, (int)fs_info.totalBytes);
  }

  LOG_INFO("Initializing device...");
  robot.init();

  multicore_launch_core1(&main_core1);
  sleep_ms(100);
  
  set_led_color(0, 20, 0);
  LOG_INFO("Initialization finished");
  LOG_INFO(" ");

  gpio_put(PIN_BUILTIN_LED, 1);

  return;

  /*
  rotencoder_wire.setSDA(PIN_ENCODER_SDA);
  rotencoder_wire.setSCL(PIN_ENCODER_SCL);
  rotencoder_wire.begin();
  rotencoder_wire.setClock(1000000);
  encoder.init();
  encoder.set_hysteresis(0x4); //0x6);
  */
}

void loop() {
  main_core0();
}
