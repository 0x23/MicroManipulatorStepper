// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include <LittleFS.h> 
#include "robot.h"
#include "hw_config.h"
#include "utilities/logging.h"
#include "utilities/utilities.h"
#include "kinematic_models/kinematic_model_delta3d.h"
#include "servo_control/homing_controller.h"
#include "servo_control/actuator_calibration.h"
#include "robot_joint/robot_joint.h"
#include "pico/multicore.h"
#include "version.h"
#include "robot_tool/pwm_tool.h"

constexpr int SPINLOCK_ID_SHARED_DATA = 0;
constexpr int SPINLOCK_ID_JOINTS = 1;

#include <NeoPixelConnect.h>
NeoPixelConnect led(PIN_BUILTIN_LED, 1);

//*** FUNCTION **************************************************************************

bool startswith(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() &&
           std::equal(prefix.begin(), prefix.end(), str.begin());
}

//*** CLASS *****************************************************************************

//--- Robot -----------------------------------------------------------------------------

Robot::Robot(float path_segment_time_step) : 
  path_planner(nullptr, path_segment_time_step), 
  motion_controller(&path_planner),
  servo_loop_frequency_counter(10000),
  motion_controller_frequency_counter(1000),
  shared_data(SPINLOCK_ID_SHARED_DATA),
  joints_spin_lock(spin_lock_instance(SPINLOCK_ID_JOINTS))
{
  kinematic_model = new KinematicModel_Delta3D();
  path_planner.set_kinematic_model(kinematic_model);

  for(int i=0; i<NUM_JOINTS; i++)
    joints[i] = nullptr;

  command_parser.set_command_processor(this);

  current_feedrate = LinearAngular(10.0f, 1.0f);
  max_acceleration = LinearAngular(500.0f, 50.0f);
  path_buffering_time_us = 50*1e3;

  for(int i=0; i<NUM_TOOLS; i++)
    robot_tools[i] = nullptr;

  state = ERobotState::IDLE;
}

Robot::~Robot() {
  if(kinematic_model != nullptr)
    delete kinematic_model;

  for(int i=0; i<NUM_JOINTS; i++) {
    if(joints[i] != nullptr)
      delete joints[i];
    joints[i] = nullptr;
  }
}

void Robot::init() {
  MT6835Encoder::setup_spi(spi0, PIN_ENCODER_SCK, PIN_ENCODER_MOSI, PIN_ENCODER_MISO, 8000000);

  // axis 1
  {
    auto* encoder = new MT6835Encoder(spi0, PIN_ENCODER1_CS);
    auto* motor_driver = new TB6612MotorDriver(
      PIN_MOTOR_EN, PIN_M1_PWM_A_POS, PIN_M1_PWM_A_NEG, PIN_MOTOR_PWMAB,
      PIN_MOTOR_EN, PIN_M1_PWM_B_POS, PIN_M1_PWM_B_NEG, PIN_MOTOR_PWMAB
    );
	encoder->set_crc_enabled(ENABLE_ENCODER_CRC);
    joints[0] = new RobotJoint(encoder, motor_driver, MOTOR1_POLE_PAIRS);
  }

  // axis 2
  {
    auto* encoder = new MT6835Encoder(spi0, PIN_ENCODER2_CS);
    auto* motor_driver = new TB6612MotorDriver(
      PIN_MOTOR_EN, PIN_M2_PWM_A_POS, PIN_M2_PWM_A_NEG, PIN_MOTOR_PWMAB,
      PIN_MOTOR_EN, PIN_M2_PWM_B_POS, PIN_M2_PWM_B_NEG, PIN_MOTOR_PWMAB
    );
	encoder->set_crc_enabled(ENABLE_ENCODER_CRC);
    joints[1] = new RobotJoint(encoder, motor_driver, MOTOR2_POLE_PAIRS);
  }

  // axis 3
  {
    auto* encoder = new MT6835Encoder(spi0, PIN_ENCODER3_CS);
    auto* motor_driver = new TB6612MotorDriver(
      PIN_MOTOR_EN, PIN_M3_PWM_A_POS, PIN_M3_PWM_A_NEG, PIN_MOTOR_PWMAB,
      PIN_MOTOR_EN, PIN_M3_PWM_B_POS, PIN_M3_PWM_B_NEG, PIN_MOTOR_PWMAB
    );
	encoder->set_crc_enabled(ENABLE_ENCODER_CRC);
    joints[2] = new RobotJoint(encoder, motor_driver, MOTOR3_POLE_PAIRS);
  }

  // initialize axes
  for(int i=0; i<NUM_JOINTS; i++) {
    joints[i]->init(i);
    joints[i]->load_calibration();
  }

  // add tools
  robot_tools[0] = new PwmTool();
  ((PwmTool*)robot_tools[0])->init(PIN_TOOL1, 8000, 8);

  robot_tools[1] = new PwmTool();
  ((PwmTool*)robot_tools[1])->init(PIN_TOOL2, 8000, 8);

  // setup timer for updating the motion controller (which evaluates joint space path
  // segments and produces the current target position for the servo loops)
  float motion_controller_update_time_us = 500;
  add_repeating_timer_us(-motion_controller_update_time_us, 
                         Robot::update_motion_controller_isr, 
                         (void*)this, 
                         &motion_controller_update_timer);
}


void Robot::update_command_parser() {
  // process serial input
  if (Serial.available()) {
    char c = Serial.read();
    command_parser.add_input_character(c);
    // Serial.write(c);
  }

  // update command parse which will queue command to the path planner
  command_parser.update();

  // TESTING:
  //sleep_ms(10);
  //float pos_error = joints[1]->servo_controller->get_position_error();
  //LOG_INFO(">pos_error [µrad]: %f\n", pos_error*1e6);
  //Serial.printf(">pos_error [µrad]: %f\n", pos_error*1e6);#
  //LOG_INFO(">pos_x [mm]: %f", joints[1]->position);
  //update_servo_controllers(0.01f);
}

/**
 * Updates the path planner, that chops up kartesian path segments into joint space
 * path segments using the inverse kinematic model. It then enqueues these joint space path
 * segments for the motion controller.
 */
void Robot::update_path_planner() {
  // check if buffering starts
  uint64_t time = time_us_64();
  if(state == ERobotState::IDLE && path_planner.input_queue_size() > 0) {
    state = ERobotState::BUFFERING_PATH;
    path_buffering_start_time = time;
  }

  // check if execution starts
  uint64_t buffering_time = time-path_buffering_start_time;
  if(state == ERobotState::BUFFERING_PATH && buffering_time > path_buffering_time_us) {
    state = ERobotState::EXECUTING_PATH;
  }

  // execute path
  if(state == ERobotState::EXECUTING_PATH) {
    // update planner and generate joint space path segments
    path_planner.process(true);

    if(path_planner.all_finished())
      state = ERobotState::IDLE;
  }
}

/**
 * Updates the motion controller with a timer interrupt in regular intervals (e.g. 2kHz).
 * The function evaluates joint space path segments and produces the current 
 * target position for the servo loops.
 */
bool Robot::update_motion_controller_isr(repeating_timer_t* timer) {
  float joint_positions[NUM_JOINTS];
  float joint_velocities[NUM_JOINTS];
  float tool_outputs[NUM_TOOLS];

  // get robot pointer
  Robot* robot = (Robot*)timer->user_data;

  // get time and delta time
  uint64_t time_us = time_us_64();
  float dt = float(time_us - robot->last_mc_update_time)*1e-6f;
  robot->last_mc_update_time = time_us;

  // get current joint position/velocity and tool outputs
  bool update_ok = robot->motion_controller.update(dt, joint_positions, joint_velocities, tool_outputs);

  // Attempt to acquire spinlock non-blocking and set new target data for the servo loops
  if (update_ok && spin_try_lock_unsafe(robot->shared_data.lock)) {
    for (int i = 0; i < NUM_JOINTS; i++) {
      robot->shared_data.joint_target_positions[i] = joint_positions[i];
      robot->shared_data.joint_target_velocities[i] = joint_velocities[i];
    }
    spin_unlock_unsafe(robot->shared_data.lock);
  }

  // update tool outputs
  if(update_ok) {
    auto& tools = robot->robot_tools;
    for(int i=0; i<tools.size(); i++) {
      auto* tool = tools[i];
      if(tool != nullptr)
        tool->set_value(tool_outputs[i]);
    }
  }

  // update frequency counter
  robot->motion_controller_frequency_counter.update(dt);

  return true; // keep repeating
}

/**
 * update servo loops, this is called from the second cpu core
 */
void Robot::update_servo_controllers(float dt) {
  float one_over_dt = 1.0f/dt;

  // update axis target position and velocity from shared data
  spin_lock_unsafe_blocking(shared_data.lock);
  for(int i=0; i<3; i++) {
    joints[i]->update_target(shared_data.joint_target_positions[i], 
                             shared_data.joint_target_velocities[i]);
  }
  spin_unlock_unsafe(shared_data.lock);

  // update servo loop for each axis
  spin_lock_unsafe_blocking(joints_spin_lock);
  for(int i=0; i<NUM_JOINTS; i++) {
    joints[i]->update(dt, one_over_dt);
  }
  spin_unlock_unsafe(joints_spin_lock);

  // update frequency counter
  servo_loop_frequency_counter.update(dt);
}

void Robot::enable_servo_control(bool enable) {
  // LOG_DEBUG(enable ? "Enable servo control" : "Disable servo sontrol");

  // update servo loop for each axis
  spin_lock_unsafe_blocking(joints_spin_lock);

  for(int i=0; i<NUM_JOINTS; i++) {
    bool en = joints[i]->is_homed && joints[i]->is_calibrated && enable;
    LOG_DEBUG(en ? "Joint-%i: servo control enabled" : "Joint-%i: servo control disabled", i);
    joints[i]->servo_controller->set_motor_update_enabled(en);
  }

  spin_unlock_unsafe(joints_spin_lock);
}

void Robot::set_pose(const Pose6DF& pose) {
  // run inverse kinematic and compute joint positions
  float joint_positions[NUM_JOINTS];
  kinematic_model->inverse(pose, joint_positions);

  while(true) {
    // Attempt to acquire spinlock non-blocking and set new target data for the servo loops
    if (spin_try_lock_unsafe(shared_data.lock)) {
      for (int i = 0; i < NUM_JOINTS; i++) {
        shared_data.joint_target_positions[i] = joint_positions[i];
        shared_data.joint_target_velocities[i] = 0.0f;
        // LOG_DEBUG("Joint-%i: set pose -> angle %f", i, joint_positions[i]);
      }
      spin_unlock_unsafe(shared_data.lock);
      break;
    }
  }

  current_pose = pose;
}

Pose6DF Robot::pose_from_joint_angles() {
  // read joint positions from encoders
  float joint_pos[NUM_JOINTS];

  spin_lock_unsafe_blocking(joints_spin_lock);
  for (int i = 0; i < NUM_JOINTS; i++) {
    joint_pos[i] = joints[i]->servo_controller->read_position(); 
  }
  spin_unlock_unsafe(joints_spin_lock);

  // run foreward kinematic model to retrieve pose from joint positions
  Pose6DF pose;
  bool ok = kinematic_model->foreward(joint_pos, pose);
  if(ok == false)
    LOG_ERROR("Foreward kinematic failed");

  return pose;
}

CommandParser* Robot::get_command_parser() {
  return &command_parser;
}

bool Robot::check_all_joints_ready() {
  bool all_ready = true;
  for(int i=0; i<NUM_JOINTS; i++) {
    all_ready &= joints[i]->is_calibrated && joints[i]->is_homed;
  }

  return all_ready;
}

bool Robot::home(uint8_t joint_mask, float retract_angles[NUM_JOINTS]) {
  HomingController homing_controller[NUM_JOINTS];
  LOG_INFO("homing...");
  enable_servo_control(false);

  // prevent servo loop updates from running during homing
  spin_lock_unsafe_blocking(joints_spin_lock);

  // initialize homing controllers
  for(int i=0; i<NUM_JOINTS; i++) {
    // only start requested joints
    if(((joint_mask>>i)&1) == 0) continue;
    LOG_DEBUG("start homing axis %i", i);
    homing_controller[i].start(joints[i]->servo_controller, 
                               -HOMING_VELOCITY, 360.0f*DEG_TO_RAD, HOMING_CURRENT,
                               ENCODER_ANGLE_TO_ROTOR_ANGLE, retract_angles[i]);
  }

  // run homing controllers
  bool all_finished = false;
  while(all_finished == false) {
    all_finished = true;
    for(int i=0; i<NUM_JOINTS; i++) {
      // only update requested joints
      if(((joint_mask>>i)&1) == 0) continue;

      // uddate
      homing_controller[i].update();
      all_finished &= homing_controller[i].is_finished();  
    }
  }

  // finalize homing controllers
  bool homing_successful = true;
  for(int i=0; i<NUM_JOINTS; i++) {
    // only check requested joints
    if(((joint_mask>>i)&1) == 0) continue;

    homing_controller[i].finalize();

    if(homing_controller[i].is_successful()) {
      joints[i]->is_homed = true;
    } else {
      LOG_ERROR("homing joint %i failed", i);
      homing_successful = false;
    }

    // set joint angles
    spin_lock_unsafe_blocking(shared_data.lock);
    shared_data.joint_target_positions[i] = joints[i]->servo_controller->read_position();
    spin_unlock_unsafe(shared_data.lock);
  }

  // servo updates may continue here
  spin_unlock_unsafe(joints_spin_lock);

  // get pose from joint angles
  set_pose(pose_from_joint_angles());

  // enable servo loops if all joints are initialized
  enable_servo_control(true);

  // check if all joints are ready
  all_joints_ready = check_all_joints_ready();

  return homing_successful;
}

bool Robot::calibrate_joint(int joint_idx, bool store_calibration, bool print_measurements) {
  if(joint_idx<0 || joint_idx >= NUM_JOINTS)
    return false;

  RobotJoint* joint = joints[joint_idx];

  // prevent servo loop updates from running during homing
  enable_servo_control(false);
  spin_lock_unsafe_blocking(joints_spin_lock);

  bool calibration_ok = joint->calibrate(print_measurements);
  if(!calibration_ok) {
    spin_unlock_unsafe(joints_spin_lock);
    return false;
  }

  // joint->servo_controller->move_to_open_loop(0.05f, 1.0);
  shared_data.joint_target_positions[joint_idx] = 0; // joint->servo_controller->read_position();

  if(store_calibration)
    joint->store_calibration();

  // servo updates may continue here
  spin_unlock_unsafe(joints_spin_lock);

  // recover pose from joint angles
  set_pose(pose_from_joint_angles());

  // enable servo loops if all joints are initialized
  enable_servo_control(true);

  // check if all joints are ready
  all_joints_ready = check_all_joints_ready();

  return true;
}

//--- G-Code Commands -------------------------------------------------------------------

bool Robot::can_process_command(const GCodeCommand& cmd) {
  if(cmd.get_command() == "G0" || 
     cmd.get_command() == "G4")
  {
    return path_planner.input_queue_full() == false;
  }

  return true;
}

void Robot::send_reply(const char* str) {
  Serial.write(str);
}

void Robot::process_command(const GCodeCommand& cmd, std::string& reply) {
  if(cmd.get_command() == "G0") process_motion_command(cmd, reply);
  else if(cmd.get_command() == "G1") process_motion_command(cmd, reply);
  else if(cmd.get_command() == "G4") process_dwell_command(cmd, reply);
  else if(cmd.get_command() == "G24") process_set_pose_command(cmd, reply);
  else if(cmd.get_command() == "G28") process_home_command(cmd, reply);
  else if(startswith(cmd.get_command(), "M")) process_machine_command(cmd, reply);
  else reply="error: unknown command\n";
}

void Robot::process_machine_command(const GCodeCommand& cmd, std::string& reply) {
  reply = "";

  // process tool output command
  if(cmd.get_command() == "M3") {
    process_tool_output_command(cmd, reply);
    return;
  }

   // enable motors
  if(cmd.get_command() == "M17") {
    // read current pose from HW and set it as current pose
    set_pose(pose_from_joint_angles());

    // enable motors
    spin_lock_unsafe_blocking(joints_spin_lock);
    for(int i=0; i<NUM_JOINTS; i++) {
      joints[i]->servo_controller->set_motor_enabled(true, true);
    }
    spin_unlock_unsafe(joints_spin_lock);

    reply = "ok\n";
    return;
  }

  // disable motors
  if(cmd.get_command() == "M18") {     
    spin_lock_unsafe_blocking(joints_spin_lock);
    for(int i=0; i<NUM_JOINTS; i++)
      joints[i]->servo_controller->set_motor_enabled(false, false);
    spin_unlock_unsafe(joints_spin_lock);
    
    reply = "ok\n";
    return;
  }

  // get current internal position (not using encoders to read physical position)
  if(cmd.get_command() == "M50") {
    reply += std::string("X") + std::to_string(current_pose.translation.x);
    reply += std::string(" Y") + std::to_string(current_pose.translation.y);
    reply += std::string(" Z") + std::to_string(current_pose.translation.z);
    reply += "\nok\n";
    return;
  }

    // get current internal position (not using encoders to read physical position)
  if(cmd.get_command() == "M51") {
    for(int i=0; i<NUM_JOINTS; i++) {
      float raw_angle = joints[i]->encoder->get_last_abs_raw_angle();
      float angle = joints[i]->encoder->get_last_abs_angle()*Constants::RAD2DEG;
      reply += std::string("Joint ")+std::to_string(i)+":  " + 
               std::to_string(angle) + " deg  (raw="+std::to_string(raw_angle)+")\n";
    }
    reply += "ok\n";
    return;
  }

  // get planner queue size
  if(cmd.get_command() == "M52") {
    int s = path_planner.input_queue_size();
    reply += std::string("Queue Size: ") + std::to_string(s) + "\n";
    reply += "ok\n";
    return;
  }

  // check if all planned motions are finished executing
  if(cmd.get_command() == "M53") {
    bool f = path_planner.all_finished();
    reply += f ? "1\n" : "0\n";
    reply += "ok\n";
    return;
  }

  // set servo loop parameters
  if(cmd.get_command() == "M55") {
    process_set_servo_parameter_command(cmd, reply);
    return;
  }

  // calibrate joint
  if(cmd.get_command() == "M56") {
    process_calibrate_joint_command(cmd, reply);
    return;
  }

  // get info
  if(cmd.get_command() == "M57") {
    uint32_t servo_loop_freq = servo_loop_frequency_counter.get();
    uint32_t mcontroler_freq = motion_controller_frequency_counter.get();

    spin_lock_unsafe_blocking(joints_spin_lock);
    for(int i=0; i<NUM_JOINTS; i++) {
      float angle = joints[i]->encoder->read_abs_angle()*Constants::RAD2DEG;
      int32_t crc_errors = joints[i]->encoder->is_crc_enabled() ? joints[i]->encoder->get_crc_error_count(false) : -1;

      reply += std::string("Joint ") + std::to_string(i)+":";
      reply += std::string("  is_homed=") + std::to_string(joints[i]->is_homed);
      reply += std::string(",  is_calibrated=") + std::to_string(joints[i]->is_calibrated);
      reply += std::string(",  encoder_angle=") + std::to_string(angle) + " deg";
      reply += std::string(",  crc_errors=") + std::to_string(crc_errors); 
      reply += std::string(",  enc_status=") + std::to_string(joints[i]->encoder->get_status()) + "\n"; 
    }
    spin_unlock_unsafe(joints_spin_lock);

    reply += std::string("Servo Loop: ") + std::to_string(servo_loop_freq/1000) + " kHz\n";
    reply += std::string("Motion Controler: ") + std::to_string(mcontroler_freq) + " Hz\n";

    for(int i=0; i<NUM_TOOLS; i++)
        reply += std::string("Tool[") + std::to_string(i) + "] output: " + std::to_string(current_tool_outputs[i]) + "\n";

    // file list
    reply += std::string("Files on flash: \n");
    auto file_list = get_file_list("/", true);
    for(auto& f : file_list) reply += std::string("  ")+f+"\n";
    reply += "ok\n";
    return;
  }

  // get firmware version
  if(cmd.get_command() == "M58") {
    reply = std::string(FIRMWARE_VERSION)+"\n";
    reply += "ok\n";
    return;
  }

  // print lookup table
  if(cmd.get_command() == "M59") {
    int idx = (int)cmd.get_value('J', 0);
    joints[idx]->servo_controller->get_enc_to_pos_lut().print_to_log();
    return;
  }

  // set linear and angular acceleration
  if(cmd.get_command() == "M204") {
    if(cmd.has_word('L')) max_acceleration.linear = cmd.get_value('L');
    if(cmd.has_word('A')) max_acceleration.angular = cmd.get_value('A');
    reply += "ok\n";
    return;
  }
}

void Robot::process_motion_command(const GCodeCommand& cmd, std::string& reply) {
  Pose6DF end_pose;
  
  #ifndef JOINT_READY_OVERRIDE
  if(!all_joints_ready) {
    reply = "error: not all joints calibrated and homed\n";
    return;
  }
  #endif
  
  if(path_planner.input_queue_full()) {
    reply = "busy\n";
    return;
  }

  // read feed rate
  current_feedrate.linear = cmd.get_value('F', current_feedrate.linear);
  current_feedrate.angular = cmd.get_value('R', current_feedrate.angular);

  if(cmd.has_word('I'))
    state = ERobotState::EXECUTING_PATH;

  // read translation
  end_pose.translation.x = cmd.get_value('X', current_pose.translation.x);
  end_pose.translation.y = cmd.get_value('Y', current_pose.translation.y);
  end_pose.translation.z = cmd.get_value('Z', current_pose.translation.z);

  // read rotation (all elements must be present)
  if(cmd.has_word('A') && cmd.has_word('B') && cmd.has_word('C')) {
    Vec3F rot_vec(cmd.get_value('A'), cmd.get_value('B'), cmd.get_value('C'));
    end_pose.rotation = QuaternionF::from_rot_vec(rot_vec);
  } else {
    end_pose.rotation = current_pose.rotation;
  }

  // create path segment
  CartesianPathSegment path_segment(current_pose, 
                                    end_pose, 
                                    current_feedrate, 
                                    max_acceleration,
                                    current_tool_outputs);

  bool ok = path_planner.add_cartesian_path_segment(path_segment);
  if(ok) {
    path_planner.run_look_ahead_planning();
    current_pose = end_pose;
    reply = "ok\n";
  } else {
    reply = "error\n";
  }
}

/**
 * Immediately sets the current pose without creating path segments or 
 * interpolating from current position. Useful for external realtime controll.
 */
void Robot::process_set_pose_command(const GCodeCommand& cmd, std::string& reply) {
  Pose6DF pose;

  if(!all_joints_ready) {
    reply = "error: not all joints calibrated and homed\n";
    return;
  }

  // read translation
  pose.translation.x = cmd.get_value('X', current_pose.translation.x);
  pose.translation.y = cmd.get_value('Y', current_pose.translation.y);
  pose.translation.z = cmd.get_value('Z', current_pose.translation.z);

  // read rotation (all elements must be present)
  if(cmd.has_word('A') && cmd.has_word('B') && cmd.has_word('C')) {
    Vec3F rot_vec(cmd.get_value('A'), cmd.get_value('B'), cmd.get_value('C'));
    pose.rotation = QuaternionF::from_rot_vec(rot_vec);
  } else {
    pose.rotation = current_pose.rotation;
  }

  // set the current pose und update target angles for servo loops
  set_pose(pose);
  reply = "ok\n";
}

void Robot::process_dwell_command(const GCodeCommand& cmd, std::string& reply) {
  #ifndef JOINT_READY_OVERRIDE
  if(!all_joints_ready) {
    reply = "error: not all joints calibrated and homed\n";
    return;
  }
  #endif

  if(path_planner.input_queue_full()) {
    reply = "busy\n";
    return;
  }

  // get dwell time
  float dwell_time = 1.0f;
  if(cmd.has_word('S')) dwell_time = cmd.get_value('S');          // time given in seconds
  if(cmd.has_word('P')) dwell_time = cmd.get_value('P')*0.001f;   // time given in milliseconds

  // create path segment
  CartesianPathSegment path_segment(current_pose, current_tool_outputs, dwell_time);
  bool ok = path_planner.add_cartesian_path_segment(path_segment);

  if(ok) {
    path_planner.run_look_ahead_planning();
    reply = "ok\n";
  } else {
    reply = "error\n";
  }
}

void Robot::process_set_servo_parameter_command(const GCodeCommand& cmd, std::string& reply) {
  // example: M55 A150 B50000 C0.2 D100 E F0.0025
  bool has_all = cmd.has_word('A') && cmd.has_word('B') && cmd.has_word('C') && 
                 cmd.has_word('D') && cmd.has_word('F');

  if(has_all == false)
    reply = "error: not all parameters given (A,B,C,D,F expected)\n";

  for(int i=0; i<NUM_JOINTS; i++) {

    joints[i]->servo_controller->velocity_lowpass.set_time_constant(cmd.get_value('F'));
    joints[i]->servo_controller->pos_controller.set_parameter(cmd.get_value('A'), cmd.get_value('B'), 0.0f, Constants::PI_F*2.0F, Constants::PI_F*0.5F);
    joints[i]->servo_controller->velocity_controller.set_parameter(cmd.get_value('C'), cmd.get_value('D'), 0.0f, Constants::PI_F*0.45f, Constants::PI_F*0.45f);
  }

  reply = "ok\n";
}

void Robot::process_home_command(const GCodeCommand& cmd, std::string& reply) {
  float retract_angles[NUM_JOINTS] = {-1.0f};

  // TODO: check parameter and build joint mask
  uint8_t joint_mask = 0;
  for(int i=0; i<NUM_JOINTS; i++) {
    char word = 'A'+i;
    if(cmd.has_word(word)) {
      joint_mask |= 1<<i;
      float retract_angle = cmd.get_value(word) * Constants::DEG2RAD;
      if(retract_angle > 1e-3f)
        retract_angles[i] = retract_angle;
    }
  }

  std::string supported_words = "A,B,C,D,E,F";
  if(cmd.contains_unsupported_words(supported_words+",G,M")) {
    reply = "error: Unsupported parameter found. Only [" + supported_words + "] are supported\n";
    return;
  }

  if(joint_mask == 0)
    joint_mask = 255;

  bool ok = home(joint_mask, retract_angles);

  reply = ok ? "ok\n" : "error\n";
}

void Robot::process_calibrate_joint_command(const GCodeCommand& cmd, std::string& reply) {
  int idx = cmd.get_value('J', 0);
  bool store_calibration = cmd.has_word('S');
  bool print_measurements = cmd.has_word('P');

  bool ok = calibrate_joint(idx, store_calibration, print_measurements);
  reply = ok ? "ok\n" : "error\n";
}

void Robot::process_tool_output_command(const GCodeCommand& cmd, std::string& reply) {
  // get tool index
  int tool_index = (int)cmd.get_value('T', 0);
  if(tool_index < 0 || tool_index >= NUM_TOOLS) {
    reply = "error: Tool index out of range\n";
    return;
  }

  // set current tool output value
  float tool_value = cmd.get_value('S', 0.0f);
  current_tool_outputs[tool_index] = tool_value;

  reply = "ok\n";
}

