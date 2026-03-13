// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

//*** INCLUDE ***************************************************************************

#include "utilities/math3d.h"

//*** CONST *****************************************************************************

constexpr int NUM_JOINTS = 3;
constexpr int NUM_TOOLS = 2;

//*** CLASS *****************************************************************************

class IKinematicModel;

//--- JointInfo -------------------------------------------------------------------------

class JointInfo {
  public:
    float max_velocity;
    float max_acceleration;
};

//--- MotionProfileConstAcc ------------------------------------------------------------

class MotionProfileConstAcc {
  public: 
    MotionProfileConstAcc() = default;
    MotionProfileConstAcc(float distance,
                          float v_start,
                          float v_end,
                          float max_velocity,
                          float max_acceleration);
    MotionProfileConstAcc(float dwell_time);

    // returns an interpolator value in range [0..1] that can be used to interpolate
    // start and end poses
    float evaluate(float time) const;

  public:
    float t1 = 0.0f;  // end time of accelleration phase
    float t2 = 0.0f;  // end time of cruise phase
    float t3 = 0.0f;  // end time of decellartion phase (total time)
    float acceleration = 1.0f;

    float v_start;
    float v_end;

    float v_peak;
    float d1;         // distance after accelleration phase
    float d2;         // distance after cruise phase
};

//--- CartesianPathSegment --------------------------------------------------------------

// A Linear motion path segment in 6DOF Cartesian Space
class CartesianPathSegment {
  public:
    CartesianPathSegment();

    CartesianPathSegment(const Pose6DF& start_pose,
                         const Pose6DF& end_pose,
                         const LinearAngular& velocity,
                         const LinearAngular& max_acceleration,
                         const float tool_outputs[NUM_TOOLS]);

    CartesianPathSegment(const Pose6DF& pose, 
                         const float tool_outputs[NUM_TOOLS],
                         float dwell_time);

    void evaluate(float time, Pose6DF& pose) const;
    float get_duration() const;
    void compute_motion_profile();

  public:
    Pose6DF start_pose;
    Pose6DF end_pose;

    LinearAngular start_velocity;
    LinearAngular target_velocity;
    LinearAngular end_velocity;
    LinearAngular max_acceleration;

    LinearAngular max_velocity_delta;
    Vec3F translation_delta_normalized;
    //Vec3F rotation_delta_axis;

    LinearAngular travel_distance;
    MotionProfileConstAcc motion_profile;

    float dwell_time;  // stay at start position for given duration if dwell_time > 0
    float tool_outputs[NUM_TOOLS];
};

//--- JointSpacePathSegment -------------------------------------------------------------

// A linear motion path segment in Joint Space
class JointSpacePathSegment {
  public:
    JointSpacePathSegment();
    JointSpacePathSegment(const float start_pos[NUM_JOINTS],
                          const float end_pos[NUM_JOINTS],
                          const float tool_outputs[NUM_TOOLS],
                          const float duration);

    void evaluate(float time, 
                  float joint_positions[NUM_JOINTS], 
                  float joint_velocity[NUM_JOINTS],
                  float tool_outputs[NUM_TOOLS]) const;

    float get_duration();

    bool is_initialized();

  public:
    bool initialized;
    float start_pos[NUM_JOINTS];
    float end_pos[NUM_JOINTS];
    float start_velocity[NUM_JOINTS];
    float end_velocity[NUM_JOINTS];
    float tool_outputs[NUM_TOOLS];

    float duration;
    float inv_duration;
};

//--- JointSpacePathSegmentGenerator ----------------------------------------------------

class JointSpacePathSegmentGenerator {
  public:
    JointSpacePathSegmentGenerator(
      const CartesianPathSegment* path_segment,
      IKinematicModel* kinematic_model,
      float time_step
    );

    void                        reset();
    bool                        generate_next(JointSpacePathSegment& js_path_segment);
    const CartesianPathSegment* get_path_segment() const;

  private:
    float delta_time;                     // time step size
    float current_time;                   // current t in range [0..1]
    float end_time;                       // end time
    float end_time_with_eps;              // end time including a small negative epsilon
    float current_joint_pos[NUM_JOINTS];  // current joint positions

    const CartesianPathSegment* path_segment = nullptr;
    IKinematicModel* kinematic_model;
};

