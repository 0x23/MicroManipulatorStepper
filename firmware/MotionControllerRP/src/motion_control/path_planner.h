// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

//*** INCLUDE ***************************************************************************

#include "path_segment.h"
#include "utilities/ringbuffer.h"

//*** CLASS *****************************************************************************

class IKinematicModel;

//--- PathPlanner -----------------------------------------------------------------------

class PathPlanner {
  public:
    static constexpr int CT_QUEUE_SIZE = 32;
    static constexpr int JS_QUEUE_SIZE = 32;
    
  public:
    PathPlanner(IKinematicModel* kinematic_model, float time_step, LinearAngular junction_deviation);
    ~PathPlanner();

    // sets the kinematic model for foreward and inverse kinematic calculations
    void set_kinematic_model(IKinematicModel* kinematic_model);

    // adds a new cartesian space path segment to the planner queue
    bool add_cartesian_path_segment(const CartesianPathSegment& path_segment);

    // runs look ahead path planning. call this everytime after one or more cartesian
    // path segments have been added
    void run_look_ahead_planning();

    // Retrieves the next joint space path segment from the queue, returns false
    // if queue is empty.
    bool pop_js_path_segment(JointSpacePathSegment& segment);

    // Processes the queued cartesian path segments and generates one 
    // joint space path segment if possible. Call this repeatedly.
    void process(bool disable_interrupts_for_queue_update);

    // returns true if all ques are empty and if everything is finished
    bool all_finished();

    // returns the number of free items in the input queue
    int input_queue_full();

    // returns the current number of queued items
    int input_queue_size();

  private:
    float compute_max_junction_velocity(
      const Vec3F& dir_in_normalized, 
      const Vec3F& dir_out_normalized, 
      float acceleration, 
      float junction_deviation);

    void print_cartesian_path_segments();

  private:
    RingBuffer<CartesianPathSegment, CT_QUEUE_SIZE> ct_path_segment_queue;
    RingBuffer<JointSpacePathSegment, JS_QUEUE_SIZE> js_path_segment_queue;

    IKinematicModel* kinematic_model;
    JointSpacePathSegmentGenerator* segment_generator = nullptr;
    float segment_time_step;
    LinearAngular junction_deviation;
};

