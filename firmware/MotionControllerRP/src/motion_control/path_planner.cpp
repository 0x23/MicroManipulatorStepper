// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#include "hardware/sync.h"

#include "path_planner.h"
#include "utilities/logging.h"

#include <algorithm>

PathPlanner::PathPlanner(IKinematicModel* kinematic_model, float time_step) {
  PathPlanner::segment_time_step = time_step;
  PathPlanner::kinematic_model = kinematic_model;

  junction_deviation.linear = 0.0001;  // mm
  junction_deviation.angular = 0.001; // rad
}

PathPlanner::~PathPlanner() {
}

void PathPlanner::set_kinematic_model(IKinematicModel* kinematic_model) {
  PathPlanner::kinematic_model = kinematic_model;
}

bool PathPlanner::add_cartesian_path_segment(const CartesianPathSegment& path_segment) {
  auto* new_segment = ct_path_segment_queue.push(path_segment);
  if(new_segment == nullptr) {
    // queue full
    return false;
  }

  return true;
}

void PathPlanner::process(bool disable_interrupts_for_queue_update) {
  // create new segment generator for next cartesian path segment
  // the segment stays in the queue until it is completed
  if(segment_generator == nullptr && ct_path_segment_queue.empty() == false) {
    auto* current_segment = ct_path_segment_queue.peek();

    // compute motion profile ( active segment can not change anymore )
    current_segment->compute_motion_profile();

    // create path segment generator
    segment_generator = new JointSpacePathSegmentGenerator(current_segment, 
                                                           kinematic_model,
                                                           segment_time_step);

    /*LOG_INFO("Starting segment: duration=%fs, (%f, %f, %f)->(%f, %f, %f) | queue size: %i", 
      current_segment->get_duration(),
      current_segment->start_pose.translation.x,
      current_segment->start_pose.translation.y,
      current_segment->start_pose.translation.z,
      current_segment->end_pose.translation.x,
      current_segment->end_pose.translation.y,
      current_segment->end_pose.translation.z,
      ct_path_segment_queue.size()); */
  }

  // generate joint space segment
  if(segment_generator != nullptr && js_path_segment_queue.full() == false) {
    JointSpacePathSegment segment;
    bool end_reached = segment_generator->generate_next(segment);

    // update output queue
    if(disable_interrupts_for_queue_update) {
      uint32_t status = save_and_disable_interrupts();
      js_path_segment_queue.push(segment);
      restore_interrupts(status); 
    } else {
      js_path_segment_queue.push(segment);
    }

   // LOG_INFO("Adding joint space segment: [%f, %f, %f] -> [%f, %f, %f]", 
   //   segment.start_pos[0], segment.start_pos[1],  segment.start_pos[2],
   //   segment.end_pos[0], segment.end_pos[1],  segment.end_pos[2]);

    // check if current cartesian path segmetn is completed
    if(end_reached) {
      // remove current cartesian path segment from ringbuffer
      ct_path_segment_queue.pop();
      // destroy segment generator
      delete segment_generator;
      segment_generator = nullptr;
    }
  }
}

/**
 * retrieve 
 */
bool PathPlanner::pop_js_path_segment(JointSpacePathSegment& segment) {
  return js_path_segment_queue.pop(segment);
}

bool PathPlanner::all_finished() {
  return js_path_segment_queue.empty() && ct_path_segment_queue.empty() && segment_generator == nullptr;
}

int PathPlanner::input_queue_full() {
  return ct_path_segment_queue.full();
}

int PathPlanner::input_queue_size() {
  return ct_path_segment_queue.size();
}

void PathPlanner::run_look_ahead_planning() {
  if(ct_path_segment_queue.empty())
    return;

  int n = ct_path_segment_queue.size();
  auto* last_segment = ct_path_segment_queue.get(n - 1);
  if(segment_generator != nullptr && last_segment == segment_generator->get_path_segment())
    return;

  // --- Reverse pass ---
  // Start from the last segment, set its end velocity to zero (or target)
  last_segment->end_velocity = LinearAngular{0, 0};
  
  // Propagate backward the feasible start velocities
  for (int i = n - 2; i >= 0; i--) {
    CartesianPathSegment* s1 = ct_path_segment_queue.get(i);
    CartesianPathSegment* s2 = ct_path_segment_queue.get(i + 1);

    // dont change active segment
    if(segment_generator != nullptr && s1 == segment_generator->get_path_segment()) {
      s2->start_velocity = s1->end_velocity;
      continue;
    }
  
    // Get minimum acceleration capability at junction
    float acc = std::min(s1->max_acceleration.linear, s2->max_acceleration.linear);

    // Compute max junction velocity feasible at junction between s1 and s2
    float max_junction_velocity_linear = compute_max_junction_velocity(
      s1->translation_delta_normalized,
      s2->translation_delta_normalized,
      acc,
      junction_deviation.linear
    );

    // no angular junction velocity limit yet
    float max_junction_velocity_angular = 1e10;

    // compute max velocity delta
    // In the reverse pass the junction is the start of s2, so the limit comes
    // from s2's ability to decelerate to its already resolved end velocity.
    float v_start_max_linear = std::sqrt(std::max(0.0f, powf(s2->end_velocity.linear, 2.0f) + 2 * s2->max_acceleration.linear * s2->travel_distance.linear));
    float v_start_max_angular = std::sqrt(std::max(0.0f, powf(s2->end_velocity.angular, 2.0f) + 2 * s2->max_acceleration.angular * s2->travel_distance.angular));

    // compute final junction velocity
    LinearAngular junction_velocity;
    junction_velocity.linear = std::min(max_junction_velocity_linear, std::min(s2->target_velocity.linear, v_start_max_linear));
    junction_velocity.angular = std::min(max_junction_velocity_angular, std::min(s2->target_velocity.angular, v_start_max_angular));
    
    // set calculated velocity to both path segments
    s1->end_velocity = junction_velocity;
    s2->start_velocity = junction_velocity;
  }

  // --- Forward pass ---
  for (int i = 0; i < n - 1; i++) {
    CartesianPathSegment* s1 = ct_path_segment_queue.get(i);
    CartesianPathSegment* s2 = ct_path_segment_queue.get(i + 1);

    // dont change active segment
    if(segment_generator != nullptr && s1 == segment_generator->get_path_segment()) {
      s2->start_velocity = s1->end_velocity;
      continue;
    }

    float v_start = s1->start_velocity.linear;
    float v_end_max_linear = std::sqrt(std::max(0.0f, powf(s1->start_velocity.linear, 2.0f) + 2 * s1->max_acceleration.linear * s1->travel_distance.linear));
    float v_end_max_angular = std::sqrt(std::max(0.0f, powf(s1->start_velocity.angular, 2.0f) + 2 * s1->max_acceleration.angular * s1->travel_distance.angular));

    LinearAngular junction_velocity;
    if(s1->end_velocity.linear > v_end_max_linear) s1->end_velocity.linear = v_end_max_linear;
    if(s1->end_velocity.angular > v_end_max_angular) s1->end_velocity.angular = v_end_max_angular;

    s2->start_velocity = s1->end_velocity;
  }

  // debug
  // print_cartesian_path_segments();
}

float PathPlanner::compute_max_junction_velocity(const Vec3F& dir_in_normalized, const Vec3F& dir_out_normalized, float acceleration, float junction_deviation) {
    const float EPSILON = 1e-6f;
    const float COS_NEAR_STRAIGHT = 0.9999f;
    const float COS_NEAR_OPPOSITE = -0.9999f;

    // Compute the cosine of the angle between the directions (negative dot product)
    float cos_theta = -dir_in_normalized.dot(dir_out_normalized);

    // Compute sin(θ/2) using half-angle identity: sin²(θ/2) = (1 - cosθ) / 2
    float sin_theta_d2 = std::sqrt(std::max(0.0f, 0.5f * (1.0f - cos_theta)));

    // Compute vmax using classical junction deviation formula
    float denom = std::max(1.0f - sin_theta_d2, EPSILON);
    float vmax = std::sqrt(acceleration * junction_deviation * sin_theta_d2 / denom);

    return vmax;
}

void PathPlanner::print_cartesian_path_segments() {
  LOG_INFO("Cartesian Path Segment Info");
  int n = ct_path_segment_queue.size();
  for (int i = 0; i < n; i++) {
    CartesianPathSegment* s = ct_path_segment_queue.get(i);
    LOG_INFO("  Segment %02i: [%f, %f, %f]->[%f, %f, %f]  l=%f  vs=%fmm/s   ve=%fmm/s", i, 
      s->start_pose.translation.x, s->start_pose.translation.y, s->start_pose.translation.z,  
      s->end_pose.translation.x, s->end_pose.translation.y, s->end_pose.translation.z,  
      s->travel_distance.linear,
      s->start_velocity.linear, s->end_velocity.linear);
  }
}

