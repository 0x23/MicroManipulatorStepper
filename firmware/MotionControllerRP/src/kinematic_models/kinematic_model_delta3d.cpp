// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

//*** INCLUDE ***************************************************************************

#include "kinematic_model_delta3d.h"
#include "utilities/math3d.h"
#include "utilities/math_constants.h"
#include "utilities/logging.h"

//*** CLASS *****************************************************************************

#define HW_VERSION4 // remove this if you are using HW-v3 (Note geometry parameters might be slightly off for HW-v3.0!)

#ifdef HW_VERSION4

/**
 * Initializes the kinematic model and its geometric parameters for hardware version v4.0.
 */
KinematicModel_Delta3D::KinematicModel_Delta3D() {
  const float D2R = Constants::DEG2RAD;

  // location of base origin in endeffector coordinate system (ee in neutral position)
  Vec3F base_offset(-47.5f, -47.5f, -47.5f);
  
  // distance between the two spehere centers of the linkage rod (arms)
  arm_length = 36.25f*2;

  // distance from arm attachment points to rotor axis
  rotor_radius = 15.0f;

  // midpoint between the two attachment spheres on the endeffector in EE coordinate system
  ee_attachment_points[0] = Vec3F(3.54f, -11.5f, 4.5f);
  ee_attachment_points[1] = Vec3F(4.5f, 3.54f, -11.5f);
  ee_attachment_points[2] = Vec3F(-11.5f, 4.5f, 3.54f);

  // set actuator transformations based on CAD model
  actuator_to_base[0].rotation = QuaternionF::from_axis_angle(Vec3F(0.0f, 0.0f, 1.0f), 90.0f*D2R);
  actuator_to_base[0].translation = Vec3F(-21.5f, 21.0f, 52.0f)+base_offset;

  actuator_to_base[1].rotation = QuaternionF::from_axis_angle(Vec3F(1.0f, 0.0f, 1.0f), 180.0f*D2R);
  actuator_to_base[1].translation = Vec3F(52.0f, -21.5f, 21.0f)+base_offset;

  actuator_to_base[2].rotation = QuaternionF::from_axis_angle(Vec3F(-1.0f, 0.0f, 0.0f), 90.0f*D2R);
  actuator_to_base[2].translation = Vec3F(21.0f, 52.0f, -21.5f)+base_offset;
  
  // angle offset from home positionto center position of the rotor
  rotor_angle_offset[0] = 42.0f*Constants::DEG2RAD;
  rotor_angle_offset[1] = 42.0f*Constants::DEG2RAD;
  rotor_angle_offset[2] = 42.0f*Constants::DEG2RAD;

  for(int i=0; i<3; i++)
    base_to_actuator[i] = actuator_to_base[i].inverse();
}

#else

// Below is the old code for HW-v3.0 for compatibility, some values might be wrong and do not reflect the actual cad model.
// If you use them please check them against your build. All these problems where fixed in HW-v4 and the new function above
// uses the correct values.
KinematicModel_Delta3D::KinematicModel_Delta3D() {
  const float D2R = Constants::DEG2RAD;

  // offset to move base origin defined in CAD to endeffector origin near neutral position
  
  // REAL DEVICE <--- These might be slightly wrong
  Vec3F base_offset(-32.5f, -32.5f, -32.5f);
  arm_length = 73.8f;
  rotor_radius = 15.0f;
  ee_attachment_points[0] = Vec3F(-0.5f, -14.5f, 2.0f);
  ee_attachment_points[1] = Vec3F(2.0f, -0.5f, -14.5f);
  ee_attachment_points[2] = Vec3F(-14.5f, 2.0f, -0.5f);

  // CAD <--- These might be be more correct but not sure
  /*
  Vec3F base_offset(-30.5f, -30.5f, -30.5f);
  arm_length = 2*36.5;
  rotor_radius = 15.0f;
  ee_attachment_points[0] = Vec3F(0.5f, -15.0f, 1.5f);
  ee_attachment_points[1] = Vec3F(1.5f, 0.5f, -15.0f);
  ee_attachment_points[2] = Vec3F(-15.0f, 1.5f, 0.5f);
  */

  // set transformation based on CAD model
  actuator_to_base[0].rotation = QuaternionF::from_axis_angle(Vec3F(0.0f, 0.0f, 1.0f), 90.0f*D2R);
  actuator_to_base[0].translation = Vec3F(-42.0f, 0.5f, 32.0f)+base_offset;

  actuator_to_base[1].rotation = QuaternionF::from_axis_angle(Vec3F(1.0f, 0.0f, 1.0f), 180.0f*D2R);
  actuator_to_base[1].translation = Vec3F(32.0f, -42.0f, 0.5f)+base_offset;

  actuator_to_base[2].rotation = QuaternionF::from_axis_angle(Vec3F(-1.0f, 0.0f, 0.0f), 90.0f*D2R);
  actuator_to_base[2].translation = Vec3F(0.5f, 32.0f, -42.0f)+base_offset;

  rotor_angle_offset[0] = 46.2f*Constants::DEG2RAD;
  rotor_angle_offset[1] = 46.2f*Constants::DEG2RAD;
  rotor_angle_offset[2] = 46.2f*Constants::DEG2RAD;

  for(int i=0; i<3; i++)
    base_to_actuator[i] = actuator_to_base[i].inverse();
}

#endif

int KinematicModel_Delta3D::get_joint_count() {
  return 3;
}

bool KinematicModel_Delta3D::foreward(const float* joint_positions, Pose6DF& pose) {
  Vec3F arm_attachment_points[3];
  for(int i=0; i<3; i++) {
    Vec3F p = arm_attachment_point(i, joint_positions[i]);
    p = actuator_to_base[i].transformPoint(p);

    // apply ee attachment point offsets so that three sphere intersection can be used 
    // to find ee position. This only works if there is no ee rotation.
    arm_attachment_points[i] = p-ee_attachment_points[i];
  }

  // compute three sphere intersection
  Vec3F intersections[2];
  bool ok = three_sphere_intersection(arm_attachment_points[0], arm_length,
                                      arm_attachment_points[1], arm_length,
                                      arm_attachment_points[2], arm_length,
                                      intersections);
  if(!ok) return false;


  // select correct solution
  Vec3F q = intersections[0].x > intersections[1].x ? intersections[0] : intersections[1];

  pose.translation = q;

  return true;
}

bool KinematicModel_Delta3D::inverse(const Pose6DF& pose, float* joint_positions) {
  for(int i=0; i<3; i++) { 
    // get ee attachment points in base coordinate system
    Vec3F intersections[2];
    Vec3F p = pose.transformPoint(ee_attachment_points[i]);
    // LOG_INFO("  p(base) = %f %f %f", p.x, p.y, p.z);

    // transform attachment point to actuator coordinates
    p = base_to_actuator[i].transformPoint(p);

    // compute intersection points
    bool ok = circle_sphere_intersection(rotor_radius, p, arm_length, intersections);
    if(!ok) return false;

    // select correct solution based on x-position (in actuator coordinates)
    Vec3F q = intersections[0].x > intersections[1].x ? intersections[0] : intersections[1];

    // compute joint angle
    float angle = -atan2(q.y, q.x);                     // joint angles are cw
    joint_positions[i] = rotor_angle_offset[i] + angle;
  }

  return true;
}

// returns the arm attachment point on the rotor for a given rotor angle
Vec3F KinematicModel_Delta3D::arm_attachment_point(int joint_idx, float rotor_angle) {
  // Note: rotor angle is defined clockwise so 0 is the retracted state
  rotor_angle -= rotor_angle_offset[joint_idx];
  return Vec3F(cos(rotor_angle), -sin(rotor_angle), 0)*rotor_radius;
}

void KinematicModel_Delta3D::test() {
  float D2R = Constants::DEG2RAD;

  {
    LOG_INFO("\n# Foreward Kinematic");
    float joint_pos[3] = {45*D2R, 45*D2R, 45*D2R};
    Pose6DF pose;
    foreward(joint_pos, pose);
    Vec3F p = pose.translation;
    LOG_INFO("%f %f %f", p.x, p.y, p.z);
  }

  {
    LOG_INFO("\n# Inverse Kinematic");
    Vec3F p(0.0, 0.0, 0.0);
    Pose6DF pose(p, QuaternionF());
    float joint_pos[3];
    inverse(pose, joint_pos);
    LOG_INFO("%f %f %f", joint_pos[0]/D2R, joint_pos[1]/D2R, joint_pos[2]/D2R);
  }
  
  LOG_INFO("\n# Rotor Attachment Points");
  for(int i=0; i<3; i++) {
    for(float angle=0.0f; angle<90.0f; angle+=10.0f) {
      Vec3F p = arm_attachment_point(i, angle*Constants::DEG2RAD);
      p = actuator_to_base[i].transformPoint(p);
      LOG_INFO("%f %f %f", p.x, p.y, p.z);
    }
  }
}

//*** FUNCTION **************************************************************************

/**
 * @brief Computes the intersection points between a circle in the XY-plane and a 3D sphere.
 *
 * Given:
 *   - A circle centered at the origin (0,0,0) in the XY-plane with radius `r1`.
 *   - A sphere centered at position `p` with radius `r2`.
 *
 * The function computes up to two 3D intersection points where the sphere intersects
 * the plane of the circle, and those points lie on the given circle.
 *
 * @param r1             Radius of the circle (must be >= 0).
 * @param p              Center of the sphere (Vec3F: x, y, z).
 * @param r2             Radius of the sphere (must be >= 0).
 * @param intersections  Output array of 2 Vec3F points. If there is an intersection,
 *                       both points are filled.
 *
 * @return true if there is at least one intersection point (either one or two),
 *         false if there is no intersection.
 */
bool circle_sphere_intersection(double r1, const Vec3F& p, double r2, Vec3F intersections[2]) {
    const double px = p.x;
    const double py = p.y;
    const double pz = p.z;
    const double r2_sq = r2 * r2;
    const double pz_sq = pz * pz;

    // Check if sphere intersects XY-plane and projected radius of sphere-circle in XY
    const double r_proj_sq = r2_sq - pz_sq;
    if (r_proj_sq < 0.0f) 
      return false;
    const double r_proj = std::sqrt(r_proj_sq);

    // Distance squared between circle centers
    const double d_sq = px * px + py * py;

    // Check if circles intersect
    const double sum_r = r1 + r_proj;
    const double diff_r = std::abs(r1 - r_proj);
    if (d_sq > sum_r * sum_r || d_sq < diff_r * diff_r)
      return false;

    // Distance between circle centers and its inverse
    const double d = std::sqrt(d_sq);
    const double inv_d = 1.0 / d;

    // a = (r1^2 - r2^2 + d^2) / (2d)
    const double r1_sq = r1 * r1;
    const double a = (r1_sq - r_proj_sq + d_sq) * 0.5f * inv_d;

    // h = sqrt(r1^2 - a^2)
    const double h_sq = r1_sq - a * a;
    if (h_sq < 1e-8f) 
      return false;  // numerical precision issue
    const double h = std::sqrt(h_sq);

    // Base point (cx2, cy2)
    const double cx2 = px * (a * inv_d);
    const double cy2 = py * (a * inv_d);

    // Offset vector
    const double rx = -py * (h * inv_d);
    const double ry =  px * (h * inv_d);

    intersections[0] = Vec3F(cx2 + rx, cy2 + ry, 0.0f);
    intersections[1] = Vec3F(cx2 - rx, cy2 - ry, 0.0f);

    return true;
}

/**
 * @brief Computes the intersection points of three spheres in 3D space.
 *
 * Given three spheres defined by their centers (p1, p2, p3) and radii (r1, r2, r3),
 * this function computes up to two points where all three spheres intersect.
 * Returns false if no real intersection exists (e.g., spheres are too far apart or nearly tangent).
 *
 * @param p1 Center of the first sphere
 * @param r1 Radius of the first sphere
 * @param p2 Center of the second sphere
 * @param r2 Radius of the second sphere
 * @param p3 Center of the third sphere
 * @param r3 Radius of the third sphere
 * @param intersections Output array of two Vec3F points where the spheres intersect
 * @return true if a real intersection exists (two points), false otherwise
 */
bool three_sphere_intersection(const Vec3F& p1, float r1,
                               const Vec3F& p2, float r2,
                               const Vec3F& p3, float r3,
                               Vec3F intersections[2])
{
  const float eps = 1e-7f;
  const float r1_sqr = r1 * r1;

  // Compute unit vector ex from p1 to p2
  Vec3F ex = p2 - p1;
  float d2 = ex.sqr_length();
  if (d2 < eps)
    return false;

  float d = std::sqrt(d2);
  float inv_d = 1.0f / d;
  ex = ex * inv_d;

  // Project p3 onto ex to compute scalar i
  Vec3F temp = p3 - p1;
  float i = ex.dot(temp);

  // Compute unit vector ey perpendicular to ex
  Vec3F ey = temp - ex * i;
  float ey2 = ey.sqr_length();
  if (ey2 < eps)
    return false;

  float inv_ey = 1.0f / std::sqrt(ey2);
  ey = ey * inv_ey;
  float j = ey.dot(temp);

  // Compute unit vector ez orthogonal to ex and ey
  Vec3F ez = ex.cross(ey);

  // Compute x and y coordinates in ex/ey plane
  float x = (r1_sqr - r2 * r2 + d * d) * 0.5f * inv_d;
  float y = (r1_sqr - r3 * r3 + i * i + j * j - 2.0f * i * x) * 0.5f * inv_ey;

  // Compute z coordinate along ez axis
  float z2 = r1_sqr - x * x - y * y;
  if (z2 < eps) return false;  // no real solution
  float z = std::sqrt(z2);

  // Compute the two possible intersection points
  Vec3F base = p1 + ex * x + ey * y;
  intersections[0] = base + ez * z;
  intersections[1] = base - ez * z;

  return true;
}