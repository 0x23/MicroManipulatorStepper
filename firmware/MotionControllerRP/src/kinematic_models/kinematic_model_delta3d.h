// --------------------------------------------------------------------------------------
// Project: MicroManipulatorStepper
// License: MIT (see LICENSE file for full description)
//          All text in here must be included in any redistribution.
// Author:  M. S. (diffraction limited)
// --------------------------------------------------------------------------------------

#pragma once

//*** INCLUDE ***************************************************************************

#include "kinematic_model_base.h"
#include "utilities/math3d.h"

//*** CLASS *****************************************************************************

class Pose6DF;

//--- KinematicModel_Delta3D ------------------------------------------------------------

class KinematicModel_Delta3D : public IKinematicModel {
  public:
    KinematicModel_Delta3D();

    int get_joint_count();
    bool foreward(const float* joint_positions, Pose6DF& pose) override;
    bool inverse(const Pose6DF& pose, float* joint_positions) override;

    void test();

  protected:
    Vec3F arm_attachment_point(int joint_idx, float rotor_angle);

  public:
    float arm_length;
    float rotor_radius;
    float rotor_angle_offset[3];   // angle offset of the neutral position from zero position in rad
    Pose6DF base_to_actuator[3];
    Pose6DF actuator_to_base[3];
    Vec3F ee_attachment_points[3]; // EE arm attachment points in endeffector coordinate system
};

//*** FUNCTION **************************************************************************

bool circle_sphere_intersection(float r1, const Vec3F& p, float r2, Vec3F intersections[2]);
bool three_sphere_intersection(const Vec3F& p1, float r1,
                               const Vec3F& p2, float r2,
                               const Vec3F& p3, float r3,
                               Vec3F intersections[2]);
