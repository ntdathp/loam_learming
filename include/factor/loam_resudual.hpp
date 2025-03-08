#ifndef LOAM_RESIDUAL_HPP
#define LOAM_RESIDUAL_HPP

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <ceres/rotation.h>

#include "utility.h"

struct LoamResidual {
    // Initialize from a LidarCoef object containing necessary data (e.g., f and target)
    LoamResidual(const LidarCoef &coef) : coef_(coef) {}

    template <typename T>
    bool operator()(const T* const pose, T* residual) const {
        // pose: [qw, qx, qy, qz, tx, ty, tz]
        T p[3];
        p[0] = T(coef_.f.x());
        p[1] = T(coef_.f.y());
        p[2] = T(coef_.f.z());

        T p_trans[3];
        // Use Ceres utility function to rotate the point with the quaternion
        ceres::QuaternionRotatePoint(pose, p, p_trans);
        // Add translation
        p_trans[0] += pose[4];
        p_trans[1] += pose[5];
        p_trans[2] += pose[6];

        // Compute residual
        residual[0] = p_trans[0];
        residual[1] = p_trans[1];
        residual[2] = p_trans[2];
        return true;
    }

private:
    const LidarCoef coef_;
};

#endif // LOAM_RESIDUAL_HPP
