#include "kalman_filter.h"
#include <iostream>
#include "tools.h"
#include <cmath>
#include <stdlib.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
using std::vector;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;

	MatrixXd I = MatrixXd::Identity(4, 4);
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */


  float z_rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  float z_phi = 0;
  float z_rho_dot =0;

  if(fabs(x_(0)) < 0.0001){
       cout << "Error1" << endl;
   }else{
       z_phi = atan2(x_(1),x_(0));
   }

   if (z_rho < 0.0001) {
       cout << "Error2" << endl;
   }else{

  z_rho_dot = (x_(0)*x_(2)+x_(1)*x_(3))/z_rho;
  }
  VectorXd h(3);
  h << z_rho,z_phi,z_rho_dot;

  VectorXd y = z - h;

  float PI = 3.1415926;

  while ( y(1) < -PI) {
          y(1)  +=  2*PI;
      }
      while (y(1) > PI) {
          y(1)  -= 2*PI;
      }

      if (y(1)  > PI || y(1) < -PI) {
          cout << "Error" << endl;
      }

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;


	MatrixXd I = MatrixXd::Identity(4, 4);
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}
