/*
 * Exact C-accelerated Venn-Abers calibration matching Algorithms 1-6
 * from Vovk, Petej & Fedorova (2015) and the official venn-abers package.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

void c_calc_p0p1(
    int k_dash,
    const double* w_cumsum,
    const double* y_cumsum,
    double* out_p0,
    double* out_p1
) {
    int n_pts = k_dash + 1;
    
    // 1. Calculate p1 (test point assumed label 1)
    double* P1_x = (double*)malloc(n_pts * sizeof(double));
    double* P1_y = (double*)malloc(n_pts * sizeof(double));
    
    P1_x[0] = 1.0;
    P1_y[0] = 1.0;
    for (int j = 1; j < n_pts; j++) {
        P1_x[j] = w_cumsum[j - 1] + 1.0;
        P1_y[j] = y_cumsum[j - 1] + 1.0;
    }
    
    double grad1 = 0.0;
    int c_point1 = 0;
    
    for (int i = 0; i < n_pts; i++) {
        P1_x[i] -= 1.0;
        P1_y[i] -= 1.0;
        
        if (i == 0) {
            double min_g = 1e300;
            for (int k = 1; k < n_pts; k++) {
                double dx = P1_x[k];
                if (dx > 0) {
                    double g = P1_y[k] / dx;
                    if (g < min_g) min_g = g;
                }
            }
            grad1 = min_g;
            out_p1[i] = grad1;
            c_point1 = 0;
        } else {
            double imp_point = P1_y[c_point1] + (P1_x[i] - P1_x[c_point1]) * grad1;
            if (P1_y[i] < imp_point) {
                double min_g = 1e300;
                int found = 0;
                for (int k = i + 1; k < n_pts; k++) {
                    double dx = P1_x[k] - P1_x[i];
                    if (dx > 0) {
                        double g = (P1_y[k] - P1_y[i]) / dx;
                        if (g < min_g) { min_g = g; found = 1; }
                    }
                }
                if (found) grad1 = min_g;
                c_point1 = i;
                out_p1[i] = grad1;
            } else {
                out_p1[i] = grad1;
            }
        }
    }
    free(P1_x);
    free(P1_y);
    
    // 2. Calculate p0 (test point assumed label 0)
    double* P0_x = (double*)malloc(n_pts * sizeof(double));
    double* P0_y = (double*)malloc(n_pts * sizeof(double));
    
    P0_x[0] = 0.0;
    P0_y[0] = 0.0;
    for (int j = 1; j < n_pts; j++) {
        P0_x[j] = w_cumsum[j - 1];
        P0_y[j] = y_cumsum[j - 1];
    }
    
    double grad0 = 0.0;
    int c_point0 = n_pts - 1;
    
    for (int i = n_pts - 1; i >= 0; i--) {
        P0_x[i] += 1.0;
        
        if (i == n_pts - 1) {
            double max_g = -1e300;
            for (int k = 0; k < n_pts - 1; k++) {
                double dx = P0_x[k] - P0_x[i];
                if (dx != 0) {
                    double g = (P0_y[k] - P0_y[i]) / dx;
                    if (g > max_g) max_g = g;
                }
            }
            grad0 = max_g;
            out_p0[i] = grad0;
            c_point0 = i;
        } else {
            double imp_point = P0_y[c_point0] + (P0_x[i] - P0_x[c_point0]) * grad0;
            if (P0_y[i] < imp_point) {
                double max_g = -1e300;
                for (int k = 0; k < i; k++) {
                    double dx = P0_x[k] - P0_x[i];
                    if (dx != 0) {
                        double g = (P0_y[k] - P0_y[i]) / dx;
                        if (g > max_g) max_g = g;
                    }
                }
                grad0 = max_g;
                c_point0 = i;
                out_p0[i] = grad0;
            } else {
                out_p0[i] = grad0;
            }
        }
    }
    free(P0_x);
    free(P0_y);
}
