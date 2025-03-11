/*******************************************************************************
 * 예시: MATLAB 코드를 C로 포팅한 샘플 (라즈베리 파이 구동 가정)
 * 실제 동작/최적화를 위해서는 추가 수정이 필요함
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "fftw3.h"
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

//=====================================================================
// 레이더 파라미터 (예시)
//=====================================================================
// 아래 파라미터들은 예시로 남겨두었으나, 실제 코드에서는 사용되지 않으므로 주석 처리함
// static const double chirp_time = 800e-6;     // 한 chirp 시간
// static const double Start_freq = 93e9;         // FMCW 시작 주파수
// static const double End_freq   = 95e9;         // FMCW 끝 주파수
// static const double fs         = 5e6;          // ADC 샘플링 속도
// static const double c          = 3e8;          // 빛의 속도
// static const int    idle_point = 20;           // 앞부분 샘플 버리기

static const int chirpperframe = 121;   // 1프레임 당 chirp 개수(120+1)
static const int Nd            = 96;    // velocity FFT 크기
static const double alpha      = 0.593328; // DOA 해석 시 스케일링

//=====================================================================
// FFT 함수 (1D)
//=====================================================================
/**
 * fft_1d: 실수 배열 입력 → 복소수 출력 (fftw_complex 활용)
 *   - in : double* 크기 n
 *   - out: fftw_complex* 크기 n
 *   - n  : 점수
 *   - norm_scale : 1/n 등의 정규화를 여기서 적용 가능
 */
void fft_1d(const double *in, fftw_complex *out, int n, double norm_scale)
{
    fftw_plan p = fftw_plan_dft_r2c_1d(n, (double*)in, out, FFTW_ESTIMATE);
    fftw_execute(p);
    for (int i = 0; i < n; i++){
        out[i] *= norm_scale;  // 복소수 전체에 스케일 적용
    }
    fftw_destroy_plan(p);
}

/**
 * ifft_1d: 복소수 배열 입력 → 복소수 출력
 *   - in : fftw_complex* 크기 n
 *   - out: fftw_complex* 크기 n
 *   - n  : 점수
 */
void ifft_1d(const fftw_complex *in, fftw_complex *out, int n, double norm_scale)
{
    fftw_plan p = fftw_plan_dft_1d(n, (fftw_complex*)in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    for (int i = 0; i < n; i++){
        out[i] *= norm_scale;
    }
    fftw_destroy_plan(p);
}

//=====================================================================
// 2D FFT 예시 (복소수 배열 입력 → 복소수 배열 출력)
//=====================================================================
/**
 * fft_2d: (nrow x ncol) 크기의 복소수 배열에 대해 2D FFT
 *        in[row][col], out[row][col]
 */
void fft_2d(fftw_complex *in, fftw_complex *out, int nrow, int ncol)
{
    fftw_complex *row_in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ncol);
    fftw_complex *row_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ncol);
    fftw_plan p_row = fftw_plan_dft_1d(ncol, row_in, row_out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int r = 0; r < nrow; r++){
        memcpy(row_in, &in[r * ncol], sizeof(fftw_complex) * ncol);
        fftw_execute(p_row);
        memcpy(&out[r * ncol], row_out, sizeof(fftw_complex) * ncol);
    }
    fftw_destroy_plan(p_row);

    fftw_complex *col_in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nrow);
    fftw_complex *col_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nrow);
    fftw_plan p_col = fftw_plan_dft_1d(nrow, col_in, col_out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int c = 0; c < ncol; c++){
        for (int r = 0; r < nrow; r++){
            col_in[r] = out[r * ncol + c];
        }
        fftw_execute(p_col);
        for (int r = 0; r < nrow; r++){
            out[r * ncol + c] = col_out[r];
        }
    }
    fftw_destroy_plan(p_col);

    fftw_free(row_in);
    fftw_free(row_out);
    fftw_free(col_in);
    fftw_free(col_out);

    double norm = 1.0 / (double)(nrow * ncol);
    for (int r = 0; r < nrow * ncol; r++){
        out[r] *= norm;
    }
}

//=====================================================================
// (예시) CSV 파일을 읽어 double 배열에 넣는 함수 - 간단 구현
//=====================================================================
int read_csv_double(const char *filename, double *out, int max_len)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        return -1;
    }
    int count = 0;
    while (!feof(fp) && count < max_len) {
        if (fscanf(fp, "%lf,", &out[count]) == 1){
            count++;
        } else {
            break;
        }
    }
    fclose(fp);
    return count;
}

//=====================================================================
// 2D CFAR 예시
//=====================================================================
void cfar2D(const double *range_doppler_in, int n_doppler, int n_range,
            int num_train_r, int num_train_d, int num_guard_r, int num_guard_d,
            double threshold_scale, int start_cfar, int end_cfar,
            int *cfar_mask_out)
{
    memset(cfar_mask_out, 0, sizeof(int) * n_doppler * n_range);

    for (int r = (num_train_r + num_guard_r); r < (n_doppler - num_train_r - num_guard_r); r++){
        for (int d = (num_train_d + num_guard_d + start_cfar);
             d < MIN(n_range - num_train_d - num_guard_d, end_cfar); d++){

            double noise_sum = 0.0;
            int count = 0;
            for (int rr = r - (num_train_r + num_guard_r); rr <= r + (num_train_r + num_guard_r); rr++){
                for (int dd = d - (num_train_d + num_guard_d); dd <= d + (num_train_d + num_guard_d); dd++){
                    if ((abs(rr - r) > num_guard_r) || (abs(dd - d) > num_guard_d)){
                        noise_sum += range_doppler_in[rr * n_range + dd];
                        count++;
                    }
                }
            }

            double noise_avg = (count > 0) ? (noise_sum / count) : 0.0;
            double threshold = noise_avg * threshold_scale;

            double val = range_doppler_in[r * n_range + d];
            if (val > threshold){
                cfar_mask_out[r * n_range + d] = 1;
            }
        }
    }
}

//=====================================================================
// main
//=====================================================================
int main(int argc, char *argv[])
{
    // 사용하지 않는 매개변수 경고 제거
    (void)argc;
    (void)argv;

    // CSV 관련 행렬 크기
    const int pt = 4000;              // fast-time 샘플 수
    const int row_size = 8 * chirpperframe; // 8 Rx * chirpperframe
    const int col_size = pt;
    double *Data_target   = (double*)malloc(sizeof(double) * row_size * col_size);
    double *Data_notarget = (double*)malloc(sizeof(double) * row_size * col_size);

    // 더미 데이터 채우기
    for (int i = 0; i < row_size * col_size; i++){
        Data_target[i]   = (double)(i % 4096);
        Data_notarget[i] = 0.0;
    }

    // Range-Doppler 맵을 위한 배열 크기
    int range_size = pt;    // fast-time
    int doppler_size = Nd;  // slow-time
    fftw_complex *range_doppler = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * range_size * doppler_size);
    fftw_complex *temp_buf      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * range_size * doppler_size);

    // 더미 데이터 (예시)
    for (int i = 0; i < range_size * doppler_size; i++){
        range_doppler[i] = 100.0 + 0.0 * I;
    }

    // 2D FFT 수행
    fft_2d(range_doppler, temp_buf, doppler_size, range_size);

    // 로그 스케일 변환
    double *range_doppler_log = (double*)malloc(sizeof(double) * range_size * doppler_size);
    double max_val = -1e9;
    for (int i = 0; i < range_size * doppler_size; i++){
        double mag = cabs(temp_buf[i]);
        double val_db = 20.0 * log10(mag / 10.0) + 30.0;
        range_doppler_log[i] = val_db;
        if (val_db > max_val)
            max_val = val_db;
    }

    // 2D CFAR 수행
    int *cfar_mask = (int*)malloc(sizeof(int) * range_size * doppler_size);
    memset(cfar_mask, 0, sizeof(int) * range_size * doppler_size);

    int num_train_range = 4;
    int num_guard_range = 2;
    int num_train_doppler = 4;
    int num_guard_doppler = 2;
    double threshold_scale = 1.25;
    int start_cfar = 20;
    int end_cfar   = 50;

    cfar2D(range_doppler_log, doppler_size, range_size,
           num_train_range, num_train_doppler,
           num_guard_range, num_guard_doppler,
           threshold_scale, start_cfar, end_cfar,
           cfar_mask);

    // CFAR 결과 처리 (검출된 타겟 처리 예시)
    for (int r = 0; r < doppler_size; r++){
        for (int d = 0; d < range_size; d++){
            if (cfar_mask[r * range_size + d] == 1){
                // 검출된 타겟: (r, d)
                // 아래 변수들은 예시로 사용하나 현재는 사용하지 않으므로 주석 처리
                // double detected_range = (double)d;
                // double detected_vel   = (double)r;
            }
        }
    }

    // 각도 추정 (Angle FFT) 예시
    fftw_complex x_data[128];
    for (int i = 0; i < 128; i++){
        x_data[i] = 1.0 + 0.0 * I;
    }

    fftw_complex residual[128];
    memcpy(residual, x_data, sizeof(residual));

    // TRx_y_position만 사용 (TRx_x_position는 미사용)
    double TRx_y_position[128];
    for (int i = 0; i < 128; i++){
        TRx_y_position[i] = i * 0.5;
    }

    int max_targetnumber = 2;
    fftw_complex steering_matrix[128 * 2];
    memset(steering_matrix, 0, sizeof(steering_matrix));

    for (int k = 0; k < max_targetnumber; k++){
        double peak_idx_x = 0.1;
        double peak_idx_y = 0.2;

        double estimated_angle_pi    = atan2(peak_idx_x, peak_idx_y) * 180.0 / M_PI;
        double estimated_angle_theta = asin(peak_idx_y / (cos(estimated_angle_pi * M_PI / 180.0) + 1e-10)) * (180.0 / M_PI);

        fftw_complex a_vec[128];
        for (int i = 0; i < 128; i++){
            double phase = 2.0 * M_PI * alpha * (TRx_y_position[i] * sin(estimated_angle_theta * M_PI / 180.0));
            a_vec[i] = cos(phase) + I * sin(phase);
        }

        for (int i = 0; i < 128; i++){
            residual[i] -= a_vec[i];
        }
    }

    // 좌표 계산 (임의 예시)
    double x_hat = 0.0, y_hat = 0.0, z_hat = 0.0;
    double range_est = 5.0;  // 예시 검출 거리
    double phi   = 30.0;     // 방위각 (azimuth)
    double theta = 10.0;     // 고각 (elevation)
    x_hat = range_est * sin(theta * M_PI / 180.0) * cos(phi * M_PI / 180.0);
    y_hat = range_est * sin(theta * M_PI / 180.0) * sin(phi * M_PI / 180.0);
    z_hat = range_est * cos(theta * M_PI / 180.0);

    printf("Estimated Target: x=%.2f, y=%.2f, z=%.2f\n", x_hat, y_hat, z_hat);

    // 메모리 해제
    free(Data_target);
    free(Data_notarget);
    fftw_free(range_doppler);
    fftw_free(temp_buf);
    free(range_doppler_log);
    free(cfar_mask);

    fftw_cleanup();

    return 0;
}
