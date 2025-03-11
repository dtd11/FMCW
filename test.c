/*******************************************************************************
 * 예시: MATLAB 코드를 C로 포팅한 샘플 (라즈베리 파이 구동 가정)
 *       - 주: 실제 동작/최적화 위해서는 추가 수정이 필요함
 ******************************************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <complex.h>
 #include "fftw3.h"
 #include <string.h>
 
 #define MAX(a,b) (((a)>(b))?(a):(b))
 #define MIN(a,b) (((a)<(b))?(a):(b))
 
 //=====================================================================
 // 레이더 파라미터 (질문에서 사용한 값들과 유사하게 정의)
 //=====================================================================
 static const int pt            = 4000;       // fast-time 샘플 수
 static const int chirpperframe = 121;        // 1프레임 당 chirp 개수(120+1)
 static const double chirp_time = 800e-6;     // 한 chirp 시간
 static const double Start_freq = 93e9;       // FMCW 시작 주파수
 static const double End_freq   = 95e9;       // FMCW 끝 주파수
 static const int Nd            = 96;         // velocity FFT 크기 (예: chirpperframe - 25)
 static const int idle_point    = 20;         // 앞부분 샘플 버리기
 static const double fs         = 5e6;        // ADC 샘플링 속도
 static const double c          = 3e8;        // 빛의 속도
 static const double alpha      = 0.593328;   // DOA 해석 시 스케일링?
 
 //=====================================================================
 // FFT 도우미 함수 (1D)
 //=====================================================================
 /** 
  * fft_1d: 실수배열 입력 → 복소출력 (fftw_complex 활용)
  *   - in : double* 크기 n
  *   - out: fftw_complex* 크기 n
  *   - n  : 점수
  * norm_scale : 1/n 등의 정규화를 여기서 적용 가능
  */
 void fft_1d(const double *in, fftw_complex *out, int n, double norm_scale)
 {
     // FFTW 계획(plan) 생성
     fftw_plan p = fftw_plan_dft_r2c_1d(n, (double*)in, out, FFTW_ESTIMATE);
     // 실행
     fftw_execute(p);
     // 정규화
     for(int i=0; i<n; i++){
         out[i][0] *= norm_scale; // 실수부
         out[i][1] *= norm_scale; // 허수부
     }
     fftw_destroy_plan(p);
 }
 
 /**
  * ifft_1d: 복소배열 입력 → 복소출력
  *   - in : fftw_complex* 크기 n
  *   - out: fftw_complex* 크기 n
  *   - n  : 점수
  */
 void ifft_1d(const fftw_complex *in, fftw_complex *out, int n, double norm_scale)
 {
     // FFTW 계획(plan) 생성 (역변환)
     fftw_plan p = fftw_plan_dft_1d(n, (fftw_complex*)in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
     // 실행
     fftw_execute(p);
     // 정규화
     for(int i=0; i<n; i++){
         out[i][0] *= norm_scale; 
         out[i][1] *= norm_scale; 
     }
     fftw_destroy_plan(p);
 }
 
 //=====================================================================
 // 2D FFT 예시 (복소배열 입력 → 복소배열 출력)
 //=====================================================================
 /**
  * fft_2d: (nrow x ncol) 크기의 복소배열에 대해 2D FFT
  *        in[row][col], out[row][col]
  */
 void fft_2d(fftw_complex *in, fftw_complex *out, int nrow, int ncol)
 {
     // 먼저 row 방향으로 FFT
     // 임시 버퍼가 필요
     fftw_complex *row_in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*ncol);
     fftw_complex *row_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*ncol);
 
     // plan은 ncol 크기에 대해 한 번 생성 후 row별로 재사용
     fftw_plan p_row = fftw_plan_dft_1d(ncol, row_in, row_out, FFTW_FORWARD, FFTW_ESTIMATE);
 
     // row별로 수행
     for(int r=0; r<nrow; r++){
         // in -> row_in
         memcpy(row_in, &in[r*ncol], sizeof(fftw_complex)*ncol);
         fftw_execute(p_row);
         // row_out -> out
         memcpy(&out[r*ncol], row_out, sizeof(fftw_complex)*ncol);
     }
     fftw_destroy_plan(p_row);
 
     // 이제 col 방향으로 FFT
     // col별로 nrow 크기의 배열
     fftw_complex *col_in  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nrow);
     fftw_complex *col_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nrow);
     fftw_plan p_col = fftw_plan_dft_1d(nrow, col_in, col_out, FFTW_FORWARD, FFTW_ESTIMATE);
 
     for(int c=0; c<ncol; c++){
         // out -> col_in
         for(int r=0; r<nrow; r++){
             col_in[r][0] = out[r*ncol + c][0];
             col_in[r][1] = out[r*ncol + c][1];
         }
         fftw_execute(p_col);
         // col_out -> 다시 out
         for(int r=0; r<nrow; r++){
             out[r*ncol + c][0] = col_out[r][0];
             out[r*ncol + c][1] = col_out[r][1];
         }
     }
     fftw_destroy_plan(p_col);
 
     fftw_free(row_in); 
     fftw_free(row_out);
     fftw_free(col_in);
     fftw_free(col_out);
 
     // 필요시 nrow*ncol로 정규화
     double norm = 1.0/(double)(nrow*ncol);
     for(int r=0; r<nrow*ncol; r++){
         out[r][0] *= norm;
         out[r][1] *= norm;
     }
 }
 
 //=====================================================================
 // (예시) CSV 파일 읽어서 double 배열에 넣는 함수 - 간단 구현
 //=====================================================================
 int read_csv_double(const char *filename, double *out, int max_len)
 {
     FILE *fp = fopen(filename, "r");
     if(!fp){
         fprintf(stderr, "Cannot open file %s\n", filename);
         return -1;
     }
     int count=0;
     while(!feof(fp) && count<max_len){
         if( fscanf(fp, "%lf,", &out[count]) == 1){
             count++;
         }
         else{
             // 구분자 형태에 맞춰서 처리
             // 여기서는 단순화
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
             int *cfar_mask_out /* same size */)
 {
     // 초기화
     memset(cfar_mask_out, 0, sizeof(int)*n_doppler*n_range);
 
     for(int r = (num_train_r + num_guard_r); r < (n_doppler - num_train_r - num_guard_r); r++){
         for(int d = (num_train_d + num_guard_d + start_cfar);
                  d < MIN(n_range - num_train_d - num_guard_d, end_cfar); d++){
 
             // 주변 cell에서 노이즈 레벨 추정
             double noise_sum = 0.0;
             int    count     = 0;
             for(int rr = r - (num_train_r+num_guard_r); rr <= r + (num_train_r+num_guard_r); rr++){
                 for(int dd = d - (num_train_d+num_guard_d); dd <= d + (num_train_d+num_guard_d); dd++){
                     // guard 영역 제외
                     if( (abs(rr-r) > num_guard_r) || (abs(dd-d) > num_guard_d) ){
                         noise_sum += range_doppler_in[rr*n_range + dd];
                         count++;
                     }
                 }
             }
 
             double noise_avg = (count>0)? (noise_sum / count): 0.0;
             double threshold = noise_avg * threshold_scale;
 
             double val = range_doppler_in[r*n_range + d];
             if(val > threshold){
                 cfar_mask_out[r*n_range + d] = 1; 
             }
         }
     }
 }
 
 //=====================================================================
 // main
 //=====================================================================
 int main(int argc, char *argv[])
 {
     // -----------------------------------------------------------
     // (1) CSV에서 데이터 읽기 (필요 시), MATLAB의 Data_target 등
     //     여기서는 예시로 raw_data_target, raw_data_notarget 에 읽는다고 가정
     // -----------------------------------------------------------
     // 실제 사용 시, CSV 크기에 맞춰 동적할당/로직 변경 필요
     // 질문 코드: 8 Rx * chirpperframe 행, pt 열
     const int row_size = 8*chirpperframe; // 968 (대략)
     const int col_size = pt;             // 4000
     double *Data_target   = (double*)malloc(sizeof(double)*row_size*col_size);
     double *Data_notarget = (double*)malloc(sizeof(double)*row_size*col_size);
 
     // 여기에 read_csv_double(...) 같은 식으로 로드(단, CSV 구조에 따라 파싱 달라짐)
     // 예) read_csv_double("3m_moving_1.csv", Data_target, row_size*col_size);
 
     // 여기선 가상의 dummy data
     for(int i=0; i<row_size*col_size; i++){
         Data_target[i]   = (double)(i%4096);
         Data_notarget[i] = 0.0; // no-target 가정
     }
 
     // -----------------------------------------------------------
     // (2) Rx별로 분리, 1번째 행/열 제거, idle_point 제거 등
     // -----------------------------------------------------------
     // 실제 MATLAB 코드에서는 많은 전처리를 거침. 여기서는 간단히 예시만.
     // 예: Rx1_data = Data_target(1:chirpperframe, :);
     //     => c에서는 직접 인덱싱해야 함
     //     (실제 크기 체크와 오프셋 처리를 주의)
 
     // 편의상 "Rx1_data"만 예시로
     // double *Rx1_data = (double*)malloc(sizeof(double)*chirpperframe*pt);
     // ...
     // for(int r=0; r<chirpperframe; r++){
     //     for(int c=0; c<pt; c++){
     //         Rx1_data[r*pt + c] = Data_target[r*pt + c];
     //     }
     // }
 
     // (추가적으로 Rx2_data, Rx3_data ... Rx8_data 같은 식)
     // (notarget 빼기, idle_point만큼 버리기 등등 생략)
 
     // -----------------------------------------------------------
     // (3) Range FFT, Doppler FFT => Range-Doppler Map
     //     여기서는 2D FFT 사용(각 Rx별로)
     // -----------------------------------------------------------
     // 예시로, "하나의 Rx"에 대해서 pt x Nd (fast-time x slow-time) 2D 배열 생성
     // 실제론 Rx별로 반복 or 평균처리
 
     int range_size = pt;    // fast-time, 4000
     int doppler_size = Nd;  // 96
     fftw_complex *range_doppler = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*range_size*doppler_size);
     fftw_complex *temp_buf      = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*range_size*doppler_size);
 
     // 임시로 더미 데이터 (실제론 Rx_data에서 변환)
     for(int i=0; i<range_size*doppler_size; i++){
         range_doppler[i][0] = 100.0;  // real
         range_doppler[i][1] = 0.0;    // imag
     }
 
     // 2D FFT (Range-Doppler)
     // => row = doppler axis, col = range axis 식으로 하려면 주의
     // MATLAB 코드에선 1) range FFT(열방향) -> 2) doppler FFT(행방향)
     // 여기서는 단순히 2D FFT만 보여줌
     fft_2d(range_doppler, temp_buf, doppler_size, range_size);
 
     // -----------------------------------------------------------
     // (4) 로그 스케일 변환
     // -----------------------------------------------------------
     double *range_doppler_log = (double*)malloc(sizeof(double)*range_size*doppler_size);
     double max_val = -1e9;
     for(int i=0; i<range_size*doppler_size; i++){
         double mag = sqrt(temp_buf[i][0]*temp_buf[i][0] + temp_buf[i][1]*temp_buf[i][1]);
         double val_db = 20.0*log10(mag/10.0) + 30.0; 
         range_doppler_log[i] = val_db;
         if(val_db > max_val) max_val = val_db;
     }
 
     // -----------------------------------------------------------
     // (5) 2D CFAR
     // -----------------------------------------------------------
     int *cfar_mask = (int*)malloc(sizeof(int)*range_size*doppler_size);
     memset(cfar_mask, 0, sizeof(int)*range_size*doppler_size);
 
     // (예시) CFAR 파라미터
     int num_train_range = 4;
     int num_guard_range = 2;
     int num_train_doppler = 4;
     int num_guard_doppler = 2;
     double threshold_scale = 1.25;
     int start_cfar = 20;
     int end_cfar   = 50;
 
     // CFAR 수행: (range_doppler_log, doppler_size, range_size) 순서 주의
     // MATLAB에선 (행=도플러, 열=거리) 형태를 많이 쓰지만, 여기선 변수명만 유지
     // 실제론 indexing을 맞춰야 합니다.
     cfar2D(range_doppler_log, doppler_size, range_size,
            num_train_range, num_train_doppler,
            num_guard_range, num_guard_doppler,
            threshold_scale, start_cfar, end_cfar,
            cfar_mask);
 
     // CFAR 결과에서 peak 위치 찾기
     for(int r=0; r<doppler_size; r++){
         for(int d=0; d<range_size; d++){
             if(cfar_mask[r*range_size + d] == 1){
                 // 검출된 지점 => (r, d)
                 // 실제 거리/속도 축은 변환이 필요
                 double detected_range = (double)d; // 예시
                 double detected_vel   = (double)r; // 예시
                 // ...
             }
         }
     }
 
     // -----------------------------------------------------------
     // (6) 각도 추정 (Angle FFT) - 매우 간단한 뼈대만
     // -----------------------------------------------------------
     // - 실제 MATLAB 코드는 8Tx*16Rx = 128 가상 채널, 2D FFT 사용, 
     //   or MUSIC/ESPRIT 등 다양한 방법으로 각도 추정을 합니다.
     // - 아래는 복잡도를 줄여서 "steering vector"를 만든 뒤 
     //   peak 찾는 로직의 틀만 시범으로 보여줍니다.
 
     // 예: x_data[128]에 한 거리 bin의 복소신호가 들어있다고 가정
     //     (MATLAB에서 target_data_cal 등등)
     //     여기서는 임의로 128개 할당
     fftw_complex x_data[128];
     for(int i=0; i<128; i++){
         x_data[i][0] = 1.0; // real
         x_data[i][1] = 0.0; // imag
     }
 
     // 단순 peak 탐색, residual 업데이트
     fftw_complex residual[128];
     memcpy(residual, x_data, sizeof(residual));
 
     // MIMO array position
     // (실제 MATLAB 코드에서 TRx_x_position, TRx_y_position 등)
     double TRx_x_position[128];
     double TRx_y_position[128];
     for(int i=0; i<128; i++){
         // 대충 임의 초기화
         TRx_x_position[i] = i*1.0;
         TRx_y_position[i] = i*0.5;
     }
 
     int max_targetnumber = 2;
     fftw_complex steering_matrix[128*2]; // 최대 타겟 2개 가정
     memset(steering_matrix, 0, sizeof(steering_matrix));
 
     // 예: 두 개 타겟 각도 가정
     for(int k=0; k<max_targetnumber; k++){
         // (가령) 2D FFT로 구한 peak idx = (k_x, k_y)
         // 여기서는 그냥 예시 각도
         double peak_idx_x = 0.1; 
         double peak_idx_y = 0.2;
 
         double estimated_angle_pi    = atan2(peak_idx_x, peak_idx_y)*180.0/M_PI; 
         double estimated_angle_theta = asin(peak_idx_y / (cos(estimated_angle_pi*M_PI/180.0)+1e-10))*(180.0/M_PI);
 
         // steering vector 생성
         fftw_complex a_vec[128];
         for(int i=0; i<128; i++){
             // phase = 2*pi*alpha*( TRx_y_position[i]*sin(theta)*cos(pi) + 
             //                      TRx_x_position[i]*sin(theta)*sin(pi) )
             // 여기선 단순화
             double phase = 2.0*M_PI*alpha * (TRx_y_position[i]*sin(estimated_angle_theta*M_PI/180.0));
             a_vec[i][0] = cos(phase);
             a_vec[i][1] = sin(phase);
         }
 
         // (residual에서 a_vec로 최소자승 등으로 추정)
         // 여기서는 단순히 residual을 0으로 만든다고 가정
         for(int i=0; i<128; i++){
             residual[i][0] -= a_vec[i][0];
             residual[i][1] -= a_vec[i][1];
         }
     }
 
     // x_hat, y_hat, z_hat 같은 좌표화는
     //  range * [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)] 형태로 계산
     double x_hat = 0.0, y_hat = 0.0, z_hat = 0.0;
     // 예시
     double range_est = 5.0;  // CFAR로 검출된 거리
     double phi   = 30.0;     // 방위각(azimuth)
     double theta = 10.0;     // 고각(elevation)
     x_hat = range_est*sin(theta*M_PI/180.0)*cos(phi*M_PI/180.0);
     y_hat = range_est*sin(theta*M_PI/180.0)*sin(phi*M_PI/180.0);
     z_hat = range_est*cos(theta*M_PI/180.0);
 
     printf("Estimated Target: x=%.2f, y=%.2f, z=%.2f\n", x_hat, y_hat, z_hat);
 
     // -----------------------------------------------------------
     // 메모리 해제
     // -----------------------------------------------------------
     free(Data_target);
     free(Data_notarget);
     fftw_free(range_doppler);
     fftw_free(temp_buf);
     free(range_doppler_log);
     free(cfar_mask);
 
     // FFTW clean-up
     fftw_cleanup();
 
     return 0;
 }
 
