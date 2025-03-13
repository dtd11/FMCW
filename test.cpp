#include <stdlib.h>
#include <fftw3.h>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <iostream>
#include "okFrontPanelDLL.h"
#include "gnuplot-iostream.h"
#include <fstream>
#include <utility>
#include <algorithm>
#include <vector>
#include <cmath>
#include <complex.h>
#include <cstdlib>
#include "mat.h"
#include "matrix.h"

#define _CRT_SECURE_NO_WARNINGS
#define BLOCK_SIZE 16  // Adjust according to your block size
#define MAX_LINE_LENGTH 1024
#define ALPHA 0.593328
#define ANGLE_FFT_LENGTH 1024
#define MAX_TARGET_NUMBER 10
#define EPSILON_VALUE 0.2
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define ROWS 40
#define COLS 85
#define TX_COUNT 8
#define RX_COUNT 16

okCFrontPanel dev;
okCFrontPanel::ErrorCode error;

typedef struct {
    double real;
    double imag;
} ComplexDouble;

typedef struct {
    int row;
    int col;
} Detection;

double rad_to_deg(double rad) {
    return rad * (180.0 / 3.14);
}

double calculate_average(double* data, int start, int end);

void transpose(int16_t* input, int16_t* output, int chirpperframe, int pt, int ch);

void save_output_to_csv(const char* filename, int16_t* output, int chirpperframe, int pt, int ch);

void spi_write(int address, int register_value);

void spi_write_DDS(int address, int register1, int register2);

void split_output_into_2d_arrays(int16_t* output, int chirpperframe, int pt,
    int16_t** RXDATA1, int16_t** RXDATA2, int16_t** RXDATA3,
    int16_t** RXDATA4, int16_t** RXDATA5, int16_t** RXDATA6,
    int16_t** RXDATA7, int16_t** RXDATA8);

double calculate_mean(double* signal, int start, int end);

void CA_CFAR(double* signal, int len, int N_TRAIN, int N_GUARD, double T, double* cfar_output, int* detected_indices, int* detected_count);

int16_t* read_csv_to_array(const char* filename, int chripperframe, int pt, int ch);

void subtract_arrays(int16_t* output, int16_t* nodata, int16_t* result, int total_size);

void perform_fft_on_raw_data(double** RAW_Rx1_data, int data_pt, fftwf_complex** fftd_data);

int load_complex_array(const char* file, const char* varname, ComplexDouble*** complex_array, size_t* m, size_t* n);

ComplexDouble complex_divide(ComplexDouble z1, ComplexDouble z2);

void convert_to_fftwf_complex(ComplexDouble** calibration_array, size_t m, size_t n, fftwf_complex*** fftwf_calibration_array);

fftwf_complex** allocate_fftwf_calibration_array(size_t m, size_t n);

void normalize_target_fft(fftwf_complex*** target_fft, fftwf_complex** fftwf_calibration_array, int target_dim1, int target_dim2, int target_dim3, fftwf_complex*** target_fft_cal);

void sum_YY_arrays(int angle_fft_legth, fftwf_complex* YY1, fftwf_complex* YY2, fftwf_complex* YY3,
    fftwf_complex* YY4, fftwf_complex* YY5, fftwf_complex* YY6,
    fftwf_complex* YY7, fftwf_complex* YY8, fftwf_complex* YY);

void d2_fftshift(fftwf_complex* data, int N);

void subtract_min_value(float** range_doppler_cfar, int num_rows, int num_cols);

void cfar_2d(float** range_doppler_cfar, int num_rows, int num_cols, int num_train_cells_range,
    int num_guard_cells_range, int num_train_cells_doppler, int num_guard_cells_doppler,
    float threshold_scale, int** cfar_result);

int find_peaks(int** cfar_result, int num_rows, int num_cols, int* row_indices, int* col_indices);

void fftshift_rows(float** array, int num_rows, int num_cols);

void remove_adjacent_targets(int* row_indices, int* col_indices, int* num_detections,
    float** range_doppler_cfar, int* row_indices_fin, int* col_indices_fin, int* num_detections_fin);

double** allocate_matrix(int rows, int cols);

fftwf_complex** allocate_complex_matrix(int rows, int cols);

void free_matrix(double** matrix, int rows);

void free_complex_matrix(fftwf_complex** matrix, int rows);

void complex_division(fftwf_complex a, fftwf_complex b, fftwf_complex result);

int main()
{   
    // 컴퓨터에서 gnuplot.exe이 있는 경로로 저장(설치 방법 따르면 거의 비슷함)
    Gnuplot gp("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\"");

    // CSV 파일 저장 경로 filepath는 저장되는 파일 / nodata_filepath는 nodata파일
    const char* filepath = "D:\\backup\\measurement\\test\\3m_rignt_20_1.csv";
    const char* nodata_filepath = "D:\\backup\\measurement\\test\\3m_rignt_20_1.csv";

    // variable 정의
    // chirp setting 포인트 수(chirp이 바뀔 때마다 세팅)
    uint16_t pt = 4000;
    uint16_t chirpperframe = (25 + 1);
    float cpt = (800e-6 - (1e-6));

    // CFAR 알고리즘 관련 변수

    int cfar_start = 30;
    int cfar_end = 60;
    int new_num_cols = cfar_end - cfar_start + 1;
    int num_train_range = 4;
    int num_guard_range = 2;
    int num_train_doppler = 4;
    int num_guard_doppler = 2;
    float threshold_scale = 2;
    int power_theshold = -75;
 
    // variable (수정 거의 안하는 부분)
    uint8_t idle_pt = 0;
    uint8_t end_idle_pt = 0;
    uint8_t idle_chirp = 0;
    uint8_t Nd = (chirpperframe - 25);
    uint16_t pointperchirp = pt + idle_pt + end_idle_pt;
    float fs = 5e6;
    float Start_freq = 93e9;
    float End_freq = 95e9;
    float f_bw = End_freq - Start_freq;
    float c = 3e8;
    uint8_t idle_point = 20;
    uint16_t data_pt = pt - idle_point;
    uint8_t ch = 8;
    uint32_t total_pt = (pt * chirpperframe * ch); // 총 샘플의 수  
    uint32_t array_size = 2 * total_pt; // 모든 샘플을 받을 바이트의 수 (16 bit이므로 총 샘플 수에서 X 2)
    float range_resolution = c / (2 * f_bw); // 거리 해상도
    float max_range = (range_resolution * data_pt / 2);
    int total_size = chirpperframe * pt * ch;
    double scaling_factor = max_range / ((data_pt / 2.0) - 1.0);
    double lambda = c / 94e9;
    double doppler_resolution = 1 / (Nd * cpt);
    double max_doppler = doppler_resolution * Nd / 2;
    double max_velocity = (lambda * max_doppler) / 2;

    int Tx_x_position[8] = { 3, 5, 4, 1, 24, 22, 23, 14 };
    int Tx_y_position[8] = { 1, 19, 28, 41, 2, 13, 23, 40 };
    int Rx_x_position[16] = { 12, 12, 12, 12, 1, 1, 1, 1, 8, 8, 8, 8, 15, 15, 15, 15 };
    int Rx_y_position[16] = { 1, 2, 3, 4, 3, 4, 5, 6, 31, 32, 33, 34, 36, 37, 38, 39 };

    int TRx_x_position[128], TRx_y_position[128];
    for (int i = 0; i < 128; i++) {
        TRx_x_position[i] = Tx_x_position[i % 8] + Rx_x_position[i % 16];
        TRx_y_position[i] = Tx_y_position[i % 8] + Rx_y_position[i % 16];
    }

    double** weight = allocate_matrix(16, 8);

    for (int row = 0; row < 16; row++) {
        for (int col = 0; col < 8; col++) {
            weight[row][col] = 1.0;
        }
    }

    double epsilon_value = 0.2;

    // notarget data 관련 변수(건드리지 말기)
    int16_t* nodata = (int16_t*)malloc(chirpperframe * pt * ch * sizeof(int16_t));
    nodata = read_csv_to_array(nodata_filepath, chirpperframe, pt, ch);

    // angle fft 관련 변수
    const char* cali_file = "C:\\Users\\samsung\\Downloads\\ADC_test\\target_fft_save.mat";
    const char* varname = "target_fft_save";
    ComplexDouble** calibration_array;
    size_t m, n;
    size_t i, j;
    load_complex_array(cali_file, varname, &calibration_array, &m, &n);
    fftwf_complex** fftwf_calibration_array = allocate_fftwf_calibration_array(m, n);
    convert_to_fftwf_complex(calibration_array, m, n, &fftwf_calibration_array);
    for (size_t i = 0; i < m; ++i) {
        free(calibration_array[i]);
    }
    free(calibration_array);



    //// Radar 의 시작 코드

//if (dev.OpenBySerial() != okCFrontPanel::NoError) {
//    std::cerr << "Failed to open device." << std::endl;
//    return -1;
//}
// bitstream 파일 넣기(프로젝트 폴더에 bitstream 파일이 있어야 함!)
//error = dev.ConfigureFPGA("240621_PLL_debugging.bit"); 
//error = dev.ConfigureFPGA("240911_Velocity_test.bit");
//error = dev.ConfigureFPGA("240704_8t16r_fix.bit");
//if (error != okCFrontPanel::NoError) {
//    std::cerr << "Failed to configure FPGA. Error code: " << error << std::endl;
//    return -1;
//}

// Chirp setting

//(93~95G,2GHz, 41.675MHz) 400us_200us step : 1000
//spi_write(4, 0x00034E27);  //# T7   triangular : 0x000300A7
//spi_write(4, 0x00001F46); // # T6
//spi_write(4, 0x00281B55); // # T5
//spi_write(4, 0x00780284); // # T4
//spi_write(4, 0x014300C3); // # T3 triangular : 0x00C20443, sawtooth : 0x00C20043
//spi_write(4, 0x07208022); // # T2 -
//spi_write(4, 0x00000001); // # T1
//spi_write(4, 0xF8136000); // # T0 -

//(93~95G,2GHz, 41.675MHz) 800us_400us step : 1000
//spi_write(4, 0x00034E27); 
//spi_write(4, 0x00001F46); 
//spi_write(4, 0x00281B55); 
//spi_write(4, 0x00780284); 
//spi_write(4, 0x014300C3); 
//spi_write(4, 0x07208042); 
//spi_write(4, 0x00000001); 
//spi_write(4, 0xF8136000);
// 반복 코드 시작

    while (1)
    {
        //debugging 용 스위칭
        // 8TX 16RX RX 스위칭 12:1, 2   2 : 1, 3    10 1, 4
        //dev.SetWireInValue(0x07, 0xffff);
        //dev.UpdateWireIns();
        //dev.SetWireInValue(0x00, 0x00000001);  // FIFO reset 1
        //dev.UpdateWireIns();
        //Sleep(0.001); // delay
        //dev.SetWireInValue(0x06, 0x00000000);  // io_trigger 0
        //dev.UpdateWireIns();
        //Sleep(0.001); // delay
        //dev.SetWireInValue(0x00, 0x00000002 + 0x00000008 * idle_chirp + 0x400000 * (chirpperframe));  // # FIFO reset 0
        ////#                        pa on        idle chirp(2)   chirp per frame
        //dev.UpdateWireIns();
        //dev.SetWireInValue(0x01, 0x00000001 * idle_pt + 0x00010000 * pt);
        ////#                         idle  pt           pt per chirp
        //dev.UpdateWireIns();
        //dev.SetWireInValue(0x06, 0x00000001); // io_trigger 1
        //dev.UpdateWireIns();
        //unsigned char* buffer = (unsigned char*)malloc(array_size * sizeof(unsigned char));
        //dev.ReadFromBlockPipeOut(0xA3, BLOCK_SIZE, array_size, buffer);  // # pipeout(fpga > PC)
        //int16_t* int16_buffer = (int16_t*)buffer; //int 16 자료형으로 변환
        //int16_t* output = (int16_t*)malloc(total_pt * sizeof(int16_t)); //공간 할당
        //transpose(int16_buffer, output, chirpperframe, pt, ch); //output으로 transpose(배열 순서 변경)
        //free(buffer);

        // CSV 파일 저장 코드
        //save_output_to_csv(filepath, output, chirpperframe, pt, ch);
        //free(buffer);
        //nodata 빼는 코드
        //int16_t* result = (int16_t*)malloc(chirpperframe * pt * ch * sizeof(int16_t));
        //subtract_arrays(output, nodata, result, total_size);

        int16_t** RXDATA1 = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA2 = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA3 = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA4 = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA5 = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA6 = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA7 = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA8 = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));

        for (int i = 0; i < chirpperframe; ++i) {
            RXDATA1[i] = (int16_t*)malloc(pt * sizeof(int16_t));
            RXDATA2[i] = (int16_t*)malloc(pt * sizeof(int16_t));
            RXDATA3[i] = (int16_t*)malloc(pt * sizeof(int16_t));
            RXDATA4[i] = (int16_t*)malloc(pt * sizeof(int16_t));
            RXDATA5[i] = (int16_t*)malloc(pt * sizeof(int16_t));
            RXDATA6[i] = (int16_t*)malloc(pt * sizeof(int16_t));
            RXDATA7[i] = (int16_t*)malloc(pt * sizeof(int16_t));
            RXDATA8[i] = (int16_t*)malloc(pt * sizeof(int16_t));
        }

        //// NODATA 안 빼는 상황
        //split_output_into_2d_arrays(output, chirpperframe, pt, RXDATA3, RXDATA4, RXDATA1, RXDATA2, RXDATA6, RXDATA5, RXDATA8, RXDATA7);
        //free(output);

        //// NODATA 빼는 상황
        //free(output);
        //split_output_into_2d_arrays(result, chirpperframe, pt, RXDATA3, RXDATA4, RXDATA1, RXDATA2, RXDATA6, RXDATA5, RXDATA8, RXDATA7);
        //free(result);

        //nodata만 쓰기(디버깅용)
        split_output_into_2d_arrays(nodata, chirpperframe, pt, RXDATA1, RXDATA2, RXDATA3, RXDATA4, RXDATA5, RXDATA6, RXDATA7, RXDATA8);
        free(nodata);

        //// rawdata 플롯(디버깅)

        //std::vector<std::pair<int, double>> plot_data1;
        //std::vector<std::pair<int, double>> plot_data2;
        //std::vector<std::pair<int, double>> plot_data3;
        //std::vector<std::pair<int, double>> plot_data4;
        //std::vector<std::pair<int, double>> plot_data5;
        //std::vector<std::pair<int, double>> plot_data6;
        //std::vector<std::pair<int, double>> plot_data7;
        //std::vector<std::pair<int, double>> plot_data8;
        //for (int i = 0; i < pt; ++i) {
        //    plot_data1.push_back(std::make_pair(i, static_cast<double>(RXDATA1[0][i])));
        //    plot_data2.push_back(std::make_pair(i, static_cast<double>(RXDATA2[0][i])));
        //    plot_data3.push_back(std::make_pair(i, static_cast<double>(RXDATA3[0][i])));
        //    plot_data4.push_back(std::make_pair(i, static_cast<double>(RXDATA4[0][i])));
        //    plot_data5.push_back(std::make_pair(i, static_cast<double>(RXDATA5[0][i])));
        //    plot_data6.push_back(std::make_pair(i, static_cast<double>(RXDATA6[0][i])));
        //    plot_data7.push_back(std::make_pair(i, static_cast<double>(RXDATA7[0][i])));
        //    plot_data8.push_back(std::make_pair(i, static_cast<double>(RXDATA8[0][i])));
        //}
        //gp << "$data1 << EOD\n";
        //for (const auto& point : plot_data1) {
        //    gp << point.first << " " << point.second << "\n";
        //}
        //gp << "EOD\n";
        //gp << "$data2 << EOD\n";
        //for (const auto& point : plot_data2) {
        //    gp << point.first << " " << point.second << "\n";
        //}
        //gp << "EOD\n";
        //gp << "$data3 << EOD\n";
        //for (const auto& point : plot_data3) {
        //    gp << point.first << " " << point.second << "\n";
        //}
        //gp << "EOD\n";
        //gp << "$data4 << EOD\n";
        //for (const auto& point : plot_data4) {
        //    gp << point.first << " " << point.second << "\n";
        //}
        //gp << "EOD\n";
        //gp << "$data5 << EOD\n";
        //for (const auto& point : plot_data5) {
        //    gp << point.first << " " << point.second << "\n";
        //}
        //gp << "EOD\n";
        //gp << "$data6 << EOD\n";
        //for (const auto& point : plot_data6) {
        //    gp << point.first << " " << point.second << "\n";
        //}
        //gp << "EOD\n";
        //gp << "$data7 << EOD\n";
        //for (const auto& point : plot_data7) {
        //    gp << point.first << " " << point.second << "\n";
        //}
        //gp << "EOD\n";
        //gp << "$data8 << EOD\n";
        //for (const auto& point : plot_data8) {
        //    gp << point.first << " " << point.second << "\n";
        //}
        //gp << "EOD\n";
        //gp << "set multiplot layout 4, 2 title 'RXDATA Channels Plot'\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "set title 'RXDATA1[0]'\n";
        //gp << "plot $data1 with lines\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "set title 'RXDATA2[0]'\n";
        //gp << "plot $data2 with lines\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "set title 'RXDATA3[0]'\n";
        //gp << "plot $data3 with lines\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "set title 'RXDATA4[0]'\n";
        //gp << "plot $data4 with lines\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "set title 'RXDATA5[0]'\n";
        //gp << "plot $data5 with lines\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "set title 'RXDATA6[0]'\n";
        //gp << "plot $data6 with lines\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "set title 'RXDATA7[0]'\n";
        //gp << "plot $data7 with lines\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "set title 'RXDATA8[0]'\n";
        //gp << "plot $data8 with lines\n";
        //gp.flush(); // 명령어 강제 전송
        //gp << "unset multiplot\n";  // 종료
        //gp.flush(); // 명령어 강제 전송

        ////////// Raw Data

        int16_t** RXDATA1_idle = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA2_idle = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA3_idle = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA4_idle = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA5_idle = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA6_idle = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA7_idle = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));
        int16_t** RXDATA8_idle = (int16_t**)malloc(chirpperframe * sizeof(int16_t*));

        for (int i = 0; i < chirpperframe; ++i) {
            RXDATA1_idle[i] = (int16_t*)malloc(data_pt * sizeof(int16_t));
            RXDATA2_idle[i] = (int16_t*)malloc(data_pt * sizeof(int16_t));
            RXDATA3_idle[i] = (int16_t*)malloc(data_pt * sizeof(int16_t));
            RXDATA4_idle[i] = (int16_t*)malloc(data_pt * sizeof(int16_t));
            RXDATA5_idle[i] = (int16_t*)malloc(data_pt * sizeof(int16_t));
            RXDATA6_idle[i] = (int16_t*)malloc(data_pt * sizeof(int16_t));
            RXDATA7_idle[i] = (int16_t*)malloc(data_pt * sizeof(int16_t));
            RXDATA8_idle[i] = (int16_t*)malloc(data_pt * sizeof(int16_t));
        }

        // idle point만큼 빼기
        for (int i = 0; i < chirpperframe; ++i) {
            for (int j = 20; j < pt; ++j) {
                RXDATA1_idle[i][j - idle_point] = RXDATA1[i][j];
                RXDATA2_idle[i][j - idle_point] = RXDATA2[i][j];
                RXDATA3_idle[i][j - idle_point] = RXDATA3[i][j];
                RXDATA4_idle[i][j - idle_point] = RXDATA4[i][j];
                RXDATA5_idle[i][j - idle_point] = RXDATA5[i][j];
                RXDATA6_idle[i][j - idle_point] = RXDATA6[i][j];
                RXDATA7_idle[i][j - idle_point] = RXDATA7[i][j];
                RXDATA8_idle[i][j - idle_point] = RXDATA8[i][j];
            }
        }

        for (int i = 0; i < chirpperframe; ++i) {
            free(RXDATA1[i]);
            free(RXDATA2[i]);
            free(RXDATA3[i]);
            free(RXDATA4[i]);
            free(RXDATA5[i]);
            free(RXDATA6[i]);
            free(RXDATA7[i]);
            free(RXDATA8[i]);
        }

        free(RXDATA1);
        free(RXDATA2);
        free(RXDATA3);
        free(RXDATA4);
        free(RXDATA5);
        free(RXDATA6);
        free(RXDATA7);
        free(RXDATA8);

        double** RXDATA1_vol = (double**)malloc(chirpperframe * sizeof(double*));
        double** RXDATA2_vol = (double**)malloc(chirpperframe * sizeof(double*));
        double** RXDATA3_vol = (double**)malloc(chirpperframe * sizeof(double*));
        double** RXDATA4_vol = (double**)malloc(chirpperframe * sizeof(double*));
        double** RXDATA5_vol = (double**)malloc(chirpperframe * sizeof(double*));
        double** RXDATA6_vol = (double**)malloc(chirpperframe * sizeof(double*));
        double** RXDATA7_vol = (double**)malloc(chirpperframe * sizeof(double*));
        double** RXDATA8_vol = (double**)malloc(chirpperframe * sizeof(double*));

        for (int i = 0; i < chirpperframe; ++i) {
            RXDATA1_vol[i] = (double*)malloc(data_pt * sizeof(double));
            RXDATA2_vol[i] = (double*)malloc(data_pt * sizeof(double));
            RXDATA3_vol[i] = (double*)malloc(data_pt * sizeof(double));
            RXDATA4_vol[i] = (double*)malloc(data_pt * sizeof(double));
            RXDATA5_vol[i] = (double*)malloc(data_pt * sizeof(double));
            RXDATA6_vol[i] = (double*)malloc(data_pt * sizeof(double));
            RXDATA7_vol[i] = (double*)malloc(data_pt * sizeof(double));
            RXDATA8_vol[i] = (double*)malloc(data_pt * sizeof(double));
        }

        // voltage로 변환
        for (int i = 0; i < chirpperframe; ++i) {
            for (int j = 0; j < data_pt; ++j) {
                RXDATA1_vol[i][j] = static_cast<double>(RXDATA1_idle[i][j]) / 2048.0;
                RXDATA2_vol[i][j] = static_cast<double>(RXDATA2_idle[i][j]) / 2048.0;
                RXDATA3_vol[i][j] = static_cast<double>(RXDATA3_idle[i][j]) / 2048.0;
                RXDATA4_vol[i][j] = static_cast<double>(RXDATA4_idle[i][j]) / 2048.0;
                RXDATA5_vol[i][j] = static_cast<double>(RXDATA5_idle[i][j]) / 2048.0;
                RXDATA6_vol[i][j] = static_cast<double>(RXDATA6_idle[i][j]) / 2048.0;
                RXDATA7_vol[i][j] = static_cast<double>(RXDATA7_idle[i][j]) / 2048.0;
                RXDATA8_vol[i][j] = static_cast<double>(RXDATA8_idle[i][j]) / 2048.0;
            }
        }

        for (int i = 0; i < chirpperframe; ++i) {
            free(RXDATA1_idle[i]);
            free(RXDATA2_idle[i]);
            free(RXDATA3_idle[i]);
            free(RXDATA4_idle[i]);
            free(RXDATA5_idle[i]);
            free(RXDATA6_idle[i]);
            free(RXDATA7_idle[i]);
            free(RXDATA8_idle[i]);
        }

        free(RXDATA1_idle);
        free(RXDATA2_idle);
        free(RXDATA3_idle);
        free(RXDATA4_idle);
        free(RXDATA5_idle);
        free(RXDATA6_idle);
        free(RXDATA7_idle);
        free(RXDATA8_idle);

        double** RAW_Rx1_data = (double**)malloc((chirpperframe - 1) * sizeof(double*));
        double** RAW_Rx2_data = (double**)malloc((chirpperframe - 1) * sizeof(double*));
        double** RAW_Rx3_data = (double**)malloc((chirpperframe - 1) * sizeof(double*));
        double** RAW_Rx4_data = (double**)malloc((chirpperframe - 1) * sizeof(double*));
        double** RAW_Rx5_data = (double**)malloc((chirpperframe - 1) * sizeof(double*));
        double** RAW_Rx6_data = (double**)malloc((chirpperframe - 1) * sizeof(double*));
        double** RAW_Rx7_data = (double**)malloc((chirpperframe - 1) * sizeof(double*));
        double** RAW_Rx8_data = (double**)malloc((chirpperframe - 1) * sizeof(double*));

        for (int i = 0; i < (chirpperframe - 1); ++i) {
            RAW_Rx1_data[i] = (double*)malloc(data_pt * sizeof(double));
            RAW_Rx2_data[i] = (double*)malloc(data_pt * sizeof(double));
            RAW_Rx3_data[i] = (double*)malloc(data_pt * sizeof(double));
            RAW_Rx4_data[i] = (double*)malloc(data_pt * sizeof(double));
            RAW_Rx5_data[i] = (double*)malloc(data_pt * sizeof(double));
            RAW_Rx6_data[i] = (double*)malloc(data_pt * sizeof(double));
            RAW_Rx7_data[i] = (double*)malloc(data_pt * sizeof(double));
            RAW_Rx8_data[i] = (double*)malloc(data_pt * sizeof(double));
        }

        // 1chirp 빼기 (최종 raw데이터)
        for (int i = 1; i < (chirpperframe); ++i) {
            for (int j = 0; j < data_pt; ++j) {
                RAW_Rx1_data[i - 1][j] = RXDATA1_vol[i][j];
                RAW_Rx2_data[i - 1][j] = RXDATA2_vol[i][j];
                RAW_Rx3_data[i - 1][j] = RXDATA3_vol[i][j];
                RAW_Rx4_data[i - 1][j] = RXDATA4_vol[i][j];
                RAW_Rx5_data[i - 1][j] = RXDATA5_vol[i][j];
                RAW_Rx6_data[i - 1][j] = RXDATA6_vol[i][j];
                RAW_Rx7_data[i - 1][j] = RXDATA7_vol[i][j];
                RAW_Rx8_data[i - 1][j] = RXDATA8_vol[i][j];
            }
        }

        for (int i = 0; i < chirpperframe; ++i) {
            free(RXDATA1_vol[i]);
            free(RXDATA2_vol[i]);
            free(RXDATA3_vol[i]);
            free(RXDATA4_vol[i]);
            free(RXDATA5_vol[i]);
            free(RXDATA6_vol[i]);
            free(RXDATA7_vol[i]);
            free(RXDATA8_vol[i]);
        }
        free(RXDATA1_vol);
        free(RXDATA2_vol);
        free(RXDATA3_vol);
        free(RXDATA4_vol);
        free(RXDATA5_vol);
        free(RXDATA6_vol);
        free(RXDATA7_vol);
        free(RXDATA8_vol);

        ///////// 2D-Range FFT code /////////

        double** range_rx1_same = (double**)malloc(Nd * sizeof(double*));

        for (int i = 0; i < Nd; i++) {
            range_rx1_same[i] = (double*)malloc(data_pt * sizeof(double));
        }
        for (int i = 24; i < (chirpperframe-1); i++) {
            for (int j = 0; j < data_pt; j++) {
                range_rx1_same[i - 24][j] = RAW_Rx1_data[i][j];
            }
        }

        fftwf_complex* input_2d = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Nd * data_pt);
        fftwf_complex* output_2d = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Nd * data_pt);

        for (int i = 0; i < Nd; i++) {
            for (int j = 0; j < data_pt; j++) {
                input_2d[i * data_pt + j][0] = (float)range_rx1_same[i][j];
                input_2d[i * data_pt + j][1] = 0.0f;
            }
        }

        fftwf_plan plan = fftwf_plan_dft_2d(Nd, data_pt, input_2d, output_2d, FFTW_FORWARD, FFTW_ESTIMATE);

        fftwf_execute(plan);

        fftwf_complex* fftd_2d = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * Nd * data_pt);

        for (int i = 0; i < (Nd * data_pt); ++i) {
            fftd_2d[i][0] = 4 * output_2d[i][0] / (double)data_pt / (double)Nd;
            fftd_2d[i][1] = 4 * output_2d[i][1] / (double)data_pt / (double)Nd;
        }

        fftwf_free(input_2d);
        fftwf_destroy_plan(plan);
        fftwf_free(output_2d);

        float* magnitude_2d = (float*)malloc((Nd * data_pt) * sizeof(float));

        for (int i = 0; i < (Nd * data_pt); ++i) {
            float real_2d = fftd_2d[i][0];
            float imag_2d = fftd_2d[i][1];
            magnitude_2d[i] = sqrt(real_2d * real_2d + imag_2d * imag_2d);
        }

        fftwf_free(fftd_2d);

        float* db_log = (float*)malloc((Nd * data_pt) * sizeof(float));

        for (int i = 0; i < (Nd * data_pt); ++i) {
            db_log[i] = 20 * log10(magnitude_2d[i] / 10) + 30;
        }

        free(magnitude_2d);

        float** db_2d = (float**)malloc(Nd * sizeof(float*));
        for (int i = 0; i < Nd; ++i) {
            db_2d[i] = (float*)malloc(data_pt * sizeof(float));
        }

        for (int i = 0; i < Nd; ++i) {
            for (int j = 0; j < data_pt; ++j) {
                db_2d[i][j] = db_log[i * data_pt + j];
            }
        }

        free(db_log);

        fftshift_rows(db_2d, Nd, data_pt);

        ///// 2D-CFAR 알고리즘

        float** range_doppler_cfar = (float**)malloc(Nd * sizeof(float*));
        for (int i = 0; i < Nd; ++i) {
            range_doppler_cfar[i] = (float*)malloc(new_num_cols * sizeof(float));
        }

        // CFAR 범위 값 집어넣기

        for (int i = 0; i < Nd; ++i) {
            for (int j = cfar_start; j <= cfar_end; ++j) {
                range_doppler_cfar[i][j - cfar_start] = db_2d[i][j];
            }
        }

        // 최솟값을 빼서 전체 데이터를 양수로 만들어주기

        subtract_min_value(range_doppler_cfar, Nd, new_num_cols);

        int** cfar_result = (int**)malloc(Nd * sizeof(int*));
        for (int i = 0; i < Nd; ++i) {
            cfar_result[i] = (int*)malloc(new_num_cols * sizeof(int));
        }

        // 2D CFAR 알고리즘 함수

        cfar_2d(range_doppler_cfar, Nd, new_num_cols, num_train_range,
            num_guard_range, num_train_doppler, num_guard_doppler,
            threshold_scale, cfar_result);

        int max_detections = Nd * new_num_cols;
        int* row_indices = (int*)malloc(max_detections * sizeof(int));
        int* col_indices = (int*)malloc(max_detections * sizeof(int));

        int num_detections = find_peaks(cfar_result, Nd, new_num_cols, row_indices, col_indices);

        int* row_indices_fin = (int*)malloc(max_detections * sizeof(int));
        int* col_indices_fin = (int*)malloc(max_detections * sizeof(int));
        int num_detections_fin = 0;

        // 인접한 포인트를 삭제하는 코드

        remove_adjacent_targets(row_indices, col_indices, &num_detections, range_doppler_cfar,
            row_indices_fin, col_indices_fin, &num_detections_fin);

        // 인접한 포인트를 삭제하지 않는 코드

        //num_detections_fin = num_detections;
        //for (int i = 0; i < num_detections_fin; i++) {
        //    row_indices_fin[i] = row_indices[i];
        //    col_indices_fin[i] = col_indices[i];
        //}

        //cfar_start의 인덱스를 맞추는 코드
        for (int i = 0; i < num_detections_fin; i++) {
            col_indices_fin[i] += cfar_start-1;
        }

        // cfar로 찾은 타겟 중 max 값 찾기

        int* new_row_indices_fin = (int*)malloc(num_detections_fin * sizeof(int));
        int* new_col_indices_fin = (int*)malloc(num_detections_fin * sizeof(int));

        int new_num_detections_fin = 0;

        for (int i = 0; i < num_detections_fin; ++i) {
            int row = row_indices_fin[i];
            int col = col_indices_fin[i];
            float value = db_2d[row][col];

            if (value >= power_theshold) {
                new_row_indices_fin[new_num_detections_fin] = row;
                new_col_indices_fin[new_num_detections_fin] = col;
                new_num_detections_fin++;
            }
        }

        // range-doppler map 에서 반으로 자르기

        float** plot_2d = (float**)malloc(Nd * sizeof(float*));
        for (int i = 0; i < Nd; ++i) {
            plot_2d[i] = (float*)malloc((data_pt / 2) * sizeof(float));
        }

        for (int i = 0; i < Nd; ++i) {
            for (int j = 0; j < data_pt / 2; ++j) {
                plot_2d[i][j] = db_2d[i][j];
            }
        }

        for (int i = 0; i < Nd; ++i) {
            free(cfar_result[i]);
        }
        free(cfar_result);
        free(row_indices);
        free(col_indices);
        for (int i = 0; i < Nd; ++i) {
            free(db_2d[i]);
        }
        free(db_2d);

        for (int i = 0; i < Nd; ++i) {
            free(range_doppler_cfar[i]);
        }
        free(range_doppler_cfar);

        ///////// range plot code ///////// 1d 거리 플롯 코드(DEBUG)

        //double* range_average = (double*)malloc(data_pt * sizeof(double));
        // //평균내기
        //for (int j = 0; j < data_pt; ++j) {
        //    double sum1 = 0.0;
        //    for (int i = 0; i < ((chirpperframe - 1)); ++i) {
        //        sum1 += RAW_Rx1_data[i][j];
        //    }
        //    range_average[j] = sum1 / ((chirpperframe - 1));
        //}
        //for (int i = 0; i < data_pt; ++i) {
        //    range_average[i] = RAW_Rx7_data[0][i];
        //}
        //for (int i = 0; i < (chirpperframe - 1); ++i) {
        //    free(RAW_Rx7_data[i]);
        //}
        //free(RAW_Rx7_data);
        //fftwf_complex* fft_input_1d = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        //fftwf_complex* fft_output_1d = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        //for (int i = 0; i < data_pt; ++i) {
        //    fft_input_1d[i][0] = range_average[i];
        //    fft_input_1d[i][1] = 0.0;
        //}
        //    free(range_average);
        //    fftwf_plan p_1d = fftwf_plan_dft_1d(data_pt, fft_input_1d, fft_output_1d, FFTW_FORWARD, FFTW_ESTIMATE);
        //    fftwf_execute(p_1d);
        //    fftwf_free(fft_input_1d);
        //    fftwf_complex* fftd_1d = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        //    for (int i = 0; i < data_pt; ++i) {
        //        fftd_1d[i][0] = 2 * fft_output_1d[i][0] / (double)data_pt;  // Real part
        //        fftd_1d[i][1] = 2 * fft_output_1d[i][1] / (double)data_pt;  // Imaginary part
        //    }
        //    fftwf_destroy_plan(p_1d);
        //    fftwf_free(fft_output_1d);
        //    double* range_1d_abs = (double*)malloc(sizeof(double) * data_pt);
        //    for (int i = 0; i < data_pt; i++)
        //    {
        //        range_1d_abs[i] = sqrt(fftd_1d[i][0] * fftd_1d[i][0] + fftd_1d[i][1] * fftd_1d[i][1]);
        //    }
        //    fftwf_free(fftd_1d);
        //    double* range_1d_log = (double*)malloc(data_pt * sizeof(double));
        //    for (int i = 0; i < data_pt; ++i) {
        //        range_1d_log[i] = 20 * log10(range_1d_abs[i]);
        //    }
        //    free(range_1d_abs);
        //    double* range_1d_plot = (double*)malloc((data_pt / 2) * sizeof(double));
        //    for (int i = 0; i < data_pt / 2; ++i) {
        //        range_1d_plot[i] = range_1d_log[i];
        //    }
        //    free(range_1d_log);
        //    float max_value = -FLT_MAX;  // Initialize to the lowest possible float value
        //    int max_index = -1;          // To store the index with the max value
        //    for (int i = 0; i < new_num_detections_fin; ++i) {
        //        int index = new_col_indices_fin[i];  // Get the column index from new_col_indices_fin
        //        float value = range_1d_plot[index];  // Access the value at the specific index in range_1d_plot[990]
        //        // Check if the current value is the maximum
        //        if (value > max_value) {
        //            max_value = value;
        //            max_index = index;
        //        }
        //    }
// (1) Gnuplot 전처리
        //gp << "set title 'CFAR 2D Heatmap'\n";
        //gp << "set xlabel 'Range (cm)'\n";
        //gp << "set ylabel 'Velocity (m/s)'\n";
        //gp << "unset key\n";
        //gp << "set view map\n";
        //gp << "set size ratio -1\n";  // 1:1 비율
        //gp << "set palette rgbformulae 33,13,10\n";

        //거리 플롯
            //double scaling_factor = max_range / (data_pt/2);
            //주파수 플롯
            //double scaling_factor = fs / data_pt;
            //adc 플롯
            //double scaling_factor = 1;
            //std::vector<std::pair<int, double>> plot_data;
            //for (int i = 0; i < data_pt/2; ++i) {
            //    double x_scaled = i * scaling_factor;
            //    plot_data.push_back(std::make_pair(x_scaled, range_1d_plot[i]));
            //}
            //gp << "set title 'Range Log Plot'\n";
            //gp << "set xlabel 'Range (cm)'\n";
            //gp << "set ylabel 'Amplitude (dBm)'\n";
            //if (new_num_detections_fin > 0) {
            //    std::vector<std::pair<int, double>> detected_points;
            //    for (int i = 0; i < new_num_detections_fin; ++i) {
            //        int index = new_col_indices_fin[i];
            //        //double x_scaled = (index - 1) * scaling_factor;
            //        double x_scaled = (index) * scaling_factor;
            //        //detected_points.push_back(std::make_pair(x_scaled, range_1d_plot[(index - 1)]));
            //        detected_points.push_back(std::make_pair(x_scaled, range_1d_plot[(index)]));
            //    }
            //    //int max_index = max_col_index;
            //    double max_x_scaled = (max_index * scaling_factor)/100;
            //    //double max_value = range_1d_plot[max_index];
            //    double x_offset = 2000;  // Move the label 0.5 units to the right
            //    double y_offset = -24;  // Move the label 1.0 units above the point
            //    // 최대 값 텍스트 레이블 추가 (텍스트만 표시)
            //    gp << "set xrange [0:15000]\n";
            //    gp << "set yrange [-110:-30]\n";
            //    gp << "set label' Range: " << std::fixed << std::setprecision(2) << max_x_scaled << " (m) "
            //        << "Max Value: " << std::fixed << std::setprecision(2) << max_value<< " (dBm)"
            //        << "' at " << (x_offset) << "," << (y_offset)
            //        << " font ',30' tc rgb 'black'\n";
            //    gp << "plot '-' with lines title 'Range_Plot', "
            //        << "'-' with points pointtype 7 pointsize 1.5 linecolor rgb 'red' title 'Detected Peaks'\n";
            //    gp.send1d(plot_data);
            //    gp.send1d(detected_points);
            //    gp << "unset label\n";
            //}
            //else {
            //    double x_offset = 2000;  // Move the label 0.5 units to the right
            //    double y_offset = -24;  // Move the label 1.0 units above the point
            //    gp << "set xrange [0:15000]\n";
            //    gp << "set yrange [-110:-30]\n";
            //    gp << "set label' Range: " << std::fixed << std::setprecision(2) << 00 << " (m) "
            //        << "Max Value: " << std::fixed << std::setprecision(2) << 00 << " (dBm)"
            //        << "' at " << (x_offset) << "," << (y_offset)
            //        << " font ',30' tc rgb 'black'\n";
            //    gp << "plot '-' with lines title 'Range_Plot'\n";
            //    gp.send1d(plot_data);
            //    gp << "unset label\n";
            //}
            //free(range_1d_plot);

        ///////// range-doppler map 코드(DEBUG)

        //gp << "set xrange [0:" << (data_pt / 2.0 - 1.0) << "]\n";
        //gp << "set yrange [0:" << (Nd - 1.0) << "]\n";
        //{
        //    double step_cm = 5.0;
        //    std::ostringstream xtics;
        //    xtics << "set xtics(";
        //    bool first = true;
        //    for (double dist = 0.0; dist <= max_range + 1e-9; dist += step_cm) {
        //        double colIndex = (dist / max_range) * ((data_pt / 2.0) - 1.0);
        //        if (!first) xtics << ", ";
        //        xtics << "\"" << dist << "\" " << colIndex;
        //        first = false;
        //    }
        //    xtics << ")\n";
        //    gp << xtics.str();
        //}
        //{
        //    double step_vel = 1.0;
        //    std::ostringstream ytics;
        //    ytics << "set ytics(";
        //    bool first = true;
        //    for (double v = -max_velocity; v <= max_velocity + 1e-9; v += step_vel) {
        //        double rowIndex =
        //            ((v + max_velocity) / (2.0 * max_velocity)) * (Nd - 1.0);
        //        if (!first) ytics << ", ";
        //        ytics << "\"" << v << "\" " << rowIndex;
        //        first = false;
        //    }
        //    ytics << ")\n";
        //    gp << ytics.str();
        //}
        //gp << "set grid xtics ytics\n";
        //gp << "plot '-' matrix with image, '-' with points pt 3 lc rgb 'red' notitle\n";
        //for (int r = 0; r < Nd; r++) {
        //    for (int c = 0; c < (data_pt / 2); c++) {
        //        gp << plot_2d[r][c] << " ";
        //    }
        //    gp << "\n";
        //}
        //gp << "\n";
        //gp << "e\n";
        //for (int i = 0; i < new_num_detections_fin; i++) {
        //    gp << col_indices_fin[i] << " " << row_indices_fin[i] << "\n";
        //}
        //gp << "e\n";
        //gp.flush();



        for (int i = 0; i < Nd; ++i) {
            free(plot_2d[i]);
        }
        free(plot_2d);
        free(row_indices_fin);
        free(col_indices_fin);
        free(new_row_indices_fin);

        ////////  ANGLE FFT  ////////        

        fftwf_complex** fftd_data_RX1 = (fftwf_complex**)malloc(24 * sizeof(fftwf_complex*));
        for (int i = 0; i < 24; ++i) {
            fftd_data_RX1[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        }
        fftwf_complex** fftd_data_RX2 = (fftwf_complex**)malloc(24 * sizeof(fftwf_complex*));
        for (int i = 0; i < 24; ++i) {
            fftd_data_RX2[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        }
        fftwf_complex** fftd_data_RX3 = (fftwf_complex**)malloc(24 * sizeof(fftwf_complex*));
        for (int i = 0; i < 24; ++i) {
            fftd_data_RX3[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        }
        fftwf_complex** fftd_data_RX4 = (fftwf_complex**)malloc(24 * sizeof(fftwf_complex*));
        for (int i = 0; i < 24; ++i) {
            fftd_data_RX4[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        }
        fftwf_complex** fftd_data_RX5 = (fftwf_complex**)malloc(24 * sizeof(fftwf_complex*));
        for (int i = 0; i < 24; ++i) {
            fftd_data_RX5[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        }
        fftwf_complex** fftd_data_RX6 = (fftwf_complex**)malloc(24 * sizeof(fftwf_complex*));
        for (int i = 0; i < 24; ++i) {
            fftd_data_RX6[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        }
        fftwf_complex** fftd_data_RX7 = (fftwf_complex**)malloc(24 * sizeof(fftwf_complex*));
        for (int i = 0; i < 24; ++i) {
            fftd_data_RX7[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        }
        fftwf_complex** fftd_data_RX8 = (fftwf_complex**)malloc(24 * sizeof(fftwf_complex*));
        for (int i = 0; i < 24; ++i) {
            fftd_data_RX8[i] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        }

        perform_fft_on_raw_data(RAW_Rx1_data, data_pt, fftd_data_RX1);
        perform_fft_on_raw_data(RAW_Rx2_data, data_pt, fftd_data_RX2);
        perform_fft_on_raw_data(RAW_Rx3_data, data_pt, fftd_data_RX3);
        perform_fft_on_raw_data(RAW_Rx4_data, data_pt, fftd_data_RX4);
        perform_fft_on_raw_data(RAW_Rx5_data, data_pt, fftd_data_RX5);
        perform_fft_on_raw_data(RAW_Rx6_data, data_pt, fftd_data_RX6);
        perform_fft_on_raw_data(RAW_Rx7_data, data_pt, fftd_data_RX7);
        perform_fft_on_raw_data(RAW_Rx8_data, data_pt, fftd_data_RX8);

        for (int i = 0; i < (chirpperframe - 1); ++i) {
            free(RAW_Rx1_data[i]);
            free(RAW_Rx2_data[i]);
            free(RAW_Rx3_data[i]);
            free(RAW_Rx4_data[i]);
            free(RAW_Rx5_data[i]);
            free(RAW_Rx6_data[i]);
            free(RAW_Rx7_data[i]);
            free(RAW_Rx8_data[i]);
        }
        free(RAW_Rx1_data);
        free(RAW_Rx2_data);
        free(RAW_Rx3_data);
        free(RAW_Rx4_data);
        free(RAW_Rx5_data);
        free(RAW_Rx6_data);
        free(RAW_Rx7_data);
        free(RAW_Rx8_data);

        fftwf_complex*** target_fft_RAW = (fftwf_complex***)malloc(ch * sizeof(fftwf_complex**));
        for (int i = 0; i < ch; ++i) {
            target_fft_RAW[i] = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*) * 24);
            for (int j = 0; j < 24; ++j) {
                target_fft_RAW[i][j] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * new_num_detections_fin);
            }
        }
        fftwf_complex*** target_fft_cal = (fftwf_complex***)malloc(ch * sizeof(fftwf_complex**));
        for (int i = 0; i < ch; ++i) {
            target_fft_cal[i] = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*) * 24);
            for (int j = 0; j < 24; ++j) {
                target_fft_cal[i][j] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * new_num_detections_fin);
            }
        }

        for (int i = 0; i < new_num_detections_fin; i++) {
            for (int j = 0; j < 24; j++) {
                target_fft_RAW[0][j][i][0] = fftd_data_RX1[j][new_col_indices_fin[i]][0];
                target_fft_RAW[0][j][i][1] = fftd_data_RX1[j][new_col_indices_fin[i]][1];
                target_fft_RAW[1][j][i][0] = fftd_data_RX2[j][new_col_indices_fin[i]][0];
                target_fft_RAW[1][j][i][1] = fftd_data_RX2[j][new_col_indices_fin[i]][1];
                target_fft_RAW[2][j][i][0] = fftd_data_RX3[j][new_col_indices_fin[i]][0];
                target_fft_RAW[2][j][i][1] = fftd_data_RX3[j][new_col_indices_fin[i]][1];
                target_fft_RAW[3][j][i][0] = fftd_data_RX4[j][new_col_indices_fin[i]][0];
                target_fft_RAW[3][j][i][1] = fftd_data_RX4[j][new_col_indices_fin[i]][1];
                target_fft_RAW[4][j][i][0] = fftd_data_RX5[j][new_col_indices_fin[i]][0];
                target_fft_RAW[4][j][i][1] = fftd_data_RX5[j][new_col_indices_fin[i]][1];
                target_fft_RAW[5][j][i][0] = fftd_data_RX6[j][new_col_indices_fin[i]][0];
                target_fft_RAW[5][j][i][1] = fftd_data_RX6[j][new_col_indices_fin[i]][1];
                target_fft_RAW[6][j][i][0] = fftd_data_RX7[j][new_col_indices_fin[i]][0];
                target_fft_RAW[6][j][i][1] = fftd_data_RX7[j][new_col_indices_fin[i]][1];
                target_fft_RAW[7][j][i][0] = fftd_data_RX8[j][new_col_indices_fin[i]][0];
                target_fft_RAW[7][j][i][1] = fftd_data_RX8[j][new_col_indices_fin[i]][1];
            }
        }
        for (int i = 0; i < 24; ++i) {
            fftwf_free(fftd_data_RX1[i]);
            fftwf_free(fftd_data_RX2[i]);
            fftwf_free(fftd_data_RX3[i]);
            fftwf_free(fftd_data_RX4[i]);
            fftwf_free(fftd_data_RX5[i]);
            fftwf_free(fftd_data_RX6[i]);
            fftwf_free(fftd_data_RX7[i]);
            fftwf_free(fftd_data_RX8[i]);
        }
        free(fftd_data_RX1);
        free(fftd_data_RX2);
        free(fftd_data_RX3);
        free(fftd_data_RX4);
        free(fftd_data_RX5);
        free(fftd_data_RX6);
        free(fftd_data_RX7);
        free(fftd_data_RX8);

        free(new_col_indices_fin);

        //calibration
        for (int i = 0; i < new_num_detections_fin; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 16; k++) {
                    fftwf_complex result;
                    complex_division(target_fft_RAW[j][k][i], fftwf_calibration_array[j][k], result);
                    target_fft_cal[j][k][i][0] = result[0]; // Real part
                    target_fft_cal[j][k][i][1] = result[1]; // Imaginary part
                }
            }
        }

        for (int i = 0; i < ch; ++i) {
            for (int j = 0; j < 24; ++j) {
                fftwf_free(target_fft_RAW[i][j]);
            }
            fftwf_free(target_fft_RAW[i]);
        }
        free(target_fft_RAW);

        fftwf_complex** x_data = (fftwf_complex**)malloc(128 * sizeof(fftwf_complex*));
        for (int i = 0; i < 128; i++) {
            x_data[i] = (fftwf_complex*)malloc(new_num_detections_fin * sizeof(fftwf_complex));
        }
        for (int k = 0; k < new_num_detections_fin; k++)
        {
            x_data[0][k][0] = target_fft_cal[0][0][k][0];
            x_data[0][k][1] = target_fft_cal[0][0][k][1];
            x_data[1][k][0] = target_fft_cal[1][0][k][0];
            x_data[1][k][1] = target_fft_cal[1][0][k][1];
            x_data[2][k][0] = target_fft_cal[2][0][k][0];
            x_data[2][k][1] = target_fft_cal[2][0][k][1];
            x_data[3][k][0] = target_fft_cal[3][0][k][0];
            x_data[3][k][1] = target_fft_cal[3][0][k][1];
            x_data[4][k][0] = target_fft_cal[4][0][k][0];
            x_data[4][k][1] = target_fft_cal[4][0][k][1];
            x_data[5][k][0] = target_fft_cal[5][0][k][0];
            x_data[5][k][1] = target_fft_cal[5][0][k][1];
            x_data[6][k][0] = target_fft_cal[6][0][k][0];
            x_data[6][k][1] = target_fft_cal[6][0][k][1];
            x_data[7][k][0] = target_fft_cal[7][0][k][0];
            x_data[7][k][1] = target_fft_cal[7][0][k][1];
            x_data[8][k][0] = target_fft_cal[4][1][k][0];
            x_data[8][k][1] = target_fft_cal[4][1][k][1];
            x_data[9][k][0] = target_fft_cal[5][1][k][0];
            x_data[9][k][1] = target_fft_cal[5][1][k][1];
            x_data[10][k][0] = target_fft_cal[6][1][k][0];
            x_data[10][k][1] = target_fft_cal[6][1][k][1];
            x_data[11][k][0] = target_fft_cal[7][1][k][0];
            x_data[11][k][1] = target_fft_cal[7][1][k][1];
            x_data[12][k][0] = target_fft_cal[4][2][k][0];
            x_data[12][k][1] = target_fft_cal[4][2][k][1];
            x_data[13][k][0] = target_fft_cal[5][2][k][0];
            x_data[13][k][1] = target_fft_cal[5][2][k][1];
            x_data[14][k][0] = target_fft_cal[6][2][k][0];
            x_data[14][k][1] = target_fft_cal[6][2][k][1];
            x_data[15][k][0] = target_fft_cal[7][2][k][0];
            x_data[15][k][1] = target_fft_cal[7][2][k][1];
            x_data[16][k][0] = target_fft_cal[0][3][k][0];
            x_data[16][k][1] = target_fft_cal[0][3][k][1];
            x_data[17][k][0] = target_fft_cal[1][3][k][0];
            x_data[17][k][1] = target_fft_cal[1][3][k][1];
            x_data[18][k][0] = target_fft_cal[2][3][k][0];
            x_data[18][k][1] = target_fft_cal[2][3][k][1];
            x_data[19][k][0] = target_fft_cal[3][3][k][0];
            x_data[19][k][1] = target_fft_cal[3][3][k][1];
            x_data[20][k][0] = target_fft_cal[4][3][k][0];
            x_data[20][k][1] = target_fft_cal[4][3][k][1];
            x_data[21][k][0] = target_fft_cal[5][3][k][0];
            x_data[21][k][1] = target_fft_cal[5][3][k][1];
            x_data[22][k][0] = target_fft_cal[6][3][k][0];
            x_data[22][k][1] = target_fft_cal[6][3][k][1];
            x_data[23][k][0] = target_fft_cal[7][3][k][0];
            x_data[23][k][1] = target_fft_cal[7][3][k][1];
            x_data[24][k][0] = target_fft_cal[4][4][k][0];
            x_data[24][k][1] = target_fft_cal[4][4][k][1];
            x_data[25][k][0] = target_fft_cal[5][4][k][0];
            x_data[25][k][1] = target_fft_cal[5][4][k][1];
            x_data[26][k][0] = target_fft_cal[6][4][k][0];
            x_data[26][k][1] = target_fft_cal[6][4][k][1];
            x_data[27][k][0] = target_fft_cal[7][4][k][0];
            x_data[27][k][1] = target_fft_cal[7][4][k][1];
            x_data[28][k][0] = target_fft_cal[4][5][k][0];
            x_data[28][k][1] = target_fft_cal[4][5][k][1];
            x_data[29][k][0] = target_fft_cal[5][5][k][0];
            x_data[29][k][1] = target_fft_cal[5][5][k][1];
            x_data[30][k][0] = target_fft_cal[6][5][k][0];
            x_data[30][k][1] = target_fft_cal[6][5][k][1];
            x_data[31][k][0] = target_fft_cal[7][5][k][0];
            x_data[31][k][1] = target_fft_cal[7][5][k][1];
            x_data[32][k][0] = target_fft_cal[0][6][k][0];
            x_data[32][k][1] = target_fft_cal[0][6][k][1];
            x_data[33][k][0] = target_fft_cal[1][6][k][0];
            x_data[33][k][1] = target_fft_cal[1][6][k][1];
            x_data[34][k][0] = target_fft_cal[2][6][k][0];
            x_data[34][k][1] = target_fft_cal[2][6][k][1];
            x_data[35][k][0] = target_fft_cal[3][6][k][0];
            x_data[35][k][1] = target_fft_cal[3][6][k][1];
            x_data[36][k][0] = target_fft_cal[4][6][k][0];
            x_data[36][k][1] = target_fft_cal[4][6][k][1];
            x_data[37][k][0] = target_fft_cal[5][6][k][0];
            x_data[37][k][1] = target_fft_cal[5][6][k][1];
            x_data[38][k][0] = target_fft_cal[6][6][k][0];
            x_data[38][k][1] = target_fft_cal[6][6][k][1];
            x_data[39][k][0] = target_fft_cal[7][6][k][0];
            x_data[39][k][1] = target_fft_cal[7][6][k][1];
            x_data[40][k][0] = target_fft_cal[4][7][k][0];
            x_data[40][k][1] = target_fft_cal[4][7][k][1];
            x_data[41][k][0] = target_fft_cal[5][7][k][0];
            x_data[41][k][1] = target_fft_cal[5][7][k][1];
            x_data[42][k][0] = target_fft_cal[6][7][k][0];
            x_data[42][k][1] = target_fft_cal[6][7][k][1];
            x_data[43][k][0] = target_fft_cal[7][7][k][0];
            x_data[43][k][1] = target_fft_cal[7][7][k][1];
            x_data[44][k][0] = target_fft_cal[4][8][k][0];
            x_data[44][k][1] = target_fft_cal[4][8][k][1];
            x_data[45][k][0] = target_fft_cal[5][8][k][0];
            x_data[45][k][1] = target_fft_cal[5][8][k][1];
            x_data[46][k][0] = target_fft_cal[6][8][k][0];
            x_data[46][k][1] = target_fft_cal[6][8][k][1];
            x_data[47][k][0] = target_fft_cal[7][8][k][0];
            x_data[47][k][1] = target_fft_cal[7][8][k][1];
            x_data[48][k][0] = target_fft_cal[0][9][k][0];
            x_data[48][k][1] = target_fft_cal[0][9][k][1];
            x_data[49][k][0] = target_fft_cal[1][9][k][0];
            x_data[49][k][1] = target_fft_cal[1][9][k][1];
            x_data[50][k][0] = target_fft_cal[2][9][k][0];
            x_data[50][k][1] = target_fft_cal[2][9][k][1];
            x_data[51][k][0] = target_fft_cal[3][9][k][0];
            x_data[51][k][1] = target_fft_cal[3][9][k][1];
            x_data[52][k][0] = target_fft_cal[4][9][k][0];
            x_data[52][k][1] = target_fft_cal[4][9][k][1];
            x_data[53][k][0] = target_fft_cal[5][9][k][0];
            x_data[53][k][1] = target_fft_cal[5][9][k][1];
            x_data[54][k][0] = target_fft_cal[6][9][k][0];
            x_data[54][k][1] = target_fft_cal[6][9][k][1];
            x_data[55][k][0] = target_fft_cal[7][9][k][0];
            x_data[55][k][1] = target_fft_cal[7][9][k][1];
            x_data[56][k][0] = target_fft_cal[4][10][k][0];
            x_data[56][k][1] = target_fft_cal[4][10][k][1];
            x_data[57][k][0] = target_fft_cal[5][10][k][0];
            x_data[57][k][1] = target_fft_cal[5][10][k][1];
            x_data[58][k][0] = target_fft_cal[6][10][k][0];
            x_data[58][k][1] = target_fft_cal[6][10][k][1];
            x_data[59][k][0] = target_fft_cal[7][10][k][0];
            x_data[59][k][1] = target_fft_cal[7][10][k][1];
            x_data[60][k][0] = target_fft_cal[4][11][k][0];
            x_data[60][k][1] = target_fft_cal[4][11][k][1];
            x_data[61][k][0] = target_fft_cal[5][11][k][0];
            x_data[61][k][1] = target_fft_cal[5][11][k][1];
            x_data[62][k][0] = target_fft_cal[6][11][k][0];
            x_data[62][k][1] = target_fft_cal[6][11][k][1];
            x_data[63][k][0] = target_fft_cal[7][11][k][0];
            x_data[63][k][1] = target_fft_cal[7][11][k][1];
            x_data[64][k][0] = target_fft_cal[0][12][k][0];
            x_data[64][k][1] = target_fft_cal[0][12][k][1];
            x_data[65][k][0] = target_fft_cal[1][12][k][0];
            x_data[65][k][1] = target_fft_cal[1][12][k][1];
            x_data[66][k][0] = target_fft_cal[2][12][k][0];
            x_data[66][k][1] = target_fft_cal[2][12][k][1];
            x_data[67][k][0] = target_fft_cal[3][12][k][0];
            x_data[67][k][1] = target_fft_cal[3][12][k][1];
            x_data[68][k][0] = target_fft_cal[4][12][k][0];
            x_data[68][k][1] = target_fft_cal[4][12][k][1];
            x_data[69][k][0] = target_fft_cal[5][12][k][0];
            x_data[69][k][1] = target_fft_cal[5][12][k][1];
            x_data[70][k][0] = target_fft_cal[6][12][k][0];
            x_data[70][k][1] = target_fft_cal[6][12][k][1];
            x_data[71][k][0] = target_fft_cal[7][12][k][0];
            x_data[71][k][1] = target_fft_cal[7][12][k][1];
            x_data[72][k][0] = target_fft_cal[4][13][k][0];
            x_data[72][k][1] = target_fft_cal[4][13][k][1];
            x_data[73][k][0] = target_fft_cal[5][13][k][0];
            x_data[73][k][1] = target_fft_cal[5][13][k][1];
            x_data[74][k][0] = target_fft_cal[6][13][k][0];
            x_data[74][k][1] = target_fft_cal[6][13][k][1];
            x_data[75][k][0] = target_fft_cal[7][13][k][0];
            x_data[75][k][1] = target_fft_cal[7][13][k][1];
            x_data[76][k][0] = target_fft_cal[4][14][k][0];
            x_data[76][k][1] = target_fft_cal[4][14][k][1];
            x_data[77][k][0] = target_fft_cal[5][14][k][0];
            x_data[77][k][1] = target_fft_cal[5][14][k][1];
            x_data[78][k][0] = target_fft_cal[6][14][k][0];
            x_data[78][k][1] = target_fft_cal[6][14][k][1];
            x_data[79][k][0] = target_fft_cal[7][14][k][0];
            x_data[79][k][1] = target_fft_cal[7][14][k][1];
            x_data[80][k][0] = target_fft_cal[0][15][k][0];
            x_data[80][k][1] = target_fft_cal[0][15][k][1];
            x_data[81][k][0] = target_fft_cal[1][15][k][0];
            x_data[81][k][1] = target_fft_cal[1][15][k][1];
            x_data[82][k][0] = target_fft_cal[2][15][k][0];
            x_data[82][k][1] = target_fft_cal[2][15][k][1];
            x_data[83][k][0] = target_fft_cal[3][15][k][0];
            x_data[83][k][1] = target_fft_cal[3][15][k][1];
            x_data[84][k][0] = target_fft_cal[4][15][k][0];
            x_data[84][k][1] = target_fft_cal[4][15][k][1];
            x_data[85][k][0] = target_fft_cal[5][15][k][0];
            x_data[85][k][1] = target_fft_cal[5][15][k][1];
            x_data[86][k][0] = target_fft_cal[6][15][k][0];
            x_data[86][k][1] = target_fft_cal[6][15][k][1];
            x_data[87][k][0] = target_fft_cal[7][15][k][0];
            x_data[87][k][1] = target_fft_cal[7][15][k][1];
            x_data[88][k][0] = target_fft_cal[4][16][k][0];
            x_data[88][k][1] = target_fft_cal[4][16][k][1];
            x_data[89][k][0] = target_fft_cal[5][16][k][0];
            x_data[89][k][1] = target_fft_cal[5][16][k][1];
            x_data[90][k][0] = target_fft_cal[6][16][k][0];
            x_data[90][k][1] = target_fft_cal[6][16][k][1];
            x_data[91][k][0] = target_fft_cal[7][16][k][0];
            x_data[91][k][1] = target_fft_cal[7][16][k][1];
            x_data[92][k][0] = target_fft_cal[4][17][k][0];
            x_data[92][k][1] = target_fft_cal[4][17][k][1];
            x_data[93][k][0] = target_fft_cal[5][17][k][0];
            x_data[93][k][1] = target_fft_cal[5][17][k][1];
            x_data[94][k][0] = target_fft_cal[6][17][k][0];
            x_data[94][k][1] = target_fft_cal[6][17][k][1];
            x_data[95][k][0] = target_fft_cal[7][17][k][0];
            x_data[95][k][1] = target_fft_cal[7][17][k][1];
            x_data[96][k][0] = target_fft_cal[0][18][k][0];
            x_data[96][k][1] = target_fft_cal[0][18][k][1];
            x_data[97][k][0] = target_fft_cal[1][18][k][0];
            x_data[97][k][1] = target_fft_cal[1][18][k][1];
            x_data[98][k][0] = target_fft_cal[2][18][k][0];
            x_data[98][k][1] = target_fft_cal[2][18][k][1];
            x_data[99][k][0] = target_fft_cal[3][18][k][0];
            x_data[99][k][1] = target_fft_cal[3][18][k][1];
            x_data[100][k][0] = target_fft_cal[4][18][k][0];
            x_data[100][k][1] = target_fft_cal[4][18][k][1];
            x_data[101][k][0] = target_fft_cal[5][18][k][0];
            x_data[101][k][1] = target_fft_cal[5][18][k][1];
            x_data[102][k][0] = target_fft_cal[6][18][k][0];
            x_data[102][k][1] = target_fft_cal[6][18][k][1];
            x_data[103][k][0] = target_fft_cal[7][18][k][0];
            x_data[103][k][1] = target_fft_cal[7][18][k][1];
            x_data[104][k][0] = target_fft_cal[4][19][k][0];
            x_data[104][k][1] = target_fft_cal[4][19][k][1];
            x_data[105][k][0] = target_fft_cal[5][19][k][0];
            x_data[105][k][1] = target_fft_cal[5][19][k][1];
            x_data[106][k][0] = target_fft_cal[6][19][k][0];
            x_data[106][k][1] = target_fft_cal[6][19][k][1];
            x_data[107][k][0] = target_fft_cal[7][19][k][0];
            x_data[107][k][1] = target_fft_cal[7][19][k][1];
            x_data[108][k][0] = target_fft_cal[4][20][k][0];
            x_data[108][k][1] = target_fft_cal[4][20][k][1];
            x_data[109][k][0] = target_fft_cal[5][20][k][0];
            x_data[109][k][1] = target_fft_cal[5][20][k][1];
            x_data[110][k][0] = target_fft_cal[6][20][k][0];
            x_data[110][k][1] = target_fft_cal[6][20][k][1];
            x_data[111][k][0] = target_fft_cal[7][20][k][0];
            x_data[111][k][1] = target_fft_cal[7][20][k][1];
            x_data[112][k][0] = target_fft_cal[0][21][k][0];
            x_data[112][k][1] = target_fft_cal[0][21][k][1];
            x_data[113][k][0] = target_fft_cal[1][21][k][0];
            x_data[113][k][1] = target_fft_cal[1][21][k][1];
            x_data[114][k][0] = target_fft_cal[2][21][k][0];
            x_data[114][k][1] = target_fft_cal[2][21][k][1];
            x_data[115][k][0] = target_fft_cal[3][21][k][0];
            x_data[115][k][1] = target_fft_cal[3][21][k][1];
            x_data[116][k][0] = target_fft_cal[4][21][k][0];
            x_data[116][k][1] = target_fft_cal[4][21][k][1];
            x_data[117][k][0] = target_fft_cal[5][21][k][0];
            x_data[117][k][1] = target_fft_cal[5][21][k][1];
            x_data[118][k][0] = target_fft_cal[6][21][k][0];
            x_data[118][k][1] = target_fft_cal[6][21][k][1];
            x_data[119][k][0] = target_fft_cal[7][21][k][0];
            x_data[119][k][1] = target_fft_cal[7][21][k][1];
            x_data[120][k][0] = target_fft_cal[4][22][k][0];
            x_data[120][k][1] = target_fft_cal[4][22][k][1];
            x_data[121][k][0] = target_fft_cal[5][22][k][0];
            x_data[121][k][1] = target_fft_cal[5][22][k][1];
            x_data[122][k][0] = target_fft_cal[6][22][k][0];
            x_data[122][k][1] = target_fft_cal[6][22][k][1];
            x_data[123][k][0] = target_fft_cal[7][22][k][0];
            x_data[123][k][1] = target_fft_cal[7][22][k][1];
            x_data[124][k][0] = target_fft_cal[4][23][k][0];
            x_data[124][k][1] = target_fft_cal[4][23][k][1];
            x_data[125][k][0] = target_fft_cal[5][23][k][0];
            x_data[125][k][1] = target_fft_cal[5][23][k][1];
            x_data[126][k][0] = target_fft_cal[6][23][k][0];
            x_data[126][k][1] = target_fft_cal[6][23][k][1];
            x_data[127][k][0] = target_fft_cal[7][23][k][0];
            x_data[127][k][1] = target_fft_cal[7][23][k][1];
        }

        for (int i = 0; i < ch; i++) {
            for (int j = 0; j < 24; j++) {
                fftwf_free(target_fft_cal[i][j]);
            }
        }
        for (int i = 0; i < ch; i++) {
            fftwf_free(target_fft_cal[i]);
        }
        free(target_fft_cal);

        fftwf_complex** residual = allocate_complex_matrix(128, new_num_detections_fin);

        for (i = 0; i < 128; i++) {
            for (j = 0; j < new_num_detections_fin; j++) {
                residual[i][j][0] = x_data[i][j][0];
                residual[i][j][1] = x_data[i][j][1];
            }
        }

        fftwf_complex** A_steeringmatrix = allocate_complex_matrix(128, MAX_TARGET_NUMBER);
        double* estimated_angle_pi = (double*)malloc(MAX_TARGET_NUMBER * sizeof(double));
        double* estimated_angle_theta = (double*)malloc(MAX_TARGET_NUMBER * sizeof(double));

        fftwf_complex** weight = allocate_complex_matrix(RX_COUNT, TX_COUNT);
        for (i = 0; i < RX_COUNT; i++) {
            for (j = 0; j < TX_COUNT; j++) {
                weight[i][j][0] = 1.0f;  // Real part
                weight[i][j][1] = 0.0f;  // Imaginary part
            }
        }

        fftwf_complex** Rx_final = allocate_complex_matrix(ROWS, COLS);
        fftwf_complex** Rx_remove_final = allocate_complex_matrix(ROWS, COLS);

        fftwf_complex Rx[TX_COUNT][RX_COUNT] = { 0 };

        for (i = 0; i < TX_COUNT; i++) {
            int Tx_x = Tx_x_position[i];
            int Tx_y = Tx_y_position[i];

            fftwf_complex** Rx_chip_1 = allocate_complex_matrix(ROWS, COLS);
            fftwf_complex** Rx_chip_2 = allocate_complex_matrix(ROWS, COLS);
            fftwf_complex** Rx_chip_3 = allocate_complex_matrix(ROWS, COLS);
            fftwf_complex** Rx_chip_4 = allocate_complex_matrix(ROWS, COLS);

            fftwf_complex** Rx_remove_chip_1 = allocate_complex_matrix(ROWS, COLS);
            fftwf_complex** Rx_remove_chip_2 = allocate_complex_matrix(ROWS, COLS);
            fftwf_complex** Rx_remove_chip_3 = allocate_complex_matrix(ROWS, COLS);
            fftwf_complex** Rx_remove_chip_4 = allocate_complex_matrix(ROWS, COLS);

            Rx_chip_1[12 + Tx_x][1 + Tx_y][0] = Rx[i][0][0];
            Rx_chip_1[12 + Tx_x][2 + Tx_y][0] = Rx[i][1][0];
            Rx_chip_1[12 + Tx_x][3 + Tx_y][0] = Rx[i][2][0];
            Rx_chip_1[12 + Tx_x][4 + Tx_y][0] = Rx[i][3][0];
            Rx_chip_1[12 + Tx_x][1 + Tx_y][1] = Rx[i][0][1];
            Rx_chip_1[12 + Tx_x][2 + Tx_y][1] = Rx[i][1][1];
            Rx_chip_1[12 + Tx_x][3 + Tx_y][1] = Rx[i][2][1];
            Rx_chip_1[12 + Tx_x][4 + Tx_y][1] = Rx[i][3][1];

            Rx_chip_2[1 + Tx_x][3 + Tx_y][0] = Rx[i][4][0];
            Rx_chip_2[1 + Tx_x][4 + Tx_y][0] = Rx[i][5][0];
            Rx_chip_2[1 + Tx_x][5 + Tx_y][0] = Rx[i][6][0];
            Rx_chip_2[1 + Tx_x][6 + Tx_y][0] = Rx[i][7][0];
            Rx_chip_2[1 + Tx_x][3 + Tx_y][1] = Rx[i][4][1];
            Rx_chip_2[1 + Tx_x][4 + Tx_y][1] = Rx[i][5][1];
            Rx_chip_2[1 + Tx_x][5 + Tx_y][1] = Rx[i][6][1];
            Rx_chip_2[1 + Tx_x][6 + Tx_y][1] = Rx[i][7][1];

            Rx_chip_3[8 + Tx_x][31 + Tx_y][0] = Rx[i][8][0];
            Rx_chip_3[8 + Tx_x][32 + Tx_y][0] = Rx[i][9][0];
            Rx_chip_3[8 + Tx_x][33 + Tx_y][0] = Rx[i][10][0];
            Rx_chip_3[8 + Tx_x][34 + Tx_y][0] = Rx[i][11][0];
            Rx_chip_3[8 + Tx_x][31 + Tx_y][1] = Rx[i][8][1];
            Rx_chip_3[8 + Tx_x][32 + Tx_y][1] = Rx[i][9][1];
            Rx_chip_3[8 + Tx_x][33 + Tx_y][1] = Rx[i][10][1];
            Rx_chip_3[8 + Tx_x][34 + Tx_y][1] = Rx[i][11][1];

            Rx_chip_4[15 + Tx_x][36 + Tx_y][0] = Rx[i][12][0];
            Rx_chip_4[15 + Tx_x][37 + Tx_y][0] = Rx[i][13][0];
            Rx_chip_4[15 + Tx_x][38 + Tx_y][0] = Rx[i][14][0];
            Rx_chip_4[15 + Tx_x][39 + Tx_y][0] = Rx[i][15][0];
            Rx_chip_4[15 + Tx_x][36 + Tx_y][1] = Rx[i][12][1];
            Rx_chip_4[15 + Tx_x][37 + Tx_y][1] = Rx[i][13][1];
            Rx_chip_4[15 + Tx_x][38 + Tx_y][1] = Rx[i][14][1];
            Rx_chip_4[15 + Tx_x][39 + Tx_y][1] = Rx[i][15][1];

            for (int r = 0; r < ROWS; r++) {
                for (int c = 0; c < COLS; c++) {
                    Rx_final[r][c][0] += Rx_chip_1[r][c][0] + Rx_chip_2[r][c][0] + Rx_chip_3[r][c][0] + Rx_chip_4[r][c][0];
                    Rx_final[r][c][1] += Rx_chip_1[r][c][1] + Rx_chip_2[r][c][1] + Rx_chip_3[r][c][1] + Rx_chip_4[r][c][1];
                }
            }

            Rx_remove_chip_1[12 + Tx_x][1 + Tx_y][0] = weight[0][i][0] * Rx[i][0][0];
            Rx_remove_chip_1[12 + Tx_x][2 + Tx_y][0] = weight[1][i][0] * Rx[i][1][0];
            Rx_remove_chip_1[12 + Tx_x][3 + Tx_y][0] = weight[2][i][0] * Rx[i][2][0];
            Rx_remove_chip_1[12 + Tx_x][4 + Tx_y][0] = weight[3][i][0] * Rx[i][3][0];
            Rx_remove_chip_1[12 + Tx_x][1 + Tx_y][1] = weight[0][i][1] * Rx[i][0][1];
            Rx_remove_chip_1[12 + Tx_x][2 + Tx_y][1] = weight[1][i][1] * Rx[i][1][1];
            Rx_remove_chip_1[12 + Tx_x][3 + Tx_y][1] = weight[2][i][1] * Rx[i][2][1];
            Rx_remove_chip_1[12 + Tx_x][4 + Tx_y][1] = weight[3][i][1] * Rx[i][3][1];

            Rx_remove_chip_2[1 + Tx_x][3 + Tx_y][0] = weight[4][i][0] * Rx[i][4][0];
            Rx_remove_chip_2[1 + Tx_x][4 + Tx_y][0] = weight[5][i][0] * Rx[i][5][0];
            Rx_remove_chip_2[1 + Tx_x][5 + Tx_y][0] = weight[6][i][0] * Rx[i][6][0];
            Rx_remove_chip_2[1 + Tx_x][6 + Tx_y][0] = weight[7][i][0] * Rx[i][7][0];
            Rx_remove_chip_2[1 + Tx_x][3 + Tx_y][1] = weight[4][i][1] * Rx[i][4][1];
            Rx_remove_chip_2[1 + Tx_x][4 + Tx_y][1] = weight[5][i][1] * Rx[i][5][1];
            Rx_remove_chip_2[1 + Tx_x][5 + Tx_y][1] = weight[6][i][1] * Rx[i][6][1];
            Rx_remove_chip_2[1 + Tx_x][6 + Tx_y][1] = weight[7][i][1] * Rx[i][7][1];

            Rx_remove_chip_3[8 + Tx_x][31 + Tx_y][0] = weight[8][i][0] * Rx[i][8][0];
            Rx_remove_chip_3[8 + Tx_x][32 + Tx_y][0] = weight[9][i][0] * Rx[i][9][0];
            Rx_remove_chip_3[8 + Tx_x][33 + Tx_y][0] = weight[10][i][0] * Rx[i][10][0];
            Rx_remove_chip_3[8 + Tx_x][34 + Tx_y][0] = weight[11][i][0] * Rx[i][11][0];
            Rx_remove_chip_3[8 + Tx_x][31 + Tx_y][1] = weight[8][i][1] * Rx[i][8][1];
            Rx_remove_chip_3[8 + Tx_x][32 + Tx_y][1] = weight[9][i][1] * Rx[i][9][1];
            Rx_remove_chip_3[8 + Tx_x][33 + Tx_y][1] = weight[10][i][1] * Rx[i][10][1];
            Rx_remove_chip_3[8 + Tx_x][34 + Tx_y][1] = weight[11][i][1] * Rx[i][11][1];

            Rx_remove_chip_4[15 + Tx_x][36 + Tx_y][0] = weight[12][i][0] * Rx[i][12][0];
            Rx_remove_chip_4[15 + Tx_x][37 + Tx_y][0] = weight[13][i][0] * Rx[i][13][0];
            Rx_remove_chip_4[15 + Tx_x][38 + Tx_y][0] = weight[14][i][0] * Rx[i][14][0];
            Rx_remove_chip_4[15 + Tx_x][39 + Tx_y][0] = weight[15][i][0] * Rx[i][15][0];
            Rx_remove_chip_4[15 + Tx_x][36 + Tx_y][1] = weight[12][i][1] * Rx[i][12][1];
            Rx_remove_chip_4[15 + Tx_x][37 + Tx_y][1] = weight[13][i][1] * Rx[i][13][1];
            Rx_remove_chip_4[15 + Tx_x][38 + Tx_y][1] = weight[14][i][1] * Rx[i][14][1];
            Rx_remove_chip_4[15 + Tx_x][39 + Tx_y][1] = weight[15][i][1] * Rx[i][15][1];

            for (int r = 0; r < ROWS; r++) {
                for (int c = 0; c < COLS; c++) {
                    Rx_remove_final[r][c][0] += Rx_remove_chip_1[r][c][0] + Rx_remove_chip_2[r][c][0] + Rx_remove_chip_3[r][c][0] + Rx_remove_chip_4[r][c][0];
                    Rx_remove_final[r][c][1] += Rx_remove_chip_1[r][c][1] + Rx_remove_chip_2[r][c][1] + Rx_remove_chip_3[r][c][1] + Rx_remove_chip_4[r][c][1];
                }
            }

            // Free dynamically allocated matrices
            free_complex_matrix(Rx_chip_1, ROWS);
            free_complex_matrix(Rx_chip_2, ROWS);
            free_complex_matrix(Rx_chip_3, ROWS);
            free_complex_matrix(Rx_chip_4, ROWS);
        }

        // Free allocated matrices
        free_complex_matrix(Rx_final, ROWS);
        free_complex_matrix(Rx_remove_final, ROWS);
        free_complex_matrix(weight, RX_COUNT);








    }
}

void spi_write(int address, int register_value)
{
    dev.SetWireInValue(0x03, register_value);
    dev.SetWireInValue(0x04, 1 << address);
    dev.UpdateWireIns();
    dev.SetWireInValue(0x04, 0x0);
    dev.UpdateWireIns();
}

void spi_write_DDS(int address, int register1, int register2)
{
    dev.SetWireInValue(0x02, register1);
    dev.SetWireInValue(0x03, register2);
    dev.SetWireInValue(0x05, 0x00000010);
    dev.UpdateWireIns();
    dev.SetWireInValue(0x05, 0x0);
    dev.UpdateWireIns();
}

void transpose(int16_t* input, int16_t* output, int chirpperframe, int pt, int ch) {
    int idx_in, idx_out;

    for (int i = 0; i < ch; ++i) {
        for (int j = 0; j < chirpperframe; ++j) {
            for (int k = 0; k < pt; ++k) {
                idx_in = j * pt * ch + k * ch + i;
                idx_out = i * chirpperframe * pt + j * pt + k;
                output[idx_out] = input[idx_in];
            }
        }
    }
}

void save_output_to_csv(const char* filename, int16_t* output, int chirpperframe, int pt, int ch) {
    FILE* file;
    errno_t err = fopen_s(&file, filename, "w");
    if (err != 0) {
        perror("Failed to open file");
        return;
    }

    for (int i = 0; i < ch; ++i) {
        for (int j = 0; j < chirpperframe; ++j) {
            for (int k = 0; k < pt; ++k) {
                int index = i * chirpperframe * pt + j * pt + k;
                fprintf(file, "%d", output[index]);
                if (k < pt - 1) {
                    fprintf(file, ",");
                }
            }
            fprintf(file, "\n");
        }
    }
    fclose(file);
}

double calculate_average(double* data, int start, int end)
{
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += data[i];
    }
    return sum / (end - start);
}

void split_output_into_2d_arrays(int16_t* output, int chirpperframe, int pt,
    int16_t** RXDATA1, int16_t** RXDATA2, int16_t** RXDATA3,
    int16_t** RXDATA4, int16_t** RXDATA5, int16_t** RXDATA6,
    int16_t** RXDATA7, int16_t** RXDATA8)
{
    int total_pts_per_channel = chirpperframe * pt;

    for (int i = 0; i < chirpperframe; ++i) {
        for (int j = 0; j < pt; ++j) {
            int index = i * pt + j;
            RXDATA1[i][j] = output[index];
            RXDATA2[i][j] = output[index + total_pts_per_channel];
            RXDATA3[i][j] = output[index + 2 * total_pts_per_channel];
            RXDATA4[i][j] = output[index + 3 * total_pts_per_channel];
            RXDATA5[i][j] = output[index + 4 * total_pts_per_channel];
            RXDATA6[i][j] = output[index + 5 * total_pts_per_channel];
            RXDATA7[i][j] = output[index + 6 * total_pts_per_channel];
            RXDATA8[i][j] = output[index + 7 * total_pts_per_channel];
        }
    }
}

double calculate_mean(double* signal, int start, int end) {
    double sum = 0.0;
    for (int i = start; i <= end; i++) {
        sum += signal[i];
    }
    return sum / (end - start + 1);
}

void CA_CFAR(double* signal, int len, int N_TRAIN, int N_GUARD, double T, double* cfar_output, int* detected_indices, int* detected_count) {
    int i;
    *detected_count = 0;
    for (i = N_TRAIN + N_GUARD; i < len - (N_TRAIN + N_GUARD); i++) {
        int start1 = i - (N_TRAIN + N_GUARD);
        int end1 = i - (N_GUARD + 1);
        int start2 = i + N_GUARD + 1;
        int end2 = i + N_GUARD + N_TRAIN;

        double mean1 = calculate_mean(signal, start1, end1);
        double mean2 = calculate_mean(signal, start2, end2);
        double noise_level = (mean1 + mean2) / 2.0;
        double threshold = T * noise_level;

        if (signal[i] > threshold) {
            cfar_output[i] = 1;
            detected_indices[*detected_count] = i;
            (*detected_count)++;
        }
        else {
            cfar_output[i] = 0;
        }
    }
}

int16_t* read_csv_to_array(const char* filename, int chripperframe, int pt, int ch) {
    int16_t* array;
    int i = 0;
    int size = chripperframe * pt * ch;
    FILE* file;
    errno_t err;
    array = (int16_t*)malloc(size * sizeof(int16_t));
    if (array == NULL) {
        printf("Error: Memory allocation failed\n");
        return NULL;
    }

    err = fopen_s(&file, filename, "r");
    if (err != 0) {
        printf("Error: Could not open file %s\n", filename);
        free(array);
        return NULL;
    }

    while (fscanf_s(file, "%d,", &array[i]) != EOF) {
        i++;
        if (i >= size) {
            size *= 2;
            array = (int16_t*)realloc(array, size * sizeof(int16_t));
            if (array == NULL) {
                printf("Error: Memory reallocation failed\n");
                fclose(file);
                return NULL;
            }
        }
    }
    fclose(file);
    return array;
}

void subtract_arrays(int16_t* output, int16_t* nodata, int16_t* result, int total_size) {
    for (int i = 0; i < total_size; ++i) {
        result[i] = output[i] - nodata[i];
    }
}

void perform_fft_on_raw_data(double** RAW_Rx1_data, int data_pt, fftwf_complex** fftd_data) {
    fftwf_plan p;
    for (int i = 0; i < 24; ++i) {
        fftwf_complex* fft_input = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        fftwf_complex* fft_output = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * data_pt);
        for (int j = 0; j < data_pt; ++j) {
            fft_input[j][0] = RAW_Rx1_data[i][j];
            fft_input[j][1] = 0.0;
        }
        p = fftwf_plan_dft_1d(data_pt, fft_input, fft_output, FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(p);
        for (int j = 0; j < data_pt; ++j) {
            fftd_data[i][j][0] = 2 * fft_output[j][0] / data_pt;
            fftd_data[i][j][1] = 2 * fft_output[j][1] / data_pt;
        }
        fftwf_free(fft_input);
        fftwf_free(fft_output);
    }
}

int load_complex_array(const char* file, const char* varname, ComplexDouble*** complex_array, size_t* m, size_t* n) {
    MATFile* pmat;
    mxArray* pa;
    ComplexDouble* data;
    size_t i, j;
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error opening file %s\n", file);
        return(EXIT_FAILURE);
    }
    pa = matGetVariable(pmat, varname);
    if (pa == NULL) {
        printf("Error reading variable %s from file %s\n", varname, file);
        matClose(pmat);
        return(EXIT_FAILURE);
    }
    *m = mxGetM(pa);
    *n = mxGetN(pa);
    if (!mxIsComplex(pa)) {
        printf("Variable %s is not a complex array.\n", varname);
        mxDestroyArray(pa);
        matClose(pmat);
        return(EXIT_FAILURE);
    }
    data = (ComplexDouble*)mxGetData(pa);
    *complex_array = (ComplexDouble**)malloc((*m) * sizeof(ComplexDouble*));
    for (i = 0; i < *m; i++) {
        (*complex_array)[i] = (ComplexDouble*)malloc((*n) * sizeof(ComplexDouble));
    }
    for (i = 0; i < *m; i++) {
        for (j = 0; j < *n; j++) {
            (*complex_array)[i][j].real = data[i + j * (*m)].real;
            (*complex_array)[i][j].imag = data[i + j * (*m)].imag;
        }
    }
    mxDestroyArray(pa);
    matClose(pmat);
    return(EXIT_SUCCESS);
}

ComplexDouble complex_divide(ComplexDouble z1, ComplexDouble z2) {
    ComplexDouble result;
    double denominator;
    denominator = z2.real * z2.real + z2.imag * z2.imag;
    if (denominator == 0.0) {
        printf("Error: Division by zero in complex division.\n");
        result.real = 0.0;
        result.imag = 0.0;
        return result;
    }
    result.real = (z1.real * z2.real + z1.imag * z2.imag) / denominator;
    result.imag = (z1.imag * z2.real - z1.real * z2.imag) / denominator;
    return result;
}

void convert_to_fftwf_complex(ComplexDouble** calibration_array, size_t m, size_t n, fftwf_complex*** fftwf_calibration_array) {
    *fftwf_calibration_array = (fftwf_complex**)malloc(m * sizeof(fftwf_complex*));
    for (size_t i = 0; i < m; ++i) {
        (*fftwf_calibration_array)[i] = (fftwf_complex*)fftwf_malloc(n * sizeof(fftwf_complex));
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            (*fftwf_calibration_array)[i][j][0] = calibration_array[i][j].real;
            (*fftwf_calibration_array)[i][j][1] = calibration_array[i][j].imag;
        }
    }
}

fftwf_complex** allocate_fftwf_calibration_array(size_t m, size_t n) {
    fftwf_complex** fftwf_calibration_array = (fftwf_complex**)malloc(m * sizeof(fftwf_complex*));
    if (fftwf_calibration_array == NULL) {
        perror("Failed to allocate memory for fftwf_calibration_array");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < m; ++i) {
        fftwf_calibration_array[i] = (fftwf_complex*)fftwf_malloc(n * sizeof(fftwf_complex));
        if (fftwf_calibration_array[i] == NULL) {
            perror("Failed to allocate memory for fftwf_calibration_array row");
            exit(EXIT_FAILURE);
        }
    }

    return fftwf_calibration_array;
}

void normalize_target_fft(fftwf_complex*** target_fft, fftwf_complex** fftwf_calibration_array, int target_dim1, int target_dim2, int target_dim3, fftwf_complex*** target_fft_cal) {
    for (int i = 0; i < target_dim3; i++) {
        for (int j = 0; j < target_dim1; j++) {
            for (int k = 0; k < target_dim2; k++) {
                double real_a = target_fft[j][k][i][0];
                double imag_a = target_fft[j][k][i][1];
                double real_b = fftwf_calibration_array[j][k][0];
                double imag_b = fftwf_calibration_array[j][k][1];
                double denominator = real_b * real_b + imag_b * imag_b;
                target_fft_cal[j][k][i][0] = (real_a * real_b + imag_a * imag_b) / denominator;
                target_fft_cal[j][k][i][1] = (imag_a * real_b - real_a * imag_b) / denominator;
            }
        }
    }
}

void sum_YY_arrays(int angle_fft_legth, fftwf_complex* YY1, fftwf_complex* YY2, fftwf_complex* YY3,
    fftwf_complex* YY4, fftwf_complex* YY5, fftwf_complex* YY6,
    fftwf_complex* YY7, fftwf_complex* YY8, fftwf_complex* YY) {

    int total_size = angle_fft_legth * angle_fft_legth;
    for (int idx = 0; idx < total_size; ++idx) {
        YY[idx][0] = YY1[idx][0] + YY2[idx][0] + YY3[idx][0] + YY4[idx][0]
            + YY5[idx][0] + YY6[idx][0] + YY7[idx][0] + YY8[idx][0];
        YY[idx][1] = YY1[idx][1] + YY2[idx][1] + YY3[idx][1] + YY4[idx][1]
            + YY5[idx][1] + YY6[idx][1] + YY7[idx][1] + YY8[idx][1];
    }
}

void d2_fftshift(fftwf_complex* data, int N) {
    int halfN = N / 2;
    fftwf_complex temp;
    for (int i = 0; i < halfN; i++) {
        for (int j = 0; j < halfN; j++) {
            temp[0] = data[i * N + j][0];
            temp[1] = data[i * N + j][1];
            data[i * N + j][0] = data[(i + halfN) * N + (j + halfN)][0];
            data[i * N + j][1] = data[(i + halfN) * N + (j + halfN)][1];
            data[(i + halfN) * N + (j + halfN)][0] = temp[0];
            data[(i + halfN) * N + (j + halfN)][1] = temp[1];
            temp[0] = data[i * N + (j + halfN)][0];
            temp[1] = data[i * N + (j + halfN)][1];
            data[i * N + (j + halfN)][0] = data[(i + halfN) * N + j][0];
            data[i * N + (j + halfN)][1] = data[(i + halfN) * N + j][1];
            data[(i + halfN) * N + j][0] = temp[0];
            data[(i + halfN) * N + j][1] = temp[1];
        }
    }
}

void subtract_min_value(float** range_doppler_cfar, int num_rows, int num_cols) {
    float min_value = FLT_MAX;
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            if (range_doppler_cfar[i][j] < min_value) {
                min_value = range_doppler_cfar[i][j];
            }
        }
    }
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            range_doppler_cfar[i][j] -= min_value;
        }
    }
}

void cfar_2d(float** range_doppler_cfar, int num_rows, int num_cols, int num_train_cells_range,
    int num_guard_cells_range, int num_train_cells_doppler, int num_guard_cells_doppler,
    float threshold_scale, int** cfar_result) {

    int total_train_range = num_train_cells_range + num_guard_cells_range;
    int total_train_doppler = num_train_cells_doppler + num_guard_cells_doppler;
    for (int r = total_train_range; r < num_rows - total_train_range; ++r) {
        for (int d = total_train_doppler; d < num_cols - total_train_doppler; ++d) {

            float noise_level = 0.0;
            int num_training_cells = 0;
            for (int i = r - total_train_range; i <= r + total_train_range; ++i) {
                for (int j = d - total_train_doppler; j <= d + total_train_doppler; ++j) {
                    if (abs(i - r) <= num_guard_cells_range && abs(j - d) <= num_guard_cells_doppler)
                        continue;
                    noise_level += range_doppler_cfar[i][j];
                    num_training_cells++;
                }
            }
            float noise_avg = noise_level / num_training_cells;
            float threshold = noise_avg * threshold_scale;
            if (range_doppler_cfar[r][d] > threshold) {
                cfar_result[r][d] = 1;
            }
            else {
                cfar_result[r][d] = 0;
            }
        }
    }
    for (int r = 0; r < num_rows; ++r) {
        for (int d = 0; d < num_cols; ++d) {
            if (r < total_train_range || r >= num_rows - total_train_range ||
                d < total_train_doppler || d >= num_cols - total_train_doppler) {
                cfar_result[r][d] = 0;
            }
        }
    }
}

int find_peaks(int** cfar_result, int num_rows, int num_cols, int* row_indices, int* col_indices) {
    int count = 0;
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            if (cfar_result[i][j] == 1) {
                row_indices[count] = i;
                col_indices[count] = j;
                count++;
            }
        }
    }

    return count;
}

void fftshift_rows(float** array, int num_rows, int num_cols) {
    int mid_row = num_rows / 2;
    for (int i = 0; i < mid_row; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            float temp = array[i][j];
            array[i][j] = array[mid_row + i][j];
            array[mid_row + i][j] = temp;
        }
    }
}

void remove_adjacent_targets(int* row_indices, int* col_indices, int* num_detections,
    float** range_doppler_cfar, int* row_indices_fin, int* col_indices_fin, int* num_detections_fin) {
    int* valid = (int*)malloc(*num_detections * sizeof(int));
    for (int i = 0; i < *num_detections; ++i) {
        valid[i] = 1;
    }
    for (int i = 0; i < *num_detections; ++i) {
        if (valid[i] == 0) {
            continue;
        }
        for (int j = i + 1; j < *num_detections; ++j) {
            if (valid[j] == 0) {
                continue;
            }
            if (abs(row_indices[i] - row_indices[j]) <= 2 && abs(col_indices[i] - col_indices[j]) <= 2) {
                if (range_doppler_cfar[row_indices[i]][col_indices[i]] >= range_doppler_cfar[row_indices[j]][col_indices[j]]) {
                    valid[j] = 0;
                }
                else {
                    valid[i] = 0;
                    break;
                }
            }
        }
    }
    *num_detections_fin = 0;
    for (int i = 0; i < *num_detections; ++i) {
        if (valid[i] == 1) {
            row_indices_fin[*num_detections_fin] = row_indices[i];
            col_indices_fin[*num_detections_fin] = col_indices[i];
            (*num_detections_fin)++;
        }
    }
    free(valid);
}

double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
    }
    return matrix;
}

// Function to allocate a 2D complex matrix
fftwf_complex** allocate_complex_matrix(int rows, int cols) {
    fftwf_complex** matrix = (fftwf_complex**)malloc(rows * sizeof(fftwf_complex*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (fftwf_complex*)malloc(cols * sizeof(fftwf_complex));
    }
    return matrix;
}

// Function to free a 2D matrix
void free_matrix(double** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to free a 2D complex matrix
void free_complex_matrix(fftwf_complex** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void complex_division(fftwf_complex a, fftwf_complex b, fftwf_complex result) {
    float denominator = (b[0] * b[0]) + (b[1] * b[1]); // |b|^2
    if (denominator == 0.0f) { // Avoid division by zero
        result[0] = 0.0f;
        result[1] = 0.0f;
    }
    else {
        result[0] = ((a[0] * b[0]) + (a[1] * b[1])) / denominator; // Real part
        result[1] = ((a[1] * b[0]) - (a[0] * b[1])) / denominator; // Imaginary part
    }
}
