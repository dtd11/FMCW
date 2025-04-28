#include <stdlib.h>
#include "fftw3.h"
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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
#include <mat.h>
#include <matrix.h>
#include <unordered_set>
#include <thread>
#include <chrono>

#define _CRT_SECURE_NO_WARNINGS
#define BLOCK_SIZE 16
#define PI 3.14159265358979323846

okCFrontPanel dev;
okCFrontPanel::ErrorCode error;

typedef struct {
    double real;
    double imag;
} ComplexDouble;

void transpose(int16_t* input, int16_t* output, int chirpperframe, int pt, int ch);

void save_output_to_csv(const char* filename, int16_t* output, int chirpperframe, int pt, int ch);

void spi_write(int address, int register_value);

void spi_write_DDS(int address, int register1, int register2);

void split_output_into_2d_arrays(int16_t* output, int chirpperframe, int pt,
    int16_t** RXDATA1, int16_t** RXDATA2, int16_t** RXDATA3,
    int16_t** RXDATA4, int16_t** RXDATA5, int16_t** RXDATA6,
    int16_t** RXDATA7, int16_t** RXDATA8);

int16_t* read_csv_to_array(const char* filename, int chripperframe, int pt, int ch);

void subtract_arrays(int16_t* output, int16_t* nodata, int16_t* result, int total_size);

void perform_fft_on_raw_data(double** RAW_Rx1_data, int data_pt, fftwf_complex** fftd_data);

int load_complex_array(const char* file, const char* varname, ComplexDouble*** complex_array, size_t* m, size_t* n);

void convert_to_fftwf_complex(ComplexDouble** calibration_array, size_t m, size_t n, fftwf_complex*** fftwf_calibration_array);

fftwf_complex** allocate_fftwf_calibration_array(size_t m, size_t n);

void cfar_2d_prefixsum(float** range_doppler_cfar, int num_rows, int num_cols,
    int num_train_cells_range, int num_guard_cells_range,
    int num_train_cells_doppler, int num_guard_cells_doppler,
    float threshold_scale, int** cfar_result);

int find_peaks(int** cfar_result, int num_rows, int num_cols, int* row_indices, int* col_indices);

void fftshift_rows(float** array, int num_rows, int num_cols);

void complex_division(fftwf_complex a, fftwf_complex b, fftwf_complex result);

void get_unique_col_indices_only_one(const int* col_indices_fin, int num_detections_fin, int* unique_col_indices, int* unique_count);

void subtract_one_from_array(int* array, int size);

double** allocate_ones_matrix(int rows, int cols);

void complex_set(fftwf_complex out, float re, float im);

void complex_add(fftwf_complex out, fftwf_complex a, fftwf_complex b);

void complex_mul(fftwf_complex out, fftwf_complex a, fftwf_complex b);

void complex_conj(fftwf_complex out, fftwf_complex a);

void complex_div(fftwf_complex out, fftwf_complex a, fftwf_complex b);

float complex_abs(fftwf_complex a);

void compute_AhA(fftwf_complex A[128][2], fftwf_complex AhA[2][2]);

void compute_Ahx(fftwf_complex A[128][2], fftwf_complex x[128], fftwf_complex Ahx[128]);

int invert_2x2(fftwf_complex m[2][2], fftwf_complex inv[2][2]);

int least_squares_custom(fftwf_complex** A_steeringmatrix, fftwf_complex** x_data, int m, fftwf_complex* s_value);

int least_squares_two_column(fftwf_complex** A_steeringmatrix, fftwf_complex** x_data, int m, fftwf_complex* s_value);

void complex_sub(fftwf_complex res, fftwf_complex a, fftwf_complex b);

void compute_residual(fftwf_complex** residual, fftwf_complex** x_data, fftwf_complex** A_steeringmatrix, fftwf_complex* s_value, int m);

int main()
{
    // 컴퓨터에서 gnuplot.exe이 있는 경로로 저장(설치 방법 따르면 거의 비슷함)
    Gnuplot gp("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\"");

    // CSV 파일 저장 경로 filepath는 저장되는 파일 / nodata_filepath는 nodata파일
    const char* filepath = "D:\\backup\\measurement\\test\\test.csv";
    const char* nodata_filepath = "D:\\backup\\measurement\\test\\250401_cal_data_1.csv";

    // variable 정의
    // chirp setting 포인트 수(chirp이 바뀔 때마다 세팅)
    uint16_t pt = 2000;
    uint16_t chirpperframe = (120 + 1);
    float cpt = (400e-6 - (1e-6));

    // CFAR 알고리즘 관련 변수

    int cfar_start = 0;
    int cfar_end = 180;
    int new_num_cols = cfar_end - cfar_start + 1;
    int num_train_range = 4;
    int num_guard_range = 0;
    int num_train_doppler = 4;
    int num_guard_doppler = 2;
    float threshold_scale = 20;

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
    float range_resolution = c / (2 * f_bw);
    float max_range = (range_resolution * data_pt / 2);
    int total_size = chirpperframe * pt * ch;
    double scaling_factor = max_range / ((data_pt / 2.0) - 1.0);
    double lambda = c / 94e9;
    double doppler_resolution = 1 / (Nd * cpt);
    double max_doppler = doppler_resolution * Nd / 2;
    double max_velocity = (lambda * max_doppler) / 2;

    // notarget data 관련 변수(건드리지 말기)
    //int16_t* nodata = (int16_t*)malloc(chirpperframe * pt * ch * sizeof(int16_t));
    //nodata = read_csv_to_array(nodata_filepath, chirpperframe, pt, ch);

    // angle fft 관련 변수(건드리지 말기)
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

    //각도 관련 변수
    int Tx_x_position[] = { 64, 59, 35, 23, 19, 5, 5, 1 };
    int Tx_y_position[] = { 5, 3, 9, 11, 2, 11, 5, 1 };
    int TRx_x_position[] = { 131,130,129,128,113,112,111,110,98,97,96,95,68,67,66,65,126,125,124,123,108,107,106,105,93,92,91,90,63,62,61,60,102,101,100,99,84,83,82,81,69,68,67,66,39,38,37,36,90,89,88,87,72,71,70,69,57,56,55,54,27,26,25,24,86,85,84,83,68,67,66,65,53,52,51,50,23,22,21,20,72,71,70,69,54,53,52,51,39,38,37,36,9,8,7,6,72,71,70,69,54,53,52,51,39,38,37,36,9,8,7,6,68,67,66,65,50,49,48,47,35,34,33,32,5,4,3,2 };
    int TRx_y_position[] = { 6,6,6,6,13,13,13,13,10,10,10,10,17,17,17,17,4,4,4,4,11,11,11,11,8,8,8,8,15,15,15,15,10,10,10,10,17,17,17,17,14,14,14,14,21,21,21,21,12,12,12,12,19,19,19,19,16,16,16,16,23,23,23,23,3,3,3,3,10,10,10,10,7,7,7,7,14,14,14,14,12,12,12,12,19,19,19,19,16,16,16,16,23,23,23,23,6,6,6,6,13,13,13,13,10,10,10,10,17,17,17,17,2,2,2,2,9,9,9,9,6,6,6,6,13,13,13,13 };
    int size_tx = sizeof(Tx_x_position) / sizeof(int);
    int size_ty = sizeof(Tx_y_position) / sizeof(int);
    subtract_one_from_array(Tx_x_position, size_tx);
    subtract_one_from_array(Tx_y_position, size_ty);
    int max_targetnumber = 2;
    float epsilon_value = 0.7;
    float alpha1 = 0.593328f;
    float alpha2 = 1.8f;
    double** weight = allocate_ones_matrix(16, 8);
    int angle_total_pt = 128;

    //// Radar 의 시작 코드

    //if (dev.OpenBySerial() != okCFrontPanel::NoError) {
    //    std::cerr << "Failed to open device." << std::endl;
    //    return -1;
    //}

    //bitstream 파일 넣기(프로젝트 폴더에 bitstream 파일이 있어야 함!)

    //error = dev.ConfigureFPGA("250331_94G_matching_ver2.bit");
    //if (error != okCFrontPanel::NoError) {
    //    std::cerr << "Failed to configure FPGA. Error code: " << error << std::endl;
    //    return -1;
    //}

    // Chirp setting

    // (93~95G,2GHz, 41.675MHz) 200us_100us
    //spi_write(4, 0x00034E27);
    //spi_write(4, 0x00001F46);
    //spi_write(4, 0x00281B55);
    //spi_write(4, 0x00780284);
    //spi_write(4, 0x014300C3);
    //spi_write(4, 0x07208012);
    //spi_write(4, 0x00000001);
    //spi_write(4, 0xF8136000);

    //(93~95G,2GHz, 41.675MHz) 400us_200us
    spi_write(4, 0x00034E27);
    spi_write(4, 0x00001F46);
    spi_write(4, 0x00281B55);
    spi_write(4, 0x00780284);
    spi_write(4, 0x014300C3);
    spi_write(4, 0x07208022);
    spi_write(4, 0x00000001);
    spi_write(4, 0xF8136000);

    //(93~95G,2GHz, 41.675MHz) 800us_400us
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
        ////ADC GAIN
        //spi_write(0, 0x000004);
        //spi_write(0, 0x99000F);
        //spi_write(0, 0x9A0019);
        //// io_trigger 1
        //dev.SetWireInValue(0x06, 0x00000001); // io_trigger 1
        //dev.UpdateWireIns();
        //unsigned char* buffer = (unsigned char*)malloc(array_size * sizeof(unsigned char));
        //dev.ReadFromBlockPipeOut(0xA3, BLOCK_SIZE, array_size, buffer);  // # pipeout(fpga > PC)
        //int16_t* int16_buffer = (int16_t*)buffer; //int 16 자료형으로 변환
        //int16_t* output = (int16_t*)malloc(total_pt * sizeof(int16_t)); //공간 할당
        //transpose(int16_buffer, output, chirpperframe, pt, ch); //output으로 transpose(배열 순서 변경)

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
        //split_output_into_2d_arrays(output, chirpperframe, pt, RXDATA7, RXDATA8, RXDATA5, RXDATA6, RXDATA3, RXDATA4, RXDATA1, RXDATA2);
        //free(output);

        //// NODATA 빼는 상황
        //free(output);
        //split_output_into_2d_arrays(result, chirpperframe, pt, RXDATA3, RXDATA4, RXDATA1, RXDATA2, RXDATA6, RXDATA5, RXDATA8, RXDATA7);
        //free(result);

        //nodata만 쓰기(디버깅용)

         //notarget data 관련 변수(건드리지 말기)
        int16_t* nodata = (int16_t*)malloc(chirpperframe * pt * ch * sizeof(int16_t));
        nodata = read_csv_to_array(nodata_filepath, chirpperframe, pt, ch);
        split_output_into_2d_arrays(nodata, chirpperframe, pt, RXDATA7, RXDATA8, RXDATA5, RXDATA4, RXDATA3, RXDATA6, RXDATA1, RXDATA2);
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
        int16_t** RXDATA[8] = { RXDATA1, RXDATA2, RXDATA3, RXDATA4, RXDATA5, RXDATA6, RXDATA7, RXDATA8 };
        double*** RAW_Rx = (double***)malloc(8 * sizeof(double**));
        for (int ch = 0; ch < 8; ch++) {
            RAW_Rx[ch] = (double**)malloc((chirpperframe - 1) * sizeof(double*));
            for (int i = 1; i < chirpperframe; i++) {
                RAW_Rx[ch][i - 1] = (double*)malloc((pt - idle_point) * sizeof(double));
                for (int j = idle_point; j < pt; j++) {
                    RAW_Rx[ch][i - 1][j - idle_point] = ((double)RXDATA[ch][i][j]) / 2048.0;
                }
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

        ///////// 2D-Range FFT code /////////

        double** range_rx1_same = (double**)malloc(Nd * sizeof(double*));
        for (int i = 0; i < Nd; i++) {
            range_rx1_same[i] = (double*)malloc(data_pt * sizeof(double));
        }

        for (int i = 24; i < (chirpperframe - 1); i++) {
            for (int j = 0; j < data_pt; j++) {
                range_rx1_same[i - 24][j] = RAW_Rx[0][i][j];
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

        for (int i = 0; i < Nd; i++) {
            free(range_rx1_same[i]);
        }
        free(range_rx1_same);

        fftwf_plan plan = fftwf_plan_dft_2d(Nd, data_pt, input_2d, output_2d, FFTW_FORWARD, FFTW_ESTIMATE);

        fftwf_execute(plan);

        fftwf_free(input_2d);
        fftwf_destroy_plan(plan);

        // absolute
        float* magnitude_2d = (float*)malloc((Nd * data_pt) * sizeof(float));

        for (int i = 0; i < (Nd * data_pt); ++i) {
            float real_2d = output_2d[i][0];
            float imag_2d = output_2d[i][1];
            magnitude_2d[i] = sqrt(real_2d * real_2d + imag_2d * imag_2d);
        }
        fftwf_free(output_2d);

        float** db_2d = (float**)malloc(Nd * sizeof(float*));
        for (int i = 0; i < Nd; ++i) {
            db_2d[i] = (float*)malloc(data_pt * sizeof(float));
        }

        for (int i = 0; i < Nd; ++i) {
            for (int j = 0; j < data_pt; ++j) {
                db_2d[i][j] = magnitude_2d[i * data_pt + j];
            }
        }

        free(magnitude_2d);

        fftshift_rows(db_2d, Nd, data_pt);

        ///// 2D-CFAR 알고리즘

        float** range_doppler_cfar = (float**)malloc(Nd * sizeof(float*));
        for (int i = 0; i < Nd; ++i) {
            range_doppler_cfar[i] = (float*)malloc(new_num_cols * sizeof(float));
        }

        for (int i = 0; i < Nd; ++i) {
            for (int j = cfar_start; j <= cfar_end; ++j) {
                range_doppler_cfar[i][j - cfar_start] = db_2d[i][j];
            }
        }

        int** cfar_result = (int**)malloc(Nd * sizeof(int*));
        for (int i = 0; i < Nd; ++i) {
            cfar_result[i] = (int*)malloc(new_num_cols * sizeof(int));
        }

        cfar_2d_prefixsum(range_doppler_cfar, Nd, new_num_cols, num_train_range,
            num_guard_range, num_train_doppler, num_guard_doppler,
            threshold_scale, cfar_result);

        int max_detections = Nd * new_num_cols;
        int* row_indices = (int*)malloc(max_detections * sizeof(int));
        int* col_indices = (int*)malloc(max_detections * sizeof(int));

        int num_detections = find_peaks(cfar_result, Nd, new_num_cols, row_indices, col_indices);

        if (num_detections == 0) {
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

            continue;
        }

        int* row_indices_fin = (int*)malloc(max_detections * sizeof(int));
        int* col_indices_fin = (int*)malloc(max_detections * sizeof(int));
        int num_detections_fin = 0;

        // 인접한 포인트를 삭제하지 않는 코드

        num_detections_fin = num_detections;
        for (int i = 0; i < num_detections_fin; i++) {
            row_indices_fin[i] = row_indices[i];
            col_indices_fin[i] = col_indices[i];
        }

        //cfar_start의 인덱스를 맞추는 코드
        for (int i = 0; i < num_detections_fin; i++) {
            col_indices_fin[i] += cfar_start;
        }

        // 2D CFAR 끝

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

        ///////// 1D - range plot code ///////// 1d 거리 플롯 코드(DEBUG)

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
        //    for (int i = 0; i < num_detections_fin; ++i) {
        //        int index = col_indices_fin[i];  // Get the column index from col_indices_fin
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
            //if (num_detections_fin > 0) {
            //    std::vector<std::pair<int, double>> detected_points;
            //    for (int i = 0; i < num_detections_fin; ++i) {
            //        int index = col_indices_fin[i];
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

        ///////// 2D - range-doppler map 코드(DEBUG)

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
        //for (int i = 0; i < num_detections_fin; i++) {
        //    gp << col_indices_fin[i] << " " << row_indices_fin[i] << "\n";
        //}
        //gp << "e\n";
        //gp.flush();

        //2D - PLOT 끝

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
        for (int i = 0; i < Nd; ++i) {
            free(plot_2d[i]);
        }
        free(plot_2d);
        free(row_indices_fin);

        ////////  ANGLE FFT  ////////

        // TARGET 값 중에서 중복되는 거리의 데이터를 제거

        int* unique_col_indices = (int*)malloc(max_detections * sizeof(int));
        int unique_count = 0;

        get_unique_col_indices_only_one(col_indices_fin, num_detections_fin, unique_col_indices, &unique_count);

        free(col_indices_fin);

        // FFT

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

        perform_fft_on_raw_data(RAW_Rx[0], data_pt, fftd_data_RX1);
        perform_fft_on_raw_data(RAW_Rx[1], data_pt, fftd_data_RX2);
        perform_fft_on_raw_data(RAW_Rx[2], data_pt, fftd_data_RX3);
        perform_fft_on_raw_data(RAW_Rx[3], data_pt, fftd_data_RX4);
        perform_fft_on_raw_data(RAW_Rx[4], data_pt, fftd_data_RX5);
        perform_fft_on_raw_data(RAW_Rx[5], data_pt, fftd_data_RX6);
        perform_fft_on_raw_data(RAW_Rx[6], data_pt, fftd_data_RX7);
        perform_fft_on_raw_data(RAW_Rx[7], data_pt, fftd_data_RX8);

        for (int ch = 0; ch < 8; ch++) {
            for (int i = 0; i < (chirpperframe - 1); i++) {
                free(RAW_Rx[ch][i]);
            }
            free(RAW_Rx[ch]);
        }
        free(RAW_Rx);

        fftwf_complex*** target_fft_RAW = (fftwf_complex***)malloc(ch * sizeof(fftwf_complex**));
        for (int i = 0; i < ch; ++i) {
            target_fft_RAW[i] = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*) * 24);
            for (int j = 0; j < 24; ++j) {
                target_fft_RAW[i][j] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * unique_count);
            }
        }
        fftwf_complex*** target_fft_cal = (fftwf_complex***)malloc(ch * sizeof(fftwf_complex**));
        for (int i = 0; i < ch; ++i) {
            target_fft_cal[i] = (fftwf_complex**)fftwf_malloc(sizeof(fftwf_complex*) * 24);
            for (int j = 0; j < 24; ++j) {
                target_fft_cal[i][j] = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * unique_count);
            }
        }

        for (int i = 0; i < unique_count; i++) {
            for (int j = 0; j < 24; j++) {
                target_fft_RAW[0][j][i][0] = fftd_data_RX1[j][unique_col_indices[i]][0];
                target_fft_RAW[0][j][i][1] = fftd_data_RX1[j][unique_col_indices[i]][1];
                target_fft_RAW[1][j][i][0] = fftd_data_RX2[j][unique_col_indices[i]][0];
                target_fft_RAW[1][j][i][1] = fftd_data_RX2[j][unique_col_indices[i]][1];
                target_fft_RAW[2][j][i][0] = fftd_data_RX3[j][unique_col_indices[i]][0];
                target_fft_RAW[2][j][i][1] = fftd_data_RX3[j][unique_col_indices[i]][1];
                target_fft_RAW[3][j][i][0] = fftd_data_RX4[j][unique_col_indices[i]][0];
                target_fft_RAW[3][j][i][1] = fftd_data_RX4[j][unique_col_indices[i]][1];
                target_fft_RAW[4][j][i][0] = fftd_data_RX5[j][unique_col_indices[i]][0];
                target_fft_RAW[4][j][i][1] = fftd_data_RX5[j][unique_col_indices[i]][1];
                target_fft_RAW[5][j][i][0] = fftd_data_RX6[j][unique_col_indices[i]][0];
                target_fft_RAW[5][j][i][1] = fftd_data_RX6[j][unique_col_indices[i]][1];
                target_fft_RAW[6][j][i][0] = fftd_data_RX7[j][unique_col_indices[i]][0];
                target_fft_RAW[6][j][i][1] = fftd_data_RX7[j][unique_col_indices[i]][1];
                target_fft_RAW[7][j][i][0] = fftd_data_RX8[j][unique_col_indices[i]][0];
                target_fft_RAW[7][j][i][1] = fftd_data_RX8[j][unique_col_indices[i]][1];
            }
        }
        float* detected_ranges = (float*)malloc(max_detections * sizeof(float));
        for (int i = 0; i < unique_count; i++) {
            detected_ranges[i] = unique_col_indices[i] * scaling_factor;
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
        free(unique_col_indices);

        //calibration
        for (int i = 0; i < unique_count; i++) {
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 24; k++) {
                    fftwf_complex result;
                    complex_division(target_fft_RAW[j][k][i], fftwf_calibration_array[j][k], result);
                    target_fft_cal[j][k][i][0] = result[0];
                    target_fft_cal[j][k][i][1] = result[1];
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

        // Mapping

        fftwf_complex** x_data = (fftwf_complex**)malloc(128 * sizeof(fftwf_complex*));
        for (int i = 0; i < 128; i++) {
            x_data[i] = (fftwf_complex*)malloc(unique_count * sizeof(fftwf_complex));
        }
        for (int k = 0; k < unique_count; k++)
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

        for (int i = 0; i < ch; ++i) {
            for (int j = 0; j < 24; ++j) {
                fftwf_free(target_fft_cal[i][j]);
            }
            fftwf_free(target_fft_cal[i]);
        }
        free(target_fft_cal);

        float** x_hat = (float**)malloc(2 * sizeof(float*));
        for (int i = 0; i < 2; ++i) {
            x_hat[i] = (float*)malloc(sizeof(float) * unique_count);
        }
        float** y_hat = (float**)malloc(2 * sizeof(float*));
        for (int i = 0; i < 2; ++i) {
            y_hat[i] = (float*)malloc(sizeof(float) * unique_count);
        }
        float** z_hat = (float**)malloc(2 * sizeof(float*));
        for (int i = 0; i < 2; ++i) {
            z_hat[i] = (float*)malloc(sizeof(float) * unique_count);
        }

        // 각도 추정 코드

        for (int m = 0; m < unique_count; m++) {

            fftwf_complex** A_steeringmatrix = (fftwf_complex**)malloc(128 * sizeof(fftwf_complex*));
            for (int i = 0; i < 128; i++) {
                A_steeringmatrix[i] = (fftwf_complex*)calloc(max_targetnumber, sizeof(fftwf_complex));
            }

            fftwf_complex* s_value = (fftwf_complex*)calloc(max_targetnumber, sizeof(fftwf_complex));

            fftwf_complex** residual = (fftwf_complex**)malloc(128 * sizeof(fftwf_complex*));
            for (int i = 0; i < 128; i++) {
                residual[i] = (fftwf_complex*)malloc(unique_count * sizeof(fftwf_complex));
                for (int j = 0; j < unique_count; j++) {
                    residual[i][j][0] = x_data[i][j][0];
                    residual[i][j][1] = x_data[i][j][1];
                }
            }

            float norm_max = 0.0f;

            for (int i = 0; i < 128; i++) {
                float re = residual[i][m][0];
                float im = residual[i][m][1];
                norm_max += re * re + im * im;
            }

            norm_max = sqrtf(norm_max);

            for (int k = 0; k < max_targetnumber; k++) {

                fftwf_complex** Rx = (fftwf_complex**)malloc(16 * sizeof(fftwf_complex*));
                for (int i = 0; i < 16; i++) {
                    Rx[i] = (fftwf_complex*)fftwf_malloc(8 * sizeof(fftwf_complex));
                }
                for (int row = 0; row < 16; row++) {
                    for (int col = 0; col < 8; col++) {
                        Rx[row][col][0] = residual[row + col * 16][m][0];
                        Rx[row][col][1] = residual[row + col * 16][m][1];
                    }
                }

                fftwf_complex* Rx_final = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * angle_total_pt * angle_total_pt);
                //fftwf_complex* Rx_remove_final = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * angle_total_pt * angle_total_pt);

                for (int q = 0; q < angle_total_pt * angle_total_pt; ++q) {
                    Rx_final[q][0] = 0.0f;
                    Rx_final[q][1] = 0.0f;
                    //Rx_remove_final[q][0] = 0.0f;
                    //Rx_remove_final[q][1] = 0.0f;
                }

                for (int i = 0; i < 8; ++i) {
                    fftwf_complex* Rx_chip_1 = (fftwf_complex*)calloc(angle_total_pt * angle_total_pt, sizeof(fftwf_complex));
                    fftwf_complex* Rx_chip_2 = (fftwf_complex*)calloc(angle_total_pt * angle_total_pt, sizeof(fftwf_complex));
                    fftwf_complex* Rx_chip_3 = (fftwf_complex*)calloc(angle_total_pt * angle_total_pt, sizeof(fftwf_complex));
                    fftwf_complex* Rx_chip_4 = (fftwf_complex*)calloc(angle_total_pt * angle_total_pt, sizeof(fftwf_complex));

                    //fftwf_complex* Rx_remove_chip_1 = (fftwf_complex*)calloc(angle_total_pt * angle_total_pt, sizeof(fftwf_complex));
                    //fftwf_complex* Rx_remove_chip_2 = (fftwf_complex*)calloc(angle_total_pt * angle_total_pt, sizeof(fftwf_complex));
                    //fftwf_complex* Rx_remove_chip_3 = (fftwf_complex*)calloc(angle_total_pt * angle_total_pt, sizeof(fftwf_complex));
                    //fftwf_complex* Rx_remove_chip_4 = (fftwf_complex*)calloc(angle_total_pt * angle_total_pt, sizeof(fftwf_complex));

                    int Tx_x = Tx_x_position[i];
                    int Tx_y = Tx_y_position[i];

                    for (int k = 0; k < 4; ++k) {
                        int idx1 = (1 + Tx_y) * angle_total_pt + (67 + Tx_x - k);
                        int idx2 = (8 + Tx_y) * angle_total_pt + (49 + Tx_x - k);
                        int idx3 = (5 + Tx_y) * angle_total_pt + (34 + Tx_x - k);
                        int idx4 = (12 + Tx_y) * angle_total_pt + (4 + Tx_x - k);

                        Rx_chip_1[idx1][0] = Rx[k][i][0]; Rx_chip_1[idx1][1] = Rx[k][i][1];
                        Rx_chip_2[idx2][0] = Rx[k + 4][i][0]; Rx_chip_2[idx2][1] = Rx[k + 4][i][1];
                        Rx_chip_3[idx3][0] = Rx[k + 8][i][0]; Rx_chip_3[idx3][1] = Rx[k + 8][i][1];
                        Rx_chip_4[idx4][0] = Rx[k + 12][i][0]; Rx_chip_4[idx4][1] = Rx[k + 12][i][1];

                        //Rx_remove_chip_1[idx1][0] = weight[k][i] * Rx[k][i][0];
                        //Rx_remove_chip_1[idx1][1] = weight[k][i] * Rx[k][i][1];
                        //Rx_remove_chip_2[idx2][0] = weight[k + 4][i] * Rx[k + 4][i][0];
                        //Rx_remove_chip_2[idx2][1] = weight[k + 4][i] * Rx[k + 4][i][1];
                        //Rx_remove_chip_3[idx3][0] = weight[k + 8][i] * Rx[k + 8][i][0];
                        //Rx_remove_chip_3[idx3][1] = weight[k + 8][i] * Rx[k + 8][i][1];
                        //Rx_remove_chip_4[idx4][0] = weight[k + 12][i] * Rx[k + 12][i][0];
                        //Rx_remove_chip_4[idx4][1] = weight[k + 12][i] * Rx[k + 12][i][1];
                    }

                    for (int j = 0; j < angle_total_pt * angle_total_pt; ++j) {
                        Rx_final[j][0] += Rx_chip_1[j][0] + Rx_chip_2[j][0] + Rx_chip_3[j][0] + Rx_chip_4[j][0];
                        Rx_final[j][1] += Rx_chip_1[j][1] + Rx_chip_2[j][1] + Rx_chip_3[j][1] + Rx_chip_4[j][1];

                        //Rx_remove_final[j][0] += Rx_remove_chip_1[j][0] + Rx_remove_chip_2[j][0] + Rx_remove_chip_3[j][0] + Rx_remove_chip_4[j][0];
                        //Rx_remove_final[j][1] += Rx_remove_chip_1[j][1] + Rx_remove_chip_2[j][1] + Rx_remove_chip_3[j][1] + Rx_remove_chip_4[j][1];
                    }

                    free(Rx_chip_1); free(Rx_chip_2); free(Rx_chip_3); free(Rx_chip_4);
                    //free(Rx_remove_chip_1); free(Rx_remove_chip_2); free(Rx_remove_chip_3); free(Rx_remove_chip_4);
                }

                for (int i = 0; i < 16; i++) {
                    fftwf_free(Rx[i]);
                }
                free(Rx);

                fftwf_complex* Rx_final_fft = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * angle_total_pt * angle_total_pt);
                //fftwf_complex* Rx_remove_final_fft = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * angle_total_pt * angle_total_pt);

                fftwf_plan plan_2d = fftwf_plan_dft_2d(angle_total_pt, angle_total_pt, Rx_final, Rx_final_fft, FFTW_FORWARD, FFTW_ESTIMATE);
                //fftwf_plan plan_remove_2d = fftwf_plan_dft_2d(angle_total_pt, angle_total_pt, Rx_remove_final, Rx_remove_final_fft, FFTW_FORWARD, FFTW_ESTIMATE);

                fftwf_execute(plan_2d);
                //fftwf_execute(plan_remove_2d);

                fftwf_complex* Rx_final_fft_shifted = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * angle_total_pt * angle_total_pt);
                //fftwf_complex* Rx_final_remove_fft_shifted = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * angle_total_pt * angle_total_pt);

                for (int i = 0; i < angle_total_pt; i++) {
                    int shifted_i = (i + angle_total_pt / 2) % angle_total_pt;
                    for (int j = 0; j < angle_total_pt; j++) {
                        int shifted_j = (j + angle_total_pt / 2) % angle_total_pt;

                        int original_idx = i * angle_total_pt + j;
                        int shifted_idx = shifted_i * angle_total_pt + shifted_j;

                        Rx_final_fft_shifted[shifted_idx][0] = Rx_final_fft[original_idx][0];
                        Rx_final_fft_shifted[shifted_idx][1] = Rx_final_fft[original_idx][1];
                        //Rx_final_remove_fft_shifted[shifted_idx][0] = Rx_remove_final_fft[original_idx][0];
                        //Rx_final_remove_fft_shifted[shifted_idx][1] = Rx_remove_final_fft[original_idx][1];
                    }
                }

                float* prod_Y_fft = (float*)malloc(sizeof(float) * angle_total_pt * angle_total_pt);
                for (int i = 0; i < angle_total_pt; i++) {
                    for (int j = 0; j < angle_total_pt; j++) {
                        int idx = i * angle_total_pt + j;
                        prod_Y_fft[idx] = sqrtf(Rx_final_fft_shifted[idx][0] * Rx_final_fft_shifted[idx][0] +
                            Rx_final_fft_shifted[idx][1] * Rx_final_fft_shifted[idx][1]);
                        //float mag_remove_rx = sqrtf(Rx_final_remove_fft_shifted[idx][0] * Rx_final_remove_fft_shifted[idx][0] +
                        //    Rx_final_remove_fft_shifted[idx][1] * Rx_final_remove_fft_shifted[idx][1]);
                    }
                }

                fftwf_destroy_plan(plan_2d);
                //fftwf_destroy_plan(plan_remove_2d);
                fftwf_free(Rx_final);
                fftwf_free(Rx_final_fft);
                fftwf_free(Rx_final_fft_shifted);
                //fftwf_free(Rx_remove_final_fft);
                //fftwf_free(Rx_final_remove_fft_shifted);
                //fftwf_free(Rx_remove_final);

                float max_val = -1.0f;
                int max_index = 0;

                for (int i = 0; i < angle_total_pt * angle_total_pt; i++) {
                    if (prod_Y_fft[i] > max_val) {
                        max_val = prod_Y_fft[i];
                        max_index = i;
                    }
                }

                int k_x = (max_index / 512) + 1;
                int k_y = (max_index % 512) + 1;

                float peak_idx_x, peak_idx_y;
                float estimated_angle_theta = 0.0f;
                float estimated_angle_phi = 0.0f;

                peak_idx_y = -((float)(k_x - 1 - angle_total_pt / 2.0f)) / (angle_total_pt * alpha2);
                peak_idx_x = (float)(k_y - 1 - angle_total_pt / 2.0f) / (angle_total_pt * alpha1);

                if (peak_idx_x >= 0 && peak_idx_y > 0) {
                    estimated_angle_theta = asinf(sqrt(peak_idx_x * peak_idx_x + peak_idx_y * peak_idx_y)) * (180.0f / PI);
                    estimated_angle_phi = atanf(peak_idx_y / peak_idx_x) * (180.0f / PI);
                }
                else if (peak_idx_x == 0 && peak_idx_y == 0) {
                    estimated_angle_theta = 0.0f;
                    estimated_angle_phi = 0.0f;
                }
                else if (peak_idx_x < 0 && peak_idx_y >= 0) {
                    estimated_angle_theta = asinf(sqrt(peak_idx_x * peak_idx_x + peak_idx_y * peak_idx_y)) * (180.0f / PI);
                    estimated_angle_phi = 180.0f + atanf(peak_idx_y / peak_idx_x) * (180.0f / PI);
                }
                else if (peak_idx_x >= 0 && peak_idx_y <= 0) {
                    estimated_angle_theta = asinf(sqrt(peak_idx_x * peak_idx_x + peak_idx_y * peak_idx_y)) * (180.0f / PI);
                    estimated_angle_phi = atanf(peak_idx_y / peak_idx_x) * (180.0f / PI);
                }
                else if (peak_idx_x <= 0 && peak_idx_y <= 0) {
                    estimated_angle_theta = asinf(sqrt(peak_idx_x * peak_idx_x + peak_idx_y * peak_idx_y)) * (180.0f / PI);
                    estimated_angle_phi = atanf(peak_idx_y / peak_idx_x) * (180.0f / PI) - 180.0f;
                }

                fftwf_complex* A_steeringvector = (fftwf_complex*)malloc(sizeof(fftwf_complex) * 128);
                float sin_theta = sinf(estimated_angle_theta * PI / 180.0f);
                float cos_phi = cosf(estimated_angle_phi * PI / 180.0f);
                float sin_phi = sinf(estimated_angle_phi * PI / 180.0f);

                for (int i = 0; i < 128; i++) {
                    float phase = 2.0f * PI * (
                        alpha1 * TRx_x_position[i] * sin_theta * cos_phi -
                        alpha2 * TRx_y_position[i] * sin_theta * sin_phi
                        );
                    A_steeringvector[i][0] = cosf(phase);
                    A_steeringvector[i][1] = sinf(phase);
                }

                for (int i = 0; i < 128; i++) {
                    A_steeringmatrix[i][k][0] = A_steeringvector[i][0];
                    A_steeringmatrix[i][k][1] = A_steeringvector[i][1];
                }

                free(A_steeringvector);


                if (k == 0) {
                    least_squares_custom(A_steeringmatrix, x_data, m, s_value);

                    for (int i = 0; i < 128; i++) {
                        fftwf_complex Ax = { 0.0f, 0.0f };
                        fftwf_complex temp;

                        complex_mul(temp, A_steeringmatrix[i][k], s_value[k]);
                        complex_add(Ax, Ax, temp);

                        complex_sub(residual[i][m], x_data[i][m], Ax);
                    }
                    x_hat[k][m] = detected_ranges[m] * sinf(estimated_angle_theta * PI / 180.0f) * cosf(estimated_angle_phi * PI / 180.0f);
                    y_hat[k][m] = detected_ranges[m] * sinf(estimated_angle_theta * PI / 180.0f) * sinf(estimated_angle_phi * PI / 180.0f);
                    z_hat[k][m] = detected_ranges[m] * cosf(estimated_angle_theta * PI / 180.0f);

                }
                if (k == 1) {
                    least_squares_two_column(A_steeringmatrix, x_data, m, s_value);
                    compute_residual(residual, x_data, A_steeringmatrix, s_value, m);
                    x_hat[k][m] = detected_ranges[m] * sinf(estimated_angle_theta * PI / 180.0f) * cosf(estimated_angle_phi * PI / 180.0f);
                    y_hat[k][m] = detected_ranges[m] * sinf(estimated_angle_theta * PI / 180.0f) * sinf(estimated_angle_phi * PI / 180.0f);
                    z_hat[k][m] = detected_ranges[m] * cosf(estimated_angle_theta * PI / 180.0f);

                }


                float residual_norm = 0.0f;

                for (int i = 0; i < 128; ++i) {
                    float re = residual[i][m][0];
                    float im = residual[i][m][1];
                    residual_norm += re * re + im * im;
                }
                residual_norm = sqrtf(residual_norm);
                if ((residual_norm / norm_max) < epsilon_value)
                    break;

            }

            for (int i = 0; i < 128; i++) {
                free(residual[i]);
            }
            free(residual);
            free(s_value);
            for (int i = 0; i < 128; i++) {
                free(A_steeringmatrix[i]);
            }
            free(A_steeringmatrix);

        }
        //전체 플롯 코드
        /////////

        std::vector<double> x0(unique_count), y0(unique_count), z0(unique_count);
        std::vector<double> x1(unique_count), y1(unique_count), z1(unique_count);
        for (int i = 0; i < unique_count; ++i) {
            x0[i] = x_hat[0][i];
            z0[i] = z_hat[0][i];
            y0[i] = y_hat[0][i];
            x1[i] = x_hat[1][i];
            y1[i] = y_hat[1][i];
            z1[i] = z_hat[1][i];
        }
        std::vector< std::vector< std::pair<double, double> > > xz_series_0(unique_count);
        std::vector< std::vector< std::pair<double, double> > > xz_series_1(unique_count);
        std::vector< std::vector< std::pair<double, double> > > yz_series_0(unique_count);
        std::vector< std::vector< std::pair<double, double> > > yz_series_1(unique_count);
        for (int i = 0; i < unique_count; ++i) {
            xz_series_0[i].push_back(std::make_pair(x0[i], z0[i]));
            xz_series_1[i].push_back(std::make_pair(x1[i], z1[i]));
            yz_series_0[i].push_back(std::make_pair(y0[i], z0[i]));
            yz_series_1[i].push_back(std::make_pair(y1[i], z1[i]));
        }
        std::vector<std::string> colors = { "red", "blue", "green", "magenta", "cyan", "orange", "brown", "violet" };
        gp << "clear\n";  // 이전 데이터를 지우고 같은 윈도우에서 새로 그림
        // multiplot 설정: 1행 2열의 레이아웃 (왼쪽: xz평면, 오른쪽: yz평면)
        gp << "set multiplot layout 1,2 title 'Target Positions on xz and yz planes (Centered at 0,0)'\n";

        //// --- xz평면 플롯 ---
        gp << "set title 'Azimuth'\n";
        gp << "set xlabel 'x (m)'\n";
        gp << "set ylabel 'z (m)'\n";
        gp << "set xrange [" << -10 << ":" << 10 << "]\n";
        gp << "set yrange [" << 0 << ":" << 10 << "]\n";
        gp << "set grid\n";
        gp << "plot ";
        // Target0 (row0)의 각 검출을 개별 시리즈로 플롯
        for (int i = 0; i < unique_count; ++i) {
            gp << "'-' with points pt 7 lc rgb '" << colors[i % colors.size()]
                << "' title 'Target0_" << i << "', ";
        }
        // Target1 (row1)의 각 검출을 개별 시리즈로 플롯
        for (int i = 0; i < unique_count; ++i) {
            gp << "'-' with points pt 7 lc rgb '" << colors[(i + unique_count) % colors.size()]
                << "' title 'Target1_" << i << "'";
            if (i != unique_count - 1) {
                gp << ", ";
            }
        }
        gp << "\n";
        // xz평면 데이터 전송
        for (int i = 0; i < unique_count; ++i) {
            gp.send1d(xz_series_0[i]);
        }
        for (int i = 0; i < unique_count; ++i) {
            gp.send1d(xz_series_1[i]);
        }
        //// --- yz평면 플롯 ---
        gp << "set title 'Elevation'\n";
        gp << "set xlabel 'y (m)'\n";
        gp << "set ylabel 'z (m)'\n";
        gp << "set xrange [" << -10 << ":" << 10 << "]\n";
        gp << "set yrange [" << -10 << ":" << 10 << "]\n";
        gp << "set grid\n";
        gp << "plot ";
        // Target0 (row0)의 각 검출을 개별 시리즈로 플롯 (yz평면)
        for (int i = 0; i < unique_count; ++i) {
            gp << "'-' with points pt 7 lc rgb '" << colors[i % colors.size()]
                << "' title 'Target" << i << "', ";
        }
        // Target1 (row1)의 각 검출
        for (int i = 0; i < unique_count; ++i) {
            gp << "'-' with points pt 7 lc rgb '" << colors[(i + unique_count) % colors.size()]
                << "' title 'Target" << i << "'";
            if (i != unique_count - 1) {
                gp << ", ";
            }
        }
        gp << "\n";
        // yz평면 데이터 전송
        for (int i = 0; i < unique_count; ++i) {
            gp.send1d(yz_series_0[i]);
        }
        for (int i = 0; i < unique_count; ++i) {
            gp.send1d(yz_series_1[i]);
        }
        gp << "unset multiplot\n";
        gp.flush();

        //2D-PLOT
        /////////
        for (int i = 0; i < 2; ++i) {
            free(x_hat[i]);
        }
        free(x_hat);
        for (int i = 0; i < 2; ++i) {
            free(y_hat[i]);
        }
        free(y_hat);
        for (int i = 0; i < 2; ++i) {
            free(z_hat[i]);
        }
        free(z_hat);
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

std::vector<std::pair<int, int>> detected_peaks;
void cfar_2d_prefixsum(float** range_doppler_cfar, int num_rows, int num_cols,
    int num_train_cells_range, int num_guard_cells_range,
    int num_train_cells_doppler, int num_guard_cells_doppler,
    float threshold_scale, int** cfar_result)
{
    // Total training 영역 training + guard
    int total_train_range = num_train_cells_range + num_guard_cells_range;
    int total_train_doppler = num_train_cells_doppler + num_guard_cells_doppler;

    // 2D prefix sum 배열 생성 num_rows+1 x num_cols+1
    float** prefix = new float* [num_rows + 1];
    for (int i = 0; i <= num_rows; ++i) {
        prefix[i] = new float[num_cols + 1];
        for (int j = 0; j <= num_cols; ++j) {
            prefix[i][j] = 0.0f;
        }
    }

    // prefixsum 계산
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            prefix[i + 1][j + 1] = range_doppler_cfar[i][j] + prefix[i][j + 1] + prefix[i + 1][j] - prefix[i][j];
        }
    }

    // CFAR 연산 수행 : 경계 부분은 전체 training 영역이 확보되어 있는 경우에만 처리
    for (int r = total_train_range; r < num_rows - total_train_range; ++r) {
        for (int d = total_train_doppler; d < num_cols - total_train_doppler; ++d) {
            // 큰 영역 (전체 training 영역: guard 포함)의 좌표 설정
            int r1 = r - total_train_range;
            int c1 = d - total_train_doppler;
            int r2 = r + total_train_range;
            int c2 = d + total_train_doppler;
            // 큰 영역의 합 구하기
            float big_sum = prefix[r2 + 1][c2 + 1]
                - prefix[r1][c2 + 1]
                - prefix[r2 + 1][c1]
                + prefix[r1][c1];

                // Guard 영역의 좌표 설정
                int gr1 = r - num_guard_cells_range;
                int gc1 = d - num_guard_cells_doppler;
                int gr2 = r + num_guard_cells_range;
                int gc2 = d + num_guard_cells_doppler;
                // Guard 영역의 합 계산
                float guard_sum = prefix[gr2 + 1][gc2 + 1]
                    - prefix[gr1][gc2 + 1]
                    - prefix[gr2 + 1][gc1]
                    + prefix[gr1][gc1];

                    // Training 영역 (guard 제외) 총 합
                    float noise_level = big_sum - guard_sum;

                    // 각 영역의 셀 개수 : 큰 영역과 guard 영역은 정해진 크기를 가짐
                    int area_big = (2 * total_train_range + 1) * (2 * total_train_doppler + 1);
                    int area_guard = (2 * num_guard_cells_range + 1) * (2 * num_guard_cells_doppler + 1);
                    int num_training_cells = area_big - area_guard;

                    float noise_avg = noise_level / num_training_cells;
                    float threshold = noise_avg * threshold_scale;
                    // CFAR 결과 적용: 중심 셀 값이 threshold보다 크면 1, 아니면 0
                    if (range_doppler_cfar[r][d] > threshold) {
                        detected_peaks.push_back(std::make_pair(r, d)); // find_peaks 함수 따로 구현할 필요 x 
                        cfar_result[r][d] = 1;
                    }
                    else {
                        cfar_result[r][d] = 0;
                    }
        }
    }

    // 경계 부분의 결과는 전부 0으로 설정
    for (int r = 0; r < num_rows; ++r) {
        for (int d = 0; d < num_cols; ++d) {
            if (r < total_train_range || r >= num_rows - total_train_range ||
                d < total_train_doppler || d >= num_cols - total_train_doppler) {
                cfar_result[r][d] = 0;
            }
        }
    }

    // 동적 할당한 prefix 배열 메모리 해제
    for (int i = 0; i <= num_rows; ++i) {
        delete[] prefix[i];
    }
    delete[] prefix;
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


void complex_division(fftwf_complex a, fftwf_complex b, fftwf_complex result) {
    float denominator = (b[0] * b[0]) + (b[1] * b[1]);
    if (denominator == 0.0f) {
        result[0] = 0.0f;
        result[1] = 0.0f;
    }
    else {
        result[0] = ((a[0] * b[0]) + (a[1] * b[1])) / denominator;
        result[1] = ((a[1] * b[0]) - (a[0] * b[1])) / denominator;
    }
}

void get_unique_col_indices_only_one(
    const int* col_indices_fin,
    int num_detections_fin,
    int* unique_col_indices,
    int* unique_count
) {
    std::unordered_set<int> seen;
    int idx = 0;

    for (int i = 0; i < num_detections_fin; ++i) {
        int col = col_indices_fin[i];
        if (seen.count(col) == 0) {
            seen.insert(col);
            unique_col_indices[idx] = col;
            idx++;
        }
    }

    *unique_count = idx;
}

void subtract_one_from_array(int* array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] -= 1;
    }
}

double** allocate_ones_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 1.0;
        }
    }
    return matrix;
}

void complex_set(fftwf_complex out, float re, float im) {
    out[0] = re;
    out[1] = im;
}

void complex_add(fftwf_complex out, fftwf_complex a, fftwf_complex b) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
}

void complex_mul(fftwf_complex out, fftwf_complex a, fftwf_complex b) {
    out[0] = a[0] * b[0] - a[1] * b[1];
    out[1] = a[0] * b[1] + a[1] * b[0];
}

void complex_conj(fftwf_complex out, fftwf_complex a) {
    out[0] = a[0];
    out[1] = -a[1];
}

void complex_div(fftwf_complex out, fftwf_complex a, fftwf_complex b) {
    float denom = b[0] * b[0] + b[1] * b[1];
    out[0] = (a[0] * b[0] + a[1] * b[1]) / denom;
    out[1] = (a[1] * b[0] - a[0] * b[1]) / denom;
}

float complex_abs(fftwf_complex a) {
    return sqrtf(a[0] * a[0] + a[1] * a[1]);
}

// A^H * A 계산 (2x2)
void compute_AhA(fftwf_complex A[128][2], fftwf_complex AhA[2][2]) {
    fftwf_complex conj_Ak, mul;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            complex_set(AhA[i][j], 0.0f, 0.0f);
            for (int k = 0; k < 128; k++) {
                complex_conj(conj_Ak, A[k][i]);
                complex_mul(mul, conj_Ak, A[k][j]);
                complex_add(AhA[i][j], AhA[i][j], mul);
            }
        }
    }
}

// A^H * x 계산 (2x1)
void compute_Ahx(fftwf_complex A[128][2], fftwf_complex x[128], fftwf_complex Ahx[128]) {
    fftwf_complex conj_Ak, mul;

    for (int i = 0; i < 2; i++) {
        complex_set(Ahx[i], 0.0f, 0.0f);
        for (int k = 0; k < 128; k++) {
            complex_conj(conj_Ak, A[k][i]);
            complex_mul(mul, conj_Ak, x[k]);
            complex_add(Ahx[i], Ahx[i], mul);
        }
    }
}

int invert_2x2(fftwf_complex m[2][2], fftwf_complex inv[2][2]) {
    fftwf_complex det, a, b, c, d, neg;
    complex_mul(det, m[0][0], m[1][1]);

    complex_mul(a, m[0][1], m[1][0]);
    a[0] = -a[0]; a[1] = -a[1];
    complex_add(det, det, a);

    if (complex_abs(det) < 1e-6) return -1;

    complex_div(inv[0][0], m[1][1], det);
    neg[0] = -m[0][1][0]; neg[1] = -m[0][1][1];
    complex_div(inv[0][1], neg, det);

    neg[0] = -m[1][0][0]; neg[1] = -m[1][0][1];
    complex_div(inv[1][0], neg, det);



    complex_div(inv[1][1], m[0][0], det);

    return 0;
}

int least_squares_custom(fftwf_complex** A_steeringmatrix, fftwf_complex** x_data, int m, fftwf_complex* s_value) {
    fftwf_complex AhA = { 0.0f, 0.0f };
    fftwf_complex Ahx = { 0.0f, 0.0f };
    fftwf_complex conj_A, mul;

    for (int i = 0; i < 128; i++) {
        complex_conj(conj_A, A_steeringmatrix[i][0]);
        complex_mul(mul, conj_A, A_steeringmatrix[i][0]);
        complex_add(AhA, AhA, mul);

        complex_mul(mul, conj_A, x_data[i][m]);
        complex_add(Ahx, Ahx, mul);
    }

    float denom = AhA[0] * AhA[0] + AhA[1] * AhA[1];
    if (denom < 1e-6f) {
        s_value[0][0] = 0.0f;
        s_value[0][1] = 0.0f;
        return -1;
    }

    complex_div(s_value[0], Ahx, AhA);

    return 0;
}

int least_squares_two_column(fftwf_complex** A_steeringmatrix, fftwf_complex** x_data, int m, fftwf_complex* s_value) {
    fftwf_complex AhA[2][2];
    fftwf_complex Ahx[2];
    fftwf_complex AhA_inv[2][2];
    fftwf_complex conj_Aik, mul;

    // 1. 초기화
    for (int i = 0; i < 2; i++) {
        Ahx[i][0] = 0.0f; Ahx[i][1] = 0.0f;
        for (int j = 0; j < 2; j++) {
            AhA[i][j][0] = 0.0f; AhA[i][j][1] = 0.0f;
        }
    }

    // 2. AhA와 Ahx 계산
    for (int i = 0; i < 128; i++) {
        for (int row = 0; row < 2; row++) {
            complex_conj(conj_Aik, A_steeringmatrix[i][row]);

            complex_mul(mul, conj_Aik, x_data[i][m]);
            complex_add(Ahx[row], Ahx[row], mul);

            for (int col = 0; col < 2; col++) {
                complex_mul(mul, conj_Aik, A_steeringmatrix[i][col]);
                complex_add(AhA[row][col], AhA[row][col], mul);
            }
        }
    }

    // 3. AhA 역행렬 구하기
    if (invert_2x2(AhA, AhA_inv) != 0) {
        printf("Matrix AhA is not invertible.\n");
        s_value[0][0] = s_value[0][1] = 0.0f;
        s_value[1][0] = s_value[1][1] = 0.0f;
        return -1;
    }

    // 4. s_value = inv(AhA) * Ahx
    for (int i = 0; i < 2; i++) {
        s_value[i][0] = 0.0f;
        s_value[i][1] = 0.0f;
        for (int j = 0; j < 2; j++) {
            complex_mul(mul, AhA_inv[i][j], Ahx[j]);
            complex_add(s_value[i], s_value[i], mul);
        }
    }

    return 0;
}

void complex_sub(fftwf_complex res, fftwf_complex a, fftwf_complex b) {
    res[0] = a[0] - b[0];
    res[1] = a[1] - b[1];
}

void compute_residual(fftwf_complex** residual, fftwf_complex** x_data, fftwf_complex** A_steeringmatrix, fftwf_complex* s_value, int m) {
    for (int i = 0; i < 128; i++) {
        fftwf_complex temp1, temp2, Ax;

        temp1[0] = A_steeringmatrix[i][0][0] * s_value[0][0] - A_steeringmatrix[i][0][1] * s_value[0][1];
        temp1[1] = A_steeringmatrix[i][0][0] * s_value[0][1] + A_steeringmatrix[i][0][1] * s_value[0][0];

        temp2[0] = A_steeringmatrix[i][1][0] * s_value[1][0] - A_steeringmatrix[i][1][1] * s_value[1][1];
        temp2[1] = A_steeringmatrix[i][1][0] * s_value[1][1] + A_steeringmatrix[i][1][1] * s_value[1][0];

        Ax[0] = temp1[0] + temp2[0];
        Ax[1] = temp1[1] + temp2[1];

        residual[i][m][0] = x_data[i][m][0] - Ax[0];
        residual[i][m][1] = x_data[i][m][1] - Ax[1];
    }
}
