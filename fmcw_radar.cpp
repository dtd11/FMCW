#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include "gnuplot-iostream.h"
#include <string>

// ------------------------------------
// CSV 읽기 
// ------------------------------------
std::vector<std::vector<double>> readCSV(const std::string &filename, bool skipFirstRow, bool skipFirstCol) {
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Failed to open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<std::vector<double>> data;
    std::string line;
    bool firstLine = true;
    while(std::getline(file, line)) {
        if(skipFirstRow && firstLine) {
            firstLine = false;
            continue;
        }
        std::stringstream ss(line);
        std::vector<double> rowvals;
        std::string val;
        int colCount=0;
        while(std::getline(ss, val, ',')) {
            // 첫 열 제거할지 여부
            if(skipFirstCol && colCount==0){
                colCount++;
                continue;
            }
            // string -> double
            try {
                double d = std::stod(val);
                rowvals.push_back(d);
            }
            catch(...) {
                // 빈 값 등
                rowvals.push_back(0.0);
            }
            colCount++;
        }
        // 빈 행이면 패스
        if(!rowvals.empty()){
            data.push_back(rowvals);
        }
    }
    file.close();
    return data;
}

// ------------------------------------
// FFTW 복소배열 편의 구조
//  - fftwf_complex는 float이지만, double FFTW는 fftw_complex
// ------------------------------------
struct ComplexVal {
    double real;
    double imag;
};

// ------------------------------------
// Idle 제거, 1-chirp 제거, voltage 변환 예시
// ------------------------------------
std::vector<std::vector<double>> removeChirpAndIdle(
    const std::vector<std::vector<double>> &in,
    int idle_point, double conv_factor)
{
    // (chirpperframe, cols)
    // 1) 첫 행(1chirp) 제거 -> (chirpperframe-1, cols)
    if(in.size() <= 1) {
        std::cerr << "[Error] not enough rows to remove 1 chirp.\n";
        exit(EXIT_FAILURE);
    }
    std::vector<std::vector<double>> step1(in.size()-1);
    for(size_t r=1; r<in.size(); r++){
        step1[r-1] = in[r];
    }
    int rowCount = (int)step1.size();
    int colCount = (int)step1[0].size();
    // 2) idle_point 열 제거
    if(colCount <= idle_point){
        std::cerr << "[Error] not enough cols to remove idle_point.\n";
        exit(EXIT_FAILURE);
    }
    std::vector<std::vector<double>> step2(rowCount, std::vector<double>(colCount - idle_point, 0.0));
    for(int r=0; r<rowCount; r++){
        for(int c=idle_point; c<colCount; c++){
            step2[r][c - idle_point] = step1[r][c];
        }
    }
    // 3) Voltage 변환
    for(int r=0; r<rowCount; r++){
        for(int c=0; c<(colCount - idle_point); c++){
            step2[r][c] *= conv_factor;
        }
    }
    return step2;
}

// ------------------------------------
// shape_adjust: start_idx만큼 행을 더 버리고, Nd = rowCount - start_idx
// total_pt = colCount
// ------------------------------------
std::vector<std::vector<double>> shapeAdjust(
    const std::vector<std::vector<double>> &in,
    int start_idx,
    int &Nd_out, int &total_pt_out)
{
    // in shape=(rows, cols)
    int rows = (int)in.size();
    int cols = (rows>0)? (int)in[0].size() : 0;
    int Nd_ = rows - start_idx;
    if(Nd_ < 0) {
        Nd_ = 0;
    }
    int total_pt_ = cols;
    // slicing
    int valid_rows = std::min(Nd_, rows - start_idx); // 혹시 초과 방지
    int valid_cols = total_pt_; // 그대로
    if(valid_rows<0) valid_rows=0;
    if(valid_cols<0) valid_cols=0;

    std::vector<std::vector<double>> out(valid_rows, std::vector<double>(valid_cols, 0.0));
    for(int r=0; r<valid_rows; r++){
        for(int c=0; c<valid_cols; c++){
            out[r][c] = in[start_idx + r][c];
        }
    }
    Nd_out = valid_rows;
    total_pt_out = valid_cols;
    return out;
}

// ------------------------------------
// 2D FFT with FFTW
//  data: shape=(Nd, total_pt)
//  1) range FFT(axis=1)
//  2) doppler FFT(axis=0)
//  3) shift+scale
// ------------------------------------
std::vector<std::vector<std::complex<double>>> fft2d(
    const std::vector<std::vector<double>> &input,
    int Nd, int total_pt)
{
    // input shape=(Nd, total_pt)
    // 1) range FFT => row-wise
    //    we'll do row by row using 1D FFT
    //    then do doppler FFT => col-wise
    // *** FFTW_PLAN_DFT_1D, or plan_many
    // 여기서는 row-wise를 루프로 구현 (간단 예시)

    std::vector<std::vector<std::complex<double>>> rangeFFT(Nd, std::vector<std::complex<double>>(total_pt, {0,0}));

    // row-wise
    for(int r=0; r<Nd; r++){
        // prepare in/out
        fftw_complex *inF = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*total_pt);
        fftw_complex *outF= (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*total_pt);
        for(int c=0; c<total_pt; c++){
            inF[c][0] = input[r][c]; // real
            inF[c][1] = 0.0;         // imag
        }
        fftw_plan p = fftw_plan_dft_1d(total_pt, inF, outF, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);

        // scaling
        for(int c=0; c<total_pt; c++){
            double re = outF[c][0] * 2.0 / (double)total_pt;
            double im = outF[c][1] * 2.0 / (double)total_pt;
            rangeFFT[r][c] = std::complex<double>(re, im);
        }
        fftw_free(inF); fftw_free(outF);
    }

    // 2) doppler FFT => col-wise => shape still (Nd, total_pt)
    // we'll allocate a new array
    std::vector<std::vector<std::complex<double>>> dopFFT(Nd, std::vector<std::complex<double>>(total_pt));

    // col by col
    for(int c=0; c<total_pt; c++){
        fftw_complex *inF = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nd);
        fftw_complex *outF= (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*Nd);
        for(int r=0; r<Nd; r++){
            inF[r][0] = rangeFFT[r][c].real();
            inF[r][1] = rangeFFT[r][c].imag();
        }
        fftw_plan p = fftw_plan_dft_1d(Nd, inF, outF, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);

        // scaling + shift
        // shift => fftshift => index -> index+Nd/2 mod Nd
        for(int r=0; r<Nd; r++){
            int shifted = (r + Nd/2) % Nd;
            double re = outF[r][0] * 2.0 / (double)Nd;
            double im = outF[r][1] * 2.0 / (double)Nd;
            dopFFT[shifted][c] = std::complex<double>(re, im);
        }
        fftw_free(inF); fftw_free(outF);
    }

    return dopFFT;
}

// ------------------------------------
// to_dB: 20*log10(|x|/10 + 1e-16) + 30
// ------------------------------------
double to_dB(const std::complex<double> &val){
    double mag = std::abs(val)/10.0 + 1e-16;
    return 20.0*std::log10(mag) + 30.0;
}

// ------------------------------------
// 2D CFAR (아주 간단)
// cfar_result: same shape, 0/1
// ------------------------------------
std::vector<std::vector<int>> cfar2d(
    const std::vector<std::vector<double>> &rd,
    int num_train_range, int num_guard_range,
    int num_train_doppler, int num_guard_doppler,
    double threshold_scale)
{
    int nr = (int)rd.size();
    if(nr==0) return {};
    int ndp = (int)rd[0].size();
    std::vector<std::vector<int>> result(nr, std::vector<int>(ndp, 0));

    int total_range = num_train_range+num_guard_range;
    int total_doppler = num_train_doppler+num_guard_doppler;
    for(int r=total_range; r<nr-total_range; r++){
        for(int d=total_doppler; d<ndp-total_doppler; d++){
            double noise_level=0.0;
            int count=0;
            for(int rr=r-total_range; rr<=r+total_range; rr++){
                for(int dd=d-total_doppler; dd<=d+total_doppler; dd++){
                    if(std::abs(rr-r)<=num_guard_range && std::abs(dd-d)<=num_guard_doppler){
                        continue;
                    }
                    noise_level += rd[rr][dd];
                    count++;
                }
            }
            double avg = noise_level/(count+1e-16);
            double thresh= avg*threshold_scale;
            if(rd[r][d] > thresh){
                result[r][d] = 1;
            }
        }
    }
    return result;
}

// ------------------------------------
// 메인
// ------------------------------------
int main(){
    // 1) CSV 파일 읽기
    std::string csv_file = "3m_rignt_20_1.csv";
    bool skipFirstRow = true; // 시나리오에 맞게
    bool skipFirstCol = true;

    // CSV -> 2D double
    std::vector<std::vector<double>> Data = readCSV(csv_file, skipFirstRow, skipFirstCol);
    std::cout << "[DEBUG] CSV shape = " << Data.size() << " x " 
              << (Data.empty()? 0: Data[0].size()) << std::endl;

    // 파라미터
    int chirpperframe = 120+1;
    int start_idx = 25;
    int idle_point = 20;
    double conv_factor = 2.0/4096.0;

    // Rx 분리 (4Rx 예시)
    // Data: shape=(chirpperframe*4, ?)
    // Rx1: Data[0..chirpperframe-1]
    // ...
    if((int)Data.size() < chirpperframe*4){
        std::cerr << "[Error] not enough rows for 4Rx.\n";
        return -1;
    }

    std::vector<std::vector<double>> Rx1_data(Data.begin(), Data.begin()+chirpperframe);
    std::vector<std::vector<double>> Rx2_data(Data.begin()+chirpperframe, Data.begin()+2*chirpperframe);
    std::vector<std::vector<double>> Rx3_data(Data.begin()+2*chirpperframe, Data.begin()+3*chirpperframe);
    std::vector<std::vector<double>> Rx4_data(Data.begin()+3*chirpperframe, Data.begin()+4*chirpperframe);

    // removeChirpAndIdle
    auto RAW_Rx1_data_Vol = removeChirpAndIdle(Rx1_data, idle_point, conv_factor);
    auto RAW_Rx2_data_Vol = removeChirpAndIdle(Rx2_data, idle_point, conv_factor);
    auto RAW_Rx3_data_Vol = removeChirpAndIdle(Rx3_data, idle_point, conv_factor);
    auto RAW_Rx4_data_Vol = removeChirpAndIdle(Rx4_data, idle_point, conv_factor);

    // shapeAdjust -> range_rx1
    int Nd1=0, total_pt1=0;
    auto range_rx1 = shapeAdjust(RAW_Rx1_data_Vol, start_idx, Nd1, total_pt1);

    // FFT 시간 측정
    clock_t fft_start = clock();
    // 2D FFT
    auto fft_rx1_2d = fft2d(range_rx1, Nd1, total_pt1);
    clock_t fft_end = clock();
    double fft_time = double(fft_end - fft_start)/CLOCKS_PER_SEC;
    std::cout << "[INFO] FFT 실행 시간: " << fft_time << " 초" << std::endl;

    // dB 변환
    // shape=(Nd1, total_pt1)
    // half => total_pt1/2
    int half_col = total_pt1/2;
    std::vector<std::vector<double>> rd1(Nd1, std::vector<double>(half_col, 0.0));
    for(int r=0; r<Nd1; r++){
        for(int c=0; c<half_col; c++){
            rd1[r][c] = to_dB( fft_rx1_2d[r][c] );
        }
    }

    // CFAR 시간 측정
    clock_t cfar_startt = clock();
    // min val 빼기
    double minv = 1e30;
    for(int r=0; r<Nd1; r++){
        for(int c=0; c<half_col; c++){
            if(rd1[r][c]<minv) minv=rd1[r][c];
        }
    }
    for(int r=0; r<Nd1; r++){
        for(int c=0; c<half_col; c++){
            rd1[r][c] -= minv;
        }
    }

    int cfar_begin=20, cfar_end=50;
    if(cfar_end>half_col) cfar_end=half_col; // guard

    // sub region
    std::vector<std::vector<double>> rd1_sub(Nd1, std::vector<double>(cfar_end-cfar_begin, 0.0));
    for(int r=0; r<Nd1; r++){
        for(int c=cfar_begin; c<cfar_end; c++){
            rd1_sub[r][c-cfar_begin] = rd1[r][c];
        }
    }

    // cfar
    auto cfar_res = cfar2d(rd1_sub, 4,2, 4,2, 1.25); // 예시

    // 인접포인트 제거
    // (루프 돌면서 cf. python처럼 or adjacency)

    clock_t cfar_endt = clock();
    double cfar_time = double(cfar_endt - cfar_startt)/CLOCKS_PER_SEC;
    std::cout << "[INFO] CFAR 실행 시간: " << cfar_time << " 초" << std::endl;

    // ---------------------------------
    // 시각화 (gnuplot)
    // ---------------------------------
    // 1) rd1을 "rd1.dat"에 저장
    {
        std::ofstream fout("rd1.dat");
        for(int r=0; r<Nd1; r++){
            for(int c=0; c<half_col; c++){
                fout << rd1[r][c] << " ";
            }
            fout << "\n";
        }
        fout.close();
    }
    // 2) cfar_res를 "cfar.dat"에 저장 => (row,col) where=1
    {
        std::ofstream fout("cfar.dat");
        for(int r=0; r<Nd1; r++){
            for(int c=0; c<(int)cfar_res[0].size(); c++){
                if(cfar_res[r][c]==1){
                    fout << c+cfar_begin << " " << r << "\n";
                }
            }
        }
        fout.close();
    }

    // 3) gnuplot 실행. (matrix plot or image)
    std::ofstream script("plot_script.gp");
    script << "set title 'Range-Doppler Map (Rx1)'\n";
    script << "set view map\n";
    script << "set size ratio -1\n";
    script << "unset key\n";
    script << "plot 'rd1.dat' matrix with image, 'cfar.dat' using 1:2 with points pt 7 lc rgb 'red'\n";
    script.close();
    int ret = system("gnuplot -persist plot_script.gp");
    if(ret == -1){
        std::cerr << "[Warning] Failed to launch gnuplot.\n";
    }
    return 0;
}
