# FMCW Radar System with Raspberry Pi 5

본 프로젝트는 **Raspberry Pi 5 (16GB, Linux 시스템)** 기반의 **FMCW (Frequency Modulated Continuous Wave) 레이더 시스템** 구현을 목표로 합니다.  
수집된 데이터를 **C++ 기반 GNUplot**을 사용하여 시각화합니다.  

## 📌 설치 방법  

### 1️⃣ FFTW & matio 라이브러리 설치  
```bash
sudo apt-get update
sudo apt-get install libfftw3-dev gnuplot
sudo apt-get install libfftw3f-dev gnuplot
sudo apt-get install libmatio-dev
```

### 2️⃣ Boost 라이브러리 설치  
```bash
sudo apt-get update
sudo apt-get install libboost-dev libboost-iostreams-dev
```

## 📊 데이터 시각화  
- **GNUplot**을 사용하여 FMCW 레이더 데이터를 그래프로 출력합니다.  
- C++ 기반으로 데이터 처리 및 시각화 코드를 작성합니다.  

## 🛠️ 주요 기술  
- **Raspberry Pi 5 (16GB, Linux)**  
- **FMCW Radar**  
- **FFTW (고속 푸리에 변환 라이브러리)**  
- **Boost C++ 라이브러리**  
- **GNUplot을 활용한 데이터 시각화**  
