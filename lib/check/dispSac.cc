#include <cstdlib>
#include <iostream>

#include "user/sacio.h"

float ZeroTol = 0.0001;

inline float absolute(float a) {
  if (a < 0) return (-a);
  return a;
}

inline bool isCloseEnough(float a, float b, float relTol = 0.001) {
  float abs_a = absolute(a);
  float abs_b = absolute(b);
  float min = (abs_a > abs_b) ? abs_b : abs_a;
  float max = (abs_a > abs_b) ? abs_a : abs_b;
  if (min < ZeroTol) {
    if (max < ZeroTol) return true;
    return false;
  }
  float diff = max - min;
  float reldiff = diff / min;
  if (reldiff < relTol) return true;
  return false;
}

static void usage() {
  std::cerr<<"Usage: dispSac <path> [options]"<<'\n';
  std::cerr<<"Options:"<<'\n';
  std::cerr<<"   -p  position"<<'\n';
  std::cerr<<"   -l  lower bound"<<'\n';
  std::cerr<<"   -u  upper bound"<<'\n';
  std::cerr<<"   -f  find number"<<'\n';
}

int main(int argc, char** argv) {
  int opt;
  int opterr = 0;

  std::string path;
  bool should_disp = false;
  int disp_pos;
  int lb = 5, ub = 5;
  bool should_find = false;
  float find_number = 10;

  if (argc < 2) {
    opterr ++;
  }

  int curr_pos = 0;
  int curr_arg_pos = 0;
  while (curr_pos < argc) {
    std::string curr = argv[curr_pos];
    if (curr == "-p") {
      curr_pos ++;
      disp_pos = atoi(argv[curr_pos]);
      should_disp = true;
      curr_pos ++;
      continue;
    }
    if (curr == "-l") {
      curr_pos ++;
      lb = atoi(argv[curr_pos]);
      curr_pos ++;
      continue;
    }
    if (curr == "-u") {
      curr_pos ++;
      ub = atoi(argv[curr_pos]);
      curr_pos ++;
      continue;
    }
    if (curr == "-f") {
      curr_pos ++;
      find_number = atof(argv[curr_pos]);
      should_find = true;
      curr_pos ++;
      continue;
    }
    if (curr_arg_pos == 1) path = argv[curr_pos];
    curr_pos ++;
    curr_arg_pos ++;
  }

  if (opterr) {
    usage();
    exit(1);
  }

  SacInput sac(path);
  //display header info
  std::cout << "sac header info: " << '\n';
  std::cout << "  dt: " << sac.header().delta << '\n';
  std::cout << "  npts: " << sac.header().npts << '\n';
  std::cout << "  t1: " << sac.header().t1 << '\n';
  std::cout << "  t2: " << sac.header().t2 << '\n';
  //display data
  if (should_disp) {
    int disp_start = disp_pos - lb - 1;
    size_t disp_start_bytes = disp_start * sizeof(float);
    size_t disp_bytes = (lb + ub + 1) * sizeof(float);
    float* data = new float[lb + ub + 1];
    sac.read((char*)data, disp_start_bytes, disp_bytes);
    //before
    std::cout << "before: " << '\n';
    for (int i = 0; i < lb; i++) 
      std::cout << data[i] << '\t';
    std::cout << '\n';
    //the point
    std::cout << "the point: " << '\n';
    std::cout << data[lb] << '\n';
    //after
    std::cout<< "after: " << '\n';
    for (int i = lb + 1; i < lb + ub + 1; i++) 
      std::cout << data[i] << '\t';
    std::cout << '\n';
    delete [] data;
  }
  //find position
  if (should_find) {
    //read data
    size_t npts = sac.header().npts;
    float* data = new float[npts];
    sac.read((char*)data, 0, npts * sizeof(float));
    for (int i = 0; i < npts; ++i) {
      if (isCloseEnough(data[i], find_number)) {
        std::cout << i << '\t' << data[i] << '\n';
      }
    }
    delete [] data;
  }

}
