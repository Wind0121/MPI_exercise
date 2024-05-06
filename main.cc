#include "mpi.h"
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iostream>
using namespace std::chrono;
constexpr int kN = 1000;
int my_id, num_proc;
int a[kN][kN], b[kN][kN], c[kN][kN];

duration<size_t, std::nano> singleMatrix() {
  memset(a, -1, sizeof(a));
  memset(b, -1, sizeof(b));
  memset(c, 0, sizeof(c));
  auto start = system_clock::now();
  for (int i = 0; i < kN; i++) {
    for (int j = 0; j < kN; j++) {
      for (int k = 0; k < kN; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  auto end = system_clock::now();
  return end - start;
}

duration<size_t, std::nano> multiMatrix() {
  memset(a, -1, sizeof(a));
  memset(b, -1, sizeof(b));
  memset(c, 0, sizeof(c));
  auto start = system_clock::now();
  int st = kN / num_proc * my_id, ed = kN / num_proc * (my_id + 1);
  for (int i = st; i < ed; i++) {
    for (int j = 0; j < kN; j++) {
      for (int k = 0; k < kN; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  MPI_Gather(&c[st][0], kN / num_proc * kN, MPI_INT, &c[0][0],
             kN / num_proc * kN, MPI_INT, 0, MPI_COMM_WORLD);
  //   if (my_id == 0) {
  //     for (int i = 1; i < 5; i++) {
  //       int sti = kN / num_proc * i;
  //       MPI_Recv(&c[sti][0], kN / num_proc * kN, MPI_INT, i, 0,MPI_COMM_WORLD, nullptr);
  //     }
  //   } else {
  //     MPI_Send(&c[st][0], kN / num_proc * kN, MPI_INT, 0, 0, MPI_COMM_WORLD);
  //   }
  auto end = system_clock::now();
  return end - start;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

  printf("process %d of %d: init\n", my_id, num_proc);

  if (my_id == 0) {
    auto dur = singleMatrix();
    std::cout << "singleMatrix: " << dur.count() << "ns\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  auto dur = multiMatrix();
  if (my_id == 0) {
    std::cout << "multiMatrix: " << dur.count() << "ns\n";
    for (int i = 0; i < kN; i++)
      for (int j = 0; j < kN; j++)
        if (c[i][j] != kN)
          std::cout << "error ";
  }

  MPI_Finalize();
  return 0;
}