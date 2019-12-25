#include <string.h>
#include <iostream>
#include <chrono>
#include <random>
#include <Windows.h>
#include <mpi.h>
#include <omp.h>

using namespace std;
using namespace chrono;

#define MAT_SIZE 2000	// 矩阵大小
#define MAX_NUM 1000	// 矩阵元素最大值
#define NUM_THREADS 8	// 线程数量

enum tags {
	init_mat_a,
	init_mat_b,
	send_mat_a,
	send_mat_b,
	calc_mat_c
};

void MainProc(const int proc_cnt) {
	MPI_Status status;

	// 设置随机因子
	const auto timeNow = time(nullptr);
	default_random_engine rand(timeNow);
	const uniform_int_distribution<unsigned> dist(1, MAX_NUM);


	// 输出每进程的线程数
	cout << "Threads per process: " << omp_get_max_threads() << endl;

	// 记录开始时间
	const auto start = system_clock::now();

	// 创建矩阵空间
	const auto a = new int*[MAT_SIZE], b = new int*[MAT_SIZE], c = new int*[MAT_SIZE];

	// 主进程生成最后一块矩阵元素
#pragma omp parallel for shared(a) shared(b)
	for (auto i = MAT_SIZE / proc_cnt * (proc_cnt - 1); i < MAT_SIZE; ++i) {
		a[i] = new int[MAT_SIZE];
		b[i] = new int[MAT_SIZE];
		c[i] = new int[MAT_SIZE]{0};
		for (auto j = 0; j < MAT_SIZE; ++j) {
			a[i][j] = dist(rand);
			b[i][j] = dist(rand);
		}
	}

	// 收取子进程生成矩阵元素
	for (auto pid = 0; pid < proc_cnt - 1; ++pid) {
		for (auto i = MAT_SIZE / proc_cnt * pid; i < MAT_SIZE / proc_cnt * (pid + 1); ++i) {
			a[i] = new int[MAT_SIZE];
			b[i] = new int[MAT_SIZE];
			MPI_Recv(a[i], MAT_SIZE, MPI_INT, pid, init_mat_a, MPI_COMM_WORLD, &status);
			MPI_Recv(b[i], MAT_SIZE, MPI_INT, pid, init_mat_b, MPI_COMM_WORLD, &status);
		}
	}

	// 发送矩阵信息到子进程		
	for (auto pid = 0; pid < proc_cnt - 1; ++pid) {
		for (auto i = 0; i < MAT_SIZE; ++i) {
			MPI_Send(a[i], MAT_SIZE, MPI_INT, pid, send_mat_a, MPI_COMM_WORLD);
			MPI_Send(b[i], MAT_SIZE, MPI_INT, pid, send_mat_b, MPI_COMM_WORLD);
		}
	}

	// 主进程计算最后一块矩阵乘积		
#pragma omp parallel for shared(a) shared(b) shared(c)
	for (auto i = MAT_SIZE / proc_cnt * (proc_cnt - 1); i < MAT_SIZE; ++i) {
		for (auto j = 0; j < MAT_SIZE; ++j) {
			for (auto k = 0; k < MAT_SIZE; ++k) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	// 收取子进程计算矩阵乘积结果
	for (auto pid = 0; pid < proc_cnt - 1; ++pid) {
		for (auto i = MAT_SIZE / proc_cnt * pid; i < MAT_SIZE / proc_cnt * (pid + 1); ++i) {
			c[i] = new int[MAT_SIZE];
			MPI_Recv(c[i], MAT_SIZE, MPI_INT, pid, calc_mat_c, MPI_COMM_WORLD, &status);
		}
	}

	// 输出总共用时
	cout << "Processed in " << proc_cnt << " process(es) uses "
		<< duration_cast<milliseconds>(system_clock::now() - start).count()
		<< " milliseconds." << endl;

	// delete分行写才能完全释放各个变量
	for (auto i = 0; i < MAT_SIZE; ++i) {
		delete[] a[i];
		delete[] b[i];
		delete[] c[i];
	}
	delete[] a;
	delete[] b;
	delete[] c;
}

void SubProc(const int proc_cnt) {
	MPI_Status status;

	// 设置随机因子
	const auto timeNow = time(nullptr);
	default_random_engine rand(timeNow);
	const uniform_int_distribution<unsigned> dist(1, MAX_NUM);

	// 子进程生成对应矩阵元素
	auto a = new int*[MAT_SIZE / proc_cnt];
	auto b = new int*[MAT_SIZE / proc_cnt];
#pragma omp parallel for shared(a) shared(b)
	for (auto i = 0; i < MAT_SIZE / proc_cnt; ++i) {
		a[i] = new int[MAT_SIZE];
		b[i] = new int[MAT_SIZE];
		for (auto j = 0; j < MAT_SIZE; ++j) {
			a[i][j] = dist(rand);
			b[i][j] = dist(rand);
		}
	}

	// 子进程发送对应矩阵元素
	for (auto i = 0; i < MAT_SIZE / proc_cnt; ++i) {
		MPI_Send(a[i], MAT_SIZE, MPI_INT, proc_cnt - 1, init_mat_a, MPI_COMM_WORLD);
		MPI_Send(b[i], MAT_SIZE, MPI_INT, proc_cnt - 1, init_mat_b, MPI_COMM_WORLD);
	}

	// 释放缓冲区
	for (auto i = 0; i < MAT_SIZE / proc_cnt; ++i) {
		delete[] a[i];
		delete[] b[i];
	}
	delete[] a;
	delete[] b;

	// 子进程收取矩阵信息
	a = new int*[MAT_SIZE];
	b = new int*[MAT_SIZE];
	const auto c = new int*[MAT_SIZE / proc_cnt];
	for (auto i = 0; i < MAT_SIZE; ++i) {
		a[i] = new int[MAT_SIZE];
		b[i] = new int[MAT_SIZE];
		MPI_Recv(a[i], MAT_SIZE, MPI_INT, proc_cnt - 1, send_mat_a, MPI_COMM_WORLD, &status);
		MPI_Recv(b[i], MAT_SIZE, MPI_INT, proc_cnt - 1, send_mat_b, MPI_COMM_WORLD, &status);
	}

	// 子进程计算对应位置矩阵乘积		
#pragma omp parallel for shared(a) shared(b) shared(c)
	for (auto i = 0; i < MAT_SIZE / proc_cnt; ++i) {
		c[i] = new int[MAT_SIZE]{0};
		for (auto j = 0; j < MAT_SIZE; ++j) {
			for (auto k = 0; k < MAT_SIZE; ++k) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	// 子进程发送计算矩阵乘积结果
	for (auto i = 0; i < MAT_SIZE / proc_cnt; ++i) {
		MPI_Send(c[i], MAT_SIZE, MPI_INT, proc_cnt - 1, calc_mat_c, MPI_COMM_WORLD);
	}

	// 释放缓冲区
	for (auto i = 0; i < MAT_SIZE; ++i) {
		delete[] a[i];
		delete[] b[i];
		if (i < MAT_SIZE / proc_cnt) {
			delete[] c[i];
		}
	}
	delete[] a;
	delete[] b;
	delete[] c;
}

int main(int argc, char* argv[]) {
	int procCnt, procId;

	// 初始化MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &procId);
	MPI_Comm_size(MPI_COMM_WORLD, &procCnt);

	// 设置线程总数
	omp_set_num_threads(NUM_THREADS);

	if (procId == procCnt - 1) {
		// 最后一个进程为主进程
		MainProc(procCnt);
	}
	else {
		// 其他进程为子进程
		SubProc(procCnt);
	}

	MPI_Finalize();
}
