#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double sum_serial(double* x, size_t n)
{
	double s = 0.0;

	for (int i = 0; i != n; ++i)
	{
		s += x[i];
	}

	return s;
}

double sum_parallel(double* x, size_t n)
{
	double s = 0.0;

	#pragma omp parallel for
	for (int i = 0; i != n; ++i)
	{
		s += x[i];
	}

	return s;
}

double sum_parallel_critical(double* x, size_t n)
{
	double s = 0.0;

	#pragma omp parallel for
	for (int i = 0; i != n; ++i)
	{
		#pragma omp critical
		s += x[i];
	}

	return s;
}

double sum_parallel_local(double* x, size_t n)
{
	double s[omp_get_max_threads()];

	#pragma omp parallel
	{
		s[omp_get_thread_num()] = 0.0;

		#pragma omp for
		for (int i = 0; i != n; ++i)
		{
			s[omp_get_thread_num()] += x[i];
		}
	}

	for (int i = 1; i != omp_get_max_threads(); ++i)
	{
		s[0] += s[i];
	}

	return s[0];
}

double sum_parallel_optimized(double* x, size_t n)
{
	double s[8 * omp_get_max_threads()];

	#pragma omp parallel
	{
		s[8 * omp_get_thread_num()] = 0.0;

		#pragma omp for
		for (int i = 0; i != n; ++i)
		{
			s[8 * omp_get_thread_num()] += x[i];
		}
	}

	for (int i = 1; i != omp_get_max_threads(); ++i)
	{
		s[0] += s[8 * i];
	}

	return s[0];
}

int main()
{
	int n = 1e8;
	double* x = malloc(n * sizeof(double));

	for (int i = 0; i != n; ++i)
	{
		x[i] = rand() / (double)RAND_MAX;
	}

	//double a = omp_get_wtime();
	//double s = sum_serial(x, n);
	//double b = omp_get_wtime();

	//double a = omp_get_wtime();
	//double s = sum_parallel(x, n);
	//double b = omp_get_wtime();

	//double a = omp_get_wtime();
	//double s = sum_parallel_critical(x, n);
	//double b = omp_get_wtime();

	//double a = omp_get_wtime();
	//double s = sum_parallel_local(x, n);
	//double b = omp_get_wtime();

	double a = omp_get_wtime();
	double s = sum_parallel_optimized(x, n);
	double b = omp_get_wtime();

	printf("Computed %.0f in %.9f seconds\n", s, b - a);

	return 0;
}