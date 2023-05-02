#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void DFT_serial(double* xr, double* xi, double* Xr, double* Xi, int n, int idft)
{
	double k = (2.0 * M_PI * (idft ? -1.0 : +1.0)) / (double)n;

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			Xr[i] += xr[j] * cos(k * (double)(i * j)) + xi[j] * sin(k * (double)(i * j));
			Xi[i] += xi[j] * cos(k * (double)(i * j)) - xr[j] * sin(k * (double)(i * j));
		}
	}

	if (idft)
	{
		for (int i = 0; i < n; ++i)
		{
			Xr[i] /= n;
			Xi[i] /= n;
		}
	}
}

void DFT_parallel(double* xr, double* xi, double* Xr, double* Xi, int n, int idft)
{
	double k = (2.0 * M_PI * (idft ? -1.0 : +1.0)) / (double)n;

	#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			Xr[i] += xr[j] * cos(k * (double)(i * j)) + xi[j] * sin(k * (double)(i * j));
			Xi[i] += xi[j] * cos(k * (double)(i * j)) - xr[j] * sin(k * (double)(i * j));
		}
	}

	if (idft)
	{
		#pragma omp parallel for
		for (int i = 0; i < n; ++i)
		{
			Xr[i] /= n;
			Xi[i] /= n;
		}
	}
}

int main()
{
	int n = 10000;
	int e = 0;

	double* xr[3];
	double* xi[3];

	xr[0] = (double*)malloc(n * sizeof(double));
	xi[0] = (double*)malloc(n * sizeof(double));
	xr[1] = (double*)malloc(n * sizeof(double));
	xi[1] = (double*)malloc(n * sizeof(double));
	xr[2] = (double*)malloc(n * sizeof(double));
	xi[2] = (double*)malloc(n * sizeof(double));

	for (int i = 0; i != n; ++i)
	{
		xr[0][i] = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
		xi[0][i] = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
	}

	for (int i = 0; i != n; ++i)
	{
		xr[1][i] = 0.0;
		xi[1][i] = 0.0;
	}

	for (int i = 0; i != n; ++i)
	{
		xr[2][i] = 0.0;
		xi[2][i] = 0.0;
	}

	//double t0 = omp_get_wtime();
	//DFT_serial(xr[0], xi[0], xr[1], xi[1], n, 0);
	//DFT_serial(xr[1], xi[1], xr[2], xi[2], n, 1);
	//double t1 = omp_get_wtime();

	double t0 = omp_get_wtime();
	DFT_parallel(xr[0], xi[0], xr[1], xi[1], n, 0);
	DFT_parallel(xr[1], xi[1], xr[2], xi[2], n, 1);
	double t1 = omp_get_wtime();

	for (int i = 0; i != n; ++i)
	{
		e += fabs(xr[0][i] - xr[2][i]) > 1e-8 || fabs(xi[0][i] - xi[2][i]) > 1e-8 ? 1 : 0;
	}

	printf("Computed %d elements in %.9f seconds with %d errors.\n", n, t1 - t0, e);

	free(xr[0]);
	free(xi[0]);
	free(xr[1]);
	free(xi[1]);
	free(xr[2]);
	free(xi[2]);

	return 0;
}