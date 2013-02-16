//Udacity HW2 Driver
#pragma once
#include "stdafx.h"
#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
						uchar4* const d_outputImageRGBA,
						const size_t numRows, const size_t numCols,
						unsigned char *d_redBlurred,
						unsigned char *d_greenBlurred,
						unsigned char *d_blueBlurred,
						const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
								const float* const h_filter, const size_t filterWidth);

void cleanup();



int _tmain(int argc, _TCHAR* argv[]) 
{
	uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
	uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
	unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

	float *h_filter;
	int    filterWidth;

	//PreProcess
	const std::string *filename = new std::string("./cinque_terre_small.jpg");
	cv::Mat imageInputRGBA;
	cv::Mat imageOutputRGBA;

	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	cv::Mat image = cv::imread(filename->c_str(), CV_LOAD_IMAGE_COLOR);
  
	if (image.empty()) 
	{
	std::cerr << "Couldn't open file: " << filename << std::endl;
	cv::waitKey(0);
	exit(1);
	}

	cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
	std::cerr << "Images aren't continuous!! Exiting." << std::endl;
	exit(1);
	}

	h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

	const size_t numPixels = image.rows * image.cols;
	//allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

	//copy input array to the GPU
	checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	//now create the filter that they will use
	const int blurKernelWidth = 9;
	const float blurKernelSigma = 2.;

	filterWidth = blurKernelWidth;

	//create and fill the filter we will convolve with
	h_filter = new float[blurKernelWidth * blurKernelWidth];

	float filterSum = 0.f; //for normalization

	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) 
	{
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c)
		{
			float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r)
	{
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c)
		{
			h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
		}
	}

	//blurred
	checkCudaErrors(cudaMalloc(&d_redBlurred,    sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(&d_greenBlurred,  sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMalloc(&d_blueBlurred,   sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_redBlurred,   0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_blueBlurred,  0, sizeof(unsigned char) * numPixels));


	allocateMemoryAndCopyToGPU(image.rows, image.cols, h_filter, filterWidth);
	GpuTimer timer;
	timer.Start();
	//call the students' code
	your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, image.rows, image.cols,
						d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	int err = printf("%f msecs.\n", timer.Elapsed());

	if (err < 0) {
	//Couldn't print! Probably the student closed stdout - bad news
	std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
	exit(1);
	}

	cleanup();

	//check results and output the blurred image
	//PostProcess

	//copy the output back to the host
	checkCudaErrors(cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	cv::Mat imageOutputBGR;
	cv::cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);
	//output the image
	cv::imwrite("./blurredResult.jpg", imageOutputBGR);

	cv::namedWindow( "Display window", CV_WINDOW_NORMAL);
	cv::imshow("Display window", imageOutputBGR);
	
	cv::waitKey(0);


	checkCudaErrors(cudaFree(d_redBlurred));
	checkCudaErrors(cudaFree(d_greenBlurred));
	checkCudaErrors(cudaFree(d_blueBlurred));

	return 0;
}