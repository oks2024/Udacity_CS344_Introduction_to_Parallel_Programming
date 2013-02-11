// Image.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "utils.h"

using namespace std;


void your_rgba_to_greyscale(uchar4 * const d_rgbaImage,
							unsigned char* const d_greyImage, size_t numRows, size_t numCols);

int _tmain(int argc, _TCHAR* argv[])
{
	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	// Initialize images.
	cv::Mat imageRGBA;
	cv::Mat imageGrey;

	cv::Mat image = cv::imread("./cinque_terre_small.jpg", CV_LOAD_IMAGE_COLOR);

	if (image.empty()) 
	{
	    cout << "Cannot load image!" << endl;
	    return -1;
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);
	
	// Shouldn't happen
	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous())
	{
	    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
	    exit(1);
	}

	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
	
	// Initialize host memory.
	h_rgbaImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	h_greyImage  = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = imageRGBA.rows * imageRGBA.cols;

	// Initialize GPU memory
	checkCudaErrors(cudaMalloc(&d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(&d_greyImage, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

	//copy input array to the GPU
	checkCudaErrors(cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	GpuTimer timer;
	timer.Start();
	//call the students' code
	your_rgba_to_greyscale(d_rgbaImage, d_greyImage, imageRGBA.rows, imageRGBA.cols);
	timer.Stop();
  
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	

	std::cout << timer.Elapsed();

	//copy the output back to the host
	checkCudaErrors(cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));


	/*postProcess("D:/Photos/result.jpg");*/
	cv::namedWindow( "Display window", CV_WINDOW_NORMAL);
	cv::imshow("Display window", imageGrey);

	cv::imwrite("./result.jpg", imageGrey);
	
	cv::waitKey(0);

	//cleanup
	cudaFree(d_greyImage);
	cudaFree(d_rgbaImage);

	return 0;
}

