// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <opencv2/opencv.hpp>


typedef std::vector < cv::Point > Contour;

int main()
{
	try
	{
		// Load the Leaf image
		cv::Mat leafImg = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/Tomato___Early_blight/Early_blight_5.jpg", cv::IMREAD_UNCHANGED);
		if (!leafImg.data)
			throw ipa::error("Cannot load image");
		
		printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n", leafImg.rows, leafImg.cols, leafImg.channels(), ipa::bitdepth(leafImg.depth()));

		ipa::imshow("Leaf image", leafImg);

		cv::Mat hsv;
		cv::cvtColor(leafImg, hsv, cv::COLOR_BGR2HSV);

		ipa::imshow("Leaf image in HSV", hsv);

		cv::Mat greenMask, yellowMask, brownMask;

		cv::inRange(hsv, cv::Scalar(30, 40, 40), cv::Scalar(85, 255, 255), greenMask);   // Verde
		cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
		cv::inRange(hsv, cv::Scalar(10, 50, 100), cv::Scalar(20, 255, 200), brownMask);   // Marrone

		cv::Mat fullMask = greenMask | yellowMask | brownMask;

		ipa::imshow("Mask", fullMask);

		cv::Mat bgmodel, fgmodel;

		cv::Mat mask(leafImg.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
		mask.setTo(cv::GC_FGD, fullMask);
	

		cv::morphologyEx(fullMask, fullMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

		ipa::imshow("After closing", fullMask);

		cv::Mat maskCopy = fullMask.clone();

		for (int y = 0; y < maskCopy.rows; y++)
		{
			unsigned char* yRow = maskCopy.ptr<unsigned char>(y);

			for (int x = 0; x < maskCopy.cols; x++)
			{
				if (yRow[x] == 0)
					yRow[x] = 255;
				else
					yRow[x] = 0;
			}
		}

		ipa::imshow("inverted img", maskCopy);


		std::vector<Contour> objects;
		cv::findContours(maskCopy, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		float areaLim = 500;

		for (int k = 0; k < objects.size(); k++)
		{
			float area = cv::contourArea(objects[k]);
			if (area < areaLim)
				cv::drawContours(maskCopy, std::vector<std::vector<cv::Point>>{objects[k]}, -1, cv::Scalar(0), cv::FILLED);
		}

		cv::morphologyEx(maskCopy, maskCopy, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

		ipa::imshow("Image Analysis results: ", maskCopy);

		std::vector<Contour> leafCounour;
		cv::findContours(maskCopy, leafCounour, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		for (int k = 0; k < leafCounour.size(); k++)
		{
			cv::drawContours(leafImg, leafCounour, k, cv::Scalar(0, 0, 255), 3, cv::FILLED);
		}

		ipa::imshow("Image Analysis results: ", leafImg);

		return 1;
	}
	catch (ipa::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

