// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include <opencv2/imgproc.hpp>
// include my project functions
#include "functions.h"

typedef std::vector <cv::Point> Contour;


namespace ipa {
	// bilateral filter parameters
	int filter_size = 15;
	int sigma_color = 150;
	int sigma_space = 75;

}
template <typename Container, typename UnaryPredicate>
void erase_if(Container& c, UnaryPredicate pred) {
	c.erase(std::remove_if(c.begin(), c.end(), pred), c.end());
}

// Given an image it returns the contours found with Otsu or Triangle Binarization
std::vector <Contour> contour_finder(cv::Mat img,bool Otsu) 
{
	if (Otsu)
	{
		int threshold = ucas::getOtsuAutoThreshold(ucas::histogram(img));
		cv::threshold(img, img, threshold, 255, cv::THRESH_BINARY);
		//ucas::imshow("Binarized image", channel);

		std::vector <Contour> objects;
		cv::findContours(img, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
		erase_if(objects, [](Contour& c) {
			return cv::contourArea(c) < 500;});
		return objects;
	}
	else
	{
		int threshold = ucas::getTriangleAutoThreshold(ucas::histogram(img));
		cv::threshold(img, img, threshold, 255, cv::THRESH_BINARY);
		//ucas::imshow("Binarized image", channel);

		std::vector <Contour> objects;
		cv::findContours(img, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
		erase_if(objects, [](Contour& c) {
			return cv::contourArea(c) < 500;});
		return objects;
	}
}

int main()
{
	srand(TIME(0));

	try
	{
		
		cv::Mat img = cv::imread(std::string(EXAMPLE_IMAGES_PATH) + "/dataset/Tomato___Bacterial_spot/Bacterial_spot_0.JPG");
		if (!img.data)
			throw ucas::Error("Cannot open image");

		ucas::imshow("Original image", img);
		
		// 1 - DENOISING
		
		// bilateral filtering

		cv::Mat denoised_img=img.clone();
		cv::bilateralFilter(img, denoised_img, ipa::filter_size, ipa::sigma_color, ipa::sigma_space);

		// 2 - SEGMENTATION
		// 2.1 - Leaf is clearly visible in HSV color space (in particular in the first two channels)
		// -> change color space 
		cv::cvtColor(denoised_img, denoised_img, cv::COLOR_BGR2HSV);

		std::vector <cv::Mat> imgChannels(3);
		cv::split(denoised_img, imgChannels);
		
		//ucas::imshow("First channel", imgChannels[0]);
		//ucas::imshow("Second channel", imgChannels[1]);
		//ucas::imshow("Third channel", imgChannels[2]);
		
		
		// The difference is considered in order to filter out other components
		cv::Mat channel=imgChannels[0]-imgChannels[1];
		// The sum is considered in order to clearly see the leaf 
		cv::Mat channel_wl= imgChannels[0] + imgChannels[1];

		//ucas::imshow("Channel", channel);
		cv::Mat img_clone = img.clone();
		
		// 2.2 - Contours finding and drawing
		std::vector<Contour> objects = contour_finder(channel, true);
			
		std::vector<Contour> objects_wl = contour_finder (channel_wl, true);

		for (int k = 0; k < objects.size(); k++)
			cv::drawContours(img, objects, k, cv::Scalar(0, 0,256), 1, cv::LINE_AA);

		for (int k = 0; k < objects_wl.size(); k++)
			cv::drawContours(img_clone, objects_wl, k, cv::Scalar(0, 0, 256), 1, cv::LINE_AA);

		ucas::imshow("Contourn", img);

		ucas::imshow("Contourn with leaf", img_clone);

		// In certain cases the channel and channel_wl contain partial leaf's contours and external contours
		// -> considering the difference leaves only the leaf
		cv::Mat difference = img - img_clone;
		cv::cvtColor(difference, difference, cv::COLOR_BGR2GRAY);
		int triangle = ucas::getOtsuAutoThreshold(ucas::histogram(difference));
		cv::threshold(difference, difference, triangle, 255, cv::THRESH_BINARY);

		ucas::imshow("Difference", difference);


		return EXIT_SUCCESS;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}