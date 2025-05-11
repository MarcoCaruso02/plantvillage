// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"

// include my project functions
#include "functions.h"

#include <opencv2/opencv.hpp>

cv::Mat invertImg(cv::Mat img);

cv::Mat gradientImageGray(const cv::Mat& frameGray);

std::vector<cv::Point> extractNonBlackPoints(const cv::Mat& maskBGR);

cv::Mat segmentGraphCutFromBGRMasks(const cv::Mat& image, const cv::Mat& fgMaskBGR, const cv::Mat& bgMaskBGR, int iterCount = 5);

void NlMeansParameters(int sigma, float& h, int& N, int& S);


std::string root = "C:/Users/marco/Desktop/New_dataset";


typedef std::vector < cv::Point > Contour;

int main()
{
	bool debug = false;
	bool see_result = false;
	bool orginalWithNewContour = true;
	std::string nclass = "7a";
	try
	{
		for (int i = 0; i <= 159; i++)
		{
			std::string leafPath = "/Tomato___Spider_mites Two-spotted_spider_mite/Spider_mites_";

			// Load the Leaf image
			cv::Mat leafImg = cv::imread(root + leafPath + std::to_string(i) + ".jpg", cv::IMREAD_UNCHANGED);
			if (!leafImg.data)
				throw ipa::error("Cannot load image");

			printf("Image loaded: dims = %d x %d, channels = %d, bitdepth = %d\n", leafImg.rows, leafImg.cols, leafImg.channels(), ipa::bitdepth(leafImg.depth()));
			
			if (debug)
				ipa::imshow("Leaf image", leafImg);

			cv::Mat leafImgCopy = leafImg.clone();

			cv::Mat hsv;
			cv::cvtColor(leafImg, hsv, cv::COLOR_BGR2HSV);

			if (debug)
				ipa::imshow("Leaf image in HSV", hsv);
			if (i == 24)
				cv::imwrite("C:/Users/marco/Desktop/" + std::to_string(i) + "HSV.jpg", hsv);
			cv::Mat greenMask, yellowMask, brownMask, redMask;

			//change mask based on the dataset

			if (leafPath == "/Tomato___Bacterial_spot/Bacterial_spot_")
			{
				//ipa::imshow("Leaf image in HSV", hsv);
				cv::inRange(hsv, cv::Scalar(30, 40, 20), cv::Scalar(85, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				cv::inRange(hsv, cv::Scalar(10, 40, 80), cv::Scalar(20, 255, 200), brownMask);   // Marrone
			}
			else if (leafPath == "/Tomato___Early_blight/Early_blight_")
			{
				cv::inRange(hsv, cv::Scalar(20, 30, 20), cv::Scalar(85, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				cv::inRange(hsv, cv::Scalar(10, 20, 80), cv::Scalar(20, 255, 200), brownMask);   // Marrone
			}
			else if (leafPath == "/Tomato___healthy/healthy_")
			{
				cv::inRange(hsv, cv::Scalar(20, 20, 20), cv::Scalar(85, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				cv::inRange(hsv, cv::Scalar(10, 50, 100), cv::Scalar(20, 255, 200), brownMask);   // Marrone
			}
			else if (leafPath == "/Tomato___Leaf_Mold/Leaf_Mold_")
			{
				//else value
				cv::inRange(hsv, cv::Scalar(30, 40, 20), cv::Scalar(85, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				cv::inRange(hsv, cv::Scalar(10, 50, 100), cv::Scalar(20, 255, 200), brownMask);   // Marrone
			}
			else if (leafPath == "/Tomato___Septoria_leaf_spot/Septoria_leaf_spot_")
			{

				cv::inRange(hsv, cv::Scalar(30, 40, 20), cv::Scalar(85, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				cv::inRange(hsv, cv::Scalar(10, 50, 20), cv::Scalar(20, 255, 200), brownMask);   // Marrone

			}
			else if (leafPath == "/Tomato___Spider_mites Two-spotted_spider_mite/Spider_mites_")
			{

				cv::inRange(hsv, cv::Scalar(20, 20, 20), cv::Scalar(85, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				//no brown mask
				cv::inRange(hsv, cv::Scalar(10, 50, 100), cv::Scalar(10, 50, 100), brownMask);   // Marrone
			}
			else if (leafPath == "/Tomato___Tomato_Yellow_Leaf_Curl_Virus/Yellow_Leaf_Curl_Virus_") {
				cv::inRange(hsv, cv::Scalar(30, 40, 20), cv::Scalar(90, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(27, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				cv::inRange(hsv, cv::Scalar(10, 50, 100), cv::Scalar(20, 255, 200), brownMask);   // Marrone
			}
			else if (leafPath == "/Tomato___Target_Spot/Target_Spot_") {
				cv::inRange(hsv, cv::Scalar(20, 20, 20), cv::Scalar(90, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				cv::inRange(hsv, cv::Scalar(10, 50, 100), cv::Scalar(20, 255, 200), brownMask);   // Marrone
			}
			else
			{
				cv::inRange(hsv, cv::Scalar(30, 40, 20), cv::Scalar(85, 255, 255), greenMask);   // Verde
				cv::inRange(hsv, cv::Scalar(20, 40, 40), cv::Scalar(30, 255, 255), yellowMask);  // Giallo
				cv::inRange(hsv, cv::Scalar(10, 50, 100), cv::Scalar(20, 255, 200), brownMask);   // Marrone
			}



			cv::Mat fullMask = greenMask | yellowMask | brownMask;
			if (leafPath == "/Tomato___Tomato_Yellow_Leaf_Curl_Virus/Yellow_Leaf_Curl_Virus_") {
				fullMask = greenMask | yellowMask;
				//printf("SONO SPECIALE\n");
			}

			if (debug)
				ipa::imshow("Mask", fullMask);

			cv::Mat bgmodel, fgmodel;

			cv::Mat mask(leafImg.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
			mask.setTo(cv::GC_FGD, fullMask);


			cv::morphologyEx(fullMask, fullMask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

			if (debug)
				ipa::imshow("After closing", fullMask);

			cv::Mat maskCopy = fullMask.clone();

			//invert image
			maskCopy = invertImg(maskCopy);

			if (debug)
				ipa::imshow("inverted img", maskCopy);


			std::vector<Contour> objects;

			float areaLim = 500;

			cv::findContours(maskCopy, objects, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

			for (int k = 0; k < objects.size(); k++)
			{
				float area = cv::contourArea(objects[k]);
				if (area < areaLim)
					cv::drawContours(maskCopy, objects, k, cv::Scalar(0), cv::FILLED);
			}

			cv::morphologyEx(maskCopy, maskCopy, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

			if (debug)
				ipa::imshow("Image Analysis results: ", maskCopy);

			std::vector<Contour> leafCounour;

			//invert again
			maskCopy = invertImg(maskCopy);
			cv::findContours(maskCopy, leafCounour, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

			for (int k = 0; k < leafCounour.size(); k++)
				cv::drawContours(leafImg, leafCounour, k, cv::Scalar(0, 0, 255), 1, cv::LINE_8 /*cv::FILLED*/);


			if (see_result)
				ipa::imshow("Image Analysis results: ", leafImg);

			cv::Mat forMask = maskCopy.clone();
			cv::Mat backMask = 255 - forMask;

			if (debug)
			{
				ucas::imshow("Foregorund mask before erode", forMask);
				ucas::imshow("Background mask before erode", backMask);
			}

			//		cv::morphologyEx(forMask, forMask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)));

			cv::morphologyEx(forMask, forMask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
			cv::morphologyEx(backMask, backMask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

			if (debug)
			{
				ucas::imshow("Foregorund mask after erode", forMask);
				ucas::imshow("Background mask after erode", backMask);
			}


			cv::Mat fMask, bMask;

			leafImgCopy.copyTo(fMask, forMask);
			leafImgCopy.copyTo(bMask, backMask);

			cv::Mat grabcutImg = segmentGraphCutFromBGRMasks(leafImgCopy, fMask, bMask);

			if (see_result)
				ipa::imshow("Graph cut result", grabcutImg);

			//final clean

			cv::Mat cleaned = grabcutImg.clone();
			cv::cvtColor(cleaned, cleaned, cv::COLOR_BGR2GRAY);
			std::vector<Contour> debries;

			for (int y = 0; y < cleaned.rows; y++)
			{
				unsigned char* yRow = cleaned.ptr<unsigned char>(y);

				for (int x = 0; x < cleaned.cols; x++)
				{
					if (yRow[x] != 0)
						yRow[x] = 255;
				}
			}
			if (debug)
				ipa::imshow("CLEAN", cleaned);

			cv::findContours(cleaned, debries, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

			for (int k = 0; k < debries.size(); k++)
			{
				float area = cv::contourArea(debries[k]);
				if (area < areaLim)
					cv::drawContours(cleaned, debries, k, cv::Scalar(0), cv::FILLED);
			}

			if (debug)
				ipa::imshow("Graph cut result", cleaned);

			cv::Mat result;
			cv::bitwise_and(grabcutImg, grabcutImg, result, cleaned);

			if (debug)
				ipa::imshow("FINAL", result);


			
			//No shadow 7th and 10th
			if (leafPath == "/Tomato___Spider_mites Two-spotted_spider_mite/Spider_mites_" || leafPath == "/Tomato___Tomato_Yellow_Leaf_Curl_Virus/Yellow_Leaf_Curl_Virus_") {
				cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
				int T_otsu = ucas::getOtsuAutoThreshold(ucas::histogram(result));
				cv::threshold(result, result, T_otsu, 255, cv::THRESH_BINARY);
				//if(leafPath == "/Tomato___Spider_mites Two-spotted_spider_mite/Spider_mites_")
					cv::morphologyEx(result, result, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)));
				//else
					//cv::morphologyEx(result, result, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21)));

				if (debug)
					ipa::imshow("No shadow", result);
				std::vector<Contour> otsuCont; 
				cv::findContours(result, otsuCont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				cv::Mat overlayed = leafImgCopy.clone(); // o leafImg

				for (int k = 0; k < otsuCont.size(); k++)
				{
					float area = cv::contourArea(otsuCont[k]);
					if (area > areaLim)
						cv::drawContours(overlayed, otsuCont, k, cv::Scalar(0,0,255), cv::FILLED);
				}
				if (debug)
					ipa::imshow("No shadow fill", overlayed);

				for (int y = 0; y < overlayed.rows; y++)
				{
					cv::Vec3b* yRow = overlayed.ptr<cv::Vec3b>(y); // every pixel is a BGR triplet

					for (int x = 0; x < overlayed.cols; x++)
					{
						cv::Vec3b& pixel = yRow[x];

						if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 255)
						{
							pixel = cv::Vec3b(255, 255, 255);
						}
						else
							pixel = cv::Vec3b(0, 0, 0);
					}
				}
				if (debug)
					ipa::imshow("Mask for Noshadow", overlayed);
				leafImgCopy.copyTo(result, overlayed);
				if(debug)
					ipa::imshow("Final final result", result);

			}
			
			if (orginalWithNewContour) {
				cv::Mat resultGray;
				cv::cvtColor(result, resultGray, cv::COLOR_BGR2GRAY);

				// Binarizza
				cv::threshold(resultGray, resultGray, 1, 255, cv::THRESH_BINARY);

				// Trova i contorni
				std::vector<std::vector<cv::Point>> resultContours;
				cv::findContours(resultGray, resultContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				// Disegna i contorni sull'immagine originale
				cv::Mat overlayed = leafImgCopy.clone(); // o leafImg
				cv::drawContours(overlayed, resultContours, -1, cv::Scalar(0, 0, 255), 1);
				//cv::imwrite("C:/Users/marco/Desktop/Result_mask2/7a/" + std::to_string(i) + ".jpg", overlayed);
				cv::imwrite("C:/Users/marco/Desktop/Result_mask2/"+nclass+"/"+std::to_string(i) + ".jpg", overlayed);

			}
				//cv::imwrite("C:/Users/giaco/OneDrive/Desktop/University 2.0/Image_analysis/leaf_segmentation/7-spider_mites/" + std::to_string(i) + ".jpg", result);
			
			else		
				cv::imwrite("C:/Users/marco/Desktop/Result_mask2/" + nclass + "/" + std::to_string(i) + ".jpg", result);
			

		}

		return 1;

	}
	catch (ipa::error& ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}

cv::Mat invertImg(cv::Mat img)
{
	for (int y = 0; y < img.rows; y++)
	{
		unsigned char* yRow = img.ptr<unsigned char>(y);

		for (int x = 0; x < img.cols; x++)
		{
			if (yRow[x] == 0)
				yRow[x] = 255;
			else
				yRow[x] = 0;
		}
	}

	return img;
}

cv::Mat gradientImageGray(const cv::Mat& frameGray)
{
	CV_Assert(frameGray.channels() == 1); // Ensure input is grayscale

	// Sobel filters
	cv::Mat sobelX = (cv::Mat_<float>(3, 3) <<
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	cv::Mat sobelY = (cv::Mat_<float>(3, 3) <<
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1);

	cv::Mat dx, dy;
	// Convolution along x-axis and y-axis
	cv::filter2D(frameGray, dx, CV_32F, sobelX);
	cv::filter2D(frameGray, dy, CV_32F, sobelY);

	cv::Mat gradMag;
	cv::magnitude(dx, dy, gradMag);

	// Normalize to [0, 255]
	cv::normalize(gradMag, gradMag, 0, 255, cv::NORM_MINMAX);

	// Convert to 8-bit image for visualization
	gradMag.convertTo(gradMag, CV_8U);

	return 3 * gradMag;
}

// Estrae i punti non-neri da una maschera BGR
std::vector<cv::Point> extractNonBlackPoints(const cv::Mat& maskBGR)
{
	CV_Assert(maskBGR.channels() == 3);

	std::vector<cv::Point> points;
	for (int y = 0; y < maskBGR.rows; ++y) {
		for (int x = 0; x < maskBGR.cols; ++x) {
			const cv::Vec3b& pixel = maskBGR.at<cv::Vec3b>(y, x);
			if (pixel != cv::Vec3b(0, 0, 0)) {
				points.emplace_back(x, y);
			}
		}
	}
	return points;
}

// Segmentazione con grabCut usando maschere BGR
cv::Mat segmentGraphCutFromBGRMasks(const cv::Mat& image,
	const cv::Mat& fgMaskBGR,
	const cv::Mat& bgMaskBGR,
	int iterCount)
{
	CV_Assert(!image.empty() && image.type() == CV_8UC3);
	CV_Assert(fgMaskBGR.size() == image.size() && bgMaskBGR.size() == image.size());

	// Estrai punti foreground e background dalle maschere
	std::vector<cv::Point> foregroundPixels = extractNonBlackPoints(fgMaskBGR);
	std::vector<cv::Point> backgroundPixels = extractNonBlackPoints(bgMaskBGR);

	// Inizializza maschera GrabCut
	cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(cv::GC_PR_BGD)); // default: probabile background

	for (const auto& pt : foregroundPixels)
		mask.at<uchar>(pt) = cv::GC_FGD;

	for (const auto& pt : backgroundPixels)
		mask.at<uchar>(pt) = cv::GC_BGD;

	// Modelli richiesti da grabCut
	cv::Mat bgModel, fgModel;
	cv::grabCut(image, mask, cv::Rect(), bgModel, fgModel, iterCount, cv::GC_INIT_WITH_MASK);

	// Estrai punti del risultato come foreground (certo o probabile)
	/*std::vector<cv::Point> resultForeground;
	for (int y = 0; y < mask.rows; ++y) {
		for (int x = 0; x < mask.cols; ++x) {
			uchar val = mask.at<uchar>(y, x);
			if (val == cv::GC_FGD || val == cv::GC_PR_FGD)
				resultForeground.emplace_back(x, y);
		}
	}*/

	cv::Mat binMask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
	cv::Mat result;
	image.copyTo(result, binMask);

	//ipa::imshow("Graph cut result", result);

	//ucas::imshow("Graph cut", mask);
	return result;
}


void NlMeansParameters(int sigma, float& h, int& N, int& S)
{
	h = 3.0f;		// OpenCV default value
	N = 7;			// OpenCV default value
	S = 21;			// OpenCV default value
	if (sigma > 0 && sigma <= 15)
	{
		h = 0.40f * sigma;
		N = 3;
		S = 21;
	}
	if (sigma > 15 && sigma <= 30)
	{
		h = 0.40f * sigma;
		N = 5;
		S = 21;
	}
	if (sigma > 30 && sigma <= 45)
	{
		h = 0.35f * sigma;
		N = 7;
		S = 35;
	}
	if (sigma > 45 && sigma <= 75)
	{
		h = 0.35f * sigma;
		N = 9;
		S = 35;
	}
	if (sigma > 75 && sigma <= 100)
	{
		h = 0.30f * sigma;
		N = 11;
		S = 35;
	}
}