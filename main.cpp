// include aia and ucas utility functions
#include "ipaConfig.h"
#include "ucasConfig.h"
#include "functions.h"
#include <opencv2/photo.hpp>
typedef std::vector <cv::Point> Contour;
std::string root = "C:/Users/marco/Desktop/University/MAGISTRALE/1oANNO/2oSemestre/Image_Processing_Analysis/Codes/plants/dataset_less";

namespace ipa {
	cv::Mat img;

	// bilateral filter parameters
	int filter_size = 15;
	int sigma_color = 150;
	int sigma_space = 75;

}
cv::Mat createMarkersFromPixels(cv::Size size,
	const std::vector<cv::Point>& foregroundPixels,
	const std::vector<cv::Point>& backgroundPixels)
{
	cv::Mat markers = cv::Mat::zeros(size, CV_32S); // inizializza tutto a 0

	// Imposta pixel foreground (es. foglia) con marker ID 1
	for (const auto& pt : foregroundPixels)
		markers.at<int>(pt) = 255;

	// Imposta pixel background (es. sfondo) con marker ID 2
	for (const auto& pt : backgroundPixels)
		markers.at<int>(pt) = 2;

	return markers;
}

int main()
{
	srand(TIME(0));

	try
	{
		cv::Mat img = cv::imread(root + "/Tomato___Bacterial_spot/Bacterial_spot_46.JPG");
		if (!img.data)
			throw ucas::Error("Cannot open image");

		//cv::resize(img, img, cv::Size(0, 0), 0.3f, 0.3f);

		float magFactor = 1.0;

		ucas::imshow("Original image", img, true, magFactor);


		//denoise
		cv::Mat img_denoised = img.clone();
		cv::bilateralFilter(img, img_denoised, ipa::filter_size, ipa::sigma_color, ipa::sigma_space);
		cv::imwrite(root + "/Tomato___Bacterial_spot/Bacterial_spot_0d.JPG", img_denoised);
		ipa::imshow("denoised", img_denoised);


		
		std::vector<cv::Point> foregroundPixels = { {103, 126}}; // foglia
		std::vector<cv::Point> backgroundPixels = {};  // sfondo

		cv::Size imgSize = cv::Size(256, 256); // dimensione immagine
		cv::Mat seedsImg = createMarkersFromPixels(imgSize, foregroundPixels, backgroundPixels);


	

		// 1. Converti immagine in Lab
		cv::Mat imgLab;
		cv::cvtColor(img_denoised, imgLab, cv::COLOR_BGR2Lab);

		// 2. Estrai il canale L
		std::vector<cv::Mat> labChannels;
		cv::split(imgLab, labChannels);
		cv::Mat a = labChannels[1]; // a

		cv::Mat hist;
		int histSize = 256; // numero di bin
		float range[] = { -128, 128 }; // intervallo di valori per il canale a
		const float* histRange = { range };
		cv::calcHist(&labChannels[1], 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

		// Mostra l'istogramma
		int hist_w = 512, hist_h = 400;
		int bin_w = cvRound((double)hist_w / histSize);
		cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

		normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
		for (int i = 1; i < histSize; i++) {
			line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
				cv::Scalar(255, 0, 0), 2, 8, 0);
		}
		cv::imshow("Histogram of a channel", histImage);
		cv::waitKey(0);


		// 3. Crea un criterio di segmentazione basato su a
		int lowerBound = 50; // valore inferiore per il verde
		int upperBound = 128;  // valore superiore per il verde

		cv::Mat aCriterion;
		cv::inRange(a, lowerBound, upperBound, aCriterion);


		cv::Mat predicate = aCriterion;
		ucas::imshow("aCriterion", aCriterion);

		//region growing

		cv::Mat seeds_curr;
		seedsImg.convertTo(seeds_curr, CV_8U);
		cv::Mat seeds_prev;
		do
		{
			seeds_prev = seeds_curr.clone();

			cv::Mat candidatePixels;
			cv::dilate(seeds_curr, candidatePixels, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
			candidatePixels = candidatePixels - seeds_curr;

			seeds_curr += candidatePixels & predicate;

			cv::imshow("Growing in progress", seeds_curr);
			//cv::waitKey(50);

		} while (cv::countNonZero(seeds_curr - seeds_prev));

		ipa::imshow("result", seeds_curr);
		
		
		return EXIT_SUCCESS;
	}
	catch (ucas::Error& ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}