#include <iostream>
#include <opencv2/opencv.hpp>
#include "proj.h"
using namespace std;
using namespace cv;


void drawRectangle(Mat& image, const MyRect& rect, const Scalar& color, int thickness) {
    line(image, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y), color, thickness);
    line(image, Point(rect.x, rect.y + rect.height), Point(rect.x + rect.width, rect.y + rect.height), color, thickness);
    line(image, Point(rect.x, rect.y), Point(rect.x, rect.y + rect.height), color, thickness);
    line(image, Point(rect.x + rect.width, rect.y), Point(rect.x + rect.width, rect.y + rect.height), color, thickness);
}

Mat extractROI(const Mat& image, const MyRect& rect) {
    Mat roi(rect.height, rect.width, image.type());
    
    for (int y = 0; y < rect.height; y++) {
        for (int x = 0; x < rect.width; x++) {
            int srcY = rect.y + y;
            int srcX = rect.x + x;

            if (srcY >= 0 && srcY < image.rows && srcX >= 0 && srcX < image.cols) {
                if (image.channels() == 1) {
                    roi.at<uchar>(y, x) = image.at<uchar>(srcY, srcX);
                } else if (image.channels() == 3) {
                    roi.at<Vec3b>(y, x) = image.at<Vec3b>(srcY, srcX);
                }
            }
        }
    }
    
    return roi;
}

int main() {

    Mat source = imread("D:\\ANUL3 _SEM2\\procesare_imagini\\projectRaul\\Project\\car9.jpg", IMREAD_COLOR);
    
    if (source.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }
    //car8 & car6

    imshow("Original Image", source);

    LicensePlateDetector detector;
    MyRect licensePlateRect = detector.detectLicensePlate(source);

    if (licensePlateRect.width > 0 && licensePlateRect.height > 0) {
        Mat result = source.clone();
        drawRectangle(result, licensePlateRect, Scalar(0, 255, 0), 2);
        imshow("License Plate Detection", result);

        Mat licensePlate = extractROI(source, licensePlateRect);
        imshow("License Plate", licensePlate);

        Mat preprocessed = detector.preprocessPlate(licensePlate);
        imshow("Preprocessed License Plate", preprocessed);
    } else {
        cout << "No license plate detected!" << endl;
    }

    waitKey(0);
    return 0;
}