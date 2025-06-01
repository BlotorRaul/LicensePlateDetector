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

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    Mat source = imread("D:\\ANUL3 _SEM2\\procesare_imagini\\projectRaul\\Project\\Images\\car8.jpg", IMREAD_COLOR);
    
    if (source.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }
    //car6

    imshow("Original Image", source);

    LicensePlateDetector detector;
    Mat gray = detector.manualGrayscaleConversion(source);
    Mat blurred = detector.manualGaussianBlur(gray, 5);
    Mat edges = detector.manualSobelOperator(blurred);
    Mat binary = detector.manualThreshold(edges, 0); // Otsu method
    Mat morphed = detector.manualMorphologicalOperation(binary);
    imshow("Gray", gray);
    imshow("Blurred", blurred);
    imshow("Edges", edges);
    imshow("Binary", binary);
    imshow("Morphed", morphed);


    Mat morphedCopy = morphed.clone();
    vector<vector<Point>> morphedContours;
    findContours(morphedCopy, morphedContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int maxArea = 0;
    Rect plateRect;
    for (const auto& c : morphedContours) {
        Rect r = boundingRect(c);
        // Placuța e lata si nu foarte inalta
        if (r.area() > maxArea && r.width > r.height * 2.5 && r.y > morphed.rows * 0.4) {
            maxArea = r.area();
            plateRect = r;
        }
    }
    if (maxArea > 0) {

        Mat binaryROI = binary(plateRect);
        vector<vector<Point>> contours = detector.manualFindContours(binaryROI);

        Mat result = source.clone();
        rectangle(result, plateRect, Scalar(0,255,0), 2);
        imshow("Detected Plate (contour)", result);
        imshow("Plate ROI (contour)", source(plateRect));
        imshow("Binary Plate ROI (contour)", binaryROI);
        imshow("Morphed Plate ROI (contour)", morphed(plateRect));
        imwrite("detected_plate_contour.jpg", source(plateRect));
        cout << "Contours in binary ROI: " << contours.size() << endl;
        if (contours.size() >= 3) {
            cout << "Zona crop-uita are cel putin 3 caractere, este placuta!" << endl;
        } else {
            cout << "Zona crop-uita NU are suficiente caractere!" << endl;
        }
    }

    waitKey(0);
    return 0;
}