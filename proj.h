#ifndef PROJ_H
#define PROJ_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


class MyRect {
public:
    int x, y, width, height;

    MyRect() : x(0), y(0), width(0), height(0) {}
    MyRect(int _x, int _y, int _width, int _height) : x(_x), y(_y), width(_width), height(_height) {}

    bool isEmpty() const {
        return width <= 0 || height <= 0;
    }
};

class LicensePlateDetector {
public:
    LicensePlateDetector();

    MyRect detectLicensePlate(const Mat& image);

    Mat preprocessPlate(const Mat& plate);

private:
    Mat manualGrayscaleConversion(const Mat& image);
    Mat manualGaussianBlur(const Mat& image, int kernelSize);
    Mat manualSobelOperator(const Mat& image);
    Mat manualThreshold(const Mat& image, int threshold);
    Mat manualMorphologicalOperation(const Mat& image);
    vector<vector<Point>> manualFindContours(const Mat& image);

    Mat preprocessImage(const Mat& image);
    vector<MyRect> findPossiblePlateRegions(const Mat& image);
    MyRect selectBestPlate(const vector<MyRect>& candidates, const Mat& image);

    double aspectRatioMin;
    double aspectRatioMax;
    double minPlateArea;
    double maxPlateArea;
};

#endif
