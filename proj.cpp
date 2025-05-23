#include <iostream>
#include <opencv2/opencv.hpp>
#include "proj.h"
#include <cmath> 
#include <queue>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

LicensePlateDetector::LicensePlateDetector() {
    aspectRatioMin = 2.0;
    aspectRatioMax = 6.0;
    minPlateArea = 1000;
    maxPlateArea = 30000;
}

MyRect LicensePlateDetector::detectLicensePlate(const Mat& image) {
    Mat preprocessed = preprocessImage(image);

    vector<MyRect> candidates = findPossiblePlateRegions(preprocessed);

    return selectBestPlate(candidates, image);
}

Mat LicensePlateDetector::manualGrayscaleConversion(const Mat& image) {
    Mat gray(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            //pixel[0] = blue ->bgr
            uchar grayValue = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];//asa vede ochiul uman
            gray.at<uchar>(i, j) = grayValue;
        }
    }
    return gray;
}
//reduce zgomot si detalii minore -> blur pe baza functiei gaussiene
Mat LicensePlateDetector::manualGaussianBlur(const Mat& image, int kernelSize) { //kernel = 7-> -3 -2 -1...3
    Mat blurred = image.clone();
    int halfKernel = kernelSize / 2;
    //stocam ponderile matricilor in kernel
    vector<vector<double>> kernel(kernelSize, vector<double>(kernelSize));
    double sigma = 1.0;//extinderea blururlui -> sigma mare blur mare
    double sum = 0.0;

    for (int i = -halfKernel; i <= halfKernel; i++) {
        for (int j = -halfKernel; j <= halfKernel; j++) {
            double exponent = -(i*i + j*j) / (2 * sigma * sigma);
            kernel[i + halfKernel][j + halfKernel] = exp(exponent) / (2 * M_PI * sigma * sigma);
            sum += kernel[i + halfKernel][j + halfKernel];
        }
    }
    //normalizare
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    Mat result = Mat::zeros(image.size(), image.type());
    
    for (int i = halfKernel; i < image.rows - halfKernel; i++) {
        for (int j = halfKernel; j < image.cols - halfKernel; j++) {
            double sum = 0.0;
            
            for (int ki = -halfKernel; ki <= halfKernel; ki++) {
                for (int kj = -halfKernel; kj <= halfKernel; kj++) {
                    sum += image.at<uchar>(i + ki, j + kj) * kernel[ki + halfKernel][kj + halfKernel];
                }
            }
            
            result.at<uchar>(i, j) = saturate_cast<uchar>(sum); //asigura converitrea intr-un uchar
        }
    }
    
    return result;
}

//detecteaza zonele de tranzitie brusca a intensitatii pixelilor
Mat LicensePlateDetector::manualSobelOperator(const Mat& image) {
    Mat result = Mat::zeros(image.size(), image.type());

    int sobelX[3][3] = { //detecteaza margini verticale prin diferentele de intensitate din stanga si dreapta a pixelului
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.cols - 1; j++) {
            int gx = 0;

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    //daca stanga mai intunecata si dreapta mai luminoasa -> gx valoarea pozitiva mare
                    gx += image.at<uchar>(i + ki, j + kj) * sobelX[ki + 1][kj + 1];
                }
            }
            result.at<uchar>(i, j) = saturate_cast<uchar>(abs(gx));
        }
    }
    
    return result;
}
//binarizare
Mat LicensePlateDetector::manualThreshold(const Mat& image, int threshold) {
    Mat result = Mat::zeros(image.size(), image.type());

    if (threshold == 0) {
        //calc histograma
        int histogram[256] = {0};
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                histogram[image.at<uchar>(i, j)]++;
            }
        }

        int total = image.rows * image.cols;

        //suma intensitatilor
        float sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += i * histogram[i];
        }

        //metoda otsu
        float sumB = 0; //sum intensitati fundal
        int wB = 0;//nr pixeli fundal
        int wF = 0; //nr pixeli obiect
        float maxVariance = 0;
        threshold = 0;
        //cauta punctul in care varianta intre background si foreground este maxima
        //->separa pixelii in doua grupuri distincte
        //varianta mare-> background si foreground sunt foarte distincte

        for (int i = 0; i < 256; i++) {
            wB += histogram[i];
            if (wB == 0) continue;
            
            wF = total - wB;
            if (wF == 0) break;
            
            sumB += i * histogram[i];
            float mB = sumB / wB;
            float mF = (sum - sumB) / wF;
            
            float variance = wB * wF * (mB - mF) * (mB - mF);
            
            if (variance > maxVariance) {
                maxVariance = variance;
                threshold = i; //am gasit pragul bun
            }
        }
    }

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            result.at<uchar>(i, j) = (image.at<uchar>(i, j) > threshold) ? 255 : 0;
        }
    }
    
    return result;
}
//uneste componentele conectate si reduce zgomotul
Mat LicensePlateDetector::manualMorphologicalOperation(const Mat& image) {

    int width = 17;
    int height = 3;
    Mat element = Mat::ones(height, width, CV_8UC1);//conecteaza componentele orizontale
    
    // Dilate
    Mat dilated = Mat::zeros(image.size(), image.type());
    int halfWidth = width / 2;
    int halfHeight = height / 2;

    //kernelul(element) este plasat peste pixel(i,j)
    //daca kernelul intalneste cel putin un pixel alb atunci (i,j) devine alb
    //daca nu, devine negru
    for (int i = halfHeight; i < image.rows - halfHeight; i++) {
        for (int j = halfWidth; j < image.cols - halfWidth; j++) {
            bool hit = false;
            for (int ki = -halfHeight; ki <= halfHeight && !hit; ki++) {
                for (int kj = -halfWidth; kj <= halfWidth && !hit; kj++) {
                    if (element.at<uchar>(ki + halfHeight, kj + halfWidth) > 0 &&
                        image.at<uchar>(i + ki, j + kj) > 0) {
                        hit = true;
                    }
                }
            }
            dilated.at<uchar>(i, j) = hit ? 255 : 0;
        }
    }
    
    // Erode-pentru rafinare componentelor albe
    //daca kernel se potriveste perfect( toti pixelii albi din kernel corespund pixelilor albi din imagine)
    //atunci pixelul (i,j) ramane alb(255)
    //daca orice pixel din kernel nu corespunde, pixelul (i,j) devine negru(0)
    Mat eroded = Mat::zeros(dilated.size(), dilated.type());
    
    for (int i = halfHeight; i < dilated.rows - halfHeight; i++) {
        for (int j = halfWidth; j < dilated.cols - halfWidth; j++) {
            bool fit = true;

            for (int ki = -halfHeight; ki <= halfHeight && fit; ki++) {
                for (int kj = -halfWidth; kj <= halfWidth && fit; kj++) {
                    if (element.at<uchar>(ki + halfHeight, kj + halfWidth) > 0 &&
                        dilated.at<uchar>(i + ki, j + kj) == 0) {
                        fit = false;
                    }
                }
            }
            
            eroded.at<uchar>(i, j) = fit ? 255 : 0;
        }
    }
    
    return eroded;
}
//cautam pixeli albi si ii exploram in BFS
//de ce alb? pentru ca intr-o imagine binara un contur este alb iar restul e negru
vector<vector<Point>> LicensePlateDetector::manualFindContours(const Mat& image) {
    Mat visited = Mat::zeros(image.size(), CV_8UC1);
    vector<vector<Point>> contours;

    int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            //pixel alb si nevizitat
            if (image.at<uchar>(i, j) == 255 && visited.at<uchar>(i, j) == 0) {
                vector<Point> contour;
                queue<Point> q; //coada pentru bfs
                q.push(Point(j, i));
                visited.at<uchar>(i, j) = 255;

                while (!q.empty()) {
                    Point p = q.front();
                    q.pop();
                    contour.push_back(p); //fiecare pixel alb este adaugat intr-un contur

                    for (int k = 0; k < 8; k++) {
                        int nx = p.x + dx[k];
                        int ny = p.y + dy[k];
                        
                        if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows &&
                            image.at<uchar>(ny, nx) == 255 && visited.at<uchar>(ny, nx) == 0) {
                            q.push(Point(nx, ny));
                            visited.at<uchar>(ny, nx) = 255;
                        }
                    }
                }

                if (contour.size() > 50) { //sunt considerate zgomot
                    contours.push_back(contour);
                }
            }
        }
    }
    
    return contours;
}

Mat LicensePlateDetector::preprocessImage(const Mat& image) {

    Mat gray = manualGrayscaleConversion(image);
    Mat blurred = manualGaussianBlur(gray, 5);
    Mat edges = manualSobelOperator(blurred);
    Mat binary = manualThreshold(edges, 0); // Otsu method
    Mat morphed = manualMorphologicalOperation(binary);

    imshow("Gray", gray);
    imshow("Blurred", blurred);
    imshow("Edges", edges);
    imshow("Binary", binary);
    imshow("Morphed", morphed);
    
    return morphed;
}

vector<MyRect> LicensePlateDetector::findPossiblePlateRegions(const Mat& image) {
    vector<vector<Point>> contours = manualFindContours(image);
    vector<MyRect> candidates;

    for (const auto& contour : contours) {
        int minX = image.cols, minY = image.rows, maxX = 0, maxY = 0;
        
        for (const auto& point : contour) {
            minX = min(minX, point.x);
            minY = min(minY, point.y);
            maxX = max(maxX, point.x);
            maxY = max(maxY, point.y);
        }

        MyRect rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
        double area = rect.width * rect.height;
        double aspectRatio = (double)rect.width / rect.height;
        
        if (area >= minPlateArea && area <= maxPlateArea && 
            aspectRatio >= aspectRatioMin && aspectRatio <= aspectRatioMax) {
            candidates.push_back(rect);
        }
    }
    
    return candidates;
}

MyRect LicensePlateDetector::selectBestPlate(const vector<MyRect>& candidates, const Mat& image) {
    if (candidates.empty()) {
        return MyRect(0, 0, 0, 0);
    }

    MyRect bestPlate = candidates[0];
    double maxArea = bestPlate.width * bestPlate.height;
    
    for (size_t i = 1; i < candidates.size(); i++) {
        double area = candidates[i].width * candidates[i].height;
        if (area > maxArea) {
            maxArea = area;
            bestPlate = candidates[i];
        }
    }
    
    return bestPlate;
}

Mat LicensePlateDetector::preprocessPlate(const Mat& plate) {//binarizare adaptiva
    Mat gray = manualGrayscaleConversion(plate);
    Mat blurred = manualGaussianBlur(gray, 5);

    Mat threshold_img = Mat::zeros(blurred.size(), blurred.type());
    int blockSize = 11; //cati vecini vreau sa iau pentru calc mediei
    int halfBlockSize = blockSize / 2;
    int C = 2; //pentru ajustarea sensibilitatea pragului
    
    for (int i = halfBlockSize; i < blurred.rows - halfBlockSize; i++) {
        for (int j = halfBlockSize; j < blurred.cols - halfBlockSize; j++) {
            int sum = 0;
            int count = 0;
            
            for (int ki = -halfBlockSize; ki <= halfBlockSize; ki++) {
                for (int kj = -halfBlockSize; kj <= halfBlockSize; kj++) {
                    sum += blurred.at<uchar>(i + ki, j + kj);
                    count++;
                }
            }
            
            int mean = sum / count;

            //pixel mai mic -> e negru
            threshold_img.at<uchar>(i, j) = (blurred.at<uchar>(i, j) < mean - C) ? 255 : 0;
        }
    }
    
    return threshold_img;
}

