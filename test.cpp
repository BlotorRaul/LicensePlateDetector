#include <iostream>
#include <opencv2/opencv.hpp>
#include "proj.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;
//[x1, y1, x2, y2]
float calculateIoU(const std::vector<int>& boxA, const std::vector<int>& boxB) {
    int xA = std::max(boxA[0], boxB[0]);
    int yA = std::max(boxA[1], boxB[1]);
    int xB = std::min(boxA[2], boxB[2]);
    int yB = std::min(boxA[3], boxB[3]);

    int interWidth = std::max(0, xB - xA);
    int interHeight = std::max(0, yB - yA);
    int interArea = interWidth * interHeight;

    int areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    int areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    float iou = float(interArea) / float(areaA + areaB - interArea);
    return iou;
}

json load_annotations(const std::string& filename) {
    fs::path exePath = fs::current_path();
    fs::path jsonPath = exePath.parent_path() / filename;
    std::ifstream inFile(jsonPath);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open annotations file" << std::endl;
        return json();
    }
    json annotations;
    inFile >> annotations;
    return annotations;
}

int main() {
    // Disable OpenCV debug messages
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    json annotations = load_annotations("annotations.json");
    if (annotations.empty()) {
        return -1;
    }

    LicensePlateDetector detector;
    float totalIoU = 0.0f;
    int validCount = 0;

    for (int i = 1; i <= 11; ++i) {
        std::string imageName = "tester" + std::to_string(i) + ".png";
        if (!annotations.contains(imageName)) {
            std::cerr << "Skipping " << imageName << ": no ground truth found.\n";
            continue;
        }

        fs::path imagePath = fs::current_path().parent_path() / "Tests" / imageName;
        cv::Mat image = cv::imread(imagePath.string());
        if (image.empty()) {
            std::cerr << "Skipping " << imageName << ": image not found or unreadable.\n";
            continue;
        }

        cv::Mat gray = detector.manualGrayscaleConversion(image);
        cv::Mat blurred = detector.manualGaussianBlur(gray, 5);
        cv::Mat edges = detector.manualSobelOperator(blurred);
        cv::Mat binary = detector.manualThreshold(edges, 0);
        cv::Mat morphed = detector.manualMorphologicalOperation(binary);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(morphed.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int maxArea = 0;
        cv::Rect plateRect;
        for (const auto& c : contours) {
            cv::Rect r = cv::boundingRect(c);
            //forma alungita + pozitionat mai jos in imagine y> 40%
            if (r.area() > maxArea && r.width > r.height * 2.5 && r.y > morphed.rows * 0.4) {
                maxArea = r.area();
                plateRect = r;
            }
        }

        if (maxArea == 0) {
            std::cout << imageName << " – No plate detected.\n";
            continue;
        }

        std::vector<int> predictedBox = {
            plateRect.x,
            plateRect.y,
            plateRect.x + plateRect.width,
            plateRect.y + plateRect.height
        };

        std::vector<int> groundTruthBox = annotations[imageName];
        float iou = calculateIoU(predictedBox, groundTruthBox);

        // Display detailed results for each image
        std::cout << "\n=== Test Results for " << imageName << " ===" << std::endl;
        std::cout << "Ground Truth: [" << groundTruthBox[0] << ", " << groundTruthBox[1]
                  << ", " << groundTruthBox[2] << ", " << groundTruthBox[3] << "]" << std::endl;
        std::cout << "Predicted:    [" << predictedBox[0] << ", " << predictedBox[1]
                  << ", " << predictedBox[2] << ", " << predictedBox[3] << "]" << std::endl;
        std::cout << "IoU Score: " << iou << std::endl;
        std::cout << "==================\n" << std::endl;

        // Visual display
        cv::Mat resultImage = image.clone();
        cv::rectangle(resultImage,
            cv::Point(groundTruthBox[0], groundTruthBox[1]),
            cv::Point(groundTruthBox[2], groundTruthBox[3]),
            cv::Scalar(0, 0, 255), 2);  // Red rectangle for ground truth

        cv::rectangle(resultImage,
            cv::Point(predictedBox[0], predictedBox[1]),
            cv::Point(predictedBox[2], predictedBox[3]),
            cv::Scalar(0, 255, 0), 2);  // Green rectangle for prediction

        cv::imshow("Detection Results - " + imageName, resultImage);
        
        totalIoU += iou;
        validCount++;

        // Check for ESC key (27 is the ASCII code for ESC)
        int key = cv::waitKey(0);
        cv::destroyWindow("Detection Results - " + imageName);
        
        if (key == 27) {  // ESC key pressed
            if (validCount > 0) {
                float averageIoU = totalIoU / validCount;
                std::cout << "\n==============================" << std::endl;
                std::cout << "Process stopped at image " << imageName << std::endl;
                std::cout << "Total images processed: " << validCount << std::endl;
                std::cout << "Average IoU: " << averageIoU << std::endl;
                std::cout << "==============================\n" << std::endl;
            }
            return 0;
        }
    }

    // Display final average IoU if we processed all images
    if (validCount > 0) {
        float averageIoU = totalIoU / validCount;
        std::cout << "\n==============================" << std::endl;
        std::cout << "Total images processed: " << validCount << std::endl;
        std::cout << "Average IoU: " << averageIoU << std::endl;
        std::cout << "==============================\n" << std::endl;
    } else {
        std::cout << "No valid detections to calculate average IoU.\n";
    }

    return 0;
}