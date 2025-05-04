#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace cv;

struct AxisEquation {
    double rho;
    double theta;
};

struct GeometricProps {
    double area;
    cv::Point2d centroid;
    AxisEquation least_second_moment;
    AxisEquation max_second_moment;
};

int CalcThreshold(const cv::Mat& image);
cv::Mat CreateIdealBinary(const cv::Mat& image, const int& binary_threshold);
cv::Mat ComputerGeometricPropsAndMarkUpImage(const cv::Mat& binary_image);

int main() {
    cv::Mat image, display_image;
    image = cv::imread("../cado_and_stapler_2.png", cv::IMREAD_GRAYSCALE);

    int binary_threshold = CalcThreshold(image);

    cv::Mat binary_image = CreateIdealBinary(image, binary_threshold);

    cv::imshow("Binary", binary_image);
    cv::waitKey();

    cv::Mat labels, stats, centroids;
    int num_objects = cv::connectedComponentsWithStats(binary_image, labels, stats, centroids);

    std::map<int, cv::Rect> component_rois;

    // Skip label 0 (background)
    for (int i = 1; i < num_objects; i++) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        component_rois[i] = cv::Rect(x, y, width, height);
    }

    for (std::pair<int, cv::Rect> item : component_rois) {
        // Turn off any pixels in cropped image that aren't from the correct label
        // This can affect the calculations as it will pull the measurments towards the separate object
        cv::Mat new_image = binary_image.clone();
        cv::MatIterator_<uchar> image_itr = new_image.begin<uchar>();
        cv::MatIterator_<uchar> end = new_image.end<uchar>();
        cv::MatConstIterator_<int> label_itr = labels.begin<int>();
        for (; image_itr < end; ++image_itr, ++label_itr) {
            if (*label_itr != item.first) {
                *image_itr = 0;
            }
        }
        cv::Mat component_image = new_image(item.second);
        display_image = ComputerGeometricPropsAndMarkUpImage(component_image);

        cv::imshow("Final", display_image);
        cv::waitKey();
    }

    return 0;
}

/*
* Calculate a threshold value for the image
* Generate a histogram of the image
* Using first derivative and then second derivate test, find the local mins and maxs in that histogram
* There should be two strong maxes and a min in between the two, the horizontal value of this min is
* our threshold value
*/
int CalcThreshold(const cv::Mat& image) {

    // Generate Histogram
    const int hist_size = 32;
    float range[] = { 0, 256 };
    const float* hist_range[] = { range };
    bool uniform = true, accumulate = false;

    cv::Mat histogram;
    cv::calcHist(&image, 1, 0, cv::Mat(), histogram, 1, &hist_size, hist_range, uniform, accumulate);

    // Calculate the first and second derivative of the histogram using convolution and custom kernel
    // which is essentially just the sobel operator

    int max_index = -1;
    int max_val = -1;
    std::vector<int> hist_vec(hist_size);
    for (int i = 0; i < histogram.rows; i++) {
        hist_vec[i] = static_cast<int>(histogram.at<uchar>(i, 1));
        if (hist_vec[i] > max_val) {
            max_val = hist_vec[i];
            max_index = i;
        }
    }

    // Enable below to display the histogram
    //
    // int hist_w = 512, hist_h = 400;
    // int bin_w = cvRound( (double) hist_w/hist_size );
    // Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    // normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    //
    // for( int i = 1; i < hist_size; i++ )
    // {
    //     line( histImage, Point( bin_w*(i-1), hist_h - cvRound(histogram.at<float>(i-1)) ),
    //           Point( bin_w*(i), hist_h - cvRound(histogram.at<float>(i)) ),
    //           Scalar( 255, 0, 0), 2, 8, 0  );
    // }
    // imshow("calcHist Demo", histImage );
    // cv::waitKey();

    constexpr int sobel_size = 3;
    int sobel_kernel[sobel_size] = { -1, 0, 1 };


    std::vector<int> first_derivative(hist_size);
    for (int i = 1; i < hist_vec.size() - 1; i++) {
        first_derivative[i] = sobel_kernel[0] * hist_vec[i - 1] + sobel_kernel[1] * hist_vec[i] + sobel_kernel[2] * hist_vec[i + 1];
    }

    std::vector<int> second_derivative(hist_size);
    for (int i = 1; i < hist_vec.size() - 1; i++) {
        second_derivative[i] = sobel_kernel[0] * first_derivative[i - 1] + first_derivative[1] * hist_vec[i] + first_derivative[2] * hist_vec[i + 1];
    }

    // Find local minima and maxima by finding areas where the derivative f'(x) is close to zero
    // Use second derivative test to determine if it is a local min or max
    // Get the associated values from f(x) as well
    bool first_non_zero = false;
    std::vector<std::pair<int, int>> local_maxima;
    std::vector<std::pair<int, int>> local_minima;
    constexpr int threshold = 20;
    for (int i = 1; i < first_derivative.size() - 1; i++) {
        if (first_derivative[i] != 0) {
            first_non_zero = true;
        }

        if (abs(first_derivative[i]) < threshold && first_non_zero) {
            if (second_derivative[i] < 0) {
                local_maxima.emplace_back(i, hist_vec[i]);
            } else if (second_derivative[i] > 0) {
                local_minima.emplace_back(i, hist_vec[i]);
            }
        }
    }

    // Get two largest peaks in histogram
    int largest_index = -1;
    int largest_val = -1;
    int second_largest_index = -1;
    int second_largest_val = -1;
    for (int i = 0; i < local_maxima.size(); i++) {
        int current_hist_index = local_maxima[i].first;
        int current_hist_val = local_maxima[i].second;
        if (largest_val < current_hist_val) {
            largest_index = current_hist_index;
            largest_val = current_hist_val;
        } else if (second_largest_val < current_hist_val) {
            second_largest_index = current_hist_index;
            second_largest_val = current_hist_val;
        }
    }

    // Find a local minima that is between our two maxes, this is our valley of interest
    int valley_bin = -1;
    int end_index = largest_index > second_largest_index ? largest_index : second_largest_index;
    int start_index = end_index == largest_index ? second_largest_index : largest_index;
    for (int i = 0; i < local_minima.size(); i++) {
        int min_bin = local_minima[i].first;
        if (min_bin > start_index && min_bin < end_index) {
            valley_bin = local_minima[i].first;
        }
    }

    int factor = 256 / hist_size;
    return (valley_bin + 1) * factor;
}

/*
 * Turn into binary image using our calculated threshold from the images histogram
 */
cv::Mat CreateIdealBinary(const cv::Mat& image, const int& binary_threshold) {
    cv::Mat binary_image;

    // Median filter gets rid of holes in thresholded image caused by specular highlights and imperfections on object
    cv::medianBlur(image, binary_image, 75);

    cv::threshold(binary_image, binary_image, binary_threshold, 255, cv::THRESH_BINARY_INV);

    // cv::imshow("Binary image", binary_image);
    // cv::waitKey(0);

    return binary_image;
}

cv::Mat ComputerGeometricPropsAndMarkUpImage(const cv::Mat& binary_image) {
    cv::Moments image_moments = cv::moments(binary_image, true);
    double hu[7];
    cv::HuMoments(image_moments, hu);

    GeometricProps props;
    props.area = image_moments.m00;
    props.centroid = { image_moments.m10 / image_moments.m00, image_moments.m01 / image_moments.m00 };

    double theta = 0.5 * atan2(2 * image_moments.mu11, image_moments.mu20 - image_moments.mu02);
    double angle_degrees = theta * 180.0 / CV_PI;

    double max_slope = tan(theta);
    double x_1 = -10.0;
    double y_1 = max_slope * (x_1 - props.centroid.x) + props.centroid.y;

    double x_2 = binary_image.cols + 10;
    double y_2 = max_slope * (x_2 - props.centroid.x) + props.centroid.y;

    double min_slope = tan(max_slope + (CV_PI / 2));
    double x_3 = -10.0;
    double y_3 = min_slope * (x_1 - props.centroid.x) + props.centroid.y;

    double x_4 = binary_image.cols + 10;
    double y_4 = min_slope * (x_2 - props.centroid.x) + props.centroid.y;

    cv::Mat display_image;
    cv::cvtColor(binary_image, display_image, COLOR_GRAY2BGR);
    cv::circle(display_image, props.centroid, 25, cv::Scalar(255, 0, 0), 5);
    cv::line(display_image, cv::Point(x_1, y_1), cv::Point(x_2, y_2), cv::Scalar(0, 0, 255), 3);
    cv::line(display_image, cv::Point(x_3, y_3), cv::Point(x_4, y_4), cv::Scalar(0, 255, 0), 3);

    return display_image;
}


