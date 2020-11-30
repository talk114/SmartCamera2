//
// Created by admin on 2018/8/17.
//

#include <opencv_utils.h>
#include "../../../opencv/include/opencv2/imgproc/imgproc_c.h"
#include <numeric>
#include "jni.h"
#include <android/log.h>

using namespace cv;
using namespace std;


static const char *const LOG_TAG = "smart_camera_lib";

#define LOG_D(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)


void drawLines(Mat &src, vector <Vec2f> &lines) {
    // 以下遍历图像绘制每一条线
    std::vector<cv::Vec2f>::const_iterator it = lines.begin();
    while (it != lines.end()) {
        // 以下两个参数用来检测直线属于垂直线还是水平线
        float rho = (*it)[0];   // 表示距离
        float theta = (*it)[1]; // 表示角度

        if (theta < CV_PI / 4. || theta > 3. * CV_PI / 4.) // 若检测为垂直线
        {
            // 得到线与第一行的交点
            cv::Point pt1(cvRound(rho / cos(theta)), 0);
            // 得到线与最后一行的交点
            cv::Point pt2(cvRound((rho - src.rows * sin(theta)) / cos(theta)), src.rows);
            // 调用line函数绘制直线
            cv::line(src, pt1, pt2, cv::Scalar(255), 1);

        } else // 若检测为水平线
        {
            // 得到线与第一列的交点
            cv::Point pt1(0, cvRound(rho / sin(theta)));
            // 得到线与最后一列的交点
            cv::Point pt2(src.cols, cvRound((rho - src.cols * cos(theta)) / sin(theta)));
            // 调用line函数绘制直线
            cv::line(src, pt1, pt2, cv::Scalar(255), 1);
        }
        ++it;
    }
}

void drawLines(Mat &src, vector <Vec4i> &lines, int offsetX, int offsetY) {
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        cv::line(src, Point(l[0] + offsetX, l[1] + offsetY), Point(l[2] + offsetX, l[3] + offsetY),
                 Scalar(255), 4, LINE_AA);
    }
}

vector <Point> findMaxContours(Mat &src) {
    vector <vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector <Point> maxAreaPoints;
    double maxArea = 0;

    vector < vector < Point >> ::const_iterator
    it = contours.begin();
    int length = contours.size();

    while (it != contours.end()) {
        vector <Point> item = *it;
        double area = contourArea(Mat(item));
        if (area > maxArea) {
            maxArea = area;
            maxAreaPoints = item;
        }

        ++it;
    }

    vector <Point> outDP;
    if (maxAreaPoints.size() > 0) {
        double arc = arcLength(maxAreaPoints, true);
        //多变形逼近
        approxPolyDP(Mat(maxAreaPoints), outDP, 0.03 * arc, true);

        if (outDP.size() == 4 && isContourConvex(Mat(outDP)))
            return outDP;
    }

    return vector<Point>();
}

//顺时针90
void matRotateClockWise90(Mat &src) {
    transpose(src, src);
    flip(src, src, 1);
}

//顺时针180
void matRotateClockWise180(Mat &src) {
    flip(src, src, -1);
}

//顺时针270
void matRotateClockWise270(Mat &src) {
    transpose(src, src);
    flip(src, src, 0);
}


Vec3f calcParams(Point2f p1, Point2f p2) // line's equation Params computation
{
    float a, b, c;
    if (p2.y - p1.y == 0) {
        a = 0.0f;
        b = -1.0f;
    } else if (p2.x - p1.x == 0) {
        a = -1.0f;
        b = 0.0f;
    } else {
        a = (p2.y - p1.y) / (p2.x - p1.x);
        b = -1.0f;
    }

    c = (-a * p1.x) - b * p1.y;
    return (Vec3f(a, b, c));
}

Point findIntersection(Vec3f params1, Vec3f params2) {
    float x = -1, y = -1;
    float det = params1[0] * params2[1] - params2[0] * params1[1];
    if (det < 0.5f && det > -0.5f) // lines are approximately parallel
    {
        return (Point(-1, -1));
    } else {
        x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det;
        y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det;
    }
    return (Point(x, y));
}

vector <Point> getQuadrilateral(Mat &input) // returns that 4 intersection points of the card
{
    Mat convexHull_mask(input.rows, input.cols, CV_8UC1);
    convexHull_mask = Scalar(0);

    vector <vector<Point>> contours;
    findContours(input, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<int> indices(contours.size());
    iota(indices.begin(), indices.end(), 0);

    sort(indices.begin(), indices.end(), [&contours](int lhs, int rhs) {
        return contours[lhs].size() > contours[rhs].size();
    });

    std::ostringstream logStr;
    logStr << "SIZE: [ contours = " << contours.size()
           << ", indices = " << indices.size()
           << " ]" << std::endl;
    string log = logStr.str();
    LOG_D("%s", log.c_str());
    if (contours.size() == 0)
        return vector<Point>();
    /// Find the convex hull object
    vector <vector<Point>> hull(1);
    convexHull(Mat(contours[indices[0]]), hull[0], false);

    vector <Vec4i> lines;
    drawContours(convexHull_mask, hull, 0, Scalar(255));
    HoughLinesP(convexHull_mask, lines, 1, CV_PI / 200, 50, 50, 10);

    if (lines.size() == 4) // we found the 4 sides
    {
        vector <Vec3f> params(4);
        for (int l = 0; l < 4; l++) {
            params.push_back(
                    calcParams(Point(lines[l][0], lines[l][1]), Point(lines[l][2], lines[l][3])));
        }

        vector <Point> corners;
        for (int i = 0; i < params.size(); i++) {
            for (int j = i;
                 j < params.size(); j++) // j starts at i so we don't have duplicated points
            {
                Point intersec = findIntersection(params[i], params[j]);
                if ((intersec.x > 0) && (intersec.y > 0) && (intersec.x < input.cols) &&
                    (intersec.y < input.rows)) {
                    corners.push_back(intersec);
                }
            }
        }

//        for (int i = 0; i < corners.size(); i++)
//        {
//            circle(output, corners[i], 3, Scalar(0, 0, 255));
//        }

        if (corners.size() == 4) // we have the 4 final corners
        {
            return (corners);
        }
    }

    return vector<Point>();
}

vector <Point> findMax2Contours(Mat &src) {

    vector <Point> card_corners = getQuadrilateral(src);
    if (card_corners.size() == 4) {
        return card_corners;
    }


    return vector<Point>();
}


int thresh = 50, N = 11;
const char *wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) /
           sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
void findSquares(Mat &image, vector<vector<Point>> &squares) {
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
    pyrUp(pyr, timg, image.size());
    vector <vector<Point>> contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++) {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for (int l = 0; l < N; l++) {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0) {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1, -1));
            } else {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l + 1) * 255 / N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector <Point> approx;

            // test each contour
            for (size_t i = 0; i < contours.size(); i++) {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 &&
                    fabs(contourArea(approx)) > 1000 &&
                    isContourConvex(approx)) {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++) {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
}
//vector<Point> findMax3Contours(Mat &src) {
//    vector<vector<Point> > squares;
//    findSquares(image, squares);
//}