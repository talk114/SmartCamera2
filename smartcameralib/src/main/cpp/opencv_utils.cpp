//
// Created by admin on 2018/8/17.
//

#include <opencv_utils.h>
#include "../../../opencv/include/opencv2/imgproc/imgproc_c.h"
#include <numeric>

using namespace cv;
using namespace std;

void drawLines(Mat &src, vector<Vec2f> &lines) {
    // 以下遍历图像绘制每一条线
    std::vector<cv::Vec2f>::const_iterator it= lines.begin();
    while (it!=lines.end())
    {
        // 以下两个参数用来检测直线属于垂直线还是水平线
        float rho= (*it)[0];   // 表示距离
        float theta= (*it)[1]; // 表示角度

        if (theta < CV_PI/4. || theta > 3.* CV_PI/4.) // 若检测为垂直线
        {
            // 得到线与第一行的交点
            cv::Point pt1(cvRound(rho/cos(theta)),0);
            // 得到线与最后一行的交点
            cv::Point pt2(cvRound((rho - src.rows*sin(theta))/cos(theta)),src.rows);
            // 调用line函数绘制直线
            cv::line(src, pt1, pt2, cv::Scalar(255), 1);

        }
        else // 若检测为水平线
        {
            // 得到线与第一列的交点
            cv::Point pt1(0,cvRound(rho/sin(theta)));
            // 得到线与最后一列的交点
            cv::Point pt2(src.cols,cvRound((rho-src.cols*cos(theta))/sin(theta)));
            // 调用line函数绘制直线
            cv::line(src, pt1, pt2, cv::Scalar(255), 1);
        }
        ++it;
    }
}

void drawLines(Mat &src, vector<Vec4i> &lines, int offsetX, int offsetY) {
    for( size_t i = 0; i < lines.size(); i++ ) {
        Vec4i l = lines[i];
        cv::line(src, Point(l[0] + offsetX, l[1] + offsetY), Point(l[2] + offsetX, l[3] + offsetY), Scalar(255), 4, LINE_AA);
    }
}

vector<Point> findMaxContours(Mat &src) {
    vector<vector<Point>> contours;
    findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<Point> maxAreaPoints;
    double maxArea = 0;

    vector<vector<Point>>::const_iterator it= contours.begin();
    int length = contours.size();

    while (it!=contours.end()) {
        vector<Point> item = *it;
        double area = contourArea(Mat(item));
        if(area > maxArea) {
            maxArea = area;
            maxAreaPoints = item;
        }

        ++it;
    }

    vector<Point> outDP;
    if(maxAreaPoints.size() > 0) {
        double arc = arcLength(maxAreaPoints, true);
        //多变形逼近
        approxPolyDP(Mat(maxAreaPoints), outDP, 0.03*arc, true);

        if(outDP.size()==4 && isContourConvex(Mat(outDP)))
           return outDP;
    }

    return vector<Point>();
}

//顺时针90
void matRotateClockWise90(Mat &src)
{
    transpose(src, src);
    flip(src, src, 1);
}

//顺时针180
void matRotateClockWise180(Mat &src)
{
    flip(src, src, -1);
}

//顺时针270
void matRotateClockWise270(Mat &src)
{
    transpose(src, src);
    flip(src, src, 0);
}


Vec3f calcParams(Point2f p1, Point2f p2) // line's equation Params computation
{
    float a, b, c;
    if (p2.y - p1.y == 0)
    {
        a = 0.0f;
        b = -1.0f;
    }
    else if (p2.x - p1.x == 0)
    {
        a = -1.0f;
        b = 0.0f;
    }
    else
    {
        a = (p2.y - p1.y) / (p2.x - p1.x);
        b = -1.0f;
    }

    c = (-a * p1.x) - b * p1.y;
    return(Vec3f(a, b, c));
}

Point findIntersection(Vec3f params1, Vec3f params2)
{
    float x = -1, y = -1;
    float det = params1[0] * params2[1] - params2[0] * params1[1];
    if (det < 0.5f && det > -0.5f) // lines are approximately parallel
    {
        return(Point(-1, -1));
    }
    else
    {
        x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det;
        y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det;
    }
    return(Point(x, y));
}

vector<Point> getQuadrilateral(Mat& input) // returns that 4 intersection points of the card
{
    Mat convexHull_mask(input.rows, input.cols, CV_8UC1);
    convexHull_mask = Scalar(0);

    vector<vector<Point>> contours;
    findContours(input, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<int> indices(contours.size());
    iota(indices.begin(), indices.end(), 0);

    sort(indices.begin(), indices.end(), [&contours](int lhs, int rhs) {
        return contours[lhs].size() > contours[rhs].size();
    });

    /// Find the convex hull object
    vector<vector<Point> >hull(1);
    convexHull(Mat(contours[indices[0]]), hull[0], false);

    vector<Vec4i> lines;
    drawContours(convexHull_mask, hull, 0, Scalar(255));
    HoughLinesP(convexHull_mask, lines, 1, CV_PI / 200, 50, 50, 10);
    cout << "lines size:" << lines.size() << endl;

    if (lines.size() == 4) // we found the 4 sides
    {
        vector<Vec3f> params(4);
        for (int l = 0; l < 4; l++)
        {
            params.push_back(calcParams(Point(lines[l][0], lines[l][1]), Point(lines[l][2], lines[l][3])));
        }

        vector<Point> corners;
        for (int i = 0; i < params.size(); i++)
        {
            for (int j = i; j < params.size(); j++) // j starts at i so we don't have duplicated points
            {
                Point intersec = findIntersection(params[i], params[j]);
                if ((intersec.x > 0) && (intersec.y > 0) && (intersec.x < input.cols) && (intersec.y < input.rows))
                {
                    cout << "corner: " << intersec << endl;
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
            return(corners);
        }
    }

    return(vector<Point>());
}

vector<Point> findMax2Contours(Mat &src)
{

    vector<Point> card_corners = getQuadrilateral(src);
    if (card_corners.size() == 4)
    {
        return card_corners;
    }


    return vector<Point>();
}
