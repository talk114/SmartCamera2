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
int thresh = 50, N = 11;

#define LOG_D(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)


double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
void drawLines(Mat &src, vector <Vec2f> &lines) {
    // 以下遍历图像绘制每一条线
    vector<Vec2f>::const_iterator it = lines.begin();
    while (it != lines.end()) {
        // 以下两个参数用来检测直线属于垂直线还是水平线
        float rho = (*it)[0];   // 表示距离
        float theta = (*it)[1]; // 表示角度

        if (theta < CV_PI / 4. || theta > 3. * CV_PI / 4.) // 若检测为垂直线
        {
            // 得到线与第一行的交点
            Point pt1(cvRound(rho / cos(theta)), 0);
            // 得到线与最后一行的交点
            Point pt2(cvRound((rho - src.rows * sin(theta)) / cos(theta)), src.rows);
            // 调用line函数绘制直线
            line(src, pt1, pt2, Scalar(255), 1);

        } else // 若检测为水平线
        {
            // 得到线与第一列的交点
            Point pt1(0, cvRound(rho / sin(theta)));
            // 得到线与最后一列的交点
            Point pt2(src.cols, cvRound((rho - src.cols * cos(theta)) / sin(theta)));
            // 调用line函数绘制直线
            line(src, pt1, pt2, Scalar(255), 1);
        }
        ++it;
    }
}

void drawLines(Mat &src, vector <Vec4i> &lines, int offsetX, int offsetY) {
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(src, Point(l[0] + offsetX, l[1] + offsetY), Point(l[2] + offsetX, l[3] + offsetY),
                 Scalar(255), 4, LINE_AA);
    }
}

vector<Point> findMaxCoutours(Mat &src) {
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
        std::ostringstream logStr;
        logStr << "Countours: [ " << item.size()

               << " ]" << std::endl;
        string log = logStr.str();
        LOG_D("%s", log.c_str());

        ++it;
    }

    vector <Point> outDP;
    if (maxAreaPoints.size() > 0) {
        double arc = arcLength(maxAreaPoints, true);

        approxPolyDP(Mat(maxAreaPoints), outDP, 0.03 * arc, true);

        if (outDP.size() == 4 )//&& isContourConvex(Mat(outDP)
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

    ostringstream logStr;
    logStr << "SIZE: [ contours = " << contours.size()
           << ", indices = " << indices.size()
           << " ]" << endl;
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

//
//int thresh = 50, N = 11;
//const char *wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
//static double angle(Point pt1, Point pt2, Point pt0) {
//    double dx1 = pt1.x - pt0.x;
//    double dy1 = pt1.y - pt0.y;
//    double dx2 = pt2.x - pt0.x;
//    double dy2 = pt2.y - pt0.y;
//    return (dx1 * dx2 + dy1 * dy2) /
//           sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
//}

// returns sequence of squares detected on the image.
//void findSquares(Mat &image, vector<vector<Point>> &squares) {
//    squares.clear();
//
//    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
//
//    // down-scale and upscale the image to filter out the noise
//    pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
//    pyrUp(pyr, timg, image.size());
//    vector <vector<Point>> contours;
//
//    // find squares in every color plane of the image
//    for (int c = 0; c < 3; c++) {
//        int ch[] = {c, 0};
//        mixChannels(&timg, 1, &gray0, 1, ch, 1);
//
//        // try several threshold levels
//        for (int l = 0; l < N; l++) {
//            // hack: use Canny instead of zero threshold level.
//            // Canny helps to catch squares with gradient shading
//            if (l == 0) {
//                // apply Canny. Take the upper threshold from slider
//                // and set the lower to 0 (which forces edges merging)
//                Canny(gray0, gray, 0, thresh, 5);
//                // dilate canny output to remove potential
//                // holes between edge segments
//                dilate(gray, gray, Mat(), Point(-1, -1));
//            } else {
//                // apply threshold if l!=0:
//                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
//                gray = gray0 >= (l + 1) * 255 / N;
//            }
//
//            // find contours and store them all as a list
//            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
//
//            vector <Point> approx;
//
//            // test each contour
//            for (size_t i = 0; i < contours.size(); i++) {
//                // approximate contour with accuracy proportional
//                // to the contour perimeter
//                approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
//
//                // square contours should have 4 vertices after approximation
//                // relatively large area (to filter out noisy contours)
//                // and be convex.
//                // Note: absolute value of an area is used because
//                // area may be positive or negative - in accordance with the
//                // contour orientation
//                if (approx.size() == 4 &&
//                    fabs(contourArea(approx)) > 1000 &&
//                    isContourConvex(approx)) {
//                    double maxCosine = 0;
//
//                    for (int j = 2; j < 5; j++) {
//                        // find the maximum cosine of the angle between joint edges
//                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
//                        maxCosine = MAX(maxCosine, cosine);
//                    }
//
//                    // if cosines of all angles are small
//                    // (all angles are ~90 degree) then write quandrange
//                    // vertices to resultant sequence
//                    if (maxCosine < 0.3)
//                        squares.push_back(approx);
//                }
//            }
//        }
//    }
//}

//void checkLines(vector<Vec4i> &lines, int checkMinLength) {
//    for (unsigned i = lines.size() - 1; i <= 0; --i) {
//        Vec4i l = lines[i];
//        int x1 = l[0];
//        int y1 = l[1];
//        int x2 = l[2];
//        int y2 = l[3];
////        int p = static_cast<int>(i);
//        float distance;
//        distance = powf((x1 - x2), 2) + powf((y1 - y2), 2);
//        distance = sqrtf(distance);
//
//        if (distance < checkMinLength) {
//            lines.erase(lines.begin() + i);
//            continue;
//        }
//        if (x2 == x1) {
//            LOG_D("X2 == X1");
//            //true
//            continue;
//        }
//
//        float angle = cvFastArctan(abs(y2 - y1), abs(x2 - x1));
//
//            if (abs(angle) >45 && abs(90 - angle) > 10) {
//                lines.erase(lines.begin() + i);
//            }
//            if (abs(angle) <45 && abs(angle) >10) {
//                lines.erase(lines.begin() + i);
//            }
//
//    }
//}
//
//vector<Point> findMax4Coutours(Mat &Processing){
//    //Image Processing
////    GaussianBlur(BW, BW, Size(15, 15), 1.5, 1.5);		//Gaussian blur
////    erode(BW, BW, Mat(), Point(-1, -1));				//Erosion
////    dilate(BW, BW, Mat(), Point(-1, -1), 10, 1, 10);	//Dilation
//
//    //Canny Edge Detection
////    Canny(BW, Processing, 0, 30, 3);
//
//    //Vector of lines, followed by Houghlines
//    vector <Vec4i> lines;
//    HoughLinesP(Processing, lines, 1, CV_PI / 180, 100, 0, 0);		//Houghlines Function(Input, Storage(output), double rho (keep 1), double theta (keep pi/180), threshold value (pix), keep 0, keep 0)
//
//    checkLines(lines, 480);
//    if (lines.size() > 3)								//if the number of lines is greater than 3, continue processing, otherwise skip. This was put in because the program errored later due to loops having values i<array-1 if the array contained no elements
//    {
//        //Declaring variables
//        Point C;					//Centre point
//        Point P[50000];				//P's of intersection, array arbitrary size 500
//        double numer3, denom3, numer4;		//doubles, used later, for calcs of gradients
//        Point pt1[5000], pt2[5000];		//output hough lines end points
//        double m1, m2, m3, m4, m5, m6, denom1, denom2, denom4, denom5;	//more doubles
//        Point f1[5000], f2[5000];			//two arrays, for confirmed perpendicular lines f1-->pt1  f2-->pt2
//        int numlines = 0;        // Number of intersecting lines detected.
//
//
//        // Show how many lines were detected by opencv
//
//        		LOG_D("%s: %d", "LINE", lines.size());			//uncomment if you want to check this variable
//
//
////        for (size_t i = 0; i < lines.size(); i++)			//loops through each line
////        {
////            // Convert from Polar to Cartesian Co-ordinates.
////            float rho = lines[i][0], theta = lines[i][1];					//
////            double a = cos(theta), b = sin(theta);							//
////            double x0 = a*rho, y0 = b*rho;									// This section does the conversion from lines array polar into cartesian end points of the lines, arbitrary length 1000, can be changed but not necessary
////            pt1[i].x = cvRound(x0 + 1000 * (-b));							//
////            pt1[i].y = cvRound(y0 + 1000 * (a));							//
////            pt2[i].x = cvRound(x0 - 1000 * (-b));							//
////            pt2[i].y = cvRound(y0 - 1000 * (a));							//
////        }
//
////        for (size_t start = 0; start < lines.size() - 1; start++)			//
////        {																	//
////            for (size_t i = start + 1; i < lines.size(); i++)				// Double looping system for comparisons
////            {
////                denom1 = (pt2[start].x - pt1[start].x);						//Gradient comparison between the lines start and i
////                denom2 = (pt2[i].x - pt1[i].x);								//
////                if (denom1 == 0.0) { denom1 = 0.00001; }					//allows for complete comparison between all elements
////                if (denom2 == 0.0) { denom2 = 0.00001; }					//
////                m1 = (pt2[start].y - pt1[start].y) / denom1;				//
////                m2 = (pt2[i].y - pt1[i].y) / denom2;						//
////                m3 = -(m1*m2);												//
//////
//////                // Check if these two lines intersect at ~90 deg. If correct save into arrays f1 (pt1) and f2 (pt2). I.E. end points of the lines
//////
////                if (m3 > 0.1  && m3 < 4.0)			// Constraints for angles. Change for a more stringent 90 degrees (tend to 1) for less stringent, widen constraint
////                {
////                    // If Lines intersect - increment the counter.
////                    f1[numlines] = pt1[start];    // And save the lines to our final output arrays.
////                    f2[numlines] = pt2[start];
////                    numlines++;                // Lines intersect - increment the counter.
////                    f1[numlines] = pt1[i];    // And save the lines to our final output arrays.
////                    f2[numlines] = pt2[i];
////                    numlines++;
////                }
////            }
////        }
////
////        if (numlines > 3)														//
////        {																		//
////            for (size_t i = 0; i < numlines - 1; i++)							//
////            {																	//
////                for (size_t r = i + 1; r < numlines; r++)						// Removes duplicates from the array, makes things simpler later on LEAVE THIS IN!
////                {																//
////                    if (f1[i].x == f1[r].x && f1[i].y == f1[r].y)				//
////                    {															//
////                        for (size_t m = r; m < numlines - 1; m++)				//
////                        {														//
////                            f1[m] = f1[m + 1];									//
////                            f2[m] = f2[m + 1];									//
////                        }
////                        numlines--;
////                        //				printf("Numlines is currently %d\n", numlines);				//uncomment if you want to check this variable
////                    }
////                }
////            }
////
////            int counter = 0;														//
////            for (size_t start = 0; start < numlines - 1; start++)					//
////            {																		// Double loop system for comparisons, same as before
////                for (size_t i = start + 1; i < numlines; i++)						//
////                {																	//
////
////                    denom4 = (f2[start].x - f1[start].x);							//
////                    denom5 = (f2[i].x - f1[i].x);									//
////                    if (denom4 == 0.0) { denom4 = 0.00001; }						//
////                    if (denom5 == 0.0) { denom5 = 0.00001; }						// Finding the gradient between the f's, used as the order (horizontal, vertical) of the array may not be right for what we need to do
////                    m4 = (f2[start].y - f1[start].y) / denom1;						// below. Hence check
////                    m5 = (f2[i].y - f1[i].y) / denom2;								//
////                    m6 = -(m4*m5);													//
////                    if (m6 > 0.1  && m6 < 4.0)										// if between, use "start" and "i"
////                    {
////                        numer3 = (((f1[start].x*f2[start].y) - (f1[start].y*f2[start].x))*(f1[i].x - f2[i].x)) - (((f1[start].x - f2[start].x)*((f1[i].x*f2[i].y) - (f1[i].y*f2[i].x))));
////                        numer4 = (((f1[start].x*f2[start].y) - (f1[start].y*f2[start].x))*(f1[i].y - f2[i].y)) - (((f1[start].y - f2[start].y)*((f1[i].x*f2[i].y) - (f1[i].y*f2[i].x))));
////                        denom3 = ((f1[start].x - f2[start].x)*(f1[i].y - f2[i].y)) - ((f1[start].y - f2[start].y)*(f1[i].x - f2[i].x));
////                        P[counter].x = (numer3 / denom3);
////                        P[counter].y = (numer4 / denom3);
////                        if ((P[counter].x > 0 && P[counter].x < Processing.cols) || (P[counter].y > 0 && P[counter].y < Processing.rows))			// if within the size of the camera feed (640x480)
////                        {
////                            counter++;												// increment counter
////                        }
////                    }
////                    else
////                    {																//else, use "start" and "i+1"
////                        numer3 = (((f1[start].x*f2[start].y) - (f1[start].y*f2[start].x))*(f1[i + 1].x - f2[i + 1].x)) - (((f1[start].x - f2[start].x)*((f1[i + 1].x*f2[i + 1].y) - (f1[i + 1].y*f2[i + 1].x))));
////                        numer4 = (((f1[start].x*f2[start].y) - (f1[start].y*f2[start].x))*(f1[i + 1].y - f2[i + 1].y)) - (((f1[start].y - f2[start].y)*((f1[i + 1].x*f2[i + 1].y) - (f1[i + 1].y*f2[i + 1].x))));
////                        denom3 = ((f1[start].x - f2[start].x)*(f1[i + 1].y - f2[i + 1].y)) - ((f1[start].y - f2[start].y)*(f1[i + 1].x - f2[i + 1].x));
////                        P[counter].x = (numer3 / denom3);
////                        P[counter].y = (numer4 / denom3);
////                        if ((P[counter].x > 0 && P[counter].x < Processing.cols) || (P[counter].y > 0 && P[counter].y < Processing.rows))			// if within the size of the camera feed (640x480)
////                        {
////                            counter++;												// increment counter
////                        }
////                    }
////
////                }
////            }
////
//////            //	printf("Final value for counter = %d\n", counter);							//uncomment if you want to check this variable
//////
////            for (size_t i = 0; i < counter - 1; i++)								//
////            {																		//
////                for (size_t r = i + 1; r < counter; r++)							//
////                {																	// Remove duplicates from the P array as was done before with f1 and f2
////                    if (P[i].x == P[r].x && P[i].y == P[r].y)						// if P = next P
////                    {																//
////                        for (size_t m = r; m < counter - 1; m++)					//
////                        {															//
////                            P[m] = P[m + 1];										// Make P = next P
////                        }															//
////                        counter--;													// Reduce size of array by 1
////                    }
////                }
////            }
////
////            int xmax = -999999, xmin = 999999, ymax = -999999, ymin = 999999;		// Starting values
////            for (size_t i = 0; i < counter; i++)
////            {
////                if (P[i].x < xmin){ xmin = P[i].x; }								// Finds smallest x
////                if (P[i].x > xmax){ xmax = P[i].x; }
////                if (P[i].y < ymin){ ymin = P[i].y; }
////                if (P[i].y > ymax){ ymax = P[i].y; }								// Finds biggest y
////            }
////
////            C.x = round(((xmax - xmin) / 2) + xmin);								// x value of C = average of xmax and xmin (plus xmin to place it in centre)
////            C.y = round(((ymax - ymin) / 2) + ymin);								// y value of C = average of ymax and ymin (plus ymin to place it in centre)
////            vector<Point> points(counter);
////            for (size_t i = 0; i < counter; i++){
////                points.push_back(P[i]);
////            }
////            return points;
////        }
//    }
//    return vector<Point>();
//}
////vector<Point> findMax3Contours(Mat &src) {
////    vector<vector<Point> > squares;
////    findSquares(image, squares);
////}




//void find_squares(Mat& image, vector<vector<Point>> &squares)
//{
//    // blur will enhance edge detection
//    Mat blurred(image);
//    medianBlur(image, blurred, 9);
//
//    Mat gray0(blurred.size(), CV_8U), gray;
//    vector<vector<Point> > contours;
//
//    // find squares in every color plane of the image
//    for (int c = 0; c < 3; c++)
//    {
//        int ch[] = {c, 0};
//        mixChannels(&blurred, 1, &gray0, 1, ch, 1);
//
//        // try several threshold levels
//        const int threshold_level = 2;
//        for (int l = 0; l < threshold_level; l++)
//        {
//            // Use Canny instead of zero threshold level!
//            // Canny helps to catch squares with gradient shading
//            if (l == 0)
//            {
//                Canny(gray0, gray, 10, 20, 3); //
//
//                // Dilate helps to remove potential holes between edge segments
//                dilate(gray, gray, Mat(), Point(-1,-1));
//            }
//            else
//            {
//                gray = gray0 >= (l+1) * 255 / threshold_level;
//            }
//
//            // Find contours and store them in a list
//            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
//
//            // Test contours
//            vector<Point> approx;
//            for (size_t i = 0; i < contours.size(); i++)
//            {
//                // approximate contour with accuracy proportional
//                // to the contour perimeter
//                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
//
//                // Note: absolute value of an area is used because
//                // area may be positive or negative - in accordance with the
//                // contour orientation
//                if (approx.size() == 4 &&
//                    fabs(contourArea(Mat(approx))) > 1000 &&
//                    isContourConvex(Mat(approx)))
//                {
//                    double maxCosine = 0;
//
//                    for (int j = 2; j < 5; j++)
//                    {
//                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
//                        maxCosine = MAX(maxCosine, cosine);
//                    }
//
//                    if (maxCosine < 0.3)
//                        squares.push_back(approx);
//                }
//            }
//        }
//    }
//}


/* findSquares: returns sequence of squares detected on the image
 */


//void findSquares(const Mat& image, vector<vector<Point> >& squares)
//{
//    squares.clear();
//
//    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
//
//    // down-scale and upscale the image to filter out the noise
//    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
//    pyrUp(pyr, timg, image.size());
//    vector<vector<Point> > contours;
//
//    // find squares in every color plane of the image
//    for( int c = 0; c < 3; c++ )
//    {
//        int ch[] = {c, 0};
//        mixChannels(&timg, 1, &gray0, 1, ch, 1);
//
//        // try several threshold levels
//        for( int l = 0; l < N; l++ )
//        {
//            // hack: use Canny instead of zero threshold level.
//            // Canny helps to catch squares with gradient shading
//            if( l == 0 )
//            {
//                // apply Canny. Take the upper threshold from slider
//                // and set the lower to 0 (which forces edges merging)
//                Canny(gray0, gray, 0, thresh, 5);
//                // dilate canny output to remove potential
//                // holes between edge segments
//                dilate(gray, gray, Mat(), Point(-1,-1));
//            }
//            else
//            {
//                // apply threshold if l!=0:
//                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
//                gray = gray0 >= (l+1)*255/N;
//            }
//
//            // find contours and store them all as a list
//            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
//
//            vector<Point> approx;
//
//            // test each contour
//            for( size_t i = 0; i < contours.size(); i++ )
//            {
//                // approximate contour with accuracy proportional
//                // to the contour perimeter
//                approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
//
//                // square contours should have 4 vertices after approximation
//                // relatively large area (to filter out noisy contours)
//                // and be convex.
//                // Note: absolute value of an area is used because
//                // area may be positive or negative - in accordance with the
//                // contour orientation
//                if( approx.size() == 4 &&
//                    fabs(contourArea(approx)) > 1000 &&
//                    isContourConvex(approx) )
//                {
//                    double maxCosine = 0;
//
//                    for( int j = 2; j < 5; j++ )
//                    {
//                        // find the maximum cosine of the angle between joint edges
//                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
//                        maxCosine = MAX(maxCosine, cosine);
//                    }
//
//                    // if cosines of all angles are small
//                    // (all angles are ~90 degree) then write quandrange
//                    // vertices to resultant sequence
//                    if( maxCosine < 0.3 )
//                        squares.push_back(approx);
//                }
//            }
//        }
//    }
//}

/* findLargestSquare: find the largest square within a set of squares
 */
void findLargestSquare(const vector<vector<Point>>& squares,
                       vector<Point>& biggest_square)
{
    if (!squares.size())
    {
        cout << "findLargestSquare !!! No squares detect, nothing to do." << endl;
        return;
    }

    int max_width = 0;
    int max_height = 0;
    int max_square_idx = 0;
    for (size_t i = 0; i < squares.size(); i++)
    {
        // Convert a set of 4 unordered Points into a meaningful Rect structure.
        Rect rectangle = boundingRect(Mat(squares[i]));

        //cout << "find_largest_square: #" << i << " rectangle x:" << rectangle.x << " y:" << rectangle.y << " " << rectangle.width << "x" << rectangle.height << endl;

        // Store the index position of the biggest square found
        if (rectangle.width * rectangle.height > max_width * max_height)
        {
            max_width = rectangle.width;
            max_height = rectangle.height;
            max_square_idx = i;
        }
    }

    biggest_square = squares[max_square_idx];
}

