//
// Created by pqpo on 2018/8/16.
//
#include <opencv2/opencv.hpp>
#include <android/bitmap.h>
#include <android_utils.h>
#include "jni.h"
#include <android/log.h>
#include <sstream>
#include <opencv_utils.h>
#include "opencv2/core/core_c.h"

using namespace cv;
using namespace std;


static const char *const kClassScanner = "me/pqpo/smartcameralib/SmartScanner";

static bool DEBUG = false;

static const char *const LOG_TAG = "smart_camera_lib";

#define LOG_D(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

static struct {
    int gaussianBlurRadius = 3;
    int cannyThreshold1 = 5;
    int cannyThreshold2 = 80;
    int thresholdThresh = 0;
    int thresholdMaxVal = 255;
    float checkMinLengthRatio = 0.5;
    int houghLinesThreshold = 110;
    int houghLinesMinLineLength = 80;
    int houghLinesMaxLineGap = 10;
    float detectionRatio = 0.1;
    float angleThreshold = 5;
} gScannerParams;

Mat cropByMask(Mat &imgMat, int rotation, int maskX, int maskY, int maskWidth, int maskHeight) {
    if (rotation == 90) {
        matRotateClockWise90(imgMat);
    } else if (rotation == 180) {
        matRotateClockWise180(imgMat);
    } else if (rotation == 270) {
        matRotateClockWise270(imgMat);
    }
    int newHeight = imgMat.rows;
    int newWidth = imgMat.cols;
    maskX = max(0, min(maskX, newWidth));
    maskY = max(0, min(maskY, newHeight));
    maskWidth = max(0, min(maskWidth, newWidth - maskX));
    maskHeight = max(0, min(maskHeight, newHeight - maskY));

    Rect rect(maskX, maskY, maskWidth, maskHeight);
    Mat croppedMat = imgMat(rect);
    return croppedMat;
}

void
processMat(void *yuvData, Mat &outMat, int width, int height, int rotation, int maskX, int maskY,
           int maskWidth, int maskHeight, float scaleRatio) {
    Mat mYuv(height + height / 2, width, CV_8UC1, (uchar *) yuvData);
    Mat imgMat(height, width, CV_8UC1);
    cvtColor(mYuv, imgMat, COLOR_YUV420sp2GRAY);

    Mat croppedMat = cropByMask(imgMat, rotation, maskX, maskY, maskWidth, maskHeight);

    if (croppedMat.cols == 0) {
        return;
    }

    Mat resizeMat;
    resize(croppedMat, resizeMat, Size(static_cast<int>(maskWidth * scaleRatio),
                                       static_cast<int>(maskHeight * scaleRatio)));

    Mat blurMat;
    GaussianBlur(resizeMat, blurMat,
                 Size(gScannerParams.gaussianBlurRadius, gScannerParams.gaussianBlurRadius), 0);

    Mat cannyMat;
    Canny(blurMat, cannyMat, gScannerParams.cannyThreshold1, gScannerParams.cannyThreshold2);
    Mat dilateMat;
    dilate(cannyMat, dilateMat, getStructuringElement(MORPH_RECT, Size(2, 2)));
    Mat thresholdMat;
    threshold(dilateMat, thresholdMat, gScannerParams.thresholdThresh,
              gScannerParams.thresholdMaxVal, THRESH_OTSU);
    outMat = thresholdMat;
}

vector <Vec4i> houghLines(Mat &scr) {
    vector <Vec4i> lines;
    HoughLinesP(scr, lines, 1, CV_PI / 180.0, gScannerParams.houghLinesThreshold,
                gScannerParams.houghLinesMinLineLength, gScannerParams.houghLinesMaxLineGap);

    return lines;
}

jclass find_class(JNIEnv *env, const char *name) {
    jclass clazz = env->FindClass(name);
    return (jclass) env->NewGlobalRef(clazz);
}

jfieldID get_field(JNIEnv *env, jclass *clazz, const char *name, const char *sig) {
    jfieldID filed = env->GetFieldID(*clazz, name, sig);
    return  filed;
}
jobjectArray parse_array(JNIEnv *env,vector<Vec4i> linesLeft){
    jclass rectClassName =find_class(env, "android/graphics/Rect");
    jobjectArray array = env->NewObjectArray(linesLeft.size(), rectClassName, NULL);
    for (unsigned i = 0; i<linesLeft.size(); i++) {
        Vec4i l = linesLeft[i];
        int x1 = l[0];
        int y1 = l[1];
        int x2 = l[2];
        int y2 = l[3];
        jmethodID cid = env->GetMethodID(rectClassName, "<init>", "(IIII)V");
        jobject rect = env->NewObject( rectClassName, cid, x1,  y1, x2, y2);
        env->SetObjectArrayElement(array,i, rect);
    }
    return array;
}
void checkLines(vector <Vec4i> &lines, int checkMinLength, bool vertical) {
    for (unsigned i = lines.size() - 1; i <= 0; --i) {
        Vec4i l = lines[i];
        int x1 = l[0];
        int y1 = l[1];
        int x2 = l[2];
        int y2 = l[3];
//        int p = static_cast<int>(i);
        float distance;
        distance = powf((x1 - x2), 2) + powf((y1 - y2), 2);
        distance = sqrtf(distance);

        if (distance < checkMinLength) {
            lines.erase(lines.begin() + i);
            continue;
        }
        if (x2 == x1) {
            LOG_D("X2 == X1");
            //true
            continue;
        }

        float angle = cvFastArctan(abs(y2 - y1), abs(x2 - x1));
        if (DEBUG) {
            std::ostringstream logStr;
            logStr << "Detection angle: [ vertical = " << vertical
                   << ", angle = " << angle << ", threshold = " << gScannerParams.angleThreshold
                   << " ]" << std::endl;
            string log = logStr.str();
            LOG_D("%s", log.c_str());
        }
        if (vertical) {
            if (abs(90 - angle) < gScannerParams.angleThreshold) {
                //true
            } else {
                lines.erase(lines.begin() + i);
            }
        } else {
            if (abs(angle) < gScannerParams.angleThreshold) {
                //true
            } else {
                lines.erase(lines.begin() + i);
            }
        }
    }
}

jobjectArray parse_array_points(JNIEnv *env,vector<Point> points){
    jclass rectClassName =find_class(env, "android/graphics/Point");
    jobjectArray array = env->NewObjectArray(points.size(), rectClassName, NULL);
    for (unsigned i = 0; i<points.size(); i++) {
        Point l = points[i];
        int x = l.x;
        int y = l.y;
        jmethodID cid = env->GetMethodID(rectClassName, "<init>", "(II)V");
        jobject rect = env->NewObject( rectClassName, cid, x, y);
        env->SetObjectArrayElement(array,i, rect);
    }
    return array;
}

extern "C"
JNIEXPORT jobject

JNICALL
Java_me_pqpo_smartcameralib_SmartScanner_previewScan(JNIEnv *env, jclass type, jbyteArray yuvData_,
                                                     jint width, jint height, jint rotation, jint x,
                                                     jint y, jint maskWidth, jint maskHeight,
                                                     jobject previewBitmap, jfloat ratio) {
    jbyte *yuvData = env->GetByteArrayElements(yuvData_, NULL);
    Mat outMat;
    processMat(yuvData, outMat, width, height, rotation, x, y, maskWidth, maskHeight, ratio);
    env->ReleaseByteArrayElements(yuvData_, yuvData, 0);

    if (outMat.cols == 0) {
        return 0;
    }

    int matH = outMat.rows;
    int matW = outMat.cols;
    int thresholdW = cvRound(gScannerParams.detectionRatio * matW);
    int thresholdH = cvRound(gScannerParams.detectionRatio * matH);
    int checkMinLengthH = static_cast<int>(matH * gScannerParams.checkMinLengthRatio);
    int checkMinLengthW = static_cast<int>(matW * gScannerParams.checkMinLengthRatio);

    //1. crop left
    Rect rect(0, 0, thresholdW, matH);
    Mat croppedMatL = outMat(rect);
    //2. crop top
    rect.x = 0;
    rect.y = 0;
    rect.width = matW;
    rect.height = thresholdH;
    Mat croppedMatT = outMat(rect);
    //3. crop right
    rect.x = matW - thresholdW;
    rect.y = 0;
    rect.width = thresholdW;
    rect.height = matH;
    Mat croppedMatR = outMat(rect);
    //4. crop bottom
    rect.x = 0;
    rect.y = matH - thresholdH;
    rect.width = matW;
    rect.height = thresholdH;
    Mat croppedMatB = outMat(rect);

    vector <Vec4i> linesLeft = houghLines(croppedMatL);
    vector <Vec4i> linesTop = houghLines(croppedMatT);
    vector <Vec4i> linesRight = houghLines(croppedMatR);
    vector <Vec4i> linesBottom = houghLines(croppedMatB);
    checkLines(linesLeft, checkMinLengthH, true);
    checkLines(linesRight, checkMinLengthH, true);
    checkLines(linesTop, checkMinLengthW, false);
    checkLines(linesBottom, checkMinLengthW, false);

    if (previewBitmap != NULL) {
        drawLines(outMat, linesLeft, 0, 0);
        drawLines(outMat, linesTop, 0, 0);
        drawLines(outMat, linesRight, matW - thresholdW, 0);
        drawLines(outMat, linesBottom, 0, matH - thresholdH);
        mat_to_bitmap(env, outMat, previewBitmap);
    }

    if (DEBUG) {
        std::ostringstream logStr;
        logStr << "Number of lines in the area: [ " << linesLeft.size()
               << " , " << linesTop.size()
               << " , " << linesRight.size()
               << " , " << linesBottom.size() << " ]" << std::endl;
        string log = logStr.str();
        LOG_D("%s", log.c_str());
    }


    if (linesLeft.size() > 0 && linesTop.size() > 0 && linesRight.size() > 0 &&
        linesBottom.size() > 0){
        jclass cls = find_class(env, "me/pqpo/smartcameralib/HoloItems");

        jfieldID left = get_field(env, &cls,"left","[Landroid/graphics/Rect;");
        jfieldID right = env->GetFieldID(cls,"right","[Landroid/graphics/Rect;");
        jfieldID top = env->GetFieldID(cls,"top","[Landroid/graphics/Rect;");
        jfieldID bottom = env->GetFieldID(cls,"bottom","[Landroid/graphics/Rect;");


        jobject classObject = env->NewObject(cls, env->GetMethodID(cls,"<init>","()V"));
        env->SetObjectField(classObject, left, parse_array(env, linesLeft));
        env->SetObjectField(classObject, right, parse_array(env, linesRight));
        env->SetObjectField(classObject, top, parse_array(env, linesTop));
        env->SetObjectField(classObject, bottom, parse_array(env, linesBottom));

        return classObject;
    }
//    if (checkLines(linesLeft, checkMinLengthH, true) && checkLines(linesRight, checkMinLengthH, true)
//        && checkLines(linesTop, checkMinLengthW, false) && checkLines(linesBottom, checkMinLengthW, false)) {
//        if (DEBUG) {
//            LOG_D("Detect passed!");
//        }
//        return 1;
//    }
    return NULL;
}

extern "C"
JNIEXPORT jobject
JNICALL
Java_me_pqpo_smartcameralib_SmartScanner_previewCourtours(JNIEnv *env, jclass type, jbyteArray yuvData_,
                                                     jint width, jint height, jint rotation, jint x,
                                                     jint y, jint maskWidth, jint maskHeight,
                                                     jobject previewBitmap, jfloat ratio) {
    jbyte *yuvData = env->GetByteArrayElements(yuvData_, NULL);
    Mat outMat;
    processMat(yuvData, outMat, width, height, rotation, x, y, maskWidth, maskHeight, ratio);
    env->ReleaseByteArrayElements(yuvData_, yuvData, 0);

    if (outMat.cols == 0) {
        return 0;
    }

    int matH = outMat.rows;
    int matW = outMat.cols;
    int checkMinLengthH = static_cast<int>(matH * gScannerParams.checkMinLengthRatio);
    int checkMinLengthW = static_cast<int>(matW * gScannerParams.checkMinLengthRatio);

    Rect rect(0, 0, matW, matH);
    Mat croppedMat = outMat(rect);

    vector<Point>  countours = findMaxContours(croppedMat);
    if (previewBitmap != NULL) {

        mat_to_bitmap(env, outMat, previewBitmap);
    }



    if (countours.size() ==4){
        jclass cls = find_class(env, "me/pqpo/smartcameralib/HoloItems");

        jfieldID points = get_field(env, &cls,"points","[Landroid/graphics/Point;");


        jobject classObject = env->NewObject(cls, env->GetMethodID(cls,"<init>","()V"));
        env->SetObjectField(classObject, points, parse_array_points(env, countours));


        if (DEBUG) {
            std::ostringstream logStr;
            logStr << "Countours: [ " << countours.size()
                    << " x1: "  << countours[0].x     << " y1: "  << countours[0].y
                    << " x2: "  << countours[1].x     << " y2: "  << countours[1].y
                    << " x3: "  << countours[2].x     << " y3: "  << countours[2].y
                    << " x4: "  << countours[3].x     << " y4: "  << countours[3].y
                   << " ]" << std::endl;
            string log = logStr.str();
        LOG_D("%s", log.c_str());
        }

        return classObject;
    }
    return NULL;
}

static void initScannerParams(JNIEnv * env) {
    jclass classDocScanner = env->FindClass(kClassScanner);
    DEBUG = env->GetStaticBooleanField(classDocScanner,
                                       env->GetStaticFieldID(classDocScanner, "DEBUG", "Z"));
    gScannerParams.gaussianBlurRadius = env->GetStaticIntField(classDocScanner,
                                                               env->GetStaticFieldID(
                                                                       classDocScanner,
                                                                       "gaussianBlurRadius", "I"));
    gScannerParams.cannyThreshold1 = env->GetStaticIntField(classDocScanner,
                                                            env->GetStaticFieldID(classDocScanner,
                                                                                  "cannyThreshold1",
                                                                                  "I"));
    gScannerParams.cannyThreshold2 = env->GetStaticIntField(classDocScanner,
                                                            env->GetStaticFieldID(classDocScanner,
                                                                                  "cannyThreshold2",
                                                                                  "I"));
    gScannerParams.checkMinLengthRatio = env->GetStaticFloatField(classDocScanner,
                                                                  env->GetStaticFieldID(
                                                                          classDocScanner,
                                                                          "checkMinLengthRatio",
                                                                          "F"));
    gScannerParams.houghLinesThreshold = env->GetStaticIntField(classDocScanner,
                                                                env->GetStaticFieldID(
                                                                        classDocScanner,
                                                                        "houghLinesThreshold",
                                                                        "I"));
    gScannerParams.houghLinesMinLineLength = env->GetStaticIntField(classDocScanner,
                                                                    env->GetStaticFieldID(
                                                                            classDocScanner,
                                                                            "houghLinesMinLineLength",
                                                                            "I"));
    gScannerParams.houghLinesMaxLineGap = env->GetStaticIntField(classDocScanner,
                                                                 env->GetStaticFieldID(
                                                                         classDocScanner,
                                                                         "houghLinesMaxLineGap",
                                                                         "I"));
    gScannerParams.detectionRatio = env->GetStaticFloatField(classDocScanner,
                                                             env->GetStaticFieldID(classDocScanner,
                                                                                   "detectionRatio",
                                                                                   "F"));
    gScannerParams.angleThreshold = env->GetStaticFloatField(classDocScanner,
                                                             env->GetStaticFieldID(classDocScanner,
                                                                                   "angleThreshold",
                                                                                   "F"));
    if (DEBUG) {
        LOG_D("load params done!");
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_me_pqpo_smartcameralib_SmartScanner_reloadParams(JNIEnv
*env,
jclass type
) {
initScannerParams(env);
}

extern "C"
JNIEXPORT jint

JNICALL
JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv * env = NULL;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_4) != JNI_OK) {
        return JNI_FALSE;
    }
    initScannerParams(env);
    return JNI_VERSION_1_4;
}

