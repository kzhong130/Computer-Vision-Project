#ifndef InputGraph_H
#define InputGraph_H

#include <QMainWindow>
#include <QFileDialog>
#include <QImage>
#include <QLabel>
#include <QTextCodec>
#include <QMessageBox>
#include <QMouseEvent>
#include <QWidget>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QtWidgets/QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <vector>
#include <QDebug>
#include <cmath>
#include <math.h>

#define NN 0
#define LINEAR 1
#define CHANNEL_R 0
#define CHANNEL_G 1
#define CHANNEL_B 2
#define SIGMA 1.2

using namespace cv;
using namespace std;

bool MatEequals(Mat &m1, Mat &m2);


class InputGraph
{
public:
    Mat image;
    QImage img;//QImage格式的图片

    Mat original;//临时保存用
    Mat show;//临时展示用

    Mat second;//导入图片

    string name;
    int xoffset;
    int yoffset;

    double **sobelX;
    double **sobelY;
    double **laplacekernel;


    std::vector<int> getPixelVal(int x, int y);

    void Dilation(int size, int ksize, double ** kernel);

    void ReDilation(int size, int ksize, double ** kernel, Mat &ma);

    void Erosion(int size,int ksize,double ** kernel);

    void open(int size,int ksize,double ** kernel);

    void close(int size,int ksize,double ** kernel);

    void openReconstruct(int size, int ksize, double ** kernel, int k, Mat &m);

    void ReErosion(int size,int ksize,double ** kernel,Mat& ma);

    void watershedCal();

    void binDilation(int size,int ksize,double ** kernel,Mat &ma);

    void binErosion(int size,int ksize,double** kernel,Mat &ma);

    void thinning(Mat& image,int size,double **kernel);

    void hit(Mat& image,int size, double **kernel);

    void ReverseMat(Mat& ma);

    void UnionMat(Mat& ma,Mat& b);

    void InsertMat(Mat &ma,Mat &b);

    void thickenning(Mat& image,int size,double ** kernel);

    void ChessDistance(Mat& image);

    void CityDistance(Mat& image);

    void Skelton(Mat & image, int size, double **kernel);

    void binOpen(int size,int ksize,double** kernel,Mat &ma);

    void binClose(int size,int ksize,double ** kernel,Mat &ma);

    void gray(Mat& image);

    void channelSplit(int rgbNum,Mat& ma);

    void HSV_H(int value,Mat& ma);

    void HSV_S(int value,Mat& ma);

    void HSV_V(int value,Mat& ma);

    void makeHist(int *hist, int channelNum, Mat mat);

    int countOtus(Mat img);

    void toOtus();

    void thre2binary(int thre1, int thre2,Mat& ma);

    void addimage(double weight1, double weight2);

    void minusimage();

    void multiplicationimage();

    void cutImage(int x1,int y1,int x2,int y2,Mat& ma);

    void resize1(int value);

    void resize2(int value);

    void spin1(int value);

    void spin2(int value);

    void linearTrans(int A,int B,int C,int D);

    void SegTrans(int A,int B,int C,int D);

    void exTrans(double A,double B,double C);

    void logTrans(double A, double B, double C);

    void Skelton2(Mat& raw);

    void average();

    void showavg();

    void softFilter(int ksize, double ** kernel, Mat &ma);

    void avgFilter(int ksize,Mat& ma);

    void meanFilter(int ksize,Mat& ma);

    void meanFilter2(int ksize,Mat& ma);


    void CalGaussianKernel(double **gaus, const int size, const double sigma);

    void gaussFilter(int ksize,Mat& ma);

    void laplace();


    void SobelGradDirction(const Mat imageSource,Mat &imageSobelX,Mat &imageSobelY,double *&pointDrection);

    void SobelAmplitude(const Mat imageGradX,const Mat imageGradY,Mat &SobelAmpXY);

    void LocalMaxValue(const Mat imageInput,Mat &imageOutput,double *pointDrection);

    void DoubleThreshold(Mat &imageIput,double lowThreshold,double highThreshold);

    void DoubleThresholdLink(Mat &imageInput,double lowThreshold,double highThreshold);

    void canny();

    void initKernel();

    void filter2D(double **kernel, int ksize);

    void sobel(int dir);

    void houghline();

    Mat houghTransform(Mat src);

    void imageColorLevel(int Highlight, int Shadow, int OutHighlight, int OutShadow, double Midtones, int rgbnum);
};

#endif // INPUTGRAPH_H

