#include <InputGraph.h>

#define START_GRADIENT 10
#define IMAGE_OUT_NAME "result.bmp"

std::vector<int> InputGraph::getPixelVal(int x, int y) {
    Mat imagepx;
    imagepx = image.clone();

    std::vector<int> ans;

    int iChannels = imagepx.channels();

    /*if(x>=iRows||y>=iCols){
        ans = { -1, -1, -1, 1};
        return ans;
    }*/

    switch (iChannels) {
    // Grayscale
        case 1: {
            int grayScaleVal = imagepx.at<uchar>(y, x);
            ans = { grayScaleVal, grayScaleVal, grayScaleVal, 0 };
        }
    // RGB
        case 3: {
            Vec3b rgbVal = imagepx.at<Vec3b>(y, x);
            ans = { rgbVal[2], rgbVal[1], rgbVal[0], 1 };
        }
    }
    return ans;
}



//只对于二值化图使用，只看一个通道
bool EmptyMat(Mat &ma){
    for(int i=0;i<ma.rows;i++){
        for(int j=0;j<ma.cols;j++){
            if(ma.at<Vec3b>(i,j)[0]!=0){
                return false;
            }
        }
    }
    return true;
}

void InputGraph::binOpen(int size, int ksize, double **kernel, Mat &ma){

    binErosion(size,ksize,kernel,ma);
    binDilation(size,ksize,kernel,ma);

}

void InputGraph::binClose(int size, int ksize, double **kernel, Mat &ma){
    binDilation(size,ksize,kernel,ma);
    binErosion(size,ksize,kernel,ma);
}

void InputGraph::Skelton(Mat &image,int size,double ** kernel){
    Mat ma;
    image.copyTo(ma);
    Mat last;
    int n=0;
    while(!EmptyMat(ma)){
        ma.copyTo(last);
        binErosion(size,2*size+1,kernel,ma);
        n++;
    }
    cout<<n<<endl;
    Mat m1;
    last.copyTo(m1);
    binOpen(size,2*size+1,kernel,m1);
    for(int i=0;i<last.rows;i++){
        for(int j=0;j<last.cols;j++){
            int val= last.at<Vec3b>(i,j)[0]-m1.at<Vec3b>(i,j)[0];
            if(val<0) val = 0;
            last.at<Vec3b>(i,j)[0]=val;
            last.at<Vec3b>(i,j)[1]=val;
            last.at<Vec3b>(i,j)[2]=val;
        }
    }
    last.copyTo(image);
}


void InputGraph::gray(Mat &image){
    int iRows = image.rows;
    int iCols = image.cols;

    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            int r = image.at<Vec3b>(i, j)[2];
            int g = image.at<Vec3b>(i, j)[1];
            int b = image.at<Vec3b>(i, j)[0];
            //利用经验公式
            int gray = (r * 299 + g * 587 + b * 114 + 500) / 1000;
            image.at<Vec3b>(i, j)[0] = gray;
            image.at<Vec3b>(i, j)[1] = gray;
            image.at<Vec3b>(i, j)[2] = gray;
        }
    }
}

void InputGraph::channelSplit(int rgbNum,Mat& ma) {

    int iRows = image.rows;
    int iCols = image.cols;


    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            for (int k = 0; k < 3; k++) {
                ma.at<Vec3b>(i, j)[k] = image.at<Vec3b>(i, j)[2-rgbNum];
            }
        }
    }
}

void InputGraph::HSV_H(int value, Mat &ma){

    Mat hsv(ma.size(), CV_8UC3);
    cvtColor(ma, hsv, CV_BGR2HSV);
    int iRows = hsv.rows;
    int iCols = hsv.cols;
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            hsv.at<Vec3b>(i, j)[0] = value;
        }
    }
    cvtColor(hsv, ma, CV_HSV2BGR);
}

void InputGraph::HSV_S(int value, Mat &ma){
    Mat hsv(ma.size(), CV_8U, Scalar(0));
    cvtColor(ma, hsv, CV_BGR2HSV);
    int iRows = hsv.rows;
    int iCols = hsv.cols;
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            int adjVal = hsv.at<Vec3b>(i, j)[1] + value;
            if (adjVal > 255) {
                adjVal = 255;
            }
            if (adjVal < 0) {
                adjVal = 0;
            }
            hsv.at<Vec3b>(i, j)[1] = adjVal;
        }
    }
    cvtColor(hsv, ma, CV_HSV2BGR);
}

void InputGraph::HSV_V(int value, Mat &ma){
    Mat hsv(ma.size(), CV_8U, Scalar(0));
    cvtColor(ma, hsv, CV_BGR2HSV);
    int iRows = hsv.rows;
    int iCols = hsv.cols;
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            int adjVal = hsv.at<Vec3b>(i, j)[2] + value;
            if (adjVal > 255) {
                adjVal = 255;
            }
            if (adjVal < 0) {
                adjVal = 0;
            }
            hsv.at<Vec3b>(i, j)[2] = adjVal;
        }
    }
    cvtColor(hsv, ma, CV_HSV2BGR);
}

//计算某个mat的某个channel的直方图
void InputGraph::makeHist(int *hist, int channelNum, Mat mat) {
    memset(hist, 0, sizeof(int)*256);
    int iChannels = mat.channels();
    int iRows = mat.rows;
    int iCols = mat.cols;
    if (iChannels == 1) { //gray
        for (int i = 0; i < iRows; i++) {
            for (int j = 0; j < iCols; j++) {
                int tmp = mat.at<uchar>(i, j);
                hist[tmp]++;
            }
        }
    }
    else { // RGB
        for (int i = 0; i < iRows; i++) {
            for (int j = 0; j < iCols; j++) {
                Vec3b tmp = mat.at<Vec3b>(i, j);
                hist[tmp[2 - channelNum]]++;
            }
        }
    }
}

int InputGraph::countOtus(Mat img) {
    int thresholdValue = 1;
    int hist[256];
    int k; // various counters
    int n, n1, n2, gmin, gmax;
    double m1, m2, sum, csum, fmax, sb;
    gmin = 255; gmax = 0;
    makeHist(hist, 0, img);
    // set up everything
    sum = csum = 0.0;
    n = 0;
    for (k = 0; k <= 255; k++){
        sum += (double)k * (double)hist[k]; // x*f(x) 质量矩
        n += hist[k]; //f(x) 质量
    }
    if (!n){
        fprintf(stderr, "NOT NORMAL thresholdValue = 160\n");
        return (160);
    }
    fmax = -1.0;
    n1 = 0;
    for (k = 0; k < 255; k++)
    {
        n1 += hist[k];
        if (!n1) { continue; }
        n2 = n - n1;
        if (n2 == 0) { break; }
        csum += (double)k *hist[k];
        m1 = csum / n1;
        m2 = (sum - csum) / n2;
        sb = (double)n1 *(double)n2 *(m1 - m2) * (m1 - m2);
        if (sb > fmax) {
            fmax = sb;
            thresholdValue = k;
        }
    }
    return(thresholdValue);
}

void InputGraph::toOtus() {
    Mat copyImg = image.clone();
    int iRows = copyImg.rows;
    int iCols = copyImg.cols;
    if (copyImg.channels() != 1) {
        // convert the image into grayscale
        cvtColor(copyImg, copyImg, CV_BGR2GRAY);
    }
    int threshod = countOtus(copyImg);
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            int tmp = copyImg.at<uchar>(i, j);
            copyImg.at<uchar>(i, j) = (tmp > threshod) ? 255 : 0;
        }
    }
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            unsigned char tmp = copyImg.at<uchar>(i, j);
            Vec3b tmpVec = { tmp, tmp, tmp };
            image.at<Vec3b>(i, j) = tmpVec;
        }
    }
}


void InputGraph::thre2binary(int thre1, int thre2,Mat& ma) {
    int iRows = ma.rows;
    int iCols = ma.cols;
    if (ma.channels() != 1) {
        // convert the image into grayscale
        cvtColor(ma, ma, CV_BGR2GRAY);
    }
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            int tmp = ma.at<uchar>(i, j);
            if( (tmp > thre1) & (tmp < thre2) ){
                ma.at<uchar>(i, j) = 255;
            }else{
                ma.at<uchar>(i, j) = 0;
            }
        }
    }
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            unsigned char tmp = ma.at<uchar>(i, j);
            Vec3b tmpVec = { tmp, tmp, tmp };
            show.at<Vec3b>(i, j) = tmpVec;
        }
    }
}


void InputGraph::addimage(double weight1,double weight2){
    int iRows1 = image.rows;
    int iCols1 = image.cols;
    for (int i = 0; i < iRows1; i++) {
        for (int j = 0; j < iCols1; j++) {
            Vec3b left = image.at<Vec3b>(i, j);
            Vec3b right = second.at<Vec3b>(i, j);
            Vec3b ans = weight1*left + weight2*right;
            for (int k = 0; k < 3; k++) {
                if (ans[k] > 255)
                    ans[k] = 255;
            }
            image.at<Vec3b>(i, j) = ans;
        }
    }
}

void InputGraph::minusimage(){
    int iRows1 = image.rows;
    int iCols1 = image.cols;
    for (int i = 0; i < iRows1; i++) {
        for (int j = 0; j < iCols1; j++) {
            Vec3b left = image.at<Vec3b>(i, j);
            Vec3b right = second.at<Vec3b>(i, j);
            Vec3b ans = left - right;
            for (int k = 0; k < 3; k++) {
                if (ans[k] < 0)
                    ans[k] = 0;
            }
            image.at<Vec3b>(i, j) = ans;
        }
    }
}

void InputGraph::multiplicationimage(){
    int iRows1 = image.rows;
    int iCols1 = image.cols;
    for (int i = 0; i < iRows1; i++) {
        for (int j = 0; j < iCols1; j++) {
            Vec3b left = image.at<Vec3b>(i, j);
            Vec3b right = second.at<Vec3b>(i, j);
            uchar tmp0 = left[0] * right[0] / 255;
            uchar tmp1 = left[1] * right[1] / 255;
            uchar tmp2 = left[2] * right[2] / 255;
            Vec3b ans = {tmp0, tmp1, tmp2};
            image.at<Vec3b>(i, j) = ans;
        }
    }
}

void InputGraph::cutImage(int x1,int y1,int x2,int y2,Mat& ma){
    Mat res((x2-x1+1),(y2-y1+1), CV_8UC3, Scalar(0));
    for (int i = x1; i < x2+1; i++) {
        for (int j = y1; j < y2+1; j++) {
            res.at<Vec3b>(i-x1, j-y1) = ma.at<Vec3b>(i, j);
        }
    }
    res.copyTo(ma);
}


void InputGraph::resize1(int value){
    double srcW = image.cols;  //原始图片宽度，如果用int可能会导致(srcH - 1)/(dstH - 1)恒为零
    double srcH = image.rows;  //原始图片高度
    int showW = srcW * value / 10;
    int showH = srcH * value / 10;
    double dstW = (double)showW;  //目标图片宽度
    double dstH = (double)showH;  //目标图片高度
    Mat tmp(showH, showW, CV_8UC3, Scalar(0));
    show = tmp.clone();
    double xm = 0;      //映射的x
    double ym = 0;      //映射的y
    int xi = 0;         //映射x整数部分
    int yi = 0;         //映射y整数部分
    int xl = 0;         //xi + 1
    int yl = 0;         //yi + 1
    double xs = 0;
    double ys = 0;

    /* 为目标图片每个像素点赋值 */
    for(int i = 0; i < showH; i ++) {
        for(int j = 0; j < showW; j ++) {
            //求出目标图像(i,j)点到原图像中的映射坐标(mapx,mapy)
            xm = (srcH - 1)/(dstH - 1) * i;
            ym = (srcW - 1)/(dstW - 1) * j;
            /* 取映射到原图的xm的整数部分 */
            xi = (int)xm;
            yi = (int)ym;
            /* 取偏移量 */
            xs = xm - xi;
            ys = ym - yi;
            xl = xi + 1;
            yl = yi + 1;
            //边缘点
            if((xi+1) > (srcH-1)) xl = xi-1;
            if((yi+1) > (srcW-1)) yl = yi-1;
            //b
            show.at<Vec3b>(i,j)[0] = (int)(image.at<Vec3b>(xi,yi)[0]*(1-xs)*(1-ys) +
                    image.at<Vec3b>(xi,yl)[0]*(1-xs)*ys +
                    image.at<Vec3b>(xl,yi)[0]*xs*(1-ys) +
                    image.at<Vec3b>(xl,yl)[0]*xs*ys);
            //g
            show.at<Vec3b>(i,j)[1] = (int)(image.at<Vec3b>(xi,yi)[1]*(1-xs)*(1-ys) +
                    image.at<Vec3b>(xi,yl)[1]*(1-xs)*ys +
                    image.at<Vec3b>(xl,yi)[1]*xs*(1-ys) +
                    image.at<Vec3b>(xl,yl)[1]*xs*ys);
            //r
            show.at<Vec3b>(i,j)[2] = (int)(image.at<Vec3b>(xi,yi)[2]*(1-xs)*(1-ys) +
                    image.at<Vec3b>(xi,yl)[2]*(1-xs)*ys +
                    image.at<Vec3b>(xl,yi)[2]*xs*(1-ys) +
                    image.at<Vec3b>(xl,yl)[2]*xs*ys);
        }
    }
    show.copyTo(image);
}

//临近
void InputGraph::resize2(int value){
    double srcW = image.cols;
    double srcH = image.rows;  //原始图片高度
    int showW = srcW * value / 10;
    int showH = srcH * value / 10;
    double rateW = value / 10.0;  //目标图片宽度
    double rateH = value / 10.0;  //目标图片高度
    Mat tmp(showH, showW, CV_8UC3, Scalar(0));
    show = tmp.clone();

    for (int i = 0; i < showH; i++){
        int tSrcH = (int)(double(i)/rateH);
        for (int j = 0; j < showW; j++){
            int tSrcW = (int)(double(j)/rateW);
            show.at<Vec3b>(i, j) = image.at<Vec3b>(tSrcH, tSrcW);
        }
    }
    show.copyTo(image);
}

//双线性
void InputGraph::spin1(int value){
    int icols = image.cols;  //原始图片宽度
    int irows = image.rows;  //原始图片高度
    double theta = value / 180.0 * M_PI;
    double SinTheta = sin(theta);
    double CosTheta = cos(theta);

    int newWidth = (abs(icols/2*CosTheta) + abs(irows/2*SinTheta))*2;   //计算新图像的长宽
    int newHeight = (abs(irows/2*CosTheta) + abs(icols/2*SinTheta))*2;

    int xr = icols/2;
    int yr = irows/2;      //旋转前的图像中心

    int xr2 = newWidth/2;   //旋转后的图像中心
    int yr2 = newHeight/2;

    Mat tmp(newHeight, newWidth, CV_8UC3, Scalar(0));
    show = tmp.clone();
    double ConstX = -xr2*CosTheta + yr2*SinTheta + xr + 0.5;
    double ConstY = -yr2*CosTheta - xr2*SinTheta + yr + 0.5;

    int    px, py;
    double tx,ty,p1,p2,p3,p4,p1_2,p3_4;

    for(int i=0; i < newHeight;i++){
        tx = - i*SinTheta - CosTheta + ConstX ;
        ty =   i*CosTheta - SinTheta + ConstY ;
        for(int j=0; j < newWidth;j++){
            tx += CosTheta;
            ty += SinTheta;
            px = int(tx);
            py = int(ty);
            if(px<0||px>icols-2 || py<0||py>irows-2){
                show.at<Vec3b>(i, j) = {255, 255, 255};
                continue;
            }
            for (int k = 0; k < 3; k++) {
                p1 = image.at<Vec3b>(py,px)[k];    //此处求出周围点的值
                p2 = image.at<Vec3b>(py,px+1)[k];
                p3 = image.at<Vec3b>((py+1),px)[k];
                p4 = image.at<Vec3b>((py+1),px+1)[k];
                p1_2 = p1 + (tx-px)*(p2-p1);
                p3_4 = p3 + (tx-px)*(p4-p3);
                show.at<Vec3b>(i, j)[k] = (int)(p1_2 + (ty - py)*(p3_4 - p1_2));
            }
        }
    }
    show.copyTo(image);
}

void InputGraph::spin2(int value){
    int icols = image.cols;  //原始图片宽度
    int irows = image.rows;  //原始图片高度
    double theta = value / 180.0 * M_PI;
    double SinTheta = sin(theta);
    double CosTheta = cos(theta);

    int newWidth = (abs(icols/2*CosTheta) + abs(irows/2*SinTheta))*2;   //计算新图像的长宽
    int newHeight = (abs(irows/2*CosTheta) + abs(icols/2*SinTheta))*2;

    int xr = icols/2;
    int yr = irows/2;      //旋转前的图像中心

    int xr2 = newWidth/2;   //旋转后的图像中心
    int yr2 = newHeight/2;

    Mat tmp(newHeight, newWidth, CV_8UC3, Scalar(0));
    show = tmp.clone();
    double ConstX = -xr2*CosTheta + yr2*SinTheta + xr + 0.5;
    double ConstY = -yr2*CosTheta - xr2*SinTheta + yr + 0.5;

    double x1;
    double y1;
    int x0;
    int y0;
    for(int i=0; i < newHeight;i++){
        x1 = - i*SinTheta - CosTheta + ConstX ;
        y1 =   i*CosTheta - SinTheta + ConstY ;
        for(int j=0; j < newWidth;j++){
            x1 += CosTheta;
            y1 += SinTheta;
            x0 = int(x1);
            y0 = int(y1);
            if(x0<0||x0>icols-1 || y0<0||y0>irows-1){
                show.at<Vec3b>(i, j) = {255, 255, 255};
                continue;
            }
            show.at<Vec3b>(i, j) = image.at<Vec3b>(y0, x0);
        }
    }
    show.copyTo(image);
}


//线性调整
void InputGraph::linearTrans(int A, int B, int C, int D){
    int iRows = image.rows;
    int iCols = image.cols;
    Mat image2 = image.clone();
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            Vec3b tmp = image2.at<Vec3b>(i, j);
            for(int k = 0;k < 3;k++){
                int value = tmp[k];
                if (value >= B) {
                    value = (value - B)*(255 - D) / (255 - B) + D;
                }else if (value >= A && value < B) {
                    value = (value - A)*(D - C) / (B - A) + C;
                }else {
                    value = value * C / A;
                }
                if (value > 255) {
                    value = 255;
                }
                if (value < 0) {
                    value = 0;
                }
                tmp[k] = value;
            }
            image.at<Vec3b>(i, j) = tmp;
        }
    }
}


//分段调整
void InputGraph::SegTrans(int A,int B,int C,int D){
    int iRows = image.rows;
    int iCols = image.cols;
    Mat image2 = image.clone();
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            Vec3b tmp = image2.at<Vec3b>(i, j);
            for(int k = 0;k < 3;k++){
                int value = tmp[k];
                if (value >= B) {
                    value = (value - B)*(255 - D) / (255 - B) + D;
                }else if (value >= A && value < B) {
                    value = (value - A)*(D - C) / (B - A) + C;
                }else {
                    value = value * C / A;
                }
                if (value > 255) {
                    value = 255;
                }
                if (value < 0) {
                    value = 0;
                }
                tmp[k] = value;
            }
            image.at<Vec3b>(i, j) = tmp;
        }
    }
}

void InputGraph::logTrans(double A, double B, double C){
    int iRows = image.rows;
    int iCols = image.cols;
    Mat image2 = image.clone();
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            Vec3b tmp = image2.at<Vec3b>(i, j);
            for(int k = 0;k < 3;k++){
                double value = tmp[k]*1.00;
                value = A + ((log(value + 1.00)) / (B*log(C)));
                if (value > 255) {
                    value = 255;
                }
                if (value < 0) {
                    value = 0;
                }
                tmp[k] = (int)value;
            }
            image.at<Vec3b>(i, j) = tmp;
        }
    }
}


void InputGraph::exTrans(double A, double B, double C){
    int iRows = image.rows;
    int iCols = image.cols;
    Mat image2 = image.clone();
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            Vec3b tmp = image2.at<Vec3b>(i, j);
            for(int k = 0;k < 3;k++){
                double value = tmp[k]*1.00;
                value = pow(B, (C*(value - A))) - 1;
                if (value > 255) {
                    value = 255;
                }
                if (value < 0) {
                    value = 0;
                }
                tmp[k] = (int)value;
            }
            image.at<Vec3b>(i, j) = tmp;
        }
    }
}


vector<int> GetFlags(int a[],int length)
{
    vector<int> vec;
    int neighbour[]={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64};
    for(int i=0;i<length;i++)
    {
        for(int j=0;j<8;j++)
        {
            int sum=0;
            for(int k=j;k<=j+a[i];k++)
                sum+=neighbour[k];
            vec.push_back(sum);
            std::cout<<sum<<" ";
        }
    }
    std::cout<<std::endl;
    return vec;
}



void InputGraph::average(){
    int iRows = image.rows;
    int iCols = image.cols;
    //Mat newImg(iCols, iRows, image.type());
    int rHist[256], gHist[256], bHist[256];
    makeHist(rHist, CHANNEL_R, image);
    makeHist(gHist, CHANNEL_G, image);
    makeHist(bHist, CHANNEL_B, image);
    int numberOfPixel = image.rows*image.cols;
    int rLUT[256], gLUT[256], bLUT[256];
    rLUT[0] = 1.0*rHist[0] / numberOfPixel * 255;
    gLUT[0] = 1.0*gHist[0] / numberOfPixel * 255;
    bLUT[0] = 1.0*bHist[0] / numberOfPixel * 255;
    int rSum = rHist[0], gSum = gHist[0], bSum = bHist[0];
    for (int i = 1; i < 256; i++) {
        rSum += rHist[i];
        rLUT[i] = 1.0 * rSum / numberOfPixel * 255;
        gSum += gHist[i];
        gLUT[i] = 1.0 * gSum / numberOfPixel * 255;
        bSum += bHist[i];
        bLUT[i] = 1.0 * bSum / numberOfPixel * 255;
    }
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            Vec3b srcTmp = image.at<Vec3b>(i, j), dstTmp;
            dstTmp[0] = bLUT[srcTmp[0]];
            dstTmp[1] = gLUT[srcTmp[1]];
            dstTmp[2] = rLUT[srcTmp[2]];
            image.at<Vec3b>(i, j) = dstTmp;
        }
    }
}

void InputGraph::showavg(){
    image.copyTo(show);
    gray(show);
    int ihist[256];
    makeHist(ihist, 0, show);
    Mat histImg = Mat(256, 512, CV_8UC3,Scalar(255,255,255));
    int maxHistVal = 0;
    for (int k = 0; k < 256; k++) {
        if (ihist[k] > maxHistVal) {
            maxHistVal = ihist[k];
        }
    }
    CvScalar color = CV_RGB(0, 0, 0);
    for (int i = 0; i < 512; i++) {
        int drawVal = ihist[i/2] * 256 / maxHistVal;
        line(histImg, Point(i, 256), Point(i, 256- drawVal), color);
    }
    imshow("histogram", histImg);
}

void InputGraph::softFilter(int ksize, double **kernel,Mat& ma){
    Mat newImg = ma.clone();
    int iRows = newImg.rows;
    int iCols = newImg.cols;
    for (int i = ksize / 2; i < iRows - ksize / 2; i++) {
        for (int j = ksize / 2; j < iCols - ksize / 2; j++) {
            Vec3b tmp = ma.at<Vec3b>(i, j);
            double rVal = 0, gVal = 0, bVal = 0;
            for (int m = 0; m < ksize; m++) {
                for (int n = 0; n < ksize; n++) {
                    int i_t = i - (ksize / 2) + m;
                    int j_t = j - (ksize / 2) + n;
                    Vec3b tmp1 = ma.at<Vec3b>(i_t, j_t);
                    rVal += kernel[m][n] * tmp1[2];
                    gVal += kernel[m][n] * tmp1[1];
                    bVal += kernel[m][n] * tmp1[0];
                }
            }
            if (rVal < 0) rVal = 0;
            if (gVal < 0) gVal = 0;
            if (bVal < 0) bVal = 0;
            if (rVal > 255) rVal = 255;
            if (gVal > 255) gVal = 255;
            if (bVal > 255) bVal = 255;
            tmp[0] = bVal;
            tmp[1] = gVal;
            tmp[2] = rVal;
            newImg.at<Vec3b>(i, j) = tmp;
        }
    }
    newImg.copyTo(ma);
}


void InputGraph::avgFilter(int ksize, Mat &ma){
    double val = 1.0 / (ksize*ksize);
    double **kernel;
    kernel = (double **)malloc(ksize * sizeof(double*));
    for (int i = 0; i < ksize; i++) {
        kernel[i] = (double *)malloc(ksize * sizeof(double));
        for (int j = 0; j < ksize; j++) {
            kernel[i][j] = val;
        }
    }
    softFilter(ksize,kernel,ma);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;
}

void InputGraph::meanFilter(int ksize,Mat& ma){
    Mat newImg = image.clone();
    int iRows = newImg.rows;
    int iCols = newImg.cols;
    int *orderR = (int *)malloc(sizeof(int)*ksize ^ 2);
    int *orderG = (int *)malloc(sizeof(int)*ksize ^ 2);
    int *orderB = (int *)malloc(sizeof(int)*ksize ^ 2);
    for (int i = ksize / 2; i < iRows - ksize / 2; i++) {
        for (int j = ksize / 2; j < iCols - ksize / 2; j++) {
            Vec3b tmp = image.at<Vec3b>(i, j);
            for (int m = 0; m < ksize; m++) {
                for (int n = 0; n < ksize; n++) {
                    int i_t = i - (ksize / 2) + m;
                    int j_t = j - (ksize / 2) + n;
                    Vec3b tmp1 = image.at<Vec3b>(i_t, j_t);
                    orderR[m*ksize + n] = tmp1[2];
                    orderG[m*ksize + n] = tmp1[1];
                    orderB[m*ksize + n] = tmp1[0];
                    //printf("%d,%d,%d\n",tmp1[2],tmp1[1],tmp1[0]);
                }
            }
            // sort
            std::sort(orderR, &orderR[ksize^2]);
            std::sort(orderG, &orderG[ksize ^ 2]);
            std::sort(orderB, &orderB[ksize ^ 2]);
            tmp[0] = orderB[(ksize ^ 2) / 2];
            tmp[1] = orderG[(ksize ^ 2) / 2];
            tmp[2] = orderR[(ksize ^ 2) / 2];
            //printf("mid:%d,%d,%d\n",tmp[2],tmp[1],tmp[0]);
            newImg.at<Vec3b>(i, j) = tmp;
        }
    }
    newImg.copyTo(ma);
}


//计算高斯算子
void InputGraph::CalGaussianKernel(double **gaus, const int size, const double sigma){
    const double PI = 4.0*atan(1.0);
    int center = size / 2;
    double sum = 0;
    for (int i = 0; i<size; i++){
        for (int j = 0; j<size; j++){
            gaus[i][j] = (1 / (2 * PI*sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
            sum += gaus[i][j];
        }
    }
    for (int i = 0; i<size; i++){
        for (int j = 0; j<size; j++){
            gaus[i][j] /= sum;
        }
    }
    return;
}
//高斯滤波
void InputGraph::gaussFilter(int ksize,Mat& ma) {
    double **kernel;
    kernel = (double **)malloc(ksize * sizeof(double*));
    for (int i = 0; i < ksize; i++) {
        kernel[i] = (double *)malloc(ksize * sizeof(double));
    }
    CalGaussianKernel(kernel, ksize, SIGMA);
    softFilter(ksize,kernel, ma);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;
}

//生成对应的算子
void InputGraph::initKernel(){
    //double **kernel;
    sobelX = (double **)malloc(3 * sizeof(double*));
    sobelY = (double **)malloc(3 * sizeof(double*));
    laplacekernel = (double **)malloc(3 * sizeof(double*));
    cout<<laplacekernel<<endl;
    for (int i = 0; i < 3; i++) {
        sobelX[i] = (double *)malloc(3 * sizeof(double));
        sobelY[i] = (double *)malloc(3 * sizeof(double));
        laplacekernel[i] = (double *)malloc(3 * sizeof(double));
    }
    sobelX[0][0] = -1;
    sobelX[0][1] = 0;
    sobelX[0][2] = 1;
    sobelX[1][0] = -2;
    sobelX[1][1] = 0;
    sobelX[1][2] = 2;
    sobelX[2][0] = -1;
    sobelX[2][1] = 0;
    sobelX[2][2] = 1;
    sobelY[0][0] = 1;
    sobelY[0][1] = 2;
    sobelY[0][2] = 1;
    sobelY[1][0] = 0;
    sobelY[1][1] = 0;
    sobelY[1][2] = 0;
    sobelY[2][0] = -1;
    sobelY[2][1] = -2;
    sobelY[2][2] = -1;
    laplacekernel[0][0] = 0;
    laplacekernel[0][1] = 1;
    laplacekernel[0][2] = 0;
    laplacekernel[1][0] = 1;
    laplacekernel[1][1] = -4;
    laplacekernel[1][2] = 1;
    laplacekernel[2][0] = 0;
    laplacekernel[2][1] = 1;
    laplacekernel[2][2] = 0;
}


//laplace边缘检测
void InputGraph::laplace(){

    initKernel();
    Mat tmp = image.clone();
    softFilter(3,laplacekernel,image);
    gray(show);
    image = tmp.clone();

}


void InputGraph::canny()
{
    int iRows = image.rows;
    int iCols = image.cols;
    Mat tmp = image.clone();
    gray(image);    //RGB转换为灰度图
    int size=5;    //定义卷积核大小
    gaussFilter(5,image);  //高斯滤波
    image = tmp.clone();
    Mat imageshow;
    if (show.channels() != 1) {
        cvtColor(show, imageshow, CV_RGB2GRAY);
    }

    Mat imageSobelY;
    Mat imageSobelX;
    double *pointDirection=new double[(imageSobelX.cols-1)*(imageSobelX.rows-1)];  //定义梯度方向角数组
    SobelGradDirction(imageshow,imageSobelX,imageSobelY,pointDirection);  //计算X、Y方向梯度和方向角

    Mat SobelGradAmpl;
    SobelAmplitude(imageSobelX,imageSobelY,SobelGradAmpl);   //计算X、Y方向梯度融合幅值

    Mat imageLocalMax;
    LocalMaxValue(SobelGradAmpl,imageLocalMax,pointDirection);  //局部非极大值抑制

    Mat cannyImage;
    cannyImage=Mat::zeros(imageLocalMax.size(),CV_8UC1);
    DoubleThreshold(imageLocalMax,90,160);        //双阈值处理

    DoubleThresholdLink(imageLocalMax,90,160);   //双阈值中间阈值滤除及连接


    // then recover the 3 channels image
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            unsigned char tmp = imageSobelX.at<uchar>(i, j);
            Vec3b tmpVec = { tmp, tmp, tmp };
            show.at<Vec3b>(i, j) = tmpVec;
        }
    }
}



void InputGraph::SobelGradDirction(const Mat imageSource,Mat &imageSobelX,Mat &imageSobelY,double *&pointDrection)
{
    pointDrection=new double[(imageSource.rows-1)*(imageSource.cols-1)];
    for(int i=0;i<(imageSource.rows-1)*(imageSource.cols-1);i++)
    {
        pointDrection[i]=0;
    }
    imageSobelX=Mat::zeros(imageSource.size(),CV_32SC1);
    imageSobelY=Mat::zeros(imageSource.size(),CV_32SC1);
    uchar *P=imageSource.data;
    uchar *PX=imageSobelX.data;
    uchar *PY=imageSobelY.data;

    int step=imageSource.step;
    int stepXY=imageSobelX.step;
    int k=0;
    int m=0;
    int n=0;
    for(int i=1;i<(imageSource.rows-1);i++)
    {
        for(int j=1;j<(imageSource.cols-1);j++)
        {
            //通过指针遍历图像上每一个像素
            double gradY=P[(i-1)*step+j+1]+P[i*step+j+1]*2+P[(i+1)*step+j+1]-P[(i-1)*step+j-1]-P[i*step+j-1]*2-P[(i+1)*step+j-1];
            PY[i*stepXY+j*(stepXY/step)]=abs(gradY);
            double gradX=P[(i+1)*step+j-1]+P[(i+1)*step+j]*2+P[(i+1)*step+j+1]-P[(i-1)*step+j-1]-P[(i-1)*step+j]*2-P[(i-1)*step+j+1];
            PX[i*stepXY+j*(stepXY/step)]=abs(gradX);
            if(gradX==0)
            {
                gradX=0.00000000000000001;  //防止除数为0异常
            }
            pointDrection[k]=atan(gradY/gradX)*57.3;//弧度转换为度
            pointDrection[k]+=90;
            k++;
        }
    }
    convertScaleAbs(imageSobelX,imageSobelX);
    convertScaleAbs(imageSobelY,imageSobelY);
}

void InputGraph::SobelAmplitude(const Mat imageGradX,const Mat imageGradY,Mat &SobelAmpXY)
{
    SobelAmpXY=Mat::zeros(imageGradX.size(),CV_32FC1);
    for(int i=0;i<SobelAmpXY.rows;i++)
    {
        for(int j=0;j<SobelAmpXY.cols;j++)
        {
            SobelAmpXY.at<float>(i,j)=sqrt(imageGradX.at<uchar>(i,j)*imageGradX.at<uchar>(i,j)+imageGradY.at<uchar>(i,j)*imageGradY.at<uchar>(i,j));
        }
    }
    convertScaleAbs(SobelAmpXY,SobelAmpXY);
}

void InputGraph::LocalMaxValue(const Mat imageInput,Mat &imageOutput,double *pointDrection)
{
    //imageInput.copyTo(imageOutput);
    imageOutput=imageInput.clone();
    int k=0;
    for(int i=1;i<imageInput.rows-1;i++)
    {
        for(int j=1;j<imageInput.cols-1;j++)
        {
            int value00=imageInput.at<uchar>((i-1),j-1);
            int value01=imageInput.at<uchar>((i-1),j);
            int value02=imageInput.at<uchar>((i-1),j+1);
            int value10=imageInput.at<uchar>((i),j-1);
            int value11=imageInput.at<uchar>((i),j);
            int value12=imageInput.at<uchar>((i),j+1);
            int value20=imageInput.at<uchar>((i+1),j-1);
            int value21=imageInput.at<uchar>((i+1),j);
            int value22=imageInput.at<uchar>((i+1),j+1);

            if(pointDrection[k]>0&&pointDrection[k]<=45)
            {
                if(value11<=(value12+(value02-value12)*tan(pointDrection[(i-1)*imageOutput.rows+j]))||(value11<=(value10+(value20-value10)*tan(pointDrection[(i-1)*imageOutput.rows+j]))))
                {
                    imageOutput.at<uchar>(i,j)=0;
                }
            }
            if(pointDrection[k]>45&&pointDrection[k]<=90)

            {
                if(value11<=(value01+(value02-value01)/tan(pointDrection[(i-1)*imageOutput.cols+j]))||value11<=(value21+(value20-value21)/tan(pointDrection[(i-1)*imageOutput.cols+j])))
                {
                    imageOutput.at<uchar>(i,j)=0;

                }
            }
            if(pointDrection[k]>90&&pointDrection[k]<=135)
            {
                if(value11<=(value01+(value00-value01)/tan(180-pointDrection[(i-1)*imageOutput.cols+j]))||value11<=(value21+(value22-value21)/tan(180-pointDrection[(i-1)*imageOutput.cols+j])))
                {
                    imageOutput.at<uchar>(i,j)=0;
                }
            }
            if(pointDrection[k]>135&&pointDrection[k]<=180)
            {
                if(value11<= (value10+(value00-value10)*tan(180-pointDrection[(i-1)*imageOutput.cols+j])) ||value11<= (value12+(value22-value11)*tan(180-pointDrection[(i-1)*imageOutput.cols+j])))
                {
                    imageOutput.at<uchar>(i,j)=0;
                }
            }
            k++;
        }
    }
}
//双阈值化
void InputGraph::DoubleThreshold(Mat &imageIput,double lowThreshold,double highThreshold)
{
    for(int i=0;i<imageIput.rows;i++)
    {
        for(int j=0;j<imageIput.cols;j++)
        {
            if(imageIput.at<uchar>(i,j)>highThreshold)
            {
                imageIput.at<uchar>(i,j)=255;
            }
            if(imageIput.at<uchar>(i,j)<lowThreshold)
            {
                imageIput.at<uchar>(i,j)=0;
            }
        }
    }
}

//双阈值化链接，去掉一些被错判的边界点
void InputGraph::DoubleThresholdLink(Mat &imageInput,double lowThreshold,double highThreshold)
{
    for(int i=1;i<imageInput.rows-1;i++)
    {
        for(int j=1;j<imageInput.cols-1;j++)
        {
            if(imageInput.at<uchar>(i,j)>lowThreshold&&imageInput.at<uchar>(i,j)<255)
            {
                if(imageInput.at<uchar>(i-1,j-1)==255||imageInput.at<uchar>(i-1,j)==255||imageInput.at<uchar>(i-1,j+1)==255||
                    imageInput.at<uchar>(i,j-1)==255||imageInput.at<uchar>(i,j)==255||imageInput.at<uchar>(i,j+1)==255||
                    imageInput.at<uchar>(i+1,j-1)==255||imageInput.at<uchar>(i+1,j)==255||imageInput.at<uchar>(i+1,j+1)==255)
                {
                    imageInput.at<uchar>(i,j)=255;
                    DoubleThresholdLink(imageInput,lowThreshold,highThreshold); //递归调用
                }
                else
            {
                    imageInput.at<uchar>(i,j)=0;
            }
            }
        }
    }
}

//sobel边缘检测
void InputGraph::sobel(int dir){
    initKernel();
    if(dir == 1){
        filter2D(sobelX,3);
        gray(show);
    }else if(dir == 2){
        filter2D(sobelY,3);
        gray(show);
    }else if(dir==3){
        Mat tmp = image.clone();
        filter2D(sobelX,3);
        image = show.clone();
        filter2D(sobelY,3);
        image = tmp.clone();
        gray(show);
    }
}



void InputGraph::filter2D(double **kernel, int ksize) {
    Mat newImg = image.clone();
    int iRows = newImg.rows;
    int iCols = newImg.cols;
    for (int i = ksize / 2; i < iRows - ksize / 2; i++) {
        for (int j = ksize / 2; j < iCols - ksize / 2; j++) {
            Vec3b tmp = image.at<Vec3b>(i, j);
            double rVal = 0, gVal = 0, bVal = 0;
            for (int m = 0; m < ksize; m++) {
                for (int n = 0; n < ksize; n++) {
                    int i_t = i - (ksize / 2) + m;
                    int j_t = j - (ksize / 2) + n;
                    Vec3b tmp1 = image.at<Vec3b>(i_t, j_t);
                    rVal += kernel[m][n] * tmp1[2];
                    gVal += kernel[m][n] * tmp1[1];
                    bVal += kernel[m][n] * tmp1[0];
                }
            }
            if (rVal < 0) rVal = 0;
            if (gVal < 0) gVal = 0;
            if (bVal < 0) bVal = 0;
            if (rVal > 255) rVal = 255;
            if (gVal > 255) gVal = 255;
            if (bVal > 255) bVal = 255;
            tmp[0] = bVal;
            tmp[1] = gVal;
            tmp[2] = rVal;
            newImg.at<Vec3b>(i, j) = tmp;
        }
    }
    show = newImg.clone();
}


//霍夫变换检测直线
void InputGraph::houghline(){
    Mat copyImg = image.clone();
    int iRows = copyImg.rows;
    int iCols = copyImg.cols;
    if (copyImg.channels() != 1) {
        // convert the image into grayscale
        cvtColor(copyImg, copyImg, CV_BGR2GRAY);
    }
    int threshod = countOtus(copyImg);
    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            int tmp = copyImg.at<uchar>(i, j);
            copyImg.at<uchar>(i, j) = (tmp > threshod) ? 255 : 0;
        }
    }
    Mat inputimage(iRows,iCols,CV_8UC1, Scalar(0));
    inputimage = houghTransform(copyImg);

    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            unsigned char tmp = inputimage.at<uchar>(i, j);
            Vec3b tmpVec = { tmp, tmp, tmp };
            show.at<Vec3b>(i, j) = tmpVec;
        }
    }

}
Mat InputGraph::houghTransform(Mat src) {

    static double PI_VALUE = M_PI;
    int hough_space = 500;
    int width = src.cols;
    int height = src.rows;

    int iRows = src.rows;
    int iCols = src.cols;
    float threshold = 0.5f;
    float scale = 1.0f;
    float offset = 0.0f;
    int centerX = width / 2;
    int centerY = height / 2;
    double hough_interval = PI_VALUE/(double)hough_space;

    int maxnum = max(width, height);
    int max_length = (int)(sqrt(2.0) * maxnum);
    int hough_1dsize = 2 * hough_space * max_length;
    int *hough_1d = (int *)malloc(sizeof(int)*2 * hough_space * max_length);

    Mat hough_2d(hough_space,2*max_length, CV_32SC1, Scalar(0));
    Mat image_2d = src.clone();

    for (int i = 0; i < iRows; i++) {
        for (int j = 0; j < iCols; j++) {
            int p = src.at<uchar>(i, j);
            p = p & 0xff;
            if(p == 0) continue;
            for(int cell=0; cell < hough_space; cell++ ) {
                //找到最大的值
                maxnum = (int)((j - centerX) * cos(cell * hough_interval) + (i - centerY) * sin(cell * hough_interval));
                maxnum += max_length;
                if (maxnum < 0 || (maxnum >= 2 * max_length)) {
                    continue;
                }
                hough_2d.at<uchar>(cell, maxnum) +=1;
            }
        }
    }

    // find the max hough value
    int max_hough = 0;
    for(int i=0; i<hough_space; i++) {
        for(int j=0; j<2*max_length; j++) {
            hough_1d[(i + j * hough_space)] = hough_2d.at<uchar>(i, j);
            if(hough_2d.at<uchar>(i, j) > max_hough) {
                max_hough = hough_2d.at<uchar>(i, j);
            }
        }
    }

    // transfer back to image pixels space from hough parameter space
    int hough_threshold = (int)(threshold * max_hough);
    for(int row = 0; row < hough_space; row++) {
        for(int col = 0; col < 2*max_length; col++) {
            if(hough_2d.at<uchar>(row, col) < hough_threshold) // discard it
                continue;
            int hough_value = hough_2d.at<uchar>(row, col);
            bool isLine = true;
            for(int i=-1; i<2; i++) {
                for(int j=-1; j<2; j++) {
                    if(i != 0 || j != 0) {
                      int yf = row + i;
                      int xf = col + j;
                      if(xf < 0) continue;
                      if(xf < 2*max_length) {
                          if (yf < 0) {
                              yf += hough_space;
                          }
                          if (yf >= hough_space) {
                              yf -= hough_space;
                          }
                          if(hough_2d.at<uchar>(yf, xf) <= hough_value) {
                              continue;
                          }
                          isLine = false;
                          break;
                      }
                    }
                }
            }
            if(!isLine) continue;

            // transform back to pixel data
            double dy = sin(row * hough_interval);
            double dx = cos(row * hough_interval);
            if ((row <= hough_space / 4) || (row >= 3 * hough_space / 4)) {
                for (int subrow = 0; subrow < height; ++subrow) {
                  int subcol = (int)((col - max_length - ((subrow - centerY) * dy)) / dx) + centerX;
                  if ((subcol < width) && (subcol >= 0)) {
                      image_2d.at<uchar>(subrow,subcol) = 155;
                  }
                }
              } else {
                for (int subcol = 0; subcol < width; ++subcol) {
                  int subrow = (int)((col - max_length - ((subcol - centerX) * dx)) / dy) + centerY;
                  if ((subrow < height) && (subrow >= 0)) {
                      image_2d.at<uchar>(subrow,subcol) =  155;
                  }
                }
              }
        }
    }

    // convert to hough 1D and return result
    for (int i = 0; i < hough_1dsize; i++)
    {
      int value = (int)(scale * hough_1d[i] + offset); // scale always equals 1
      if (value < 0)
          value = 0;
      else if (value > 255) {
          value = 255;
      }
      hough_1d[i] = (0xFF000000 | value + (value << 16) + (value << 8));
    }
    return image_2d;
}

//获取某个通道的色阶表
bool getColorLevelTable(int Highlight, int Shadow,int OutHighlight,int OutShadow,double Midtones,unsigned char* clTable)
{
    int diff = (int)(Highlight - Shadow);
    int outDiff = (int)(OutHighlight - OutShadow);

    double coef = 255.0 / diff;
    double outCoef = outDiff / 255.0;
    double exponent = 1.0 / Midtones;

    for (int i = 0; i < 256; i++){
        int v;
        if (clTable[i] <= (unsigned char)Shadow)
            v = 0;
        else{
            v = (int)((clTable[i] - Shadow) * coef + 0.5);
            if (v > 255)
                v = 255;
        }
        v = (int)(pow(v / 255.0, exponent) * 255.0 + 0.5);
        clTable[i] = (unsigned char)(v * outCoef + OutShadow + 0.5);
    }

    return true;
}
//检查是否正常获取四个通道的色阶表
bool checkColorLevelData( int Highlight, int Shadow,int OutHighlight,int OutShadow,double Midtones, unsigned char clTables[][256])
{
    bool result = false;
    int i, j;
    for (i = 0; i < 3; i++){
        for (j = 0; j < 256; j++)
            clTables[i][j] = (unsigned char)j;
    }

    int item = 0;
    //分别获取B G R三个通道的色阶表
    for (i = 0; i < 3; i++, item++){
        if (getColorLevelTable(Highlight, Shadow,OutHighlight, OutShadow,Midtones, clTables[i]))
            result = true;
    }
    //获取RGB通道的色阶表
    for (i = 0; i < 3; i++){
        if (!getColorLevelTable(Highlight, Shadow,OutHighlight, OutShadow,Midtones, clTables[i]))
            break;
        result = true;
    }

    return result;
}
void InputGraph::imageColorLevel(int Highlight, int Shadow,int OutHighlight,int OutShadow,double Midtones,int rgbnum)
{

    int iRows = image.rows;
    int iCols = image.cols;
    unsigned char clTables[3][256];

    //色阶表替换法替换三个通道的数据
    if (checkColorLevelData(Highlight, Shadow,OutHighlight, OutShadow,Midtones, clTables)){
        if(rgbnum == 0){
            for (int i = 0; i < iRows; i++){
                for (int j = 0; j < iCols; j++){
                    Vec3b tmp = image.at<Vec3b>(i, j);
                    tmp[2] = clTables[2][tmp[2]];
                    show.at<Vec3b>(i, j) = tmp;
                }
            }

        }else if(rgbnum == 1){
            for (int i = 0; i < iRows; i++){
                for (int j = 0; j < iCols; j++){
                    Vec3b tmp = image.at<Vec3b>(i, j);
                    tmp[1] = clTables[1][tmp[1]];
                    show.at<Vec3b>(i, j) = tmp;
                }
            }

        }else if(rgbnum == 2){
            for (int i = 0; i < iRows; i++){
                for (int j = 0; j < iCols; j++){
                    Vec3b tmp = image.at<Vec3b>(i, j);
                    tmp[0] = clTables[0][tmp[0]];
                    show.at<Vec3b>(i, j) = tmp;
                }
            }

        }else{
            for (int i = 0; i < iRows; i++){
                for (int j = 0; j < iCols; j++){
                    Vec3b tmp = image.at<Vec3b>(i, j);
                    tmp[0] = clTables[0][tmp[0]];	//将输入源对应到色阶表中
                    tmp[1] = clTables[1][tmp[1]];
                    tmp[2] = clTables[2][tmp[2]];
                    show.at<Vec3b>(i, j) = tmp;
                }
            }
        }
    }
}


void InputGraph::Dilation(int size,int ksize,double** kernel){
    int result1=-1;
    int result2=-1;
    int result3=-1;
    Mat mk = Mat(image.rows,image.cols,CV_8UC3);    //内参数K Mat类型变量
    for(int i=0; i < image.rows;i++){
        for(int j=0;j<image.cols;j++){

            result1=-1;
            result2=-1;
            result3=-1;
            for(int m=-size;m<=size;m++){
                for(int n=-size;n<=size;n++){
                    //超出边界部分设为0

                    int v1;
                    int v2;
                    int v3;
                    if(i-m<0||j-n<0||i-m>=image.rows||j-n>=image.cols){
                        v1=0;
                        v2=0;
                        v3=0;
                    }
                    else{
                        v1 = image.at<Vec3b>(i-m,j-n)[0];
                        v2 = image.at<Vec3b>(i-m,j-n)[1];
                        v3 = image.at<Vec3b>(i-m,j-n)[2];
                        //cout<<v1<<" "<<v2<<" "<<v3<<endl;
                    }
                    result1=(result1>v1+kernel[m+size][n+size])?result1:v1+kernel[m+size][n+size];
                    result2=(result2>v2+kernel[m+size][n+size])?result2:v2+kernel[m+size][n+size];
                    result3=(result3>v3+kernel[m+size][n+size])?result3:v3+kernel[m+size][n+size];
                    if(result1>=255) result1=255;
                    if(result2>=255) result2=255;
                    if(result3>=255) result3=255;
                    //cout<<result1<<" "<<result2<<" "<<result3<<endl;
                }
            }

            //return;
            mk.at<Vec3b>(i,j)[0]=result1;
            mk.at<Vec3b>(i,j)[1]=result2;
            mk.at<Vec3b>(i,j)[2]=result3;
        }
    }

    mk.copyTo(image);

}


void InputGraph::Erosion(int size,int ksize,double ** kernel){
    int result1=-1;
    int result2=-1;
    int result3=-1;
    Mat mk = Mat(image.rows,image.cols,CV_8UC3);    //内参数K Mat类型变量
    for(int i=0; i < image.rows;i++){
        for(int j=0;j<image.cols;j++){

            result1=256;
            result2=256;
            result3=256;
            for(int m=-size;m<=size;m++){
                for(int n=-size;n<=size;n++){
                    //超出边界部分设为0

                    int v1;
                    int v2;
                    int v3;
                    if(i+m<0||j+n<0||i+m>=image.rows||j+n>=image.cols){
                        v1=0;
                        v2=0;
                        v3=0;
                    }
                    else{
                        v1 = image.at<Vec3b>(i+m,j+n)[0];
                        v2 = image.at<Vec3b>(i+m,j+n)[1];
                        v3 = image.at<Vec3b>(i+m,j+n)[2];
                        //cout<<v1<<" "<<v2<<" "<<v3<<endl;
                    }
                    result1=(result1<v1-kernel[m+size][n+size])?result1:v1-kernel[m+size][n+size];
                    result2=(result2<v2-kernel[m+size][n+size])?result2:v2-kernel[m+size][n+size];
                    result3=(result3<v3-kernel[m+size][n+size])?result3:v3-kernel[m+size][n+size];
                    if(result1>=255) result1=255;
                    if(result2>=255) result2=255;
                    if(result3>=255) result3=255;
                    if(result1<=0) result1=0;
                    if(result2<=0) result2=0;
                    if(result3<=0) result3=0;
                    //cout<<result1<<" "<<result2<<" "<<result3<<endl;
                }
            }

            //return;
            mk.at<Vec3b>(i,j)[0]=result1;
            mk.at<Vec3b>(i,j)[1]=result2;
            mk.at<Vec3b>(i,j)[2]=result3;
        }
    }
    mk.copyTo(image);
}

void InputGraph::ReErosion(int size,int ksize,double ** kernel,Mat& ma){
    int result1=-1;
    int result2=-1;
    int result3=-1;

    Mat mk = Mat(ma.rows,ma.cols,CV_8UC3);    //内参数K Mat类型变量
    for(int i=0; i < ma.rows;i++){
        for(int j=0;j<ma.cols;j++){

            result1=256;
            result2=256;
            result3=256;
            for(int m=-size;m<=size;m++){
                for(int n=-size;n<=size;n++){
                    //超出边界部分设为0

                    int v1;
                    int v2;
                    int v3;
                    if(i+m<0||j+n<0||i+m>=ma.rows||j+n>=ma.cols){
                        v1=0;
                        v2=0;
                        v3=0;
                    }
                    else{
                        v1 = ma.at<Vec3b>(i+m,j+n)[0];
                        v2 = ma.at<Vec3b>(i+m,j+n)[1];
                        v3 = ma.at<Vec3b>(i+m,j+n)[2];
                        //cout<<v1<<" "<<v2<<" "<<v3<<endl;
                    }
                    result1=(result1<v1-kernel[m+size][n+size])?result1:v1-kernel[m+size][n+size];
                    result2=(result2<v2-kernel[m+size][n+size])?result2:v2-kernel[m+size][n+size];
                    result3=(result3<v3-kernel[m+size][n+size])?result3:v3-kernel[m+size][n+size];
                    if(result1>=255) result1=255;
                    if(result2>=255) result2=255;
                    if(result3>=255) result3=255;
                    if(result1<=0) result1=0;
                    if(result2<=0) result2=0;
                    if(result3<=0) result3=0;
                    //cout<<result1<<" "<<result2<<" "<<result3<<endl;
                }
            }

            //return;
            mk.at<Vec3b>(i,j)[0]=result1;
            mk.at<Vec3b>(i,j)[1]=result2;
            mk.at<Vec3b>(i,j)[2]=result3;
        }
    }
    mk.copyTo(ma);

}



void InputGraph::ReDilation(int size,int ksize,double** kernel,Mat& ma){
    int result1=-1;
    int result2=-1;
    int result3=-1;
    Mat mk = Mat(ma.rows,ma.cols,CV_8UC3);    //内参数K Mat类型变量
    for(int i=0; i < ma.rows;i++){
        for(int j=0;j<ma.cols;j++){

            result1=-1;
            result2=-1;
            result3=-1;
            for(int m=-size;m<=size;m++){
                for(int n=-size;n<=size;n++){
                    //超出边界部分设为0

                    int v1;
                    int v2;
                    int v3;
                    if(i-m<0||j-n<0||i-m>=ma.rows||j-n>=ma.cols){
                        v1=0;
                        v2=0;
                        v3=0;
                    }
                    else{
                        v1 = ma.at<Vec3b>(i-m,j-n)[0];
                        v2 = ma.at<Vec3b>(i-m,j-n)[1];
                        v3 = ma.at<Vec3b>(i-m,j-n)[2];
                        //cout<<v1<<" "<<v2<<" "<<v3<<endl;
                    }
                    result1=(result1>v1+kernel[m+size][n+size])?result1:v1+kernel[m+size][n+size];
                    result2=(result2>v2+kernel[m+size][n+size])?result2:v2+kernel[m+size][n+size];
                    result3=(result3>v3+kernel[m+size][n+size])?result3:v3+kernel[m+size][n+size];
                    if(result1>=255) result1=255;
                    if(result2>=255) result2=255;
                    if(result3>=255) result3=255;
                    //cout<<result1<<" "<<result2<<" "<<result3<<endl;
                }
            }

            //return;
            mk.at<Vec3b>(i,j)[0]=result1;
            mk.at<Vec3b>(i,j)[1]=result2;
            mk.at<Vec3b>(i,j)[2]=result3;
        }
    }

    mk.copyTo(ma);

}



void InputGraph::open(int size, int ksize, double **kernel){
    Erosion(size,ksize,kernel);
    Dilation(size,ksize,kernel);


}

void InputGraph::close(int size, int ksize, double **kernel){

    Dilation(size,ksize,kernel);
    Erosion(size,ksize,kernel);
}


bool MatEequals(Mat& m1,Mat& m2){
    if(m1.channels()!=m2.channels()) {cout<<"not equ"<<endl;return false;}
    assert(m1.channels()==3);
    if(m1.rows!=m2.rows) return false;
    if(m1.cols!=m2.cols) return false;
    for(int i=0;i<m1.rows;i++){
        for(int j=0;j<m1.cols;j++){
            //printf("%d,%d\n",m1.at<Vec3b>(i,j)[0],m2.at<Vec3b>(i,j)[0]);
            if(m1.at<Vec3b>(i,j)[0]!=m2.at<Vec3b>(i,j)[0]) {printf("mat not eq1 %d,%d\n",m1.at<Vec3b>(i,j)[0],m2.at<Vec3b>(i,j)[0]);return false;}
            if(m1.at<Vec3b>(i,j)[1]!=m2.at<Vec3b>(i,j)[1]) {printf("mat not eq2 %d,%d\n",m1.at<Vec3b>(i,j)[1],m2.at<Vec3b>(i,j)[1]);return false;}
            if(m1.at<Vec3b>(i,j)[2]!=m2.at<Vec3b>(i,j)[2]) {printf("mat not eq3 %d,%d\n",m1.at<Vec3b>(i,j)[2],m2.at<Vec3b>(i,j)[2]);return false;}

        }
    }
    //cout<<m1.rows<<" "<<m1.cols<<endl;
    return true;
}

void InputGraph::openReconstruct(int size,int ksize,double ** kernel,int k,Mat& m){
    Mat last;
    m.copyTo(last);
    while(true){
        Mat temp;
        m.copyTo(temp);
        for(int i=0;i<k;i++){
            ReErosion(size,ksize,kernel,temp);
        }
        for(int i=0;i<m.rows;i++){
            for(int j=0;j<m.cols;j++){
                m.at<Vec3b>(i,j)[0]=(m.at<Vec3b>(i,j)[0]<temp.at<Vec3b>(i,j)[0])?m.at<Vec3b>(i,j)[0]:temp.at<Vec3b>(i,j)[0];
                m.at<Vec3b>(i,j)[1]=(m.at<Vec3b>(i,j)[1]<temp.at<Vec3b>(i,j)[1])?m.at<Vec3b>(i,j)[1]:temp.at<Vec3b>(i,j)[1];
                m.at<Vec3b>(i,j)[2]=(m.at<Vec3b>(i,j)[2]<temp.at<Vec3b>(i,j)[2])?m.at<Vec3b>(i,j)[2]:temp.at<Vec3b>(i,j)[2];

            }
        }
        if(MatEequals(last,m)) break;
        cout<<1<<endl;
        m.copyTo(last);
    }
}


void InputGraph::watershedCal(){

}



void InputGraph::binDilation(int size,int ksize,double ** kernel,Mat &ma){
    int result1=0;
    int result2=0;
    int result3=0;
    Mat mk = Mat(ma.rows,ma.cols,CV_8UC3);    //内参数K Mat类型变量
    for(int i=0; i < ma.rows;i++){
        for(int j=0;j<ma.cols;j++){

            result1=0;
            result2=0;
            result3=0;
            for(int m=-size;m<=size;m++){
                for(int n=-size;n<=size;n++){
                    //超出边界部分设为0

                    int v1;
                    int v2;
                    int v3;
                    if(i-m<0||j-n<0||i-m>=ma.rows||j-n>=ma.cols){
                        v1=255;
                        v2=255;
                        v3=255;
                    }
                    else{
                        v1 = ma.at<Vec3b>(i-m,j-n)[0];
                        v2 = ma.at<Vec3b>(i-m,j-n)[1];
                        v3 = ma.at<Vec3b>(i-m,j-n)[2];
                    }
                    //assert(v1!=0&&v1!=255);
                    //if(v1!=0&&v1!=255)  cout<<v1<<endl;
                    if(v1==255&&kernel[m+size][n+size]==1) result1=255;
                    if(v2==255&&kernel[m+size][n+size]==1) result2=255;
                    if(v3==255&&kernel[m+size][n+size]==1) result3=255;
                    //cout<<v1<<" "<<v2<<" "<<v3<<endl;

                }
            }
            //cout<<"result "<<result1<<" "<<result2<<" "<<result3<<endl;

            //return;
            mk.at<Vec3b>(i,j)[0]=result1;
            mk.at<Vec3b>(i,j)[1]=result2;
            mk.at<Vec3b>(i,j)[2]=result3;
        }
    }

    mk.copyTo(ma);

}

void InputGraph::binErosion(int size,int ksize,double** kernel,Mat &ma){
    int result1=0;
    int result2=0;
    int result3=0;
    Mat mk = Mat(ma.rows,ma.cols,CV_8UC3);    //内参数K Mat类型变量
    for(int i=0; i < ma.rows;i++){
        for(int j=0;j<ma.cols;j++){

            result1=255;
            result2=255;
            result3=255;
            for(int m=-size;m<=size;m++){
                for(int n=-size;n<=size;n++){
                    //超出边界部分设为0

                    int v1;
                    int v2;
                    int v3;
                    if(i-m<0||j-n<0||i-m>=ma.rows||j-n>=ma.cols){
                        v1=0;
                        v2=0;
                        v3=0;
                    }
                    else{
                        v1 = ma.at<Vec3b>(i-m,j-n)[0];
                        v2 = ma.at<Vec3b>(i-m,j-n)[1];
                        v3 = ma.at<Vec3b>(i-m,j-n)[2];
                    }
                    //assert(v1!=0&&v1!=255);
                    if(v1!=255&&kernel[m+size][n+size]==1) result1=0;
                    if(v2!=255&&kernel[m+size][n+size]==1) result2=0;
                    if(v3!=255&&kernel[m+size][n+size]==1) result3=0;
                    //cout<<v1<<" "<<v2<<" "<<v3<<endl;

                }
            }
            //cout<<"result "<<result1<<" "<<result2<<" "<<result3<<endl;

            //return;
            mk.at<Vec3b>(i,j)[0]=result1;
            mk.at<Vec3b>(i,j)[1]=result2;
            mk.at<Vec3b>(i,j)[2]=result3;
        }
    }

    mk.copyTo(ma);

}


void InputGraph::thinning(Mat& image, int size, double **kernel)
{
    Mat ma;
    image.copyTo(ma);
    hit(ma,size,kernel);
    if(MatEequals(ma,image)) cout<<23212<<endl;
    ReverseMat(ma);
    InsertMat(ma,image);
    if(MatEequals(ma,image)) cout<<12222<<endl;
    ma.copyTo(image);
}


//hit or miss kernel (0 or 255)
void InputGraph::hit(Mat& image, int size, double **kernel){
    bool leave=true;
    Mat ma;
    image.copyTo(ma);
    for(int i=size;i<ma.rows-size;i++){
        for(int j=size;j<ma.cols-size;j++){
            leave=true;
            for(int m=-size;m<=size;m++){
                for(int n=-size;n<=size;n++){
                    if(ma.at<Vec3b>(i+m,j+n)[0]!=kernel[size+n][size+m]){
                        //printf("not equals  %d %d\n",ma.at<Vec3b>(i+m,j+n)[0],(int)kernel[size+n][size+m]);

                        leave = false;
                        break;
                    }
                }
                if(leave==false) break;
            }
            //全匹配置为（黑色）
            if(leave == true) {
                //cout<<"orginal 0:"<<endl;
                //printf("%d\n",image.at<Vec3b>(i,j)[0]);
                image.at<Vec3b>(i,j)[0]=0;
                image.at<Vec3b>(i,j)[1]=0;
                image.at<Vec3b>(i,j)[2]=0;

            }
            //否则为白色
            else{
                //cout<<"orginal 0:"<<endl;
                //printf("%d %d\n",image.at<Vec3b>(i,j)[0],ma.at<Vec3b>(i,j)[0]);
                image.at<Vec3b>(i,j)[0]=255;
                image.at<Vec3b>(i,j)[1]=255;
                image.at<Vec3b>(i,j)[2]=255;
                //printf("%d %d\n",image.at<Vec3b>(i,j)[0],ma.at<Vec3b>(i,j)[0]);

            }
        }
    }
    cout<<MatEequals(image,ma)<<endl;
}

void InputGraph::ReverseMat(Mat &ma){
    for(int i=0;i<ma.cols;i++){
        for(int j=0;j<ma.rows;j++){
            if(ma.at<Vec3b>(i,j)[0]==0) {
                ma.at<Vec3b>(i,j)[0]=255;
                ma.at<Vec3b>(i,j)[1]=255;
                ma.at<Vec3b>(i,j)[2]=255;
            }
            else{
                ma.at<Vec3b>(i,j)[0]=0;
                ma.at<Vec3b>(i,j)[1]=0;
                ma.at<Vec3b>(i,j)[2]=0;
            }
        }
    }
}

void InputGraph::UnionMat(Mat &ma, Mat &b){
    for(int i=0;i<ma.cols;i++){
        for(int j=0;j<ma.rows;j++){
            if(ma.at<Vec3b>(i,j)[0]==0&&b.at<Vec3b>(i,j)[0]==0){
                ma.at<Vec3b>(i,j)[0]=0;
                ma.at<Vec3b>(i,j)[1]=0;
                ma.at<Vec3b>(i,j)[2]=0;
            }
            else{
                ma.at<Vec3b>(i,j)[0]=255;
                ma.at<Vec3b>(i,j)[1]=255;
                ma.at<Vec3b>(i,j)[2]=255;
            }

        }
    }
}


void InputGraph::InsertMat(Mat &ma, Mat &b){
    for(int i=0;i<ma.cols;i++){
        for(int j=0;j<ma.rows;j++){
            if(ma.at<Vec3b>(i,j)[0]==255&&b.at<Vec3b>(i,j)[0]==255){
                ma.at<Vec3b>(i,j)[0]=255;
                ma.at<Vec3b>(i,j)[1]=255;
                ma.at<Vec3b>(i,j)[2]=255;
            }
            else{
                ma.at<Vec3b>(i,j)[0]=0;
                ma.at<Vec3b>(i,j)[1]=0;
                ma.at<Vec3b>(i,j)[2]=0;
            }

        }
    }
}

void InputGraph::thickenning(Mat &image, int size, double **kernel){
    Mat ma;
    image.copyTo(ma);
    hit(ma,size,kernel);
    UnionMat(image,ma);
}


struct indexPair{
    int x;
    int y;
};

void InputGraph::ChessDistance(Mat &image){
    Mat ma;
    image.copyTo(ma);
    //只针对二值图，所以这里只算一个通道的值
    vector<struct indexPair> in;  //内部点集
    vector<struct indexPair> out; //外部点集
    vector<struct indexPair> notin; //非内部点
    for(int i=1;i<ma.rows-1;i++){
        for(int j=1;j<ma.cols-1;j++){
            if(ma.at<Vec3b>(i,j)[0]==0){
                if(ma.at<Vec3b>(i-1,j-1)[0]==0 && ma.at<Vec3b>(i-1,j+1)[0]==0 &&
                   ma.at<Vec3b>(i-1,j+1)[0]==0 && ma.at<Vec3b>(i+1,j+1)[0]==0){
                       struct indexPair a;
                       a.x=i;
                       a.y=j;
                       //if(i==135&&j==364) cout<<"yes"<<endl;
                       in.push_back(a);
                }
                else{
                    if(ma.at<Vec3b>(i-1,j-1)[0]==255 && ma.at<Vec3b>(i-1,j+1)[0]==255 &&
                        ma.at<Vec3b>(i-1,j+1)[0]==255 && ma.at<Vec3b>(i+1,j+1)[0]==255){
                        struct indexPair a;
                        a.x=i;
                        a.y=j;
                        notin.push_back(a);
                    }
                    else{
                        struct indexPair a;
                        a.x=i;
                        a.y=j;
                        out.push_back(a);
                        notin.push_back(a);
                    }
                }
            }
            else{
                struct indexPair a;
                a.x=i;
                a.y=j;
                notin.push_back(a);
            }
        }
    }
    vector<int> min; //记录内部点到其他的距离，内部点按公式算，其他的设为0 黑
    vector<int> max;
    for(int i=0;i<in.size();i++){
        int maxv=-1;
        int minv=0x7fffffff;
        for(int j=0;j<out.size();j++){
            int temp = abs(in[i].x-out[j].x)+abs(in[i].y-out[j].y);
            //cout<<temp<<endl;
            if(temp<minv) minv=temp;

            if(temp>maxv) maxv=temp;
        }
        min.push_back(minv);
        max.push_back(maxv);
    }

    int maxv=-1;
    int minv=0x0fffffff;
    for(int i=0;i<min.size();i++){
        //cout<<min[i]<<" "<<max[i]<<endl;
        if(min[i]>maxv) maxv=min[i];
        if(min[i]<minv) minv=min[i];
    }
    //assert(maxv==minv);

    //cout<<maxv<<" "<<minv<<endl;

    for(int i=0;i<in.size();i++){
        int x=in[i].x;
        int y=in[i].y;
        //assert(maxv==minv);

        int val=255*abs(min[i]-minv)/(maxv-minv);
        image.at<Vec3b>(x,y)[0]=val;
        image.at<Vec3b>(x,y)[1]=val;
        image.at<Vec3b>(x,y)[2]=val;
    }

    for(int i=0;i<notin.size();i++){
        int x=notin[i].x;
        int y=notin[i].y;
        int val=0;
        image.at<Vec3b>(x,y)[0]=val;
        image.at<Vec3b>(x,y)[1]=val;
        image.at<Vec3b>(x,y)[2]=val;
    }

    //TODO：处理外部点

}

void InputGraph::CityDistance(Mat & image){
    Mat ma;
    image.copyTo(ma);
    //只针对二值图，所以这里只算一个通道的值
    vector<struct indexPair> in;  //内部点集
    vector<struct indexPair> out; //外部点集
    vector<struct indexPair> notin; //非内部点
    for(int i=1;i<ma.rows-1;i++){
        for(int j=1;j<ma.cols-1;j++){
            if(ma.at<Vec3b>(i,j)[0]==0){
                if(ma.at<Vec3b>(i-1,j-1)[0]==0 && ma.at<Vec3b>(i-1,j+1)[0]==0 &&
                   ma.at<Vec3b>(i-1,j+1)[0]==0 && ma.at<Vec3b>(i+1,j+1)[0]==0){
                       struct indexPair a;
                       a.x=i;
                       a.y=j;
                       //if(i==135&&j==364) cout<<"yes"<<endl;
                       in.push_back(a);
                }
                else{
                    if(ma.at<Vec3b>(i-1,j-1)[0]==255 && ma.at<Vec3b>(i-1,j+1)[0]==255 &&
                        ma.at<Vec3b>(i-1,j+1)[0]==255 && ma.at<Vec3b>(i+1,j+1)[0]==255){
                        struct indexPair a;
                        a.x=i;
                        a.y=j;
                        notin.push_back(a);
                    }
                    else{
                        struct indexPair a;
                        a.x=i;
                        a.y=j;
                        out.push_back(a);
                        notin.push_back(a);
                    }
                }
            }
            else{
                struct indexPair a;
                a.x=i;
                a.y=j;
                notin.push_back(a);
            }
        }
    }
    vector<int> min; //记录内部点到其他的距离，内部点按公式算，其他的设为0 黑
    vector<int> max;
    for(int i=0;i<in.size();i++){
        int maxv=-1;
        int minv=0x7fffffff;
        for(int j=0;j<out.size();j++){
            int temp = (abs(in[i].x-out[j].x)>abs(in[i].y-out[j].y))?abs(in[i].x-out[j].x):abs(in[i].y-out[j].y);

            //cout<<temp<<endl;
            if(temp<minv) minv=temp;

            if(temp>maxv) maxv=temp;
        }
        min.push_back(minv);
        max.push_back(maxv);
    }

    int maxv=-1;
    int minv=0x0fffffff;
    for(int i=0;i<min.size();i++){
        //cout<<min[i]<<" "<<max[i]<<endl;
        if(min[i]>maxv) maxv=min[i];
        if(min[i]<minv) minv=min[i];
    }
    //assert(maxv==minv);

    //cout<<maxv<<" "<<minv<<endl;

    for(int i=0;i<in.size();i++){
        int x=in[i].x;
        int y=in[i].y;
        //assert(maxv==minv);

        int val=255*abs(min[i]-minv)/(maxv-minv);
        image.at<Vec3b>(x,y)[0]=val;
        image.at<Vec3b>(x,y)[1]=val;
        image.at<Vec3b>(x,y)[2]=val;
    }

    for(int i=0;i<notin.size();i++){
        int x=notin[i].x;
        int y=notin[i].y;
        int val=0;
        image.at<Vec3b>(x,y)[0]=val;
        image.at<Vec3b>(x,y)[1]=val;
        image.at<Vec3b>(x,y)[2]=val;
    }

    //TODO：处理外部点
}

//用K3M算法
void skeleton(cv::Mat &Input) //Input-binary image
{
    int a0[]={1,2,3,4,5,6};
    int a1[]={2};
    int a2[]={2,3};
    int a3[]={2,3,4};
    int a4[]={2,3,4,5};
    int a5[]={2,3,4,5,6};
    vector<int> A0=GetFlags(a0,6);

    vector<int> A1=GetFlags(a1,1);

    vector<int> A2=GetFlags(a2,2);
    vector<int> A3=GetFlags(a3,3);
    vector<int> A4=GetFlags(a4,4);
    vector<int> A5=GetFlags(a5,5);
    vector<cv::Point2i> border;
    bool modify=true;
    int neighbour[3][3]={
        {128,1,2},
        {64,0,4},
        {32,16,8}
    };
    int row=Input.rows;
    int col=Input.cols;
    while(modify)
    {
        modify=false;
        // flag the border Pharse 0
        for(int m=1;m<row-1;++m)
        {
            for(int n=1;n<col-1;++n)
            {
                int weight=0;
                for(int j=-1;j<=1;++j)
                {
                    for(int k=-1;k<=1;k++)
                    {
                        weight+=neighbour[j+1][k+1]*Input.at<uchar>(m+j,n+k);
                    }
                }
                if(std::find(A0.begin(),A0.end(),weight)!=A0.end())
                    border.push_back(cv::Point2i(m,n));
            }
        }
        //Pharse 1
        vector<cv::Point2i>::iterator first=border.begin();
        while(first!=border.end())
        {
            int weight=0;
            for(int j=-1;j<=1;++j)
            {
                for(int k=-1;k<=1;k++)
                {
                    weight+=neighbour[j+1][k+1]*Input.at<uchar>((*first).x+j,(*first).y+k);
                }
            }
            if(std::find(A1.begin(),A1.end(),weight)!=A1.end())
            {
                Input.at<uchar>((*first).x,(*first).y)=0;
                first=border.erase(first);
            }
            else
                ++first;
        }
        //Pharse2
        first=border.begin();
        while(first!=border.end())
        {
            int weight=0;
            for(int j=-1;j<=1;++j)
            {
                for(int k=-1;k<=1;k++)
                {
                    weight+=neighbour[j+1][k+1]*Input.at<uchar>((*first).x+j,(*first).y+k);
                }
            }
            if(std::find(A2.begin(),A2.end(),weight)!=A2.end())
            {
                Input.at<uchar>((*first).x,(*first).y)=0;
                first=border.erase(first);
            }
            else
                ++first;
        }
        //Pharse3
        first=border.begin();
        while(first!=border.end())
        {
            int weight=0;
            for(int j=-1;j<=1;++j)
            {
                for(int k=-1;k<=1;k++)
                {
                    weight+=neighbour[j+1][k+1]*Input.at<uchar>((*first).x+j,(*first).y+k);
                }
            }
            if(std::find(A3.begin(),A3.end(),weight)!=A3.end())
            {
                Input.at<uchar>((*first).x,(*first).y)=0;
                first=border.erase(first);
            }
            else
                ++first;
        }
        //Pharse4
        first=border.begin();
        while(first!=border.end())
        {
            int weight=0;
            for(int j=-1;j<=1;++j)
            {
                for(int k=-1;k<=1;k++)
                {
                    weight+=neighbour[j+1][k+1]*Input.at<uchar>((*first).x+j,(*first).y+k);
                }
            }
            if(std::find(A4.begin(),A4.end(),weight)!=A4.end())
            {
                Input.at<uchar>((*first).x,(*first).y)=0;
                first=border.erase(first);
            }
            else
                ++first;
        }
        //Pharse5
        first=border.begin();
        while(first!=border.end())
        {
            int weight=0;
            for(int j=-1;j<=1;++j)
            {
                for(int k=-1;k<=1;k++)
                {
                    weight+=neighbour[j+1][k+1]*Input.at<uchar>((*first).x+j,(*first).y+k);
                }
            }
            if(std::find(A5.begin(),A5.end(),weight)!=A5.end())
            {
                Input.at<uchar>((*first).x,(*first).y)=0;
                first=border.erase(first);
                modify=true;
            }
            else
                ++first;
        }
        //Pharse6
        border.clear();
    }
    for(int m=1;m<row-1;++m)
    {
        for(int n=1;n<col-1;++n)
        {
            int weight=0;
            for(int j=-1;j<=1;++j)
            {
                for(int k=-1;k<=1;k++)
                {
                    weight+=neighbour[j+1][k+1]*Input.at<uchar>(m+j,n+k);
                }
            }
            if(std::find(A0.begin(),A0.end(),weight)!=A0.end())
                Input.at<uchar>(m,n)=0;;
        }
    }

}

void InputGraph::Skelton2(Mat& raw){
    cv::Mat image(raw.rows,raw.cols,CV_8UC1);
    cv::cvtColor(raw,image,CV_RGB2GRAY);
    cv::Mat binaryImage(image.rows,image.cols,CV_8UC1);
    cv::threshold(image,binaryImage,150,1,CV_THRESH_BINARY_INV);

    //cout<<2333<<endl;
    skeleton(binaryImage);
    for(int p=0;p<binaryImage.rows;p++)
    {
        for(int q=0;q<binaryImage.cols;q++)
        {
            if(binaryImage.at<uchar>(p,q)==1)
                binaryImage.at<uchar>(p,q)=0;
            else
                binaryImage.at<uchar>(p,q)=255;
        }
    }
    cv::imshow("output",binaryImage);

    binaryImage.copyTo(image);
}


