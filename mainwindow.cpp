#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <math.h>
#include <algorithm>
using namespace std;
using namespace cv;


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->setMouseTracking(true);
    ui->centralWidget->setMouseTracking(true);
    ui->scrollArea->setMouseTracking(true);
    ui->scrollAreaWidgetContents->setMouseTracking(true);
    ui->labelimage->setMouseTracking(true);
}

MainWindow::~MainWindow()
{
    delete ui;
}

QImage cvMat2QImage(const cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if(mat.type() == CV_8UC4)
    {
        qDebug() << "CV_8UC4";
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}


//显示图像
void MainWindow::DisplayMat(cv::Mat image)
{
    int width = image.cols;
    int height = image.rows;

    cvgraph.img = cvMat2QImage(image);

    ui->scrollAreaWidgetContents->setMinimumHeight(height);
    ui->scrollAreaWidgetContents->setMinimumWidth(width);

    ui->labelimage->setMinimumHeight(height);
    ui->labelimage->setMinimumWidth(width);

    QString sizeStr;
    sizeStr.sprintf("%d * %d px", width, height);

    ///TODO
    ui->labelpoishow_2->setText(sizeStr);

    ui->labelimage->setAlignment(Qt::AlignHCenter|Qt::AlignTop);
    ui->labelimage->setPixmap(QPixmap::fromImage(cvgraph.img));//setPixelmap(QPixmap::fromImage(img).scaled(ui->labelimage->size(),Qt::KeepAspectRatio));
    cvgraph.xoffset = (ui->labelimage->contentsRect().width()-ui->labelimage->pixmap()->rect().width())/2;
    cvgraph.yoffset = (ui->labelimage->contentsRect().height()-ui->labelimage->pixmap()->rect().height())/2 + 135;
}


//读取图片
void MainWindow::on_open_clicked()
{

    QString filename = QFileDialog::getOpenFileName(this,tr("Open Image"),".",tr("Image File (*.jpg *.png *.bmp)"));
    QTextCodec *code = QTextCodec::codecForName("gb18030");
    cvgraph.name = code->fromUnicode(filename).data();//filename.toAscii().data()
    cvgraph.image = cv::imread(cvgraph.name);
    cvgraph.original = cvgraph.image.clone();
    cvgraph.show = cvgraph.image.clone();
    if(!cvgraph.image.data)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Image Data Is Null"));
        msgBox.exec();
    }
    else {
       DisplayMat(cvgraph.image);
       //cvgraph.playnum = 1;
    }
}

void MainWindow::on_save_clicked()
{
    if(!cvgraph.image.data)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Image Data Is Null"));
        msgBox.exec();
    }
    else{
        cv::imwrite(cvgraph.name,cvgraph.image);
    }
}



void MainWindow::on_saveAs_clicked()
{
    if(!cvgraph.image.data)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Image Data Is Null"));
        msgBox.exec();
    }
    else{
        QString filename = QFileDialog::getSaveFileName(this,tr("Save Image"),"",tr("Images (*.png *.bmp *.jpg)")); //选择路径
        QTextCodec *code = QTextCodec::codecForName("gb18030");
        std::string fileAsSave = code->fromUnicode(filename).data();
        //std::string fileAsSave = filename.toStdString();
        cv::imwrite(fileAsSave,cvgraph.image);
    }
}


void MainWindow::mouseMoveEvent(QMouseEvent *event){
    QPoint globalpoi = ui->scrollAreaWidgetContents->pos();

    QPoint temp=ui->labelimage->pos();
    //处理未加载图片的情况
    if(cvgraph.original.empty()) return;
    int x,y;
    x = event->pos().x() - temp.x() - globalpoi.x()-cvgraph.xoffset-61;
    y = event->pos().y() - globalpoi.y() - temp.y()-215;

    if (x >= 0 && x < ui->labelimage->pixmap()->rect().width() &&
        y >= 0 && y < ui->labelimage->pixmap()->rect().height())
    {
        vector<int> pxBGR = cvgraph.getPixelVal(x,y);
        QString labelinfo;
        labelinfo.sprintf( "x=%d, y=%d, RGB=(%d,%d,%d)",
            x, y, pxBGR[0], pxBGR[1], pxBGR[2]);
        ui->labelpoishow->setText(labelinfo);
    }
    else{
        ui->labelpoishow->setText( "" );
    }
}


void MainWindow::on_dilation_clicked()
{
    int typeIndex = ui->structType->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            printf("%d\n",ksize);

            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }

    cvgraph.Dilation(size,ksize,kernel);
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;

}

void MainWindow::on_erosion_clicked()
{
    int typeIndex = ui->structType->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }

    cvgraph.Erosion(size,ksize,kernel);
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;
}




void MainWindow::on_open_2_clicked()
{
    int typeIndex = ui->structType->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }

    cvgraph.open(size,ksize,kernel);
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;

}

void MainWindow::on_close_clicked()
{
    int typeIndex = ui->structType->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }


    cvgraph.close(size,ksize,kernel);
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;
}

void MainWindow::on_openReconstruct_clicked()
{
    Mat m;
    cvgraph.image.copyTo(m);
    int typeIndex = ui->structType->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }


    cvgraph.openReconstruct(size,ksize,kernel,5,m);
    //展示结果
    DisplayMat(m);
    cout<<MatEequals(m,cvgraph.image)<<endl;
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;
}

void MainWindow::on_watershed_clicked()
{
    cvgraph.watershedCal();
}

//二值化图形膨胀
void MainWindow::on_dilation_2_clicked()
{
    int typeIndex = ui->structType_2->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            //printf("%d\n",ksize);

            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.binDilation(size,ksize,kernel,cvgraph.image);
    //cvgraph.binDilation(size,ksize,kernel,ma);
    cout<<MatEequals(cvgraph.image,ma)<<endl;
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;


}

void MainWindow::on_erosion_2_clicked()
{
    int typeIndex = ui->structType_2->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            //printf("%d\n",ksize);

            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.binErosion(size,ksize,kernel,cvgraph.image);
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;
}


//thin
void MainWindow::on_hilditch_clicked()
{
    int typeIndex = ui->structType_2->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            //printf("%d\n",ksize);

            double val = 255.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择十字形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 255.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }
    Mat ma;
    cvgraph.image.copyTo(ma);

    ///TODO
    //assert(!MatEequals(ma,cvgraph.image));
    if(MatEequals(ma,cvgraph.image)) cout<<1222<<endl;
    cvgraph.thinning(cvgraph.image,size,kernel);
    //cout<<1<<endl;
    if(MatEequals(ma,cvgraph.image)) cout<<2333<<endl;
    DisplayMat(cvgraph.image);
}

void MainWindow::on_open_3_clicked()
{
    int typeIndex = ui->structType_2->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            //printf("%d\n",ksize);

            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.binOpen(size,ksize,kernel,cvgraph.image);
    //cvgraph.binDilation(size,ksize,kernel,ma);
    cout<<MatEequals(cvgraph.image,ma)<<endl;
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;


}


void MainWindow::on_thick_clicked()
{
    int typeIndex = ui->structType_2->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec

    //kernel的构造不一样
    if(typeIndex==1){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            //printf("%d\n",ksize);

            double val = 0.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择十字形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 0.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=255;
                }
            }

        }

    }
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.thickenning(cvgraph.image,size,kernel);
    //cvgraph.hit(cvgraph.image,size,kernel);
    DisplayMat(cvgraph.image);
}

void MainWindow::on_distance_clicked()
{
    int index = ui->distance_type->currentIndex();
    Mat ma;
    cvgraph.image.copyTo(ma);

    if(index==1){
        cvgraph.ChessDistance(ma);
    }

    if(index==2){
        cvgraph.CityDistance(ma);
    }

    DisplayMat(ma);
}


void MainWindow::on_close_2_clicked()
{
    int typeIndex = ui->structType_2->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            //printf("%d\n",ksize);

            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.binClose(size,ksize,kernel,cvgraph.image);
    //cvgraph.binDilation(size,ksize,kernel,ma);
    cout<<MatEequals(cvgraph.image,ma)<<endl;
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;


}

void MainWindow::on_skeleton_clicked()
{
    int typeIndex = ui->structType_2->currentIndex();
    if(typeIndex==0){
        QMessageBox msgBox;
        msgBox.setText(tr("请选择结构体类型"));
        msgBox.exec();
        return;
    }
    int size;
    double **kernel;
    int ksize;
    //Rec
    if(typeIndex==1){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            //printf("%d\n",ksize);

            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    kernel[i][j] = val;
                }
            }
        }

    }
    if(typeIndex==2){
        size = ui->RecSize_2->currentIndex();
        if(size==0){
            QMessageBox msgBox;
            msgBox.setText(tr("请选择矩形结构体大小"));
            msgBox.exec();
            return;
        }
        else{
            ksize = 2*size+1;
            double val = 1.0;

            kernel = (double **)malloc(ksize * sizeof(double*));
            for (int i = 0; i < ksize; i++) {
                kernel[i] = (double *)malloc(ksize * sizeof(double));
                for (int j = 0; j < ksize; j++) {
                    if(j==size)kernel[i][j] = val;
                    else if(i==size)kernel[i][j]=val;
                    else kernel[i][j]=0;
                }
            }

        }

    }
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.Skelton2(ma);
    //cvgraph.binDilation(size,ksize,kernel,ma);
    //cout<<MatEequals(cvgraph.image,ma)<<endl;
    ma.copyTo(cvgraph.image);
    DisplayMat(cvgraph.image);
    for (int i = 0; i < ksize; i++) {
        delete kernel[i];
    }
    delete kernel;

}

void MainWindow::on_showGraph_clicked()
{
    int index=ui->graphType->currentIndex();
    if(index==4){
        Mat ma;
        cvgraph.image.copyTo(ma);
        cvgraph.gray(ma);
        DisplayMat(ma);
        ma.copyTo(cvgraph.image);
    }
    else if(index!=0){
        Mat ma;
        cvgraph.image.copyTo(ma);
        //BGR
        cvgraph.channelSplit(index-1, ma);
        DisplayMat(ma);
    }
}



void MainWindow::on_horizontalSlider_H_valueChanged(int value)
{
    cvgraph.HSV_H(value,cvgraph.image);
    DisplayMat(cvgraph.image);
    QString poi;
    poi.sprintf( "%d",value);
    ui->label_h->setText(poi);
}


void MainWindow::on_horizontalSlider_S_valueChanged(int value)
{
    cvgraph.HSV_S(value,cvgraph.image);
    DisplayMat(cvgraph.image);
    QString poi;
    poi.sprintf( "%d",value);
    ui->label_s->setText(poi);
}



void MainWindow::on_horizontalSlider_V_valueChanged(int value)
{
    cvgraph.HSV_V(value,cvgraph.image);
    DisplayMat(cvgraph.image);
    QString poi;
    poi.sprintf( "%d",value);
    ui->label_v->setText(poi);
}

void MainWindow::on_otus_clicked()
{
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.toOtus();
    DisplayMat(ma);
}

void MainWindow::on_yu1_valueChanged(int value)
{
    int value_2 = ui->yu2->value();
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.thre2binary(value,value_2,ma);
    DisplayMat(cvgraph.show);
    QString poi;
    poi.sprintf( "%d",value);
    ui->label_yu1->setText(poi);
}



void MainWindow::on_yu2_valueChanged(int value)
{
    int value_1 = ui->yu1->value();
    Mat ma;
    cvgraph.image.copyTo(ma);
    cvgraph.thre2binary(value_1,value,ma);
    DisplayMat(cvgraph.show);
    QString poi;
    poi.sprintf( "%d",value);
    ui->label_yu2->setText(poi);
}



void MainWindow::on_save_2_clicked()
{
    cvgraph.show.copyTo(cvgraph.image);
}

void MainWindow::on_openSecond_clicked()
{
    //导入第二张图片

        QString filename = QFileDialog::getOpenFileName(this,tr("Open Image"),".",tr("Image File (*.jpg *.png *.bmp)"));
        QTextCodec *code = QTextCodec::codecForName("gb18030");
        std::string inputname = code->fromUnicode(filename).data();//filename.toAscii().data()
        cvgraph.second = cv::imread(inputname);
        if(!cvgraph.second.data)
        {
            QMessageBox msgBox;
            msgBox.setText(tr("Image Data Is Null"));
            msgBox.exec();
        }
        else {
            //DisplayMat(cvgraph.second);
            //ui->label_whichview->setText("This is input picture");
            //cvgraph.playnum = 3;
            if((cvgraph.second.rows!=cvgraph.image.rows) && (cvgraph.second.cols!=cvgraph.image.cols)){
                QMessageBox msgBox;
                msgBox.setText(tr("图像大小不一样"));
                msgBox.exec();
            }
        }

}



void MainWindow::on_add_clicked()
{
    if((!cvgraph.image.data)||(!cvgraph.second.data))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("图像是空的"));
        msgBox.exec();
    }
    else{
        double weight1 = ui->add_weight->value();
        double weight2 = 1.0 - weight1;


        //cvgraph.original = cvgraph.image.clone();
        cvgraph.addimage(weight1,weight2);
        DisplayMat(cvgraph.image);

    }
}

void MainWindow::on_mltipulicate_clicked()
{

    cvgraph.multiplicationimage();
    DisplayMat(cvgraph.image);

}



void MainWindow::on_minus_clicked()
{
    cvgraph.minusimage();
    DisplayMat(cvgraph.image);
}

void MainWindow::on_pushButton_4_clicked()
{
    int x1=ui->left1->value();
    int y1=ui->left2->value();
    int x2=ui->right1->value();
    int y2=ui->right2->value();
    cvgraph.cutImage(x1,y1,x2,y2,cvgraph.image);
    DisplayMat(cvgraph.image);
}



void MainWindow::on_resize1_clicked()
{
    int rate=ui->rate->value()/10;
    int value=ui->rate->value();
    cvgraph.resize1(value);
    DisplayMat(cvgraph.image);
    QString poi;
    poi.sprintf( "x%f",value/10.0);
    ui->times->setText(poi);
}

void MainWindow::on_resize2_clicked()
{
    int rate=ui->rate->value()/10;
    int value=ui->rate->value();
    cvgraph.resize2(value);
    DisplayMat(cvgraph.image);
    QString poi;
    poi.sprintf( "x%f",value/10.0);
    ui->times->setText(poi);
}

void MainWindow::on_spin1_clicked()
{
    int value=ui->dial->value();
    cvgraph.spin1(value);
    DisplayMat(cvgraph.image);
}



void MainWindow::on_spin2_clicked()
{
    int value=ui->dial->value();
    cvgraph.spin2(value);
    DisplayMat(cvgraph.image);
}



void MainWindow::on_line_clicked()
{
    int A=ui->A->value();
    int B=ui->B->value();
    int C=ui->C->value();
    int D=ui->D->value();
    cvgraph.linearTrans(A,B,C,D);
    DisplayMat(cvgraph.image);
}



void MainWindow::on_seg_clicked()
{
    int A=ui->A->value();
    int B=ui->B->value();
    int C=ui->C->value();
    int D=ui->D->value();
    cvgraph.SegTrans(A,B,C,D);
    DisplayMat(cvgraph.image);
}

void MainWindow::on_log_clicked()
{
    double A=ui->A_2->value();
    double B=ui->B_2->value();
    double C=ui->C_2->value();
    cvgraph.logTrans(A,B,C);
    DisplayMat(cvgraph.image);
}



void MainWindow::on_ex_clicked()
{
    double A=ui->A_2->value();
    double B=ui->B_2->value();
    double C=ui->C_2->value();
    cvgraph.exTrans(A,B,C);
    DisplayMat(cvgraph.image);
}

void MainWindow::on_average_clicked()
{
    cvgraph.original = cvgraph.image.clone();
    cvgraph.average();
    DisplayMat(cvgraph.image);
}



void MainWindow::on_showavg_clicked()
{
    cvgraph.showavg();
}

void MainWindow::on_avg_clicked()
{
    int size=ui->avgFilter->currentIndex();
    if(size>0){
        Mat ma;
        cvgraph.image.copyTo(ma);
        cvgraph.avgFilter(2*size+1,ma);
        DisplayMat(ma);
    }
}

void MainWindow::on_mean_clicked()
{
    int size=ui->meanFilter->currentIndex();
    if(size>0){
        Mat ma;
        cvgraph.image.copyTo(ma);
        cvgraph.meanFilter(2*size+1,ma);
        DisplayMat(ma);
    }
}



void MainWindow::on_gauss_clicked()
{
    int size=ui->GaussFilter->currentIndex();
    if(size>0){
        Mat ma;
        cvgraph.image.copyTo(ma);
        cvgraph.gaussFilter(2*size+1,ma);
        DisplayMat(ma);
    }
}

void MainWindow::on_pushButton_15_clicked()
{
    cvgraph.laplace();
    DisplayMat(cvgraph.image);
}

void MainWindow::on_canny_clicked()
{
    cvgraph.canny();
    DisplayMat(cvgraph.show);
}

void MainWindow::on_sobel_clicked()
{
    int index = ui->sobel_2->currentIndex();
    cvgraph.sobel(index);
    DisplayMat(cvgraph.show);
}

void MainWindow::on_linecheck_clicked()
{
    cvgraph.original = cvgraph.image.clone();
    cvgraph.houghline();
    DisplayMat(cvgraph.show);
    cvgraph.show.copyTo(cvgraph.image);
}

void MainWindow::on_color_clicked()
{

    int white = ui->white->value();
    int black = ui->black->value();
    int whiteout = ui->white_2->value();
    int blackout = ui->black_2->value();
    double gray = ui->gray->value();
    int index=ui->colorlv->currentIndex();
    if(index==1){
        cvgraph.original = cvgraph.image.clone();
        cvgraph.imageColorLevel(white,black,whiteout,blackout,gray,3);
        DisplayMat(cvgraph.show);
        //cvgraph.playnum = 2;
        cvgraph.image = cvgraph.show.clone();
    }
    if(index==2){
        cvgraph.original = cvgraph.image.clone();
        cvgraph.imageColorLevel(white,black,whiteout,blackout,gray,0);
        DisplayMat(cvgraph.show);
        //cvgraph.playnum = 2;
        cvgraph.image = cvgraph.show.clone();
    }
    if(index==3){
        cvgraph.original = cvgraph.image.clone();
        cvgraph.imageColorLevel(white,black,whiteout,blackout,gray,1);
        DisplayMat(cvgraph.show);
        //cvgraph.playnum = 2;
        cvgraph.image = cvgraph.show.clone();
    }
    if(index==4){
        cvgraph.original = cvgraph.image.clone();
         cvgraph.imageColorLevel(white,black,whiteout,blackout,gray,2);
         DisplayMat(cvgraph.show);
         //cvgraph.playnum = 2;
         cvgraph.image = cvgraph.show.clone();
    }
}
