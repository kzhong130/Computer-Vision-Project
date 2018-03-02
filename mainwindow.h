#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QImage>
#include <QLabel>
#include <QTextCodec>
#include <QMessageBox>
#include <QMouseEvent>
#include <QWidget>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/cvwimage.h>
#include <InputGraph.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void DisplayMat(cv::Mat image);
    void on_open_clicked();

    void on_save_clicked();

    void on_saveAs_clicked();
    void mouseMoveEvent(QMouseEvent *event);

    void on_dilation_clicked();

    void on_erosion_clicked();

    void on_open_2_clicked();

    void on_close_clicked();

    void on_openReconstruct_clicked();

    void on_watershed_clicked();

    void on_dilation_2_clicked();

    void on_erosion_2_clicked();

    void on_hilditch_clicked();

    void on_open_3_clicked();


    void on_thick_clicked();

    void on_distance_clicked();

    void on_close_2_clicked();

    void on_skeleton_clicked();

    void on_showGraph_clicked();

    void on_horizontalSlider_H_valueChanged(int value);

    void on_horizontalSlider_S_valueChanged(int value);

    void on_horizontalSlider_V_valueChanged(int value);

    void on_otus_clicked();

    void on_yu1_valueChanged(int value);

    void on_yu2_valueChanged(int value);

    void on_save_2_clicked();

    void on_openSecond_clicked();

    void on_add_clicked();

    void on_mltipulicate_clicked();

    void on_minus_clicked();

    void on_pushButton_4_clicked();


    void on_resize1_clicked();

    void on_resize2_clicked();

    void on_spin1_clicked();

    void on_spin2_clicked();

    void on_line_clicked();

    void on_seg_clicked();

    void on_log_clicked();

    void on_ex_clicked();

    void on_average_clicked();

    void on_showavg_clicked();

    void on_avg_clicked();

    void on_mean_clicked();

    void on_gauss_clicked();

    void on_pushButton_15_clicked();

    void on_canny_clicked();

    void on_sobel_clicked();

    void on_linecheck_clicked();

    void on_color_clicked();

private:
    Ui::MainWindow *ui;
    InputGraph cvgraph;

};

#endif // MAINWINDOW_H
