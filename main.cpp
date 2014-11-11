#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <opencv/cv.h>
#include <opencv/ml.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

using namespace cv;

struct options {
    double ml, bl, mh, bh;
    std::vector<double> ly, hy;
};

std::vector<int> findPeaks(std::vector<int> hist) {
    int max = hist[0], maxi = 0;
    std::vector<int> localMaxima;
    for (unsigned int i = 0; i < hist.size(); i++) {
        if (hist[i] > max) {
            max = hist[i];
            maxi = i;
        }
        if ((i <= 0 || hist[i] >= hist[i-1]) && (i >= hist.size() - 1 || hist[i] >= hist[i+1]) && (hist[i] > 0)) {
            localMaxima.push_back(i);
        }
    }
    std::cout << "maximacount: " << localMaxima.size() << std::endl;
    int maxSquare = 0;
    int maxSquarei = 0;
    for (unsigned int i = 0; i < localMaxima.size(); i++) {
        if (localMaxima[i] == maxi)
            continue;
        
        int startx, stepx, height = hist[localMaxima[i]];
        // left or right side
        if (localMaxima[i] < maxi) {
            startx = localMaxima[i] + 1;
            stepx = 1;
        } else {
            startx = localMaxima[i] - 1;
            stepx = -1;
        }
        
        int square = 0;
        for (int x = startx; x != maxi; x += stepx) {
            if (hist[x] >= height)
                break;
            square += height - hist[x];
        }
        std::cout << "i: " << i << ", square: " << square << std::endl;
        if (square > maxSquare) {
            maxSquare = square;
            maxSquarei = i;
        }
    }
    std::cout << "maxSquarei: " << maxSquarei << ", maxSquare: " << maxSquare << std::endl;
    std::vector<int> peaks;
    peaks.push_back(maxi);
    peaks.push_back(localMaxima[maxSquarei]); 
    
    return peaks;
}

Vec3b color;
Vec3b antialiase(Mat img, double x, double y) {
    //std::cout << "x: " << x << ", y: " << y << ", rows: " << img.rows << ", cols: " << img.cols << std::endl;
    int xi = x;
    int yi = y;
    double xw = x - xi;
    double yw = y - yi;
    color=0;
    if (x >= 0 && x < img.cols && y >= 0 && y < img.rows - 1) {
        color += img.at<Vec3b>(Point(x, y)) * (1 - xw) * (1 - yw);
        color += img.at<Vec3b>(Point(x+1, y)) * xw * (1 - yw);
        color += img.at<Vec3b>(Point(x, y+1)) * (1 - xw) * yw;
        color += img.at<Vec3b>(Point(x+1, y+1)) * xw * yw;
    } 
    return color;
}

void straighten(VideoCapture& srcVideo, VideoWriter& outVideo, options opt) {//double ml, double bl, double mh, double bh, std::vector<double>& hy, std::vector<double>& ly) {
    double ml = opt.ml, bl = opt.bl, mh = opt.mh, bh = opt.bh;
    std::vector<double> ly = opt.ly;
    std::vector<double> hy = opt.hy;
    Mat frame;
    
    //namedWindow("view", CV_WINDOW_NORMAL);
    int counter = 0;
    double xl, xh, xrange, hyl, lyl, yrange, oy, ox, x, y;
    
    bool hasFrame = srcVideo.read(frame);
    Mat tempframe = frame.clone();
    //Mat tempframe = Mat(frame);
    while(true) {
        std::cout << "calculating frame " << counter++ << std::endl;
        if (hasFrame == false) 
            break;
        //tempframe = frame.clone();
        //tempframe = Mat(frame);
        for (y = 0; y < frame.rows; y++) {
            xl = ml * y + bl;
            xh = mh * y + bh;
            xrange = xh - xl;
            for (x = 0; x < frame.cols; x++) {
                ox = (x / frame.cols) * xrange + xl;
                tempframe.at<Vec3b>(Point(x, y)) = antialiase(frame, ox, y);
            }
        }
	//std::cout << "foo" << std::endl;
        for (x = 0; x < frame.cols; x++) {
            hyl = hy[x];
            lyl = ly[x];
            yrange = hyl - lyl;
            for (y = 0; y < frame.rows; y++) {
                oy = (y / frame.rows) * yrange + lyl;
                frame.at<Vec3b>(Point(x, y)) = antialiase(tempframe, x, oy);
            }
        }
        outVideo.write(frame);
        //imshow("view", newframe);
        //waitKey(1);
        hasFrame = srcVideo.read(frame);
    }
}

options getParameters(const Mat& img) {
    Mat grey;
    cvtColor(img, grey, CV_BGR2GRAY);
    
    std::vector<int> hist(256);
    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
            //std::cout << (int)grey.at<uchar>(Point(i,j)) << std::endl;
            uchar brightness = grey.at<uchar>(Point(i, j));
            hist[brightness]++;
            //Vec3b color = img.at<Vec3b>(Point(i, j));
            //std::cout << color << std::endl;
        }
    }
    
    Mat histimg(256, 256, CV_8UC3);
    int min = hist[0], max = hist[0];
    for (unsigned int i = 0; i < hist.size(); i++) {
        if (hist[i] < min)
            min = hist[i];
        if (hist[i] > max)
            max = hist[i];
    } 
    int range = max - min;
    //std::cout << "min: " << (int)min << ", max: " << (int)max << ", range: " << (int)range << std::endl;
    
    std::vector<int> peaks = findPeaks(hist);
    
    int border = (peaks[0] + peaks[1]) / 2;
    
    for (unsigned int i = 0; i < hist.size(); i++) {
        for (unsigned int j = 0; j < (double)hist[i] / range * 255; j++) {
           histimg.at<Vec3b>(Point(i, 255-j)) = Vec3b(255, 0, 0); 
        }
    }
    for (unsigned int i = 0; i < peaks.size(); i++) {
        histimg.at<Vec3b>(Point(peaks[i], 255)) = Vec3b(0, 255, 0);
    }
    histimg.at<Vec3b>(Point(border, 255)) = Vec3b(0, 0, 255);
   
    std::vector<Point> brightPoints;
    for (int x = 0; x < img.cols; x++) {
        for (int y = 0; y < img.rows; y++) {
            Point p(x, y);
            uchar brightness = grey.at<uchar>(p);
            if (brightness > border) {
                brightPoints.push_back(p);
            }
        }
    }
    
    Point ll, lh, hl, hh;
    int llMaxLength = img.cols * img.cols * 2, lhMaxLength = img.cols * img.cols * 2, hlMaxLength = img.cols * img.cols * 2, hhMaxLength = img.cols * img.cols * 2;
    
    for (unsigned int i = 0; i < brightPoints.size(); i++) {
        Point p = brightPoints[i];
        int llLength = p.x * p.x + p.y * p.y;
        int lhLength = p.x * p.x + (img.rows - p.y) * (img.rows - p.y);
        int hlLength = (img.cols - p.x) * (img.cols - p.x) + p.y * p.y;
        int hhLength = (img.cols - p.x) * (img.cols - p.x) + (img.rows - p.y) * (img.rows - p.y);
        //std::cout << "ll: " << llLength << ", lh: " << lhLength << ", hlLength: " << hlLength << ", hhLength: " << hhLength << std::endl;
        if (llLength < llMaxLength) {
            llMaxLength = llLength;
            ll = p;
        }
        if (lhLength < lhMaxLength) {
            lhMaxLength = lhLength;
            lh = p;
        }
        if (hlLength < hlMaxLength) {
            hlMaxLength = hlLength;
            hl = p;
        }
        if (hhLength < hhMaxLength) {
            hhMaxLength = hhLength;
            hh = p;
        }

    }
    
    //std::cout << "ll: " << ll << std::endl;
    //std::cout << "lh: " << lh << std::endl;
    //std::cout << "hl: " << hl << std::endl;
    //std::cout << "hh: " << hh << std::endl;
    
    //img.at<Vec3b>(ll) = Vec3b(255, 0, 0);
    //img.at<Vec3b>(lh) = Vec3b(255, 0, 0);
    //img.at<Vec3b>(hl) = Vec3b(255, 0, 0);
    //img.at<Vec3b>(hh) = Vec3b(255, 0, 0);
    
    // f(y) = m*y+b
    double ml = ((double)lh.x - ll.x) / (lh.y - ll.y);
    double bl = ll.x;
    double mh = ((double)hh.x - hl.x) / (hh.y - hl.y);
    double bh = hl.x;
    
    
    //std::cout << "ml: " << ml << ", bl: " << bl << std::endl;
    //std::cout << "mh: " << mh << ", bh: " << bh << std::endl;
     
    Mat borderimg(img.rows, img.cols, CV_8UC3);
    
    for (int y = 0; y < img.rows; y++) {
        int xl = ml * y + bl;
        int xh = mh * y + bh;
        borderimg.at<Vec3b>(Point(xl, y)) = Vec3b(255, 0, 0);
        borderimg.at<Vec3b>(Point(xh, y)) = Vec3b(255, 0, 0);
    }
    
    Mat tempimg = img.clone();
    
    double xl, xh, xrange, ox;
    for (int y = 0; y < img.rows; y++) {
        xl = ml * y + bl;
        xh = mh * y + bh;
        xrange = xh - xl;
        for (int x = 0; x < img.cols; x++) {
            ox = ((double)x / img.cols) * xrange + xl;
            tempimg.at<Vec3b>(Point(x, y)) = antialiase(img, ox, y);
        }
    }
    
    Mat newimg = tempimg.clone();
    Mat tempgrey;
    cvtColor(tempimg, tempgrey, CV_BGR2GRAY);
    
    std::vector<double> ly, hy;
    
    double lyl, oy, hyl;
    for (int x = 0; x < img.cols; x++) {
        lyl = img.rows - 1;
	hyl = 0;
        for (int y = 0; y < img.rows; y++) {
            int brightness = tempgrey.at<uchar>(Point(x, y));
            if (brightness > border) {
                if(y < lyl) {
                    lyl = y;
                } else if(y > hyl) {
                    hyl = y;
                }
            } 
        }
        //std::cout << "ly: " << ly << ", hy: " << hy << std::endl;
        double range = hyl - lyl;
        ly.push_back(lyl);
        hy.push_back(hyl);
        //tempimg.at<Vec3b>(Point(x, ly)) = Vec3b(0, 255, 0);
        //tempimg.at<Vec3b>(Point(x, hy)) = Vec3b(0, 0, 255);
        for (int y = 0; y < img.rows; y++) {
            oy = ((double)y / img.rows) * range + lyl;
            //std::cout << "x: " << x << ", y: " << y << ", oy: " << oy << std::endl;
            newimg.at<Vec3b>(Point(x, y)) = antialiase(tempimg, x, oy);
        }
    }
    
    
    
    namedWindow("image", CV_WINDOW_NORMAL);
    imshow("image", img);
    
    //namedWindow("grey", CV_WINDOW_NORMAL);
    //imshow("grey", grey);
    
    //namedWindow("hist", CV_WINDOW_NORMAL);
    //imshow("hist", histimg);
    
    //namedWindow("border", CV_WINDOW_NORMAL);
    //imshow("border", borderimg);
    
    //namedWindow("temp", CV_WINDOW_NORMAL);
    //imshow("temp", tempimg);

    //namedWindow("tempgrey", CV_WINDOW_NORMAL);
    //imshow("tempgrey", tempgrey);
    
    namedWindow("new", CV_WINDOW_NORMAL);
    imshow("new", newimg);
     
    options opt;
    opt.ml = ml;
    opt.bl = bl;
    opt.mh = mh;
    opt.bh = bh;
    opt.ly = ly;
    opt.hy = hy;
    
    return opt;
}

void saveOptions(options opt, std::string filename) {
    std::ofstream outfile(filename.c_str());
    outfile << opt.ml << std::endl << opt.bl << std::endl << opt.mh << std::endl << opt.bh << std::endl;
    outfile << "ly" << std::endl;
    for (unsigned int i = 0; i < opt.ly.size(); i++) {
        outfile << opt.ly[i] << std::endl;
    }
    outfile << "hy" << std::endl;
    for (unsigned int i = 0; i < opt.hy.size(); i++) {
        outfile << opt.hy[i] << std::endl;
    }  
    outfile << "end" << std::endl;
    outfile.close();
}

options loadOptions(std::string filename) {
    std::ifstream infile(filename.c_str());
    double ml, bl, mh, bh;
    std::vector<double> ly, hy;
    infile >> ml;
    infile >> bl;
    infile >> mh;
    infile >> bh;
    std::string temps;
    std::vector<double>* tempvec;
    while (true) {
        infile >> temps;
        if (temps == "ly") {
            tempvec = &ly;
        } else if (temps == "hy") {
            tempvec = &hy;
        } else if (temps == "end") {
            break;
        } else {
            tempvec->push_back(std::stof(temps));
        }
    }
    
    infile.close();
    options opt;
    opt.ml = ml;
    opt.bl = bl;
    opt.mh = mh;
    opt.bh = bh;
    opt.ly = ly;
    opt.hy = hy;

    return opt;
}

int main(int argc, char** argv) {
    // program optfile img
    if (argc == 3) {
        options opt;
        Mat img = imread(std::string(argv[2]));
        opt = getParameters(img);
        std::cout << "got opt: " << opt.ml << std::endl;
        saveOptions(opt, std::string(argv[1]));
    } 
    // program optfile source out
    else if (argc == 4) {
        options opt = loadOptions(std::string(argv[1]));
        
        VideoCapture cap;
        cap.open(std::string(argv[2]));
        VideoWriter wri;
        wri.open(std::string(argv[3]), CV_FOURCC('M', 'J', 'P', 'G'), 25, Size(1920, 1080));
//	std::cout << wri.isOpened();
        std::cout << "straightening" << std::endl;
        straighten(cap, wri, opt);
        std::cout << "done" << std::endl;
        
    }
    
}
