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

int global_threshold = -1; // -1 : otsu

double timer()
{
	return cv::getTickCount() / cv::getTickFrequency();
}

struct options {
	cv::Point2d cornertl, cornertr, cornerbl, cornerbr;
	std::vector<float> fntop, fnbottom, fnleft, fnright;

	options() {}

	options(char const *filename) { this->load(std::string(filename)); }

	options(std::string filename) { this->load(filename); }

	void save(std::string filename)
	{
		if (filename.substr(filename.length() - 4, 4) != ".xml")
			filename += ".xml";

		cv::FileStorage optfile(filename, cv::FileStorage::WRITE);
		if (!optfile.isOpened())
		{
			std::cerr << "optfile could not be opened!" << std::endl;
			std::cerr << "optfile: " << filename << std::endl;

			exit(-1);
		}

		optfile << "cornertl" << cornertl;
		optfile << "cornertr" << cornertr;
		optfile << "cornerbl" << cornerbl;
		optfile << "cornerbr" << cornerbr;

		optfile << "fntop"    << fntop;
		optfile << "fnbottom" << fnbottom;
		optfile << "fnleft"   << fnleft;
		optfile << "fnright"  << fnright;

		optfile.release();
	}

	void load(std::string filename)
	{
		if (filename.substr(filename.length() - 4, 4) != ".xml")
			filename += ".xml";

		cv::FileStorage optfile(filename, cv::FileStorage::READ);
		assert(optfile.isOpened());

		optfile["cornertl"] >> cornertl;
		optfile["cornertr"] >> cornertr;
		optfile["cornerbl"] >> cornerbl;
		optfile["cornerbr"] >> cornerbr;

		optfile["fntop"]    >> fntop;
		optfile["fnbottom"] >> fnbottom;
		optfile["fnleft"]   >> fnleft;
		optfile["fnright"]  >> fnright;

		optfile.release();
	}
};

template <typename T>
T sgn(T a) { return a >= 0 ? +1 : -1; }

int best_point(std::vector<cv::Point2i> contour, float gx, float gy)
{
	int bestindex = -1;
	float bestgain = 0;

	for (int k = 0; k < contour.size(); k += 1)
	{
		cv::Point2i &pt = contour[k];
		float const gain = gx*pt.x + gy*pt.y;

		if (gain > bestgain || bestindex == -1)
		{
			bestgain = gain;
			bestindex = k;
		}
	}

	return bestindex;
}

std::vector<cv::Point2i> slice_contour(std::vector<cv::Point2i> contour, int begin, int end)
{
	std::vector<cv::Point2i> res;
	auto const it = contour.begin();

	if (begin > end) // two segments
	{
		res.insert(res.end(), it + (begin+1), contour.end());
		res.insert(res.end(), contour.begin(), it + (end-1));
	}
	else // one segment
	{
		res.insert(res.end(), it + (begin+1), it + (end-1));
	}

	return res;
}

float linefit(int degree, int free, std::vector<cv::Point2i> points, std::vector<float> &coeffs)
{
	cv::Mat coords(points);
	coords = coords.reshape(1);
	
	cv::Mat A(points.size(), degree+1, CV_32F);
	cv::Mat b(points.size(), 1, CV_32F);

	for (int k = 0; k < points.size(); k += 1)
	{
		float const x = coords.at<int>(k, free);
		float const y = coords.at<int>(k, 1-free);

		b.at<float>(k) = y;

		float acc = 1;
		for (int deg = 0; deg <= degree; deg += 1)
		{
			A.at<float>(k, deg) = acc;
			acc *= x;
		}
	}

	Mat x;
	bool rv = cv::solve(A, b, x, DECOMP_QR);
	if (!rv)
		exit(-1);

	x.copyTo(coeffs); // coeffs

	//std::cout << residual << std::endl;
	return cv::norm(A*x, b, NORM_L1) / b.rows;
}

float evaluate(float x, std::vector<float> const &coeffs)
{
	if (coeffs.size() == 2)
		return (x * coeffs[1]) + coeffs[0];

	if (coeffs.size() == 3)
		return ((x * coeffs[2]) + coeffs[1]) * x + coeffs[0];

	float acc = 0;
	for (int k = coeffs.size()-1; k >= 0; k -= 1)
		acc = (acc * x) + coeffs[k];
	return acc;
}

cv::Point2f intersect(std::vector<float> x, std::vector<float> y)
{
	// *1) Y = x0 + x1 X + x2 X^2
	// *2) X = y0 + y1 Y => Y = (X-y0) / y1
	// *1 - *2 = 0

	//std::vector<double> x; xmat.copyTo(x);
	//std::vector<double> y; ymat.copyTo(y);

	if (abs(y[1]) < 1e-5) // to be tweaked
	{
		double const X = y[0];
		double const Y = ((x[2] * X) + x[1]) * X + x[0];
		return cv::Point2f(X, Y);
	}

	double const a = x[2];
	double const b = x[1] - 1/y[1];
	double const c = x[0] + y[0]/y[1];

	double const sD = sqrt(b*b - 4*a*c);

	double const x1 = (-b - sD) / (2*a);
	double const x2 = (-b + sD) / (2*a);

	double const X = (abs(x1) < abs(x2) ? x1 : x2);

	double const Y = ((x[2] * X) + x[1]) * X + x[0];

	return cv::Point2d(X, Y);
}

template <typename T>
void draw_polynomial(cv::Mat image, int free, int xmin, int xmax, std::vector<float> const &coeffs, T color)
{
	for (int x = xmin; x < xmax; x += 1)
	{
		float const fy = evaluate((float)x, coeffs);
		int const y = (int)(fy + 0.5);

		int const u = (free == 0 ? y : x);
		int const v = (free == 0 ? x : y);

		if (0 <= u && u < image.rows && 0 <= v && v < image.cols)
			cv::circle(image, cv::Point(v,u), 2, cv::Scalar(color), -1);
			//image.at<T>(u, v) = color;
	}
}

bool contour_size(vector<Point> first, vector<Point> second)
{
    return cv::contourArea(first) > cv::contourArea(second);
}

options find_sides(Mat const &image)
{
	options opt;

	Mat disp(image);

	Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);

	// THRESHOLDING, GET CONTOUR

	Mat tmp;
	Mat mask;
	double auto_threshold = -1;

	std::cout << "threshold input: " << global_threshold << "(" << (global_threshold == -1 ? "" : "not") << " using Otsu)" << std::endl;

	if (global_threshold == -1)
		auto_threshold = cv::threshold(gray, mask, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
	else
		auto_threshold = cv::threshold(gray, mask, global_threshold, 255, cv::THRESH_BINARY);

	std::cout << "threshold used: " << auto_threshold << std::endl;

	//cv::pyrDown(mask, tmp); cv::imshow("threshold", tmp);

	Mat kernel = cv::getStructuringElement(MORPH_RECT, cv::Size(3,3));
	cv::morphologyEx(mask, mask, cv::MORPH_GRADIENT, kernel);

	//cv::pyrDown(mask, tmp); cv::imshow("gradient", tmp);

	std::vector<std::vector<cv::Point2i>> contours;
	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	std::sort(contours.begin(), contours.end(), contour_size);

	if (contours.size() != 1)
	{
		std::cerr << "expecting exactly one contour, not " << contours.size() << ". check if this is what you want." << std::endl;

		if (contours.size() >= 1)
			cv::drawContours(disp, contours, 0, cv::Scalar(0, 255, 0), 2);

		for (int k = 1; k < contours.size(); k += 1)
			cv::drawContours(disp, contours, k, cv::Scalar(255, 0, 0), 2);

		cv::pyrDown(disp, tmp); cv::imshow("contours", tmp);

		while (cv::waitKey(100) == -1);

		cv::destroyWindow("contours");
	}

	if (contours.size() == 0)
	{
		exit(-1);
	}

	auto const &contour = contours[0];

	// CONTOUR -> FOUR SIDES
	// (find corners)

	int const topleft     = best_point(contour, -1, -1);
	int const topright    = best_point(contour, +1, -1);
	int const bottomleft  = best_point(contour, -1, +1);
	int const bottomright = best_point(contour, +1, +1);

	//std::cout << "contour " << contour.size() << " : " << topleft << " " << topright << " " << bottomright << " " << bottomleft << std::endl;

	// test direction of contour
	int direction = sgn(topright - topleft) + sgn(bottomright - topright) + sgn(bottomleft - bottomright) + sgn(topleft - bottomleft);
	//std::cout << "direction " << direction << std::endl;

	std::vector<cv::Point2i> ctop, cbottom, cleft, cright;

	if (direction < 0)
	{
		ctop    = slice_contour(contour, topright,    topleft);
		cleft   = slice_contour(contour, topleft,     bottomleft);
		cbottom = slice_contour(contour, bottomleft,  bottomright);
		cright  = slice_contour(contour, bottomright, topright);
	}
	else if (direction > 0)
	{
		ctop    = slice_contour(contour, topleft,     topright);
		cright  = slice_contour(contour, topright,    bottomright);
		cbottom = slice_contour(contour, bottomright, bottomleft);
		cleft   = slice_contour(contour, bottomleft,  topleft);
	}

	std::vector<float> ftop, fbottom, fleft, fright;
	float etop, ebottom, eleft, eright;
	etop    = linefit(2, 0, ctop,    ftop);
	ebottom = linefit(2, 0, cbottom, fbottom);
	eleft   = linefit(1, 1, cleft,   fleft);
	eright  = linefit(1, 1, cright,  fright);

	std::cout << "top fit: " << Mat(ftop).t() << std::endl;
	std::cout << "top error: " << etop << std::endl;
	std::cout << "bottom fit: " << Mat(fbottom).t() << std::endl;
	std::cout << "bottom error: " << ebottom << std::endl;
	std::cout << "left fit: " << Mat(fleft).t() << std::endl;
	std::cout << "left error: " << eleft << std::endl;
	std::cout << "right fit: " << Mat(fright).t() << std::endl;
	std::cout << "right error: " << eright << std::endl;

	if (eright > 1.0 || eleft > 1.0 || etop > 1.0 || ebottom > 1.0)
	{
		std::cout << "WARNING: errors unacceptably large! edit the picture to enhance contrast and remove clutter." << std::endl;
	}

	opt.fntop    = ftop;
	opt.fnbottom = fbottom;
	opt.fnleft   = fleft;
	opt.fnright  = fright;

	// compute corners as intersections
	opt.cornertl = intersect(ftop, fleft);
	opt.cornertr = intersect(ftop, fright);
	opt.cornerbl = intersect(fbottom, fleft);
	opt.cornerbr = intersect(fbottom, fright);

	if (contours.size() != 1)
	{
		auto color = Vec3b(255, 0, 255);
		draw_polynomial(disp, 0, 0, 1920, ftop, color);
		draw_polynomial(disp, 0, 0, 1920, fbottom, color);
		draw_polynomial(disp, 1, 0, 1080, fleft, color);
		draw_polynomial(disp, 1, 0, 1080, fright, color);

		//*
		cv::namedWindow("contours", CV_WINDOW_NORMAL);
		cv::pyrDown(disp, tmp); cv::imshow("contours", tmp);
		while (cv::waitKey(100) == -1);
		//*/
	}

	return opt;
}

inline Point2f pixmap(options const &opt, float fx, float fy)
{
	float const xtop    = (opt.cornertr.x - opt.cornertl.x) * fx + opt.cornertl.x;
	float const xbottom = (opt.cornerbr.x - opt.cornerbl.x) * fx + opt.cornerbl.x;

	float const ytop    = evaluate(xtop,    opt.fntop);
	float const ybottom = evaluate(xbottom, opt.fnbottom);

	float const ox = (xbottom - xtop) * fy + xtop;
	float const oy = (ybottom - ytop) * fy + ytop;

	return Vec2f(ox, oy);
}

void straighten(options const opt, VideoCapture &invid, VideoWriter &outvid)
{
	volatile double t0, t1, t2, dt1, dt2;

	cv::Mat inframe;
	cv::Mat outframe;

	int const imwidth  = (int)invid.get(CV_CAP_PROP_FRAME_WIDTH);
	int const imheight = (int)invid.get(CV_CAP_PROP_FRAME_HEIGHT);
	int const nframes  = (int)invid.get(CV_CAP_PROP_FRAME_COUNT);

	// init remap()
	cv::Mat xmap(imheight, imwidth, CV_32FC1);
	cv::Mat ymap(imheight, imwidth, CV_32FC1);

	for (int y = 0; y < imheight; y++)
	{
		float const fy = y / (float)(imheight-1);

		for (int x = 0; x < imwidth; x++)
		{
			float const fx = x / (float)(imwidth-1);

			auto const px = pixmap(opt, fx, fy);
			xmap.at<float>(y, x) = px.x;
			ymap.at<float>(y, x) = px.y;

			//outframe.at<cv::Vec3b>(y,x) = interpolate(inframe, px);
		}
	}

	// fetching frames, crunching
	int lastcount = 0;
	volatile double lasttime = timer();
	double const dt = 1.0;
	for (int counter = 0; ; counter += 1)
	{
		if (!invid.read(inframe))
			break;

		outframe.create(inframe.size(), inframe.type());
		//outframe.setTo(cv::Scalar(0));

		//t1 = timer();
		cv::remap(inframe, outframe, xmap, ymap, INTER_CUBIC);
		//t2 = timer();

		/*
		Mat tmp;
		cv::pyrDown(outframe, tmp);
		cv::imshow("output", outframe);
		int const key = cv::waitKey(1);
		if (key != -1)
			break;
		//*/
		
		outvid.write(outframe);

		double now = timer();
		if (now >= lasttime + dt)
		{
			int const dcnt = counter - lastcount;
			double fps = dcnt / (now - lasttime);

			std::cout << "\r" << (counter+1) << " of " << nframes << " frames done (" << std::fixed << std::setprecision(1) << (100.0 * counter / nframes) << "%)";
			std::cout << ", " << std::fixed << std::setprecision(1) << fps << " fps, ETA " << ((nframes - counter) / fps / 60) << " min  " << std::flush;

			lasttime += dt;
			lastcount = counter;
		}
	}

	std::cout << std::endl;
}

std::string fourcc2str(int fourcc)
{
	char buf[5];
	*(uint32_t*)buf = fourcc;
	buf[4] = '\0';
	
	for (int i = 0; i < 4; i += 1)
	{
		if (buf[i] < 32 || buf[i] >= 128)
			buf[i] = '?';
	}

	return std::string(buf);
}

int main(int argc, char **argv)
{
	int fourcc = CV_FOURCC('M', 'J', 'P', 'G');
	//fourcc = -1;
	//fourcc = CV_FOURCC('L','A','G','S');
	//fourcc = CV_FOURCC('L', 'A', 'G', 'S');
	//fourcc = CV_FOURCC('X', '2', '6', '4');

	std::vector<std::string> args;
	
	for (int k = 1; k < argc; k += 1)
	{
		auto arg = std::string(argv[k]);

		if (arg[0] != '-')
		{
			args.push_back(arg);
			continue;
		}

		if (arg == "-codecpicker")
			fourcc = -1;

		else if (arg == "-codec")
		{
			auto const codec = std::string(argv[++k]);
			if (codec == "pick" || codec == "-1")
				fourcc = -1;
			else if (codec == "H264" || codec == "h264" || codec == "H.264" || codec == "h.264")
				fourcc = CV_FOURCC('X', '2', '6', '4');
			else
				fourcc = CV_FOURCC(codec[0], codec[1], codec[2], codec[3]);
		}
		else if (arg == "-threshold")
		{
			global_threshold = atoi(argv[++k]);
		}
	}

	// program optfile img
	if (args.size() == 2)
	{
		auto const optfname = args[0];
		auto const maskfname = args[1];

		Mat img = cv::imread(maskfname);
		options opt = find_sides(img);
		opt.save(optfname);
	} 

	// program optfile source out
	else if (args.size() == 3)
	{
		auto const optfname = args[0];
		auto const invideoname = args[1];
		auto const outvideoname = args[2];

		options opt(optfname);
		
		VideoCapture invid(invideoname);
		std::cout << "reader open? " << (invid.isOpened() ? "yes" : "NO") << std::endl;
		if (!invid.isOpened())
			return -1;

		std::cout << "fourcc = " << fourcc << " (" << fourcc2str(fourcc) << ")" << std::endl;

		VideoWriter outvid(
			outvideoname,
			fourcc,
			25, // invid.get(CV_CAP_PROP_FPS),
			Size(
				invid.get(CV_CAP_PROP_FRAME_WIDTH),
				invid.get(CV_CAP_PROP_FRAME_HEIGHT)));

		std::cout << "writer open? " << (outvid.isOpened() ? "yes" : "NO") << std::endl;
		if (!outvid.isOpened())
			return -1;

		std::cout << "straightening" << std::endl;
		straighten(opt, invid, outvid);
		std::cout << "done" << std::endl;
	}

	// usage
	else {
		std::cout << "Usage: straighten [optfile] [white picture]" << std::endl;
		std::cout << "Usage: straighten [optfile] [source video] [output video]" << std::endl;
	}

	return 0;
}
