#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

string	origin_name		= "原图像";
string	aniso_name		= "各向异性扩散图像";
string	equal_name		= "直方图均衡化后的图像";
string	integral_name	= "积分图像";
string	k_means_name	= "kmeans分割图像";
string	result_name		= "结果图像";
int		frame_num		= 0;	//输入帧的数量

//********各向异性扩散滤波参数********
Mat AnisoDiff(const Mat& src, int iter, int k, float lambda);
float   lambda			= 0.23;	//扩散速度控制，最大取0.25
int     kappa			= 10;	//热传导系数，取值范围待经验确认
int		iteration		= 1;	//迭代次数

//********帧间差分及积分参数********
Mat FrameDiff(const deque<Mat>& src, int difference_interval, int integral_interval);
Mat ThreeFrameDiff(const deque<Mat>& src, int difference_interval, int integral_interval);
int		difference_interval = 2;	//差分帧间间隔
int		integral_interval	= 5;	//积分帧间间隔
int		min_diff_threshold	= 60;

//********k-means图像分割参数********
Mat KMeans(const Mat& src, int k, TermCriteria criteria);
int		k				= 3;	//聚类的种类
TermCriteria criteria	= TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);	//迭代终止条件

int main()
{
	//VideoCapture cap("C:/Users/admin/Desktop/论文/test.avi");
	VideoCapture cap(0);
	Mat frame, image;
	deque<Mat> image_deque;

	for (int i = 0; i < difference_interval + integral_interval; i++)
	{
		cap >> frame;
		frame_num++;
		if (frame.empty())
			break;
		imshow(origin_name, frame);

		frame.copyTo(image);
		cvtColor(image, image, COLOR_BGR2GRAY);
		//1.直方图均衡化
		equalizeHist(image, image);
		//2.各向异性扩散滤波
		image = AnisoDiff(image, iteration, kappa, lambda);
		image_deque.push_back(image);
	}
	
	for (;;)
	{
		Mat img_aniso = image_deque.at(difference_interval);
		imshow(aniso_name, img_aniso);
		

		//3.帧间差分和积分
		//Mat img_integral = FrameDiff(image_deque, difference_interval, integral_interval);
		Mat img_integral = ThreeFrameDiff(image_deque, difference_interval, integral_interval);
		threshold(img_integral, img_integral, min_diff_threshold, 255.0, cv::THRESH_BINARY);
		dilate(img_integral, img_integral, Mat());		//膨胀			
		erode(img_integral, img_integral, Mat());		//腐蚀
		imshow(integral_name, img_integral);
		createTrackbar("threshold", integral_name, &min_diff_threshold, 255, 0);	//创建过滤最小阈值的调节条

		////4.k-means
		//Mat img_kmeans = KMeans(img_integral, k, criteria);
		//imshow(k_means_name, img_kmeans);
		
		//5.彩色渲染和融合
		Mat img_result = frame.clone();
		for(int i = 0; i < img_result.rows; i++)
			for (int j = 0; j < img_result.cols; j++)
			{
				if (img_integral.at<uchar>(i, j) != 0)
					img_result.at<Vec3b>(i, j) = Vec3b(64, 128, 255);
			}
		imshow(result_name, img_result);
		//6.读取下一帧，再次进行第1步、第2步
		cap >> frame;
		if (frame.empty())
			break;
		imshow(origin_name, frame);

		frame.copyTo(image);
		cvtColor(image, image, COLOR_BGR2GRAY);
		cout << "当前帧："<<frame_num++ << endl;
		image_deque.pop_front();
		equalizeHist(image, image);
		image = AnisoDiff(image, iteration, kappa, lambda);
		image_deque.push_back(image);

		char key = (char)waitKey(10);
		if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
			break;
	}

	return 0;
}

//********各向异性扩散滤波（灰度图像版本）（已完成）********
//参数说明：src--原图像，iter--迭代次数，k--kappa（传导系数），lambda--扩散速度
Mat AnisoDiff(const Mat& src, int iter, int k, float lambda)
{
	float ei, si, wi, ni;	//东南西北4个梯度
	float ce, cs, cw, cn;	
	int rows = src.rows;
	int cols = src.cols;
	//Mat temp = src.clone();
	Mat temp = src;	
	Mat output = cv::Mat::zeros(rows, cols, CV_8UC1);

	if (temp.channels() != 1)
	{
		cvtColor(temp, temp, COLOR_BGR2GRAY);	//判断通道数，转换为灰度图像
		cout << "图像不是单通道灰度图，已转换为灰度图!" << endl;
	}
	//if(temp.type() != CV_8UC1)
	//	cvtColor(temp, temp, COLOR_BGR2GRAY);	//判断图像类型，转换为灰度图像

	for (int n = 0; n < iter; n++)
	{
		for(int i = 1; i < rows - 1; i++)
			for (int j = 1; j < cols - 1; j++)
			{
				uchar cur = temp.at<uchar>(i, j);
				ei = temp.at<uchar>(i - 1, j) - cur;
				si = temp.at<uchar>(i, j + 1) - cur;
				wi = temp.at<uchar>(i + 1, j) - cur;
				ni = temp.at<uchar>(i, j - 1) - cur;

				ce = exp(-ei * ei / (k * k));
				cs = exp(-si * si / (k * k));
				cw = exp(-wi * wi / (k * k));
				cn = exp(-ni * ni / (k * k));

				output.at<uchar>(i, j) = cur + lambda * (ce * ei + cs * si + cw * wi + cn * ni);
			}
		output.copyTo(temp);
	}
	return output;
}

//********帧间差分及积分（已完成）********
//参数说明：src--原图像，difference_interval--差分图像采样间隔，integral_interval--积分间隔
Mat FrameDiff(const deque<Mat>& src, int difference_interval, int integral_interval)
{
	int num_frame_stored = difference_interval + integral_interval;	//需要存储的帧的数量
	if (src.size() != num_frame_stored)
	{
		cout << "存储帧数量与需要帧数量不符，请检查程序代码！" << endl;
	}
	Mat difference_result = cv::Mat::zeros(src.at(0).rows, src.at(0).cols, src.at(0).type());	//差分结果
	Mat integral_result = cv::Mat::zeros(src.at(0).rows, src.at(0).cols, src.at(0).type());	//积分结果
	//Mat integral_result = cv::Mat::zeros(src.at(0).rows, src.at(0).cols, CV_8UC1);
	for (int r = 0; r < integral_interval; r++)
	{
		absdiff(src.at(r), src.at(r + difference_interval), difference_result);
		//去除微小变化（此步骤暂缺）
		integral_result = integral_result + difference_result;
	}
	
	return integral_result;
}

//********三帧差分及积分（已完成）********
//参数说明：src--原图像，difference_interval--差分图像采样间隔，integral_interval--积分间隔
Mat ThreeFrameDiff(const deque<Mat>& src, int difference_interval, int integral_interval)
{
	int num_frame_stored = difference_interval + integral_interval;	//需要存储的帧的数量
	if (src.size() != num_frame_stored)
	{
		cout << "存储帧数量与需要帧数量不符，请检查程序代码！" << endl;
	}
	Mat difference_result_1	= cv::Mat::zeros(src.at(0).rows, src.at(0).cols, src.at(0).type());	//差分结果1
	Mat difference_result_2 = cv::Mat::zeros(src.at(0).rows, src.at(0).cols, src.at(0).type());	//差分结果2
	Mat difference_result	= cv::Mat::zeros(src.at(0).rows, src.at(0).cols, src.at(0).type());
	Mat integral_result		= cv::Mat::zeros(src.at(0).rows, src.at(0).cols, src.at(0).type());	//积分结果
	//Mat integral_result = cv::Mat::zeros(src.at(0).rows, src.at(0).cols, CV_8UC1);
	for (int r = difference_interval; r < integral_interval; r++)
	{
		int previous = r - difference_interval, future = r + difference_interval;
		absdiff(src.at(r), src.at(r - difference_interval), difference_result_1);
		absdiff(src.at(r), src.at(r + difference_interval), difference_result_2);
		bitwise_and(difference_result_1, difference_result_2, difference_result);
		integral_result = integral_result + difference_result;
	}

	return integral_result;
}

//********k-means聚类（已完成）********
//参数说明：src--原图像，k--聚类类别数量，criteria--迭代终止条件
Mat KMeans(const Mat& src, int k, TermCriteria criteria = TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1))
{

	Mat centers;	//centers：最终分割后每个聚类的中心位置
	Mat dst(src.size(), src.type());
	Mat src_array = src.clone();
	src_array.convertTo(src_array, CV_32FC1);
	src_array = src_array.reshape(0, src.rows * src.cols);	//转化为1维向量，通道数不变，reshape按行序列化（即按src的行取值，填入dst的行中）
	Mat_<int> labels(src_array.size(), CV_32SC1);	//labels：最终分类每个样本的标签
	
	////去除大量0元素，提升速度（此步骤暂缺）
	//vector<float> src_vec;
	//for (int i = 0; i < src_array.rows; i++)
	//{
	//	if(src_array.at<float>(i, 0) != 0)
	//	{
	//		//cout << src_array.at<float>(i, 0) << endl;
	//		src_vec.push_back(src_array.at<float>(i, 0));
	//	}
	//}
	//src_array = Mat(src_vec, true);

	//kmeans
	double compactness = kmeans(src_array, k, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);	//kmeans按行组织样本，每一行为一个样本数据，列表示样本维度，kmeans只接受CV_32F类型
	//double compactness = kmeans(src_array, k, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);
	//将每一类的像素转化为聚类中心的像素值
	MatIterator_<uchar> itd = dst.begin<uchar>(), itd_end = dst.end<uchar>();	//这里还是人工设定了dst是灰度图，像素类型是uchar，不够“泛型”
	for (int i = 0; itd != itd_end; i++, itd++)
	{
		uchar color = centers.at<float>(labels.at<int>(i, 0), 0);
		(*itd) = saturate_cast<uchar>(color);
	}
	return dst;
}