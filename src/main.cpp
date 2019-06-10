#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include "sys/time.h"
#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxParserRuntime.h"
#include "NvOnnxConfig.h"
#include <time.h>
using namespace nvinfer1;

static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;
static int gUseDLACore{-1};

struct LayerInfo
{
	std::vector<int> dim;
	std::string name;
	int index;
	int size;
};
nvinfer1::IExecutionContext* context;
nvinfer1::IRuntime* runtime;
nvinfer1::ICudaEngine* engine;
cudaStream_t stream;
std::vector<LayerInfo> output_layer;
int input_size;
//std::vector<int> m_output_size;
void* buffers[10];
int inputIndex;


float m_nms_threshold = 0.4;
float data0[8] = { -248,-248,263,263,-120,-120,135,135 };
float data1[8] = { -56,-56,71,71,-24,-24,39,39 };
float data2[8] = { -8,-8,23,23,0,0,15,15 };




class Anchor {
public:
	bool operator<(const Anchor &t) const {
		return score < t.score;
	}

	bool operator>(const Anchor &t) const {
		return score > t.score;
	}

	float& operator[](int i) {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	float operator[](int i) const {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	cv::Rect_< float > anchor; // x1,y1,x2,y2
	float reg[4]; // offset reg
	cv::Point center; // anchor feat center
	float score; // cls score
	std::vector<cv::Point2f> pts; // pred pts

	cv::Rect_< float > finalbox; // final box res
};

void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& filterOutBoxes) {
	filterOutBoxes.clear();
	if (boxes.size() == 0)
		return;
	std::vector<size_t> idx(boxes.size());

	for (unsigned i = 0; i < idx.size(); i++)
	{
		idx[i] = i;
	}

	//descending sort
	sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

	while (idx.size() > 0)
	{
		int good_idx = idx[0];
		filterOutBoxes.push_back(boxes[good_idx]);

		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++)
		{
			int tmp_i = tmp[i];
			float inter_x1 = std::max(boxes[good_idx][0], boxes[tmp_i][0]);
			float inter_y1 = std::max(boxes[good_idx][1], boxes[tmp_i][1]);
			float inter_x2 = std::min(boxes[good_idx][2], boxes[tmp_i][2]);
			float inter_y2 = std::min(boxes[good_idx][3], boxes[tmp_i][3]);

			float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
			float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
			float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
			float o = inter_area / (area_1 + area_2 - inter_area);
			if (o <= threshold)
				idx.push_back(tmp_i);
		}
	}
}

class CRect2f {
public:
	CRect2f(float x1, float y1, float x2, float y2) {
		val[0] = x1;
		val[1] = y1;
		val[2] = x2;
		val[3] = y2;
	}

	float& operator[](int i) {
		return val[i];
	}

	float operator[](int i) const {
		return val[i];
	}

	float val[4];

	void print() {
		printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
	}
};

class AnchorGenerator {
public:
	void Init(int stride, int num, float* data)
	{
		anchor_stride = stride; // anchor tile stride
		preset_anchors.push_back(CRect2f(data[0], data[1], data[2], data[3]));
		preset_anchors.push_back(CRect2f(data[4], data[5], data[6], data[7]));
		anchor_num = num; // anchor type num
	}
	// filter anchors and return valid anchors
	int FilterAnchor(float* cls, float* reg, float* pts, int w, int h, int c, std::vector<Anchor>& result)
	{
		int pts_length = 0;

		pts_length = c / anchor_num / 2;

		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				int id = i * w + j;
				for (int a = 0; a < anchor_num; ++a)
				{
					float score = cls[(anchor_num + a)*w*h + id];
					if (score >= m_cls_threshold) {
						CRect2f box(j * anchor_stride + preset_anchors[a][0],
							i * anchor_stride + preset_anchors[a][1],
							j * anchor_stride + preset_anchors[a][2],
							i * anchor_stride + preset_anchors[a][3]);
						//printf("%f %f %f %f\n", box[0], box[1], box[2], box[3]);
						CRect2f delta(reg[(a * 4 + 0)*w*h + id],
							reg[(a * 4 + 1)*w*h + id],
							reg[(a * 4 + 2)*w*h + id],
							reg[(a * 4 + 3)*w*h + id]);

						Anchor res;
						res.anchor = cv::Rect_< float >(box[0], box[1], box[2], box[3]);
						bbox_pred(box, delta, res.finalbox);
						//printf("bbox pred\n");
						res.score = score;
						res.center = cv::Point(j, i);

						//printf("center %d %d\n", j, i);

						if (1) {
							std::vector<cv::Point2f> pts_delta(pts_length);
							for (int p = 0; p < pts_length; ++p) {
								pts_delta[p].x = pts[(a*pts_length * 2 + p * 2)*w*h + id];
								pts_delta[p].y = pts[(a*pts_length * 2 + p * 2 + 1)*w*h + id];
							}
							//printf("ready landmark_pred\n");
							landmark_pred(box, pts_delta, res.pts);
							//printf("landmark_pred\n");
						}
						result.push_back(res);
					}
				}
			}
		}
		return 0;
	}

private:
	void bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect_< float >& box)
	{
		float w = anchor[2] - anchor[0] + 1;
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		float dx = delta[0];
		float dy = delta[1];
		float dw = delta[2];
		float dh = delta[3];

		float pred_ctr_x = dx * w + x_ctr;
		float pred_ctr_y = dy * h + y_ctr;
		float pred_w = std::exp(dw) * w;
		float pred_h = std::exp(dh) * h;

		box = cv::Rect_< float >(pred_ctr_x - 0.5 * (pred_w - 1.0),
			pred_ctr_y - 0.5 * (pred_h - 1.0),
			pred_ctr_x + 0.5 * (pred_w - 1.0),
			pred_ctr_y + 0.5 * (pred_h - 1.0));
	}

	void landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts)
	{
		float w = anchor[2] - anchor[0] + 1;
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		pts.resize(delta.size());
		for (int i = 0; i < delta.size(); ++i) {
			pts[i].x = delta[i].x*w + x_ctr;
			pts[i].y = delta[i].y*h + y_ctr;
		}
	}

	int anchor_stride; // anchor tile stride
	std::vector<CRect2f> preset_anchors;
	int anchor_num; // anchor type num
	float m_cls_threshold = 0.8;
};



float* cls[3];
float* reg[3];
float* pts[3];
AnchorGenerator ac[3];
std::vector<int> get_dim_size(Dims dim)
{
	std::vector<int> size;
	for (int i = 0; i < dim.nbDims; ++i)
		size.emplace_back(dim.d[i]);
	return size;
}

int total_size(std::vector<int> dim)
{
	int size = 1 * sizeof(float);
	for (auto d : dim)
		size *= d;
	return size;
}
class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kINFO)
		: reportableSeverity(severity)
	{
	}

	void log(Severity severity, const char* msg) override
	{
		// suppress messages with severity enum value greater than the reportable
		if (severity > reportableSeverity)
			return;

		switch (severity)
		{
		case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
		case Severity::kERROR: std::cerr << "ERROR: "; break;
		case Severity::kWARNING: std::cerr << "WARNING: "; break;
		case Severity::kINFO: std::cerr << "INFO: "; break;
		default: std::cerr << "UNKNOWN: "; break;
		}
		std::cerr << msg << std::endl;
	}

	Severity reportableSeverity;
};

void load_onnx(std::string model)
{
	Logger gLogger;
	IBuilder* builder = createInferBuilder(gLogger);
	// parse the onnx model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	nvonnxparser::IParser* parser = nvonnxparser::createParser(network, gLogger);

	std::ifstream onnx_file(model.c_str(), std::ios::binary | std::ios::ate);
	std::streamsize file_size = onnx_file.tellg();
	onnx_file.seekg(0, std::ios::beg);
	std::vector<char> onnx_buf(file_size);
	onnx_file.read(onnx_buf.data(), onnx_buf.size());
	if (!parser->parse(onnx_buf.data(), onnx_buf.size()))
	{
		int nerror = parser->getNbErrors();
		for (int i = 0; i < nerror; ++i)
		{
			nvonnxparser::IParserError const* error = parser->getError(i);
			std::cerr << "ERROR: "
				<< error->file() << ":" << error->line()
				<< " In function " << error->func() << ":\n"
				<< "[" << static_cast<int>(error->code()) << "] " << error->desc()
				<< std::endl;
		}
	}
	ITensor* tensor_input = network->getInput(0);
	Dims dim_input = tensor_input->getDimensions();
	input_size = total_size(get_dim_size(dim_input));
	int outnode_size = network->getNbOutputs();
	//m_output_size.resize(m_outnode_size);
	for (int i = 0; i < outnode_size; ++i)
	{
		LayerInfo l;
		ITensor* tensor_output = network->getOutput(i);
		l.name = tensor_output->getName();
		Dims dim_output = tensor_output->getDimensions();
		l.dim = get_dim_size(dim_output);
		l.size = total_size(l.dim);
		output_layer.emplace_back(l);
	}
	int num_layer = network->getNbLayers();
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 20);
	//builder->allowGPUFallback(true);
	//builder->setDebugSync(true);
	engine = builder->buildCudaEngine(*network);
	runtime = createInferRuntime(gLogger);
	int gUseDLACore = -1;
	if (gUseDLACore >= 0)
	{
		//	m_runtime->setDLACore(gUseDLACore);
	}
	context = engine->createExecutionContext();
	for (int b = 0; b < engine->getNbBindings(); ++b)
	{
		if (engine->bindingIsInput(b))
			inputIndex = b;
		else
			output_layer[b - 1].index = b;
	}

	cudaStreamCreate(&stream);
	cudaMalloc(&buffers[inputIndex], input_size);   // data
	for (int i = 0; i < output_layer.size(); ++i)
		cudaMalloc(&buffers[output_layer[i].index], output_layer[i].size); // bbox_pred
	network->destroy();
	builder->destroy();

	std::cout << "RT init done!" << std::endl;
}




void doInference(cv::Mat img, std::vector<Anchor>& faces)
{
	cv::Mat image = img.clone();
	cv::Mat image_temp;
	cv::cvtColor(image, image, CV_BGR2RGB);
	cv::Mat image_resize(cv::Size(640, 640), CV_8UC3);
	float resize_scale = 1;
	if (image.cols >= image.rows&&image.cols > 640)
	{
		resize_scale = 640 / image.cols;
		cv::resize(image, image_temp, cv::Size(0, 0), resize_scale, resize_scale);
	}
	else if (image.cols < image.rows&&image.rows>640)
	{
		resize_scale = 640 / image.rows;
		cv::resize(image, image_temp, cv::Size(0, 0), resize_scale, resize_scale);
	}
	else
	{
		image_temp = image.clone();
	}
	cv::Mat imageROI0(image_resize(cv::Rect(0, 0, image_temp.cols, image_temp.rows)));
	image_temp.copyTo(imageROI0);
	int total_size = image_resize.rows*image_resize.cols*image_resize.channels();
	std::vector<float> input;
	input.resize(total_size);
	for (int k = 0; k < 3; k++)
		for (int i = 0; i < image_resize.rows; i++)
			for (int j = 0; j < image_resize.cols; j++)
			{
				input[i * image_resize.cols + j + k * image_resize.cols * image_resize.rows] =
					(float)image_resize.data[(i * image_resize.cols + j) * 3 + k];
			}
	std::vector<std::vector<float>> output;
	output.resize(output_layer.size());
	for (int i = 0; i < output_layer.size(); ++i)
		output[i].resize(output_layer[i].size / sizeof(float));
	cudaMemcpyAsync(buffers[inputIndex], input.data(),  input_size, cudaMemcpyHostToDevice, stream);
	context->enqueue(1, buffers, stream, nullptr);
	for (int i = 0; i < output_layer.size(); ++i)
		cudaMemcpyAsync(output[i].data(), buffers[output_layer[i].index], output_layer[i].size, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);




	for (int i = 0; i < 3; ++i)
	{
		reg[i] = output[i * 3 + 0].data();
		pts[i] = output[i * 3 + 1].data();
		cls[i] = output[i * 3 + 2].data();
	}
	std::vector<Anchor> proposals;
	for (int i = 0; i < 3; i++)
	{
		ac[i].FilterAnchor(cls[i], reg[i], pts[i], output_layer[i * 3 + 1].dim[2],
			output_layer[i * 3 + 1].dim[1], output_layer[i * 3 + 1].dim[0], proposals);
	}
	faces.clear();
	nms_cpu(proposals, m_nms_threshold, faces);
	std::sort(faces.begin(), faces.end(), [&](Anchor a, Anchor b)
	{
		return a.finalbox.area() > b.finalbox.area();
	});
	for (auto &face : faces)
	{
		face.finalbox.width *= resize_scale;
		face.finalbox.x *= resize_scale;
		face.finalbox.height *= resize_scale;
		face.finalbox.y *= resize_scale;
		for (int i = 0; i < 5; ++i)
		{
			face.pts[i].x *= resize_scale;
			face.pts[i].y *= resize_scale;
		}
	}

}


int main(int argc, char** argv)
{

	float data0[8] = { -248,-248,263,263,-120,-120,135,135 };
	float data1[8] = { -56,-56,71,71,-24,-24,39,39 };
	float data2[8] = { -8,-8,23,23,0,0,15,15 };
	ac[0].Init(32, 2, data0);
	ac[1].Init(16, 2, data1);
	ac[2].Init(8, 2, data2);
    // create a TensorRT model from the onnx model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};
    load_onnx("retina.onnx");
	cv::Mat image = cv::imread("test.bmp");
	std::vector<Anchor> faces;
#ifdef __linux__
	struct  timeval  m_start;
	struct  timeval  m_end;
#endif
	for (int i = 0; i < 100; i++) {
#ifdef __linux__
		gettimeofday(&m_start, NULL);
#endif
		doInference(image, faces);
#ifdef __linux__
		gettimeofday(&m_end, NULL);
		double runtime = (1000000 * (m_end.tv_sec - m_start.tv_sec) + m_end.tv_usec - m_start.tv_usec) / 1000;
		std::stringstream ss;
		ss  << "cost time = " << runtime << "ms" << std::endl;
		std::string  str = ss.str();
		std::cout << str;
#endif
	}
	for (int i = 0; i < faces.size(); i++)
	{
		cv::rectangle(image, cv::Point((int)faces[i].finalbox.x, (int)faces[i].finalbox.y), cv::Point((int)faces[i].finalbox.width, (int)faces[i].finalbox.height), cv::Scalar(0, 255, 255), 2, 8, 0);
		for (int j = 0; j < faces[i].pts.size(); ++j) {
			cv::circle(image, cv::Point((int)faces[i].pts[j].x, (int)faces[i].pts[j].y), 1, cv::Scalar(225, 0, 225), 2, 8);
		}
	}

	//cv::imshow("img", image);
	cv::imwrite("result.jpg", image);
	//cv::waitKey(0);
}
