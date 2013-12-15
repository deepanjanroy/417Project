#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;

static const std::string OPENCV_WINDOW = "Aqua Tracker";

std::string fdetector;
std::string fextractor;


// Bunch of globals that magically stops segfaults
std::vector<std::vector<Point2f> *> lines;
Mat im_show;
std::vector<DMatch> good_matches;
std::vector<Point2f> *line_ptr;
std::vector<bool> indices_to_remove;
Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> extractor;
double metric = 0.0;
int numFrames = 0;




// Helper functions

int findLine(Point2f &point) {
  for (int i=0; i < lines.size(); i++) {
    double dx = fabs(lines[i]->back().x - point.x);
    double dy = fabs(lines[i]->back().y - point.y);
    if (dx < 1 && dy < 1)
      return i;
  }
  std::vector<Point2f> *new_line = new std::vector<Point2f>;

  new_line->push_back(point);
  lines.push_back(new_line);
  indices_to_remove.push_back(true);
  return lines.size() - 1;
}


void drawLines(Mat &img) {
  for (int i=0; i < lines.size(); i++) {
    std::vector<Point2f> *line_cur = lines[i];
    if (line_cur->size() < 6) continue;
    for(int j=1; j < line_cur->size(); j++) {
      Point2f prev = (*line_cur)[j-1];
      Point2f cur = (*line_cur)[j];
      line(img, prev, cur, CV_RGB(255, 255, 0), 2);
    }
  }
}


double calculateMetric() {
  double retVal = 0;
  int i=0;
  while(i < lines.size()){
    std::vector<Point2f> *line_cur = lines[i];
    retVal += line_cur->size();
    i++;
  }
  if (lines.size() > 0) retVal = retVal / lines.size();
  return retVal;
}


void removeDuplicates(std::vector<DMatch> &matches) {
  for (int i=0; i<matches.size(); i++) {
    double bestDistance = 10000000;
    int bestIndex = -1;
    std::vector<int> duplicates;
    for (int j=i; j< matches.size(); j++) {
      if (matches[i].trainIdx == matches[j].trainIdx ||
          matches[i].queryIdx == matches[j].queryIdx) {
        duplicates.push_back(i);
        if (matches[i].distance < bestDistance) {
          bestDistance = matches[i].distance;
          bestIndex = i;
        }
      }
    }
    for (int j=duplicates.size()-1; j >=0; j--) {
      if (j != bestIndex) {
        matches.erase(matches.begin() + j);
      }
    }
  }
}

// The main class
class AquaRos

// Image Conversion code used in this class has been borrowed from
// http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages

{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;

public:
  AquaRos()
    : it_(nh_),
      matcher(NORM_L2),
      min_dist(100)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/camera_front_center/image_rect_color", 1,
      &AquaRos::image_callback, this);

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~AquaRos()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void image_callback(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    img_cur = cv_ptr->image; // This is terrible
    detector->detect(img_cur, points_cur);
    extractor->compute(img_cur, points_cur, descriptor_cur);

    if (points_prev.size() > 1) {


      matcher.match(descriptor_prev, descriptor_cur, matches);

      //pick the good matches
      good_matches.clear();
      for (int i=0; i < matches.size(); i++) {
        Point2f prev = points_prev[matches[i].trainIdx].pt;
        Point2f cur = points_cur[matches[i].queryIdx].pt;
        double dx = fabs(prev.x - cur.x);
        double dy = fabs(prev.y - cur.y);
        if (dx+dy < 35) {
          good_matches.push_back(matches[i]);
        }
      }

      indices_to_remove.clear();
      for (int j=0; j<lines.size(); j++) {
        indices_to_remove.push_back(true);
      }

      for (int i=0; i < good_matches.size(); i++) {

        Point2f old = points_prev[good_matches[i].trainIdx].pt;
        int line_index = findLine(old); //returns a new list if not found


        indices_to_remove[line_index] = false;

        line_ptr = lines[line_index];
        Point2f new_point = points_cur[good_matches[i].queryIdx].pt; // = 3
        line_ptr->push_back(new_point);
      }

      for (int i=lines.size() -1; i >=0; i--) {
        if (indices_to_remove[i]) {
          delete lines[i];
          lines.erase(lines.begin() + i);
        }
      }

      // im_show = img_cur.clone();
      drawLines(img_cur);
      // drawKeypoints(im_show, points_cur, im_show);
      cv::imshow(OPENCV_WINDOW, cv_ptr->image);
      imshow(OPENCV_WINDOW, img_cur);
      double metric_ret = calculateMetric();
      metric += metric_ret;
      numFrames += 1;
      if (waitKey(3) > 0) return;
    }


    points_prev = points_cur;
    descriptor_prev = descriptor_cur;
    img_prev = img_cur;

    // if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
    //   cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
    // Draw an example circle on the video stream

    // Update GUI Window
    cv::waitKey(3);

    if (numFrames > 0) metric /= numFrames;
    std::cout << fdetector << "," << fextractor << ", " << metric << std::endl;

  }

  // public fields

  std::vector<KeyPoint> points_prev, points_cur;
  Mat img_prev, img_cur;
  SurfDescriptorExtractor extractordfsfdsf; //WE DON"T KNOW WHAT THIS DOES BUT IT SEGFAULTS WITHOUT IT D:
  Mat descriptor_prev, descriptor_cur;
  BFMatcher matcher;
  std::vector<DMatch> matches;
  Mat img_matches;
  int min_dist;

};

int main(int argc, char** argv)
{
  fdetector = argv[2];
  fextractor = argv[3];
  ros::init(argc, argv, "image_converter");
  AquaRos ic;

  detector = FeatureDetector::create(argv[2]);
  extractor = DescriptorExtractor::create(argv[3]);


  ros::spin();
  return 0;
}
