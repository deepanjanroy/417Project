#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace cv;

std::vector<std::vector<Point2f> *> lines;
Mat im_show;
std::vector<DMatch> good_matches;
std::vector<Point2f> *line_ptr;
std::vector<bool> indices_to_remove;
Ptr<FeatureDetector> detector;
Ptr<DescriptorExtractor> extractor;
double metric = 0.0;
int numFrames = 0;

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

int main( int argc, char** argv ) {
    VideoCapture vid(argv[1]);
    // Load images
    if( !vid.isOpened()) {
        std::cout<< " --(!) Error reading video" << argv[1] << std::endl;
        return -1;
    }

    detector = FeatureDetector::create(argv[2]);
    extractor = DescriptorExtractor::create(argv[3]);

    namedWindow("Tracking", 1);

    std::vector<KeyPoint> points_prev, points_cur;
    Mat img_prev, img_cur;

    SurfDescriptorExtractor extractordfsfdsf; //WE DON"T KNOW WHAT THIS DOES BUT IT SEGFAULTS WITHOUT IT D:
    Mat descriptor_prev, descriptor_cur;

    BFMatcher matcher(NORM_L2);
    std::vector<DMatch> matches;

    Mat img_matches;
    int min_dist = 100;

    for(;;) {
      //grab a video frame
      if ( !vid.read(img_cur)) break;


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

         im_show = img_cur.clone();
         drawLines(im_show);
       //  drawKeypoints(im_show, points_cur, im_show);
         imshow("Tracking", im_show);
         double metric_ret = calculateMetric();

         metric += metric_ret;
         numFrames += 1;
         if (waitKey(3) > 0) break;
      }



      points_prev = points_cur;
      descriptor_prev = descriptor_cur;
      img_prev = img_cur;
    }

    if (numFrames > 0) metric /= numFrames;
    std::cout<<argv[2]<<","<<argv[3]<<", "<<metric<<std::endl;
}
