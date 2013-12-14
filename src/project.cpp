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

int findLine(Point2f &point) {
  for (int i=0; i < lines.size(); i++) {
    double dx = fabs(lines[i]->back().x - point.x);
    double dy = fabs(lines[i]->back().y - point.y);
    if (dx < 20 && dy < 20)
      return i;
  }
  std::vector<Point2f> *new_line = new std::vector<Point2f>; 
  new_line->push_back(point);
  lines.push_back(new_line);
  return lines.size() - 1;
}

void drawLines(Mat &img) {
  for (int i=0; i < lines.size(); i++) {
    std::vector<Point2f> *line_cur = lines[i];
    for(int i=1; i < line_cur->size(); i++) {
      Point2f prev = (*line_cur)[i-1];
      Point2f cur = (*line_cur)[i];
      line(img, prev, cur, CV_RGB(0, 0, 0));
      std::cout<< "drew line from (" << prev.x<<","<<prev.y<<") to (" <<cur.x<< "," <<cur.y<<")"<<std::endl;
    }
  }
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

//void testDrawLines(Mat &img) {
//  std::vector<Point2f> *line1 = new std::vector<Point2f>;
//  std::vector<Point2f> *line2 = new std::vector<Point2f>;
//  std::vector<Point2f> *line3 = new std::vector<Point2f>;
//  std::vector<Point2f> *line4 = new std::vector<Point2f>;
//  for (int i=0; i<50; i++) {
//    Point2f point1(i*5, i*5);
//    line1->push_back(point1);
//    Point2f point2(200 - i, 200 - (2*i));
//    line2->push_back(point2);
//    Point2f point3(50 + i, 200 - i);
//    line3->push_back(point3);
//  }
//  lines.push_back(line1);
//  lines.push_back(line2);
//  lines.push_back(line3);
//  drawLines(img);
//  namedWindow("Test", 1);
//  while (true) {
//    imshow("Test", img);
//    waitKey(30);
//  }
//}
//
      
int main( int argc, char** argv ) {

    VideoCapture vid(argv[1]);
    // Load images
    if( !vid.isOpened()) {
        std::cout<< " --(!) Error reading video" << argv[1] << std::endl;
        return -1;
    }
    
//    Mat img;
//    vid >> img;
//    testDrawLines(img);
//
    namedWindow("Tracking", 1);
    namedWindow("Keypoints", 1);

    std::vector<KeyPoint> points_prev, points_cur;
    Mat img_prev, img_cur;
    
    SurfFeatureDetector detector(2000,4);
    SurfDescriptorExtractor extractor;
    Mat descriptor_prev, descriptor_cur;

    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;

    Mat img_matches;
    int min_dist = 100;

    for(;;) {
      
      //grab a video frame
      vid >> img_cur;
      detector.detect(img_cur, points_cur);
      extractor.compute(img_cur, points_cur, descriptor_cur);

      if (points_prev.size() > 1) {
        matcher.match(descriptor_prev, descriptor_cur, matches);

      //  removeDuplicates(matches);

        double min_dist = 1000, max_dist = 0;
        for (int i=0; i<matches.size(); i++) {
          double dist = matches[i].distance;
          if (dist < min_dist) {
            min_dist = dist;
          } if (dist > max_dist) {
            max_dist = dist;
          }
        }
        
        double thresh = (min_dist + max_dist) / 2;

        std::cout<< "min_dist: " << min_dist << std::endl;
       
        std::vector<DMatch> good_matches;
        for (int i=0; i < matches.size(); i++) {
          Point2f prev = points_prev[matches[i].trainIdx].pt;
          Point2f cur = points_cur[matches[i].queryIdx].pt;
          double dx = fabs(prev.x - cur.x);
          double dy = fabs(prev.y - cur.y);
          //matches[i].distance  <= thresh
          if (dx + dy < 50) {
            good_matches.push_back(matches[i]);
          } else {
            std::cout << "distance was: " << matches[i].distance << std::endl;
          }
        }

        std::vector<bool> indices_to_remove(lines.size(), true);
        for (int i=0; i < good_matches.size(); i++) {

          Point2f old = points_prev[good_matches[i].queryIdx].pt;
          int line_index = findLine(old); //returns a new list if not found
          indices_to_remove[line_index] = false;

          std::vector<Point2f> *line = lines[line_index];
          Point2f new_point = points_cur[good_matches[i].trainIdx].pt; // = 3
          line->push_back(new_point);
        }

       // for (int i=lines.size() -1; i >=0; i--) {
       //   if (indices_to_remove[i]) {
       //     delete lines[i];
       //     lines.erase(lines.begin() + i);
       //   }
       // }

        for (int i=0; i < lines.size(); i++) {
          std::vector<Point2f> *line = lines[i];
          std::cout << "Line " << i << " size: " << lines[i]->size() <<" ";
          for (int i=0; i< line->size(); i++) {
            std::cout << "(" << (*line)[i].x << "," << (*line)[i].y << ") ";
          }
          std::cout<<std::endl;
        }

        std::vector<KeyPoint> matching;
        for (int i=0; i<matches.size(); i++) {
          matching.push_back(points_cur[matches[i].trainIdx]);
        }

        Mat img_matching;
        drawKeypoints(img_cur, matching, img_matching);

        drawLines(img_cur);
        drawKeypoints(img_cur, points_cur, img_cur);
        //Mat out_img;
        //drawMatches(img_prev, points_prev, img_cur, points_cur, good_matches, out_img);
        imshow("Tracking", img_cur);
        imshow("Keypoints", img_matching);
        if (waitKey(30) >= 0) break;
      }

      points_prev = points_cur;
      descriptor_prev = descriptor_cur;
      img_prev = img_cur;
    }
//
//    std::vector<KeyPoint> keypointsA, keypointsB;
//    Mat descriptorsA, descriptorsB;
//    std::vector<DMatch> matches;
//
//    // DETECTION
//    // Any openCV detector such as
//    SurfFeatureDetector detector(2000,4);
//
//    // DESCRIPTOR
//    // Our proposed FREAK descriptor
//    // (roation invariance, scale invariance, pattern radius corresponding to SMALLEST_KP_SIZE,
//    // number of octaves, optional vector containing the selected pairs)
//    // FREAK extractor(true, true, 22, 4, std::vector<int>());
//    FREAK extractor;
//
//    // MATCHER
//    // The standard Hamming distance can be used such as
//    // BruteForceMatcher<Hamming> matcher;
//    // or the proposed cascade of hamming distance using SSSE3
//    BruteForceMatcher<Hamming> matcher;
//
//    // detect
//    double t = (double)getTickCount();
//    detector.detect( imgA, keypointsA );
//    detector.detect( imgB, keypointsB );
//    t = ((double)getTickCount() - t)/getTickFrequency();
//    std::cout << "detection time [s]: " << t/1.0 << std::endl;
//
//    // extract
//    t = (double)getTickCount();
//    extractor.compute( imgA, keypointsA, descriptorsA );
//    extractor.compute( imgB, keypointsB, descriptorsB );
//    t = ((double)getTickCount() - t)/getTickFrequency();
//    std::cout << "extraction time [s]: " << t << std::endl;
//
//    // match
//    t = (double)getTickCount();
//    matcher.match(descriptorsA, descriptorsB, matches);
//    t = ((double)getTickCount() - t)/getTickFrequency();
//    std::cout << "matching time [s]: " << t << std::endl;
//
//    // Draw matches
//    Mat imgMatch;
//    drawMatches(imgA, keypointsA, imgB, keypointsB, matches, imgMatch);
//
//    namedWindow("matches", CV_WINDOW_KEEPRATIO);
//    imshow("matches", imgMatch);
    waitKey(0);
}
