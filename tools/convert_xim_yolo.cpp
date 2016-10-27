// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 160, "Width images are resized to");
DEFINE_int32(resize_height, 90, "Height images are resized to");

DEFINE_int32(input_width, 1280, "Height of input");
DEFINE_int32(input_height, 720, "Height of input");

DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

using namespace cv;
void split(const std::string &s,
           char delim,
           vector<std::string> &elems) {
  elems.clear();
  stringstream ss(s);
  string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

/**  * Parser the string format roi into Rect struct.  */
Rect parse_roi(const string& roi_str){
  vector<string> pos;
  split(roi_str, ',', pos);
  vector<int> pos_int;
  for(size_t i = 0; i < pos.size(); ++i){
    pos_int.push_back(atoi(pos[i].c_str()));
  }
  return Rect(pos_int[0], pos_int[1], pos_int[2], pos_int[3]);
}

void parse_line(const string& line, string& file, vector<Rect>& rois){
  rois.clear();
  vector<string> segs;
  split(line, ' ', segs);
  file = segs[0];
  rois.clear();
  for(size_t i = 1; i < segs.size(); ++i){
    rois.push_back(parse_roi(segs[i]));
  }
}

Point inline center_of_rect(Rect& rect) {
    return Point(rect.x + rect.width/2, rect.y + rect.height/2);
}

void mask2label(const string& line,
                string& img_file,
                vector<float>& labels_of_image,
                int input_width,
                int input_height, const Rect& roi){
  vector<Rect> rects;
  parse_line(line, img_file, rects);
  bool found = false;
  long min_distance = input_width*input_width+input_height*input_height;
  Rect center;
  for(int i = 0; i < rects.size(); ++i){
      Rect& rect = rects[i];
      Point pc = center_of_rect(rect);
      if (pc.x > input_width/2 - 1.5*rect.width &&
          pc.x < input_width/2 + 1.5*rect.width &&
          pc.y > input_height*(0.4) &&
          pc.y < input_height){
          long distance = (pc.x-input_width/2)*(pc.x-input_width/2)+(pc.y - input_height/2)*(pc.y - input_height/2);
          if (distance < min_distance){
              center  = rect;
              found = true;
              min_distance = distance;
          }
      }
  }
  if (found){
      labels_of_image.push_back(1);
      labels_of_image.push_back((center.x+center.width/4 - roi.x)/(float)roi.width);
      labels_of_image.push_back((center.y+center.height/4 - roi.y)/(float)roi.height);
      labels_of_image.push_back((center.x+center.width/2 - roi.x) /(float)roi.width);
      labels_of_image.push_back((center.x+center.height/2- roi.y) /(float)roi.height);
  }else{
      labels_of_image.push_back(0);
      labels_of_image.push_back(0);
      labels_of_image.push_back(0);
      labels_of_image.push_back(0);
      labels_of_image.push_back(0);
  }

}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, std::vector<float> > > lines;

  cv::Rect roi(FLAGS_input_width/4, FLAGS_input_height/3, FLAGS_input_width/2, FLAGS_input_height*2/3);
  std::string line;
  while (std::getline(infile,line)) {
      std::string filename;
      std::vector<float> vec_label;
      mask2label(line, filename, vec_label,
                 FLAGS_input_width, FLAGS_input_height, roi);
      lines.push_back(std::make_pair(filename, vec_label));
  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    //std::cout << line_id << std::endl;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, resize_height, resize_width, is_color,
        enc, &datum, roi);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 100 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 100 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
