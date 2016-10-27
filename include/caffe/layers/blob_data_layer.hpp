#ifndef CAFFE_BLOB_DATA_LAYER_HPP_
#define CAFFE_BLOB_DATA_LAYER_HPP_

#include <vector>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"

#include "caffe/blobdata_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"



namespace caffe {

template <typename Dtype>
class BlobBatch {
    public:
        vector<Blob<Dtype>* > data_;
        vector<Blob<Dtype>* > labels_;
        BlobBatch(size_t data_size, size_t labels_size){
            for (size_t i = 0; i < data_size; i++) {
	        data_.push_back(new Blob<Dtype>());
            }
            for (size_t i = 0; i < labels_size; i++) {
                labels_.push_back(new Blob<Dtype>());
            }
        }

        ~BlobBatch(){
            for (size_t i = 0; i < data_.size(); i++) {
                delete data_[i];
            }
            for (size_t i = 0; i < labels_.size(); i++) {
                delete labels_[i];
            }
        }
  
        void WarmUpCPU(){
            for (size_t i = 0; i < data_.size(); i++) {
                data_[i]->mutable_cpu_data();
            }
            for (size_t i = 0; i < labels_.size(); i++) {
                labels_[i]->mutable_cpu_data();
            }
        }

        void WarmUpGPU(){
            for (size_t i = 0; i < data_.size(); i++) {
                data_[i]->mutable_gpu_data();
            }
            for (size_t i = 0; i < labels_.size(); i++) {
                labels_[i]->mutable_gpu_data();
            }
        }
};

// New data_layer from scratch.
template <typename Dtype>
class BlobDataLayer : public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BlobDataLayer(const LayerParameter& param);
  virtual ~BlobDataLayer();
  void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;

  // BlobDataLayer uses BlobDataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "UniData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const {
      return uni_data_param.data_shape_size()
        + uni_data_param.labels_shape_size();
  }
  virtual inline int MaxTopBlobs() const {
      return uni_data_param.data_shape_size()
        + uni_data_param.labels_shape_size();
    }

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(BlobBatch<Dtype>* batch);

  vector< BlobBatch<Dtype>* > prefetch_;

  BlockingQueue<BlobBatch<Dtype>*> prefetch_free_;
  BlockingQueue<BlobBatch<Dtype>*> prefetch_full_;
  Blob<Dtype> transformed_data_;
  UniDataParameter uni_data_param;
  BlobDataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_BLOB_DATA_LAYER_HPP_
