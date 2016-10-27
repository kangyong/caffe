#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/blob_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {


template <typename Dtype>
BlobDataLayer<Dtype>::BlobDataLayer(const LayerParameter& param)
  : BaseDataLayer<Dtype>(param),
    prefetch_free_(),
    prefetch_full_(),
    reader_(param) {
        this->uni_data_param = param.uni_data_param();

        int data_size = this->uni_data_param.data_shape_size();
        int labels_size = this->uni_data_param.labels_shape_size();
        for (size_t i = 0; i < PREFETCH_COUNT; i++) {
            prefetch_.push_back(new BlobBatch<Dtype>(data_size, labels_size));
        }
        for (size_t i = 0; i < PREFETCH_COUNT; ++i) {
            prefetch_free_.push(prefetch_[i]);
        }
}

template <typename Dtype>
BlobDataLayer<Dtype>::~BlobDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BlobDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    for (size_t j = 0; j < prefetch_[i]->data_.size(); j++) {
        prefetch_[i]->WarmUpCPU();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      for (size_t j = 0; j < prefetch_[i]->data_.size(); j++) {
            prefetch_[i]->WarmUpGPU();
        }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BlobDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  MultiDatum& multi_datum = *(reader_.full().peek());

  int top_index = 0;
  for(size_t i = 0; i < multi_datum.data_size(); ++i, ++top_index){
        const Datum& datum = multi_datum.data(i);
        // Use data_transformer to infer the expected blob shape from datum.
        vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
        this->transformed_data_.Reshape(top_shape);
        // Reshape top[0] and prefetch_data according to the batch_size.
        top_shape[0] = batch_size;
        top[top_index]->Reshape(top_shape);
        for (int prefetch_id = 0; prefetch_id < this->PREFETCH_COUNT; ++prefetch_id) {
	  CHECK_GT(prefetch_[prefetch_id]->data_.size(), 0);
            this->prefetch_[prefetch_id]->data_[i]->Reshape(top_shape);
        }
	DLOG(INFO) << "output data size: " << top[top_index]->num() << ","
            << top[top_index]->channels() << "," << top[top_index]->height() << ","
		  << top[top_index]->width();
  }
  // labels
  //  if (this->output_labels_) {
    for (int i = 0; i < uni_data_param.labels_shape_size(); ++i, ++top_index){
        vector<int> label_shape;
        label_shape.push_back(batch_size);
        label_shape.push_back(uni_data_param.labels_shape(i).channels());
        label_shape.push_back(uni_data_param.labels_shape(i).height());
        label_shape.push_back(uni_data_param.labels_shape(i).width());
        top[top_index]->Reshape(label_shape);
        for (int prefetch_id = 0; prefetch_id < this->PREFETCH_COUNT; ++prefetch_id) {
	  this->prefetch_[prefetch_id]->labels_[i]->Reshape(label_shape);
	}
    }
    //}
}
  

// This function is called on prefetch thread
template<typename Dtype>
void BlobDataLayer<Dtype>::load_batch(BlobBatch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

  CHECK(batch->data_.size());
  for (size_t i = 0; i < batch->data_.size(); i++) {
      CHECK(batch->data_[i]->count());
  }
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  MultiDatum& data = *(reader_.full().peek());
  for (size_t i = 0; i < data.data_size(); i++) {
      const Datum& datum = data.data(i);
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_[i]->Reshape(top_shape);
  }

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    MultiDatum& data = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();

    for (size_t i = 0; i < data.data_size(); i++) {
        Dtype* top_data = batch->data_[i]->mutable_cpu_data();
        const Datum& datum = data.data(i);
        int offset = batch->data_[i]->offset(item_id);
        this->transformed_data_.set_cpu_data(top_data+offset);
        this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    // Copy label.
    if (this->output_labels_) {
        long long accumulate_offset = 0;
        for (size_t i = 0; i < this->uni_data_param.labels_shape_size(); i++) {
            Dtype* top_label = batch->labels_[i]->mutable_cpu_data();
            const UniDataShape& shape = uni_data_param.labels_shape(i);
            long long offset = shape.channels()*shape.width()*shape.height();
            for(int j = 0; j < offset; ++j){
		top_label[item_id*offset + j] = data.labels(accumulate_offset+j);
            }
            accumulate_offset += offset;
        }
    }
    trans_time += timer.MicroSeconds();
    reader_.free().push(const_cast<MultiDatum*>(&data));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void BlobDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif
  try {
    while (!must_stop()) {
      BlobBatch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
          for(int i = 0; i < batch->data_.size(); ++i){
              batch->data_[i]->data()->async_gpu_push(stream);
              CUDA_CHECK(cudaStreamSynchronize(stream));
          }
          //batch->data_.data().get()->async_gpu_push(stream);
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}


template <typename Dtype>
void BlobDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BlobBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  int top_index = 0;
  for (size_t i = 0; i < batch->data_.size(); i++, top_index++) {
      top[i]->ReshapeLike(*(batch->data_[i]));
      // Copy the data
      caffe_copy(batch->data_[i]->count(), batch->data_[i]->cpu_data(),
        top[top_index]->mutable_cpu_data());
  }
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    for (size_t i = 0; i < batch->labels_.size(); i++, top_index++) {
        // Reshape to loaded labels.
        top[top_index]->ReshapeLike(*(batch->labels_[i]));
        // Copy the labels.
        caffe_copy(batch->labels_[i]->count(), batch->labels_[i]->cpu_data(),
            top[top_index]->mutable_cpu_data());
    }
  }
  prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BlobDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BlobDataLayer);
REGISTER_LAYER_CLASS(BlobData);

}  // namespace caffe
