#include <vector>

#include "caffe/layers/blob_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BlobDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BlobBatch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  int top_index = 0;
  for (size_t i = 0; i < batch->data_.size(); i++, top_index++) {
      top[i]->ReshapeLike(*(batch->data_[i]));
      // Copy the data
      caffe_copy(batch->data_[i]->count(), batch->data_[i]->cpu_data(),
        top[top_index]->mutable_cpu_data());
  }

  if (this->output_labels_) {
    for (size_t i = 0; i < batch->labels_.size(); i++, top_index++) {
        // Reshape to loaded labels.
        top[top_index]->ReshapeLike(*(batch->labels_[i]));
        // Copy the labels.
        caffe_copy(batch->labels_[i]->count(), batch->labels_[i]->cpu_data(),
            top[top_index]->mutable_cpu_data());
    }
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BlobDataLayer);

}  // namespace caffe
