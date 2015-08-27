#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SemiLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  
  alpha_ = this->layer_param_.semi_loss_param().alpha();
  beta_ = this->layer_param_.semi_loss_param().beta();
  gamma_ = this->layer_param_.semi_loss_param().gamma();
  positive_weights_.Reshape(bottom[0]->num(), 1, 1, 1);
  negative_weights_.Reshape(bottom[0]->num(), 1, 1, 1);
  weakly_weights_.Reshape(bottom[0]->num(), 1, 1, 1);
  num_p = Dtype(0);
  num_n = Dtype(0);
  num_w = Dtype(0);
		
}

template <typename Dtype>
void SemiLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  
  Dtype* pw = positive_weights_.mutable_cpu_data();
  Dtype* nw = negative_weights_.mutable_cpu_data();
  Dtype* ww = weakly_weights_.mutable_cpu_data();
  
  int num = bottom[0]->num();
  /*number of samples 
   *i.e. one batch has 50 samples, num = 50*/
  int count = bottom[0]->count();
  int dim = count / num;
  /*dimension of the sample's score(network's output) 
   *i.e. there are 20 categories in all, dim = 20*/ 
   /*in this case, it's a binary classification task, so dim = 1*/
   
  for (int i = 0; i < num; ++i) {
	if (label[i] == Dtype(1)) {
	  pw[i] = alpha_;
      nw[i] = Dtype(0);
      ww[i] = Dtype(0);	  
	}
	if (label[i] == Dtype(-1)) {
	  nw[i] = beta_;
	  pw[i] = Dtype(0);
	  ww[i] = Dtype(0);
	}
	if (label[i] == Dtype(0)) {
	  ww[i] = gamma_;
	  pw[i] = Dtype(0);
	  nw[i] = Dtype(0);
	}
  } 
  
  Dtype* loss = (*top)[0]->mutable_cpu_data();
  
  num_p = caffe_cpu_asum(count, pw) / static_cast<float>(alpha_);
  num_n = caffe_cpu_asum(count, nw) / static_cast<float>(beta_);
  num_w = caffe_cpu_asum(count, ww) / static_cast<float>(gamma_);
  float loss_p = 0;
  float loss_n = 0;
  float loss_w = 0;
  switch (this->layer_param_.semi_loss_param().norm()) {
  case SemiLossParameter_Norm_L1:
    if (num_p != Dtype(0)) {
      loss_p = caffe_cpu_dot(count, bottom_data, pw) / static_cast<float>(num_p);
	}
	if (num_n != Dtype(0)) {
	  loss_n = caffe_cpu_dot(count, bottom_data, nw) / static_cast<float>(num_n);
	}
	if (num_w != Dtype(0)) {
	  loss_w = caffe_cpu_dot(count, bottom_data, ww) / static_cast<float>(num_w);
	}
    loss[0] = loss_p + loss_n + loss_w;
    break;
  case SemiLossParameter_Norm_L2:
    caffe_mul(count, bottom_data, bottom_data, bottom_diff);
    //caffe_copy(count, bottom_data, bottom_diff);
	
    if (num_p != Dtype(0)) {
      loss_p = caffe_cpu_dot(count, bottom_diff, pw) / static_cast<float>(num_p);
	}
	if (num_n != Dtype(0)) {
	  loss_n = caffe_cpu_dot(count, bottom_diff, nw) / static_cast<float>(num_n);
	}
	if (num_w != Dtype(0)) {
	  loss_w = caffe_cpu_dot(count, bottom_diff, ww) / static_cast<float>(num_w);
	}
    loss[0] = loss_p + loss_n + loss_w;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void SemiLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* label = (*bottom)[1]->cpu_data();
	
	Dtype* pw = positive_weights_.mutable_cpu_data();
    Dtype* nw = negative_weights_.mutable_cpu_data();
    Dtype* ww = weakly_weights_.mutable_cpu_data();
	
	int num = (*bottom)[0]->num();
    int count = (*bottom)[0]->count();
    int dim = count / num;
	
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.semi_loss_param().norm()) {
    case SemiLossParameter_Norm_L1:
	  if (num_p != Dtype(0)) {
        caffe_scal(count, loss_weight / static_cast<float>(num_p), pw);
	  }
	  if (num_n != Dtype(0)) {
	    caffe_scal(count, loss_weight / static_cast<float>(num_n), nw);
	  }
	  if (num_w != Dtype(0)) {
	    caffe_scal(count, loss_weight / static_cast<float>(num_w), ww);
	  }
	  
	  caffe_set(count, Dtype(0), bottom_diff);
	  caffe_add(count, bottom_diff, pw, bottom_diff);
	  caffe_add(count, bottom_diff, nw, bottom_diff);
	  caffe_add(count, bottom_diff, ww, bottom_diff);
      break;
    case SemiLossParameter_Norm_L2:
	  caffe_mul(count, bottom_data, pw, pw);
	  if (num_p != Dtype(0)) {
        caffe_scal(count, loss_weight * 2/ static_cast<float>(num_p), pw);
	  }
	  caffe_mul(count, bottom_data, nw, nw);
      if (num_n != Dtype(0)) {
        caffe_scal(count, loss_weight * 2/ static_cast<float>(num_n), nw);
	  }
	  caffe_mul(count, bottom_data, ww, ww);
      if (num_w != Dtype(0)) {
        caffe_scal(count, loss_weight * 2/ static_cast<float>(num_w), ww);
	  }
	  
	  caffe_set(count, Dtype(0), bottom_diff);
	  caffe_add(count, bottom_diff, pw, bottom_diff);
	  caffe_add(count, bottom_diff, nw, bottom_diff);
	  caffe_add(count, bottom_diff, ww, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(SemiLossLayer);

}  // namespace caffe
