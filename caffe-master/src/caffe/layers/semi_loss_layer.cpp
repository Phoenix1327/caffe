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
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  
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
  const Dtype* clabel = bottom[1]->cpu_data();//-1(negative);0(weakly);1(positive)
  const Dtype* wlabel = bottom[2]->cpu_data();//-1(pos & neg);0,1,2,3...(weakly bag idx)
  
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
   
  //data permutation: positive/negative -> weakly_bags1 ->weakly_bags2 -> ...
  caffe_set(count, Dtype(0), ww);
  bool start_count = true;
  int num_int_p = 0;//number of positive samples
  int num_int_n = 0;//number of negative samples  
  int num_int_w = 0;//number of bags
  int max_idx = 0;
  float loss_p = 0;
  float loss_n = 0;
  float loss_w = 0;
  for (int i = 0; i < num; ++i) {
	
	//positive/negative
	if (clabel[i] == Dtype(1)) {
	  num_int_p += 1;
	  loss_p += -log(1 / (1 + exp(-bottom_data[i])));
	  pw[i] = Dtype(1);
      nw[i] = Dtype(0);
	}
	if (clabel[i] == Dtype(-1)) {
	  num_int_n += 1;
	  loss_n += -log(1 / (1 + exp(bottom_data[i])));
	  nw[i] = Dtype(1);
	  pw[i] = Dtype(0);
	}
	//weakly_bags
	if (clabel[i] == Dtype(0)) {
	  //the start of a weakly_bag
	  if (start_count == true) {
		num_int_w += 1;
		start_count = false;
		max_idx = i;
		pw[i] = Dtype(0);
		nw[i] = Dtype(0);
	  }
	  //not start
	  else {
		//the end of a weakly_bag, the start of a new weakly_bag
	    if (wlabel[i] != wlabel[i-1]){
			ww[max_idx] = Dtype(1);
			start_count = true;
			i -= 1;
			loss_w += -log(1 / (1 + exp(-bottom_data[max_idx])));
			pw[i] = Dtype(0);
			nw[i] = Dtype(0);
		}
		//not end
		else {
		  if (static_cast<float>(bottom_data[i]) >= static_cast<float>(bottom_data[i-1])) {
			  max_idx = i;
		  }
		  pw[i] = Dtype(0);
		  nw[i] = Dtype(0);
		}
	  }
	}
  }
  //the end of the data 
  ww[max_idx] = Dtype(1);
  loss_w += -log(1 / (1 + exp(-bottom_data[max_idx])));
  
  num_p = Dtype(num_int_p);
  num_n = Dtype(num_int_n);
  num_w = Dtype(num_int_w);
  if (num_int_p != 0) {
	loss_p /= num_int_p;
  }
  if (num_int_n != 0) {
	loss_n /= num_int_n;
  }
  if (num_int_w != 0) {
	loss_w /= num_int_w;
  }
  
  Dtype* loss = (*top)[0]->mutable_cpu_data();
  loss[0] = loss_p + loss_n + loss_w;    
}

template <typename Dtype>
void SemiLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[2]) {
	LOG(FATAL) << this->type_name()
	           << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	
	Dtype* pw = positive_weights_.mutable_cpu_data();
    Dtype* nw = negative_weights_.mutable_cpu_data();
    Dtype* ww = weakly_weights_.mutable_cpu_data();
	
    int count = (*bottom)[0]->count();
	
	int num_int_p = static_cast<float>(num_p);
	int num_int_n = static_cast<float>(num_n);
	int num_int_w = static_cast<float>(num_w);
	
    const Dtype loss_weight = top[0]->cpu_diff()[0];
	for (int i = 0; i < count; ++i) {
	  if (num_int_p != 0 && pw[i] != 0) {
	    bottom_diff[i] = loss_weight * alpha_ / num_int_p * (1 / (1 + exp(-bottom_data[i])) - 1);
	  }
	  else if (num_int_n != 0 && nw[i] != 0) {
		bottom_diff[i] = loss_weight * beta_ / num_int_n * (1 - 1 / (1 + exp(-bottom_data[i])));
	  }
	  else if (num_int_w != 0 && ww[i] != 0) {
		bottom_diff[i] = loss_weight * gamma_ / num_int_w * (1 / (1 + exp(-bottom_data[i])) - 1);
	  }
	  else {
		bottom_diff[i] = 0;
	  }
	}
  }
}

INSTANTIATE_CLASS(SemiLossLayer);

}  // namespace caffe
