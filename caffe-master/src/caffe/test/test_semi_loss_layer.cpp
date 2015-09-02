#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SemiLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SemiLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(20, 1, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(20, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(20);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < 5; ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = (caffe_rng_rand() % 2) - 2;//-2 or -1,positive or negative
    }
	for (int i = 5; i < 10; ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = 0;//weakly bag, img idx = 0
    }
	for (int i = 10; i < 15; ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = (caffe_rng_rand() % 2) - 2;//-2 or -1,positive or negative
    }
	for (int i = 15; i < 20; ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = 1;//weakly bag, img idx = 1
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SemiLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SemiLossLayerTest, TestDtypesAndDevices);


TYPED_TEST(SemiLossLayerTest, TestGradientL1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SemiLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 2e-3, 1701);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0);
}


}  // namespace caffe
