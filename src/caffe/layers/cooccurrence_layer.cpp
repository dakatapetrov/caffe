#include <vector>

#include <fstream>
#include <string>
#include <iostream>

#include "caffe/layers/cooccurrence_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define quote(x) #x

namespace caffe {

/* template <typename Dtype> */
/* void CooccurrenceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, */
/*       const vector<Blob<Dtype>*>& top) { */
/*   const CooccurrenceParameter& concat_param = this->layer_param_.concat_param(); */
/*   CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim())) */
/*       << "Either axis or concat_dim should be specified; not both."; */
/* } */

template <typename Dtype>
void CooccurrenceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /* const int num_axes = bottom[0]->num_axes(); */
  /* const CooccurrenceParameter& concat_param = this->layer_param_.concat_param(); */
  /* if (concat_param.has_concat_dim()) { */
  /*   concat_axis_ = static_cast<int>(concat_param.concat_dim()); */
  /*   // Don't allow negative indexing for concat_dim, a uint32 -- almost */
  /*   // certainly unintended. */
  /*   CHECK_GE(concat_axis_, 0) << "casting concat_dim from uint32 to int32 " */
  /*       << "produced negative result; concat_dim must satisfy " */
  /*       << "0 <= concat_dim < " << kMaxBlobAxes; */
  /*   CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range."; */
  /* } else { */
  /*   concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis()); */
  /* } */
  /* // Initialize with the first blob. */
  /* vector<int> top_shape = bottom[0]->shape(); */
  /* num_concats_ = bottom[0]->count(0, concat_axis_); */
  /* concat_input_size_ = bottom[0]->count(concat_axis_ + 1); */
  /* int bottom_count_sum = bottom[0]->count(); */
  /* for (int i = 1; i < bottom.size(); ++i) { */
  /*   CHECK_EQ(num_axes, bottom[i]->num_axes()) */
  /*       << "All inputs must have the same #axes."; */
  /*   for (int j = 0; j < num_axes; ++j) { */
  /*     if (j == concat_axis_) { continue; } */
  /*     CHECK_EQ(top_shape[j], bottom[i]->shape(j)) */
  /*         << "All inputs must have the same shape, except at concat_axis."; */
  /*   } */
  /*   bottom_count_sum += bottom[i]->count(); */
  /*   top_shape[concat_axis_] += bottom[i]->shape(concat_axis_); */
  /* } */
  /* top[0]->Reshape(top_shape); */
  /* CHECK_EQ(bottom_count_sum, top[0]->count()); */
  /* if (bottom.size() == 1) { */
  /*   top[0]->ShareData(*bottom[0]); */
  /*   top[0]->ShareDiff(*bottom[0]); */
  /* } */
}

template <typename Dtype>
void CooccurrenceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /* if (bottom.size() == 1) { return; } */
  /* Dtype* top_data = top[0]->mutable_cpu_data(); */
  /* int offset_concat_axis = 0; */
  /* const int top_concat_axis = top[0]->shape(concat_axis_); */
  std::ofstream out;
  out.open("/home/lifelogging/code/caffe_cooc/output.txt", std::ofstream::out | std::ofstream::app);
  out << "new forward\n";
  out << "bottom size: " << bottom.size() << "\n";
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    const int num_axes = bottom[i]->num_axes();
    out << "axes: " << num_axes << "\n";
    for (int j = 0; j < num_axes; j++) {
        out << "axis[" << j << "]: " << bottom[i]->shape(j) << "\n";
    }
    out << "height: " << bottom[i]->height() << "\n";
    out << "width: " << bottom[i]->width() << "\n";
    const int x = bottom[i]->shape(0);
    const int y = bottom[i]->shape(1);
    const int h = bottom[i]->shape(2);
    const int w = bottom[i]->shape(3);
    std::ofstream dt;
    /* dt.open("/home/lifelogging/code/caffe_cooc/conf_data.csv", std::ofstream::out | std::ofstream::app); */
    /* dt << "-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n"; */
    /* for (int p = 0; p < y; p++) { */
    /*     for (int q = 0; q < x; q++) { */
    /*         dt << bottom[i]->data_at(q, p, 0, 0) << ","; */
    /*     } */
    /*     dt << "\n"; */
    /* } */
    dt.open("/home/lifelogging/code/caffe_cooc/label_data.csv", std::ofstream::out | std::ofstream::app);
    dt << "-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    for (int p = 0; p < w; p++) {
        for (int q = 0; q < h; q++) {
            dt << bottom[i]->data_at(0, 0, h, w) << ",";
        }
        dt << "\n";
    }
    dt << "-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    dt.close();
    out << typeid(bottom_data).name()<<"\t"<< quote(bottom_data) <<"\n";
    /* out << "bottom: " << *bottom[i] << "\n"; */
    out << "data: " << *bottom_data << "\n";
    /* const int bottom_concat_axis = bottom[i]->shape(concat_axis_); */
    /* for (int n = 0; n < num_concats_; ++n) { */
    /*   caffe_copy(bottom_concat_axis * concat_input_size_, */
    /*       bottom_data + n * bottom_concat_axis * concat_input_size_, */
    /*       top_data + (n * top_concat_axis + offset_concat_axis) */
    /*           * concat_input_size_); */
    /* } */
    /* offset_concat_axis += bottom_concat_axis; */
  }
  out << "end forward\n";
  out.close();
}

/* template <typename Dtype> */
/* void CooccurrenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, */
/*       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { */
/*   if (bottom.size() == 1) { return; } */
/*   const Dtype* top_diff = top[0]->cpu_diff(); */
/*   int offset_concat_axis = 0; */
/*   const int top_concat_axis = top[0]->shape(concat_axis_); */
/*   for (int i = 0; i < bottom.size(); ++i) { */
/*     const int bottom_concat_axis = bottom[i]->shape(concat_axis_); */
/*     if (propagate_down[i]) { */
/*       Dtype* bottom_diff = bottom[i]->mutable_cpu_diff(); */
/*       for (int n = 0; n < num_concats_; ++n) { */
/*         caffe_copy(bottom_concat_axis * concat_input_size_, top_diff + */
/*             (n * top_concat_axis + offset_concat_axis) * concat_input_size_, */
/*             bottom_diff + n * bottom_concat_axis * concat_input_size_); */
/*       } */
/*     } */
/*     offset_concat_axis += bottom_concat_axis; */
/*   } */
/* } */

/* #ifdef CPU_ONLY */
/* STUB_GPU(CooccurrenceLayer); */
/* #endif */

INSTANTIATE_CLASS(CooccurrenceLayer);
REGISTER_LAYER_CLASS(Cooccurrence);

}  // namespace caffe
