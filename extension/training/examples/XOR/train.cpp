/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <gflags/gflags.h>
#include <random>

#pragma clang diagnostic ignored \
    "-Wbraced-scalar-init" // {0} below upsets clang.

using executorch::extension::FileDataLoader;
using executorch::extension::training::optimizer::SGD;
using executorch::extension::training::optimizer::SGDOptions;
using executorch::runtime::Error;
using executorch::runtime::Result;
DEFINE_string(model_path, "cifar10.pte", "Model serialized in flatbuffer format.");

const int IMAGE_SIZE = 32 * 32 * 3; // 3072 bytes per image
const int BATCH_SIZE = 32;
const int NUM_IMAGES = 10000;       // Each batch has 10,000 images

struct CIFARImage {
    int64_t label;
    std::vector<float> data;  // Store image data as float
};


// compuate batch accruacy
float accuracy(std::vector<long int> labels, std::vector<long int> preds){
      assert(labels.size() == preds.size());

      int64_t num_samples = preds.size();
      float accuracy = 0.0;
      for (int i = 0; i < num_samples; i++) {
        auto pred = preds[i];
        auto label = labels[i];      
        if (pred == label){
          accuracy += 1.0;
        }
      }
      return accuracy/num_samples;  
}

// Function to load a CIFAR-10 batch and normalize pixel values to [0,1]
// Requirement: Download CIFAR-10 in binary format and extract: 
// mkdir -p ~/Data/CIFAR10
// cd ~/Data/CIFAR10
// wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
// tar -xzf cifar-10-binary.tar.gz
std::vector<CIFARImage> load_cifar10_batch(const std::string& filename) {
    std::vector<CIFARImage> dataset;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return dataset;
    }

    for (int i = 0; i < NUM_IMAGES; i++) {
        CIFARImage image;
        image.data.resize(IMAGE_SIZE);

        uint8_t raw_data[IMAGE_SIZE];

        // Read label (1 byte)
        file.read(reinterpret_cast<char*>(&image.label), 1);
        
        // Read image data (3072 bytes)
        file.read(reinterpret_cast<char*>(raw_data), IMAGE_SIZE);

        // Convert uint8_t data to float in the range [0, 1]
        for (int j = 0; j < IMAGE_SIZE; j++) {
            image.data[j] = static_cast<float>(raw_data[j]) / 255.0f;
        }

        image.label = static_cast<int64_t>(image.label);

        dataset.push_back(image);
    }

    file.close();
    return dataset;
}


int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args: ";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  std::string file_path = "/localhome/local-hroth/Data/CIFAR10/cifar-10-batches-bin/data_batch_1.bin"; // Update the path as needed
  std::vector<CIFARImage> cifar10_dataset = load_cifar10_batch(file_path);

  if (!cifar10_dataset.empty()) {
      std::cout << "Loaded " << cifar10_dataset.size() << " images from " << file_path << std::endl;
      std::cout << "First image label: " << static_cast<int>(cifar10_dataset[0].label) << std::endl;
  }  

  // Load the model file.
  executorch::runtime::Result<executorch::extension::FileDataLoader>
      loader_res =
          executorch::extension::FileDataLoader::from(FLAGS_model_path.c_str());
  if (loader_res.error() != Error::Ok) {
    ET_LOG(Error, "Failed to open model file: %s", FLAGS_model_path.c_str());
    return 1;
  }
  auto loader = std::make_unique<executorch::extension::FileDataLoader>(
      std::move(loader_res.get()));

  auto mod = executorch::extension::training::TrainingModule(std::move(loader));

  // Create full data set of input and labels.
  std::vector<std::pair<
      executorch::extension::TensorPtr,
      executorch::extension::TensorPtr>>
      data_set;
  // convert dataset to executorch tensors
  for (int i = 0; i < NUM_IMAGES-BATCH_SIZE; i++) {
    std::vector<float> image_batch; 
    image_batch.reserve(BATCH_SIZE * cifar10_dataset[i].data.size());
    std::vector<int64_t> label_batch;
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
      image_batch.insert(image_batch.end(), cifar10_dataset[i+bs].data.begin(), cifar10_dataset[i+bs].data.end());
      label_batch.push_back(cifar10_dataset[i+bs].label);
    }
    if (i==0){
      std::cout << "Built image_batch " << image_batch.size() << std::endl;
      std::cout << "Built label_batch " << label_batch.size() << std::endl;
    }
    data_set.push_back( 
        {
          executorch::extension::make_tensor_ptr<float>({BATCH_SIZE, 3, 32, 32}, {image_batch}),
          executorch::extension::make_tensor_ptr<int64_t>({BATCH_SIZE}, {label_batch})
        }
      );
  }

  // Create optimizer.
  // Get the params and names
  auto param_res = mod.named_parameters("forward");
  if (param_res.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get named parameters");
    return 1;
  }

  SGDOptions options{0.005, 0.9};  // lr, momentum
  SGD optimizer(param_res.get(), options);

  // Randomness to sample the data set.
  std::default_random_engine URBG{std::random_device{}()};
  std::uniform_int_distribution<int> dist{
      0, static_cast<int>(data_set.size()) - 1};

  // Train the model.
  size_t num_epochs = 100000;
  for (int i = 0; i < num_epochs; i++) {
    int index = dist(URBG);
    
    auto& data = data_set[index];
    const auto& results =
        mod.execute_forward_backward("forward", {*data.first, *data.second});

    if (results.error() != Error::Ok) {
      ET_LOG(Error, "Failed to execute forward_backward");
      return 1;
    }

    // Log the progress
    if (i % 100 == 0 || i == num_epochs - 1) {
      // Inference on validation data 
      //const auto& val_results = mod.forward(*data.first);

      // Compute batch accuracies
      std::vector<long int> train_labels(data.second->const_data_ptr<int64_t>(), data.second->const_data_ptr<int64_t>() + BATCH_SIZE);
      std::vector<long int> train_preds(results.get()[1].toTensor().const_data_ptr<int64_t>(), results.get()[1].toTensor().const_data_ptr<int64_t>() + BATCH_SIZE);
      
      //std::vector<long int> val_labels(data.second->const_data_ptr<int64_t>(), data.second->const_data_ptr<int64_t>() + BATCH_SIZE);
      //std::vector<long int> val_preds(val_results.get()[1].toTensor().const_data_ptr<int64_t>(), val_results.get()[1].toTensor().const_data_ptr<int64_t>() + BATCH_SIZE);      
      
      float train_accuracy = accuracy(train_labels, train_preds);
      //float val_accuracy = accuracy(train_labels, train_preds);
      
      ET_LOG(
          Info,
          "Step %d, Loss %f, Input [%ld, %ld, %ld, %ld], Prediction %ld, Label %ld, train accuracy %.2f, val accuracy %.2f",
          i,
          results.get()[0].toTensor().const_data_ptr<float>()[0],
          data.first->size(0),
          data.first->size(1),
          data.first->size(2),
          data.first->size(3),
          results.get()[1].toTensor().const_data_ptr<int64_t>()[0],
          data.second->const_data_ptr<int64_t>()[0],
          train_accuracy,
          -1.0 //val_accuracy
          );
    }
    optimizer.step(mod.named_gradients("forward").get());
  }

  //std::map<std::string, exec_aten::Tensor> param_map;
  //for (auto& param : param_res.get()) {
  //  param_map.insert(std::pair<std::string, exec_aten::Tensor>{
  //      std::string(param.first.data()), param.second});
  // }

  //executorch::extension::flat_tensor::save_ptd("xor.ptd", param_map, 16);
}
