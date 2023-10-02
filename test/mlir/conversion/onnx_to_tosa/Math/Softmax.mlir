// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_softmax_v13(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.Softmax"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK-LABEL:  func.func @test_softmax_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.exp [[PARAM_0_]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_sum [[VAR_0_]] {axis = 2 : i32} : (tensor<13x21x3xf32>) -> tensor<13x21x1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.reciprocal [[VAR_1_]] : (tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.mul [[VAR_0_]], [[VAR_2_]] {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
// CHECK:           return [[VAR_3_]] : tensor<13x21x3xf32>
}

// -----

func.func @test_softmax_v13_axis_one(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK-LABEL:  func.func @test_softmax_v13_axis_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.exp [[PARAM_0_]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_sum [[VAR_0_]] {axis = 1 : i32} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.reciprocal [[VAR_1_]] : (tensor<13x1x3xf32>) -> tensor<13x1x3xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.mul [[VAR_0_]], [[VAR_2_]] {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
// CHECK:           return [[VAR_3_]] : tensor<13x21x3xf32>
}

// -----

func.func @test_softmax_before_v13(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.SoftmaxV11"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK-LABEL:  func.func @test_softmax_before_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.exp [[PARAM_0_]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_sum [[VAR_0_]] {axis = 1 : i32} : (tensor<13x21x3xf32>) -> tensor<13x1x3xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.reduce_sum [[VAR_1_]] {axis = 2 : i32} : (tensor<13x1x3xf32>) -> tensor<13x1x1xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.reciprocal [[VAR_2_]] : (tensor<13x1x1xf32>) -> tensor<13x1x1xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.mul [[VAR_0_]], [[VAR_3_]] {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<13x1x1xf32>) -> tensor<13x21x3xf32>
// CHECK:           return [[VAR_4_]] : tensor<13x21x3xf32>
}

// -----

func.func @test_softmax_before_v13_axis_zero(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %2 = "onnx.SoftmaxV11"(%arg0) {axis = 0 : si64}: (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %2 : tensor<13x21x3xf32>
// CHECK-LABEL:  func.func @test_softmax_before_v13_axis_zero
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.exp [[PARAM_0_]] : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_sum [[VAR_0_]] {axis = 0 : i32} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.reduce_sum [[VAR_1_]] {axis = 1 : i32} : (tensor<1x21x3xf32>) -> tensor<1x1x3xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.reduce_sum [[VAR_2_]] {axis = 2 : i32} : (tensor<1x1x3xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_4_:%.+]] = tosa.reciprocal [[VAR_3_]] : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_5_:%.+]] = tosa.mul [[VAR_0_]], [[VAR_4_]] {shift = 0 : i32} : (tensor<13x21x3xf32>, tensor<1x1x1xf32>) -> tensor<13x21x3xf32>
// CHECK:           return [[VAR_5_]] : tensor<13x21x3xf32>
}