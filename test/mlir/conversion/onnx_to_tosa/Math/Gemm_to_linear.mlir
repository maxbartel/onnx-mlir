// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

// -----


func.func @gemm_to_fc(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @gemm_to_fc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>, [[PARAM_2_:%.+]]: tensor<4xf32>) -> tensor<1x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.fully_connected [[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]] : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
// CHECK:           return [[VAR_0_]] : tensor<1x4xf32>
// CHECK:         }
}

// -----


func.func @gemm_to_fc_broadcast(%arg0: tensor<2x5xf32>, %arg1: tensor<4x5xf32>, %arg2: tensor<1xf32>) -> tensor<2x4xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<2x5xf32>, tensor<4x5xf32>, tensor<1xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @gemm_to_fc_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>, [[PARAM_2_:%.+]]: tensor<1xf32>) -> tensor<2x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.fully_connected [[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]] : (tensor<2x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.reshape [[PARAM_2_]] {new_shape = array<i64: 1, 1>} : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.add [[VAR_1_]], [[VAR_2_]] : (tensor<2x4xf32>, tensor<1x1xf32>) -> tensor<2x4xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x4xf32>
// CHECK:         }
}

// -----


func.func @gemm_to_fc_opt(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @gemm_to_fc_opt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x5xf32>, [[PARAM_1_:%.+]]: tensor<4x5xf32>) -> tensor<1x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.fully_connected [[PARAM_0_]], [[PARAM_1_]], [[VAR_1_]] : (tensor<1x5xf32>, tensor<4x5xf32>, tensor<4xf32>) -> tensor<1x4xf32>
// CHECK:           return [[VAR_2_]] : tensor<1x4xf32>
// CHECK:         }
}
