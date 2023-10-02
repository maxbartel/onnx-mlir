// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s


func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x15x15xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], pads = [1, 1, 1, 1], strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x15x15xf32>
  return %0 : tensor<5x2x15x15xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_stride_13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<2x3x64x64xf32>, [[PARAM_2_:%.+]]: tensor<2xf32>) -> tensor<5x2x15x15xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.conv2d [[VAR_1_]], [[VAR_2_]], [[PARAM_2_]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x15x15x2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_3_]], [[VAR_4_]] : (tensor<5x15x15x2xf32>, tensor<4xi32>) -> tensor<5x2x15x15xf32>
// CHECK:           return [[VAR_5_]] : tensor<5x2x15x15xf32>
}

// -----
func.func @test_onnx_conv2d_novalue(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<2x3x64x64xf32>) ->  tensor<5x2x197x199xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x256x256xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x197x199xf32>
  return %0 : tensor<5x2x197x199xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_novalue
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<2x3x64x64xf32>) -> tensor<5x2x197x199xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.conv2d [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 3, 2, 4>, stride = array<i64: 1, 1>} : (tensor<5x256x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x197x199x2xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_6_:%.+]] = tosa.transpose [[VAR_4_]], [[VAR_5_]] : (tensor<5x197x199x2xf32>, tensor<4xi32>) -> tensor<5x2x197x199xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x2x197x199xf32>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad(%arg0: tensor<5x3x256x256xf32>, %arg1 : tensor<7x3x64x64xf32>) ->   tensor<5x7x15x15xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) {strides = [13, 13]} : (tensor<5x3x256x256xf32>, tensor<7x3x64x64xf32>, none) ->  tensor<5x7x15x15xf32>
  return %0 :  tensor<5x7x15x15xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_no_dilation_pad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x256xf32>, [[PARAM_1_:%.+]]: tensor<7x3x64x64xf32>) -> tensor<5x7x15x15xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<7x3x64x64xf32>, tensor<4xi32>) -> tensor<7x64x64x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<7xf32>}> : () -> tensor<7xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.conv2d [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 13, 13>} : (tensor<5x256x256x3xf32>, tensor<7x64x64x3xf32>, tensor<7xf32>) -> tensor<5x15x15x7xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_6_:%.+]] = tosa.transpose [[VAR_4_]], [[VAR_5_]] : (tensor<5x15x15x7xf32>, tensor<4xi32>) -> tensor<5x7x15x15xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x7x15x15xf32>
}

// -----
func.func @test_onnx_conv2d_no_dilation_pad_stride(%arg0: tensor<5x3x256x260xf32>, %arg1 : tensor<2x3x60x64xf32>) ->  tensor<5x2x197x197xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) : (tensor<5x3x256x260xf32>, tensor<2x3x60x64xf32>, none) ->  tensor<5x2x197x197xf32>
  return %0 : tensor<5x2x197x197xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_no_dilation_pad_stride
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x256x260xf32>, [[PARAM_1_:%.+]]: tensor<2x3x60x64xf32>) -> tensor<5x2x197x197xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x256x260xf32>, tensor<4xi32>) -> tensor<5x256x260x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<2x3x60x64xf32>, tensor<4xi32>) -> tensor<2x60x64x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.conv2d [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x256x260x3xf32>, tensor<2x60x64x3xf32>, tensor<2xf32>) -> tensor<5x197x197x2xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_6_:%.+]] = tosa.transpose [[VAR_4_]], [[VAR_5_]] : (tensor<5x197x197x2xf32>, tensor<4xi32>) -> tensor<5x2x197x197xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x2x197x197xf32>
}

// -----
func.func @test_onnx_conv2d_group(%arg0: tensor<5x64x256x256xf32>, %arg1 : tensor<12x16x45x45xf32>, %arg2: tensor<12xf32>) ->  tensor<5x12x17x17xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 1, 1, 1], strides = [13, 13], group = 4 : si64} : (tensor<5x64x256x256xf32>, tensor<12x16x45x45xf32>, tensor<12xf32>) ->  tensor<5x12x17x17xf32>
  return %0 : tensor<5x12x17x17xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_group
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x64x256x256xf32>, [[PARAM_1_:%.+]]: tensor<12x16x45x45xf32>, [[PARAM_2_:%.+]]: tensor<12xf32>) -> tensor<5x12x17x17xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x64x256x256xf32>, tensor<4xi32>) -> tensor<5x256x256x64xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<12x16x45x45xf32>, tensor<4xi32>) -> tensor<12x45x45x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.slice [[VAR_1_]] {size = array<i64: 5, 256, 256, 16>, start = array<i64: 0, 0, 0, 0>} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.slice [[VAR_2_]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 0, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.slice [[PARAM_2_]] {size = array<i64: 3>, start = array<i64: 0>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.conv2d [[VAR_3_]], [[VAR_4_]], [[VAR_5_]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tosa.slice [[VAR_1_]] {size = array<i64: 5, 256, 256, 16>, start = array<i64: 0, 0, 0, 16>} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = tosa.slice [[VAR_2_]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 3, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tosa.slice [[PARAM_2_]] {size = array<i64: 3>, start = array<i64: 3>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = tosa.conv2d [[VAR_7_]], [[VAR_8_]], [[VAR_9_]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tosa.slice [[VAR_1_]] {size = array<i64: 5, 256, 256, 16>, start = array<i64: 0, 0, 0, 32>} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tosa.slice [[VAR_2_]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 6, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = tosa.slice [[PARAM_2_]] {size = array<i64: 3>, start = array<i64: 6>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]] = tosa.conv2d [[VAR_11_]], [[VAR_12_]], [[VAR_13_]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = tosa.slice [[VAR_1_]] {size = array<i64: 5, 256, 256, 16>, start = array<i64: 0, 0, 0, 48>} : (tensor<5x256x256x64xf32>) -> tensor<5x256x256x16xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = tosa.slice [[VAR_2_]] {size = array<i64: 3, 45, 45, 16>, start = array<i64: 9, 0, 0, 0>} : (tensor<12x45x45x16xf32>) -> tensor<3x45x45x16xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = tosa.slice [[PARAM_2_]] {size = array<i64: 3>, start = array<i64: 9>} : (tensor<12xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_18_:%.+]] = tosa.conv2d [[VAR_15_]], [[VAR_16_]], [[VAR_17_]] {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 13, 13>} : (tensor<5x256x256x16xf32>, tensor<3x45x45x16xf32>, tensor<3xf32>) -> tensor<5x17x17x3xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = tosa.concat [[VAR_6_]], [[VAR_10_]], [[VAR_14_]], [[VAR_18_]] {axis = 3 : i32} : (tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>, tensor<5x17x17x3xf32>) -> tensor<5x17x17x12xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_21_:%.+]] = tosa.transpose [[VAR_19_]], [[VAR_20_]] : (tensor<5x17x17x12xf32>, tensor<4xi32>) -> tensor<5x12x17x17xf32>
// CHECK:           return [[VAR_21_]] : tensor<5x12x17x17xf32>
}

// -----
func.func @test_onnx_conv2d_autopad(%arg0: tensor<5x3x125x256xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) ->  tensor<5x2x125x256xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "SAME_LOWER"} : (tensor<5x3x125x256xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) ->  tensor<5x2x125x256xf32>
  return %0 : tensor<5x2x125x256xf32>
// CHECK-LABEL:  func.func @test_onnx_conv2d_autopad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x125x256xf32>, [[PARAM_1_:%.+]]: tensor<2x3x64x64xf32>, [[PARAM_2_:%.+]]: tensor<2xf32>) -> tensor<5x2x125x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x3x125x256xf32>, tensor<4xi32>) -> tensor<5x125x256x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.transpose [[PARAM_1_]], [[VAR_0_]] : (tensor<2x3x64x64xf32>, tensor<4xi32>) -> tensor<2x64x64x3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.conv2d [[VAR_1_]], [[VAR_2_]], [[PARAM_2_]] {dilation = array<i64: 1, 1>, pad = array<i64: 32, 31, 32, 31>, stride = array<i64: 1, 1>} : (tensor<5x125x256x3xf32>, tensor<2x64x64x3xf32>, tensor<2xf32>) -> tensor<5x125x256x2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_3_]], [[VAR_4_]] : (tensor<5x125x256x2xf32>, tensor<4xi32>) -> tensor<5x2x125x256xf32>
// CHECK:           return [[VAR_5_]] : tensor<5x2x125x256xf32>
}