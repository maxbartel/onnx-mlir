// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

/// Test the default behavior of Max Pool with no padding
func.func @test_default_maxpoolsingleout(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  "func.return"(%0) : (tensor<5x5x30x30xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_maxpoolsingleout
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.max_pool2d [[VAR_1_]] {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>) -> tensor<5x30x30x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<5x30x30x5xf32>, tensor<4xi32>) -> tensor<5x5x30x30xf32>
// CHECK:           return [[VAR_4_]] : tensor<5x5x30x30xf32>
}

// -----

/// Test the behavior of Max Pool with uniform padding
func.func @test_default_maxpoolsingleout_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  "func.return"(%0) : (tensor<5x5x32x32xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_maxpoolsingleout_pad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.max_pool2d [[VAR_1_]] {kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<5x32x32x5xf32>, tensor<4xi32>) -> tensor<5x5x32x32xf32>
// CHECK:           return [[VAR_4_]] : tensor<5x5x32x32xf32>
}

// -----

/// Test the behavior of Max Pool with non uniform padding
func.func @test_default_maxpoolsingleout_pad_nonunif(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [0, 1, 2, 3] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32>
  "func.return"(%0) : (tensor<5x5x30x34xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_maxpoolsingleout_pad_nonunif
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x30x34xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.max_pool2d [[VAR_1_]] {kernel = array<i64: 5, 3>, pad = array<i64: 0, 2, 1, 3>, stride = array<i64: 1, 1>} : (tensor<5x32x32x5xf32>) -> tensor<5x30x34x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<5x30x34x5xf32>, tensor<4xi32>) -> tensor<5x5x30x34xf32>
// CHECK:           return [[VAR_4_]] : tensor<5x5x30x34xf32>
}

// -----

/// Test the behavior of Max Pool with strides set
func.func @test_default_maxpoolsingleout_strides(%arg0 : tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3,3], pads = [1, 1, 1, 1], strides = [2, 2] } : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_maxpoolsingleout_strides
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x32x32xf32>, tensor<4xi32>) -> tensor<5x32x32x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.max_pool2d [[VAR_1_]] {kernel = array<i64: 3, 3>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<5x32x32x5xf32>) -> tensor<5x16x16x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<5x16x16x5xf32>, tensor<4xi32>) -> tensor<5x5x16x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<5x5x16x16xf32>
}



// -----

/// Test the behavior of Max Pool with strides and non uniform padding 
func.func @test_default_maxpoolsingleout_strides_nonunifpad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
  "func.return"(%0) : (tensor<5x5x15x16xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_maxpoolsingleout_strides_nonunifpad
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x30x32xf32>, tensor<4xi32>) -> tensor<5x30x32x5xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.slice [[VAR_1_]] {size = array<i64: 5, 29, 32, 5>, start = array<i64: 0, 0, 0, 0>} : (tensor<5x30x32x5xf32>) -> tensor<5x29x32x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.max_pool2d [[VAR_2_]] {kernel = array<i64: 2, 2>, pad = array<i64: 1, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<5x29x32x5xf32>) -> tensor<5x15x16x5xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_5_:%.+]] = tosa.transpose [[VAR_3_]], [[VAR_4_]] : (tensor<5x15x16x5xf32>, tensor<4xi32>) -> tensor<5x5x15x16xf32>
// CHECK:           return [[VAR_5_]] : tensor<5x5x15x16xf32>
}

// -----

/// Test the behavior of Max Pool with ceiling set (Should change the result shape)
func.func @test_default_maxpoolsingleout_strides_nonunifpad_ceil(%arg0 : tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {ceil_mode = 1 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
  "func.return"(%0) : (tensor<5x5x16x16xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_maxpoolsingleout_strides_nonunifpad_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x30x32xf32>, tensor<4xi32>) -> tensor<5x30x32x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.max_pool2d [[VAR_1_]] {kernel = array<i64: 2, 2>, pad = array<i64: 1, 1, 0, 0>, stride = array<i64: 2, 2>} : (tensor<5x30x32x5xf32>) -> tensor<5x16x16x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<5x16x16x5xf32>, tensor<4xi32>) -> tensor<5x5x16x16xf32>
// CHECK:           return [[VAR_4_]] : tensor<5x5x16x16xf32>
}

// -----

func.func @test_default_maxpoolsingleout_autopad_valid(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "VALID", kernel_shape = [3,3]} : (tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32>
  "func.return"(%0) : (tensor<5x5x14x11xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_maxpoolsingleout_autopad_valid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x16x13xf32>) -> tensor<5x5x14x11xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x16x13xf32>, tensor<4xi32>) -> tensor<5x16x13x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.max_pool2d [[VAR_1_]] {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x16x13x5xf32>) -> tensor<5x14x11x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<5x14x11x5xf32>, tensor<4xi32>) -> tensor<5x5x14x11xf32>
// CHECK:           return [[VAR_4_]] : tensor<5x5x14x11xf32>
}

// -----

func.func @test_default_maxpoolsingleout_same_upper_ceil_mode(%arg0 : tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_UPPER", ceil_mode = 1 : si64, kernel_shape = [4,4], strides = [4, 4] } : (tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32>
  "func.return"(%0) : (tensor<5x5x4x4xf32>) -> ()
// CHECK-LABEL:  func.func @test_default_maxpoolsingleout_same_upper_ceil_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x16x13xf32>) -> tensor<5x5x4x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_1_:%.+]] = tosa.transpose [[PARAM_0_]], [[VAR_0_]] : (tensor<5x5x16x13xf32>, tensor<4xi32>) -> tensor<5x16x13x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.max_pool2d [[VAR_1_]] {kernel = array<i64: 4, 4>, pad = array<i64: 0, 0, 1, 2>, stride = array<i64: 4, 4>} : (tensor<5x16x13x5xf32>) -> tensor<5x4x4x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
// CHECK:           [[VAR_4_:%.+]] = tosa.transpose [[VAR_2_]], [[VAR_3_]] : (tensor<5x4x4x5xf32>, tensor<4xi32>) -> tensor<5x5x4x4xf32>
// CHECK:           return [[VAR_4_]] : tensor<5x5x4x4xf32>
}
