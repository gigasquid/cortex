(ns cortex.tensor.operations
  "tensor operations with syntatic sugar"
  (:refer-clojure :exclude [max min * - +])
  (:require [clojure.core.matrix :as m]
            [cortex.tensor :as tensor]
            [think.datatype.core :as dtype]))

(defn max
  "Takes an input tensor and returns the max of x or the value given, mutates the output tensor and returns it"
  ([output max-value]
   (max output output max-value))
  ([output input max-value]
   (tensor/binary-op! output 1.0 input 1.0 max-value :max)))

(defn min
  "Takes an input tensor and returns the min of x or the value given, mutates the output tensor and returns it"
  ([output min-value]
   (min output output min-value))
  ([output input min-value]
   (tensor/binary-op! output 1.0 input 1.0 min-value :min)))

(defn ceil
  "Takes an tensor returns the mutated tensor with the value with the ceiling function applied"
  ([output]
   (ceil output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :ceil)))

(defn logistic
  "Takes an tensor returns the mutated tensor with the value with the logistic function applied"
  ([output]
   (logistic output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :logistic)))

(defn tanh
  "Takes an tensor returns the mutated tensor with the value with the tanh function applied"
  ([output]
   (tanh output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :tanh)))

(defn exp
  "Takes an tensor returns the mutated tensor with the value with the exp function applied"
  ([output]
   (exp output output))
  ([output input]
   (tensor/unary-op! output 1.0 input :exp)))


(defn *
  "Takes and x1 and x2 multiples them together and puts the result in the mutated output"
  ([output x1]
   (* output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :*)))

(defn -
  "Takes and x1 and x2 subtracts them and puts the result in the mutated output"
  ([output x1]
   (- output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :-)))

(defn +
  "Takes and x1 and x2 adds them and puts the result in the mutated output"
  ([output x1]
   (+ output output x1))
  ([output x1 x2]
   (tensor/binary-op! output 1.0 x1 1.0 x2 :+)))

(defn new-tensor
  "Returns a new tensor of the same shape and type of the given output tensor"
  [output]
  (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output)))

