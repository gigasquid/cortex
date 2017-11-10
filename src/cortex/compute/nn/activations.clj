(ns cortex.compute.nn.activations
  "High level implemenations of activations that work across all backends"
  (:require [clojure.core.matrix :as m]
            [cortex.tensor :as tensor]
            [think.datatype.core :as dtype]))


;;;; Forward

(defn logistic [input output]
  (tensor/unary-op! output 1.0 input :logistic))

(defn tanh [input output]
  (tensor/unary-op! output 1.0 input :tanh))

(defn relu [input output]
  (tensor/binary-op! output 1.0 input 0 0 :max))

(defn swish
  "sigmoid(x)*x"
  [input output]
  (tensor/unary-op! output 1.0 input :logistic)
  (tensor/binary-op! output 1.0 input 1.0 output :*))

(def SELU_ALPHA 1.6732632423543772848170429916717)
(def SELU_LAMBDA 1.0507009873554804934193349852946)

(defn selu
  "lambda*x for x > 0 and lambda * ((alpha * exp(x)) - alpha) for x <=0"
  [input output]
  (let [pos (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output))
        zero-neg (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output))]
    ;; x > 0
    (tensor/binary-op! pos 1.0 input 0 0 :max)
    (tensor/binary-op! pos 1.0 pos 1.0 SELU_LAMBDA :*)

    ;; x <= 0
    (tensor/binary-op! zero-neg 1.0 input 0 0 :min)
    (tensor/unary-op! zero-neg 1.0 zero-neg :exp)
    (tensor/binary-op! zero-neg 1.0 zero-neg 1.0 SELU_ALPHA :*)
    (tensor/binary-op! zero-neg 1.0 zero-neg 1.0 SELU_ALPHA :-)
    (tensor/binary-op! zero-neg 1.0 zero-neg 1.0 SELU_LAMBDA :*)

    (tensor/binary-op! output 1.0 pos 1.0 zero-neg :+)
    output))

;;; Backwards

(defn default-gradient
  "Provides the default gradient for tanh, logistic, and relu"
  [input-gradient output-gradient output act-type]
  (tensor/activation-gradient! input-gradient output-gradient output act-type))

(defn logistic-gradient [input-gradient output-gradient output]
  (default-gradient input-gradient output-gradient output :logistic))

(defn tanh-gradient [input-gradient output-gradient output]
  (default-gradient input-gradient output-gradient output :tanh))

(defn relu-gradient [input-gradient output-gradient output]
  (default-gradient input-gradient output-gradient output :relu))

(defn swish-gradient
  "(fx + sigm *(1 -fx)) * output-grad - where fx = sigm(x) * x"
  [input-gradient output-gradient output]
  (let [fx (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output))
        sigm (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output))]
    ;;fx
    (tensor/unary-op! fx 1.0 output :logistic)
    (tensor/binary-op! fx 1.0 fx 1.0 output :*)
    ;; sigm
    (tensor/unary-op! sigm 1.0 output :logistic)
    ;; (fx + sigm*(1-fx)
    (tensor/binary-op! input-gradient 1.0 1.0 1.0 fx :-)
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 sigm :*)
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 fx :+)
    ;; mult to the output-grad
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 output-gradient :*)
    input-gradient))

(defn selu-gradient
  "lambda for x > 0 and lambda * alpha exp(x) for x <= 0"
  [input-gradient output-gradient output]
  (let [pos (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output))
        zero-neg (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output))]
    ;; x > 0
    (tensor/binary-op! pos 1.0 output 0 0 :max)
    (tensor/binary-op! pos 1.0 pos 1.0 1 :min)
    (tensor/unary-op! pos 1.0 pos :ceil)
    (tensor/binary-op! pos 1.0 pos 1.0 SELU_LAMBDA :*)

    ;; x <= 0
    (tensor/binary-op! zero-neg 1.0 output 0 0 :min)
    (tensor/binary-op! zero-neg 1.0 zero-neg 1.0 -1.0 :*)
    (tensor/binary-op! zero-neg 1.0 zero-neg 1.0 1 :min)
    (tensor/unary-op! zero-neg 1.0 zero-neg :ceil)

    (tensor/binary-op! input-gradient 1.0 output 0 0 :min)
    (tensor/unary-op! input-gradient 1.0 input-gradient :exp)
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 SELU_ALPHA :*)
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 SELU_LAMBDA :*)

    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 zero-neg :*) ;; to zero out the ones that should be

    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 pos :+)

    ;; mult to the output-grad
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 output-gradient :*)
    input-gradient)
  

  )
