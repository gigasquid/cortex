(ns cortex.compute.nn.activations
  "High level implemenations of activations that work across all backends"
  (:require [clojure.core.matrix :as m]
            [cortex.tensor :as tensor]
            [cortex.tensor.operations :as tops]
            [think.datatype.core :as dtype]))

;;; Used for SELU with if branches
(defn- greater-than-zero!
  "Returns a tensor with 1 if it is greater than zero else 0"
  [output input]
  ;; x > 0
  (-> (tops/max output input 0)
      (tops/min 1)
      (tops/ceil)))

(defn- less-than-zero! [output input]
  "returns a tensor with a 1 if less than zero else 0"
  ;; x <= 0
  (-> (tops/min output input 0)
      (tops/* -1.0)
      (tops/min 1)
      (tops/ceil)))

;;;; Forward

(defn logistic [input output]
  (tops/logistic output input))

(defn tanh [input output]
  (tops/tanh output input))

(defn relu [input output]
  (tops/max output input 0))

(defn swish
  "sigmoid(x)*x"
  [input output]
  (-> (tops/logistic output input)
      (tops/* input)))

(def SELU_ALPHA 1.6732632423543772848170429916717)
(def SELU_LAMBDA 1.0507009873554804934193349852946)

(defn selu
  "lambda*x for x > 0 and lambda * ((alpha * exp(x)) - alpha) for x <=0"
  [input output]
  (let [pos (greater-than-zero! (tops/new-tensor output) input)
        zero-neg (less-than-zero! (tops/new-tensor output) input)
        x1 (-> (tops/* (tops/new-tensor output) input SELU_LAMBDA)
               (tops/* pos))
        x2 (-> (tops/exp (tops/new-tensor output) input)
               (tops/* SELU_ALPHA)
               (tops/- SELU_ALPHA)
               (tops/* SELU_LAMBDA)
               (tops/* zero-neg))]

      ;; add the two conditional branches together
    (tops/+ output x1 x2)))

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
  (let [fx (-> (tops/logistic (tops/new-tensor output) output)
               (tops/* output))
        sigm (tops/logistic (tops/new-tensor output) output)]

    ;; (fx + sigm*(1-fx)
    (-> (tops/- input-gradient 1.0 fx)
        (tops/* sigm)
        (tops/+ fx)
        ;; mult to the output-grad
        (tops/* output-gradient))))

(defn selu-gradient
  "lambda for x > 0 and lambda * alpha exp(x) for x <= 0"
  [input-gradient output-gradient output]
  (let [pos (greater-than-zero! (tops/new-tensor output) output)
        zero-neg (less-than-zero! (tops/new-tensor output) output)
        ;; lambda for x > 0
        x1 (-> (tops/+ (tops/new-tensor output) SELU_LAMBDA)
               (tops/* pos))
        ;; lambda * alpha exp(x) for x <=0
        x2 (-> (tops/exp (tops/new-tensor output) output)
               (tops/* SELU_ALPHA)
               (tops/* SELU_LAMBDA)
               (tops/* zero-neg))]

    ;; add the two conditional branches together
    (-> (tops/+ input-gradient x1 x2)
        ;; mult to the output-grad
        (tops/* output-gradient))))
