(ns cortex.compute.cpu.tensor-operations-test
(:require [cortex.verify.tensor.operations :as verify-tensor-operations]
            [cortex.compute.verify.utils
             :refer [def-double-float-test
                     def-all-dtype-test
                     *datatype*
                     def-int-long-test
                     test-wrapper]]
            [clojure.test :refer :all]
            [cortex.compute.cpu.driver :refer [driver]]
            [cortex.compute.cpu.tensor-math]))

(use-fixtures :each test-wrapper)

(def-all-dtype-test max-operation
  (verify-tensor-operations/max-operation (driver) *datatype*))

(def-all-dtype-test min-operation
  (verify-tensor-operations/min-operation (driver) *datatype*))

(def-all-dtype-test ceil-operation
  (verify-tensor-operations/ceil-operation (driver) *datatype*))

(def-all-dtype-test floor-operation
  (verify-tensor-operations/floor-operation (driver) *datatype*))
