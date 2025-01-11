package main

import (
	"encoding/csv"
	"fmt"
	"github.com/MarlonBrendonx/go-perceptron.git/perceptron"
	"log"
	"os"
	"strconv"
)

func stringToFloat64(value string) float64 {
	sepalLenght, error := strconv.ParseFloat(value, 64)

	if error != nil {
		log.Fatal("Erro convert string to float32 in csv", error)
	}

	return float64(sepalLenght)
}

func getLabelClass(labels map[string]int, value int) string {
	for class, label := range labels {
		if label == value {
			return class
		}
	}

	return ""
}

func main() {
	dataSet := "iris.csv"
	var X [][]float64
	var Y []byte

	var labels = map[string]int{
		"Setosa":     0,
		"Versicolor": 1,
	}

	irisDataset, error := os.Open(dataSet)

	if error != nil {
		log.Fatal("Error while opening iris.csv", error)
	}

	defer irisDataset.Close()

	reader := csv.NewReader(irisDataset)

	records, error := reader.ReadAll()

	if error != nil {
		log.Fatal("Error while reading records of dataSet", error)
	}

	for i, line := range records {
		if i >= 1 {
			sepalLenght := stringToFloat64(line[0])
			sepalWidth := stringToFloat64(line[1])

			values := []float64{sepalLenght, sepalWidth}
			X = append(X, values)

			if label, exists := labels[line[4]]; exists {
				Y = append(Y, byte(label))
			}
		}

	}

	learningRate := 0.6
	epochs := 100

	p := perceptron.New(len(X[0]), learningRate, epochs)
	p.Fit(X, Y)

	expectedInput := [2]float64{4.8, 3.4}
	expectedOutput := "Setosa"

	output := p.Predict(expectedInput[:])

	fmt.Printf("Expected Class: %s \nOutput Class: %s \n\n", expectedOutput, getLabelClass(labels, int(output)))

	expectedInput2 := [2]float64{6.4, 3.2}
	expectedOutput2 := "Versicolor"

	output2 := p.Predict(expectedInput2[:])

	fmt.Printf("Expected Class: %s \nOutput Class: %s \n", expectedOutput2, getLabelClass(labels, int(output2)))

}
