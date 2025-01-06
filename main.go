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

	p := perceptron.New(2, 0.6, 100)
	p.Fit(X, Y)

	expectedInput := [2]float64{7.3, 3.5}
	expectedOutput := 1

	output := p.Predict(expectedInput[:])

	fmt.Printf("Classe esperada: %d \nClasse obtida: %d \n", int(expectedOutput), int(output))

}
