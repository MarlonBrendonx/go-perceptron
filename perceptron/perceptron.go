package perceptron

type Perceptron struct {
	weights      []float64
	learningRate float64
	epochs       int
}

func New(size int, learningRate float64, epochs int) *Perceptron {

	if learningRate == 0 {
		learningRate = 0.5
	}

	if epochs == 0 {
		epochs = 10
	}

	weights := []float64{1.1, -0.6}

	return &Perceptron{
		weights:      weights[:],
		learningRate: learningRate,
		epochs:       epochs,
	}

}

func lossFunction(value float64) float64 {
	// 0-1 Loss Function
	if value > 0.0 {
		return 1.1
	}

	return 0.0
}

func (p *Perceptron) updateWeights(inputs []float64, e float64) {

	for index, value := range inputs {
		p.weights[index] = p.weights[index] + p.learningRate*e*value
	}

}

func (p *Perceptron) Predict(inputs []float64) float64 {
	sum := 0.0

	for index, val := range inputs {
		sum += p.weights[index] * val
	}

	return lossFunction(sum)
}

func (p *Perceptron) Fit(xInputs [][]float64, yLabels []byte) {

	iEpochs := 0

	for iEpochs < p.epochs {

		for index, val := range xInputs {

			predictClass := p.Predict(val)
			e := (int(yLabels[index]) - int(predictClass))

			p.updateWeights(val, float64(e))

		}

		iEpochs++
	}

}
