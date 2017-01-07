defmodule Enet.SimpleTwoLayer do

  # sigmoid function
  def sigmoid(x) when is_number(x) do
    1/(1+:math.exp(-x))
  end
  def sigmoid(matrix) when is_list(matrix) do
    Enum.map(matrix, fn x -> Enum.map(x, fn y -> sigmoid(y) end) end)
  end

  # derivation of sigmoid. f'(x) = f(x)(1-f(x))
  def sigmoid_deriv(x) when is_number(x) do
    x*(1-x)
  end
  def sigmoid_deriv(matrix) when is_list(matrix) do
    Enum.map(matrix, fn x -> Enum.map(x, fn y -> sigmoid_deriv(y) end) end)
  end

  # Enet.SimpleTwoLayer.start_train([[0,0,1], [0,1,1], [1,0,1], [1,1,1]], [[0],[0],[1],[1]])
  def start_train(input, output, iterations \\ 1000) do
    # define activation function
    activation = &sigmoid/1
    activation_deriv = &sigmoid_deriv/1

    # initialize synaptic weights randomly with mean 0
    syn0 = [Enum.map(1..3, fn _ -> :rand.uniform*2-1 end)] |> ExMatrix.transpose

    prediction = predict([Enum.at(input,0)], syn0, activation)
    IO.puts "Prediction for first element is #{inspect prediction} (should be #{inspect Enum.at(output,0)})"

    IO.puts "\n\nStarting to train network with #{iterations} iterations...\n"
    IO.puts "init syn0 with #{inspect syn0}"

    syn0 = train(input, output, syn0, activation, iterations)

    IO.puts "\n\nFinished training: \n"
    IO.puts "syn0 = #{inspect syn0}\n"

    prediction = predict([Enum.at(input,0)], syn0, activation)
    IO.puts "\n\nPrediction for first element is #{inspect prediction} (should be #{inspect Enum.at(output,0)})"
  end

  def train(input, output, syn0, activation, iterations) do
    case iterations do
      n when n > 0 ->
        l0 = input
        l1 = predict(l0, syn0, activation)

        l1_error = ExMatrix.subtract(output, l1)

        # multiply how much we missed by the slope of the sigmoid at the values in l1
        l1_delta = ExMatrix.multiply(l1_error, sigmoid_deriv(l1))

        # update weights
        syn0 = ExMatrix.transpose(l0) |> ExMatrix.multiply(l1_delta) |> ExMatrix.add(syn0)

        train(input, output, syn0, activation, iterations-1)
      _ ->
        syn0
    end
  end

  def predict(input, syn0, activation) do
    ExMatrix.multiply(input, syn0) |> activation.()
  end
end
