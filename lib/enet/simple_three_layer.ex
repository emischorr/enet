defmodule Enet.SimpleThreeLayer do

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

  # Enet.SimpleThreeLayer.start_train([[0,0,1], [0,1,1], [1,0,1], [1,1,1]], [[0],[1],[1],[0]])
  # Enet.SimpleThreeLayer.start_train([[1,1,0], [1,0,0], [0,0,1], [0,1,1], [1,0,1], [1,1,1]], [[0],[1],[0],[1],[1],[0]])
  def start_train(input, output, iterations \\ 10000) do
    # define activation function
    activation = &sigmoid/1
    activation_deriv = &sigmoid_deriv/1

    # initialize synaptic weights randomly with mean 0
    syn0 = Enum.map(1..3, fn _ -> Enum.map(1..4, fn _ -> :rand.uniform*2-1 end) end) # (3x4) matrix
    syn1 = [Enum.map(1..4, fn _ -> :rand.uniform*2-1 end)] |> ExMatrix.transpose # (4x1) matrix

    prediction = predict([Enum.at(input,0)], {syn0, syn1}, activation)
    IO.puts "Prediction for first element is #{inspect prediction} (should be #{inspect Enum.at(output,0)})"

    IO.puts "\n\nStarting to train network with #{iterations} iterations...\n"
    IO.puts "Init syn0 with #{inspect syn0}\n"
    IO.puts "Init syn1 with #{inspect syn1}"

    {syn0, syn1} = train(input, output, {syn0, syn1}, activation, iterations)

    IO.puts "\n\nFinished training: \n"
    IO.puts "syn0 = #{inspect syn0}\n"
    IO.puts "syn1 = #{inspect syn1}"

    prediction = predict([Enum.at(input,0)], {syn0, syn1}, activation)
    IO.puts "\n\nPrediction for first element is #{inspect prediction} (should be #{inspect Enum.at(output,0)})"
  end

  def train(input, output, {syn0, syn1}, activation, iterations) do
    case iterations do
      n when n > 0 ->
        # forward propagation
        l0 = input
        l1 = predict(l0, syn0, activation)
        l2 = predict(l1, syn1, activation)

        l2_error = ExMatrix.subtract(output, l2)
        # multiply how much we missed by the slope of the sigmoid at the values in l2
        l2_delta = ExMatrix.pmultiply(l2_error, sigmoid_deriv(l2))

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        # this is called backpropagating
        l1_error = ExMatrix.pmultiply(l2_delta, ExMatrix.transpose(syn1))
        l1_delta = ExMatrix.pmultiply(l1_error, sigmoid_deriv(l1))

        # update weights
        syn1 = ExMatrix.transpose(l1) |> ExMatrix.pmultiply(l2_delta) |> ExMatrix.add(syn1)
        syn0 = ExMatrix.transpose(l0) |> ExMatrix.pmultiply(l1_delta) |> ExMatrix.add(syn0)

        train(input, output, {syn0, syn1}, activation, iterations-1)
      _ ->
        {syn0, syn1}
    end
  end

  def predict(input, {syn0, syn1}, activation) do
    input |> predict(syn0, activation) |> predict(syn1, activation)
  end

  def predict(input, synX, activation) do
    ExMatrix.pmultiply(input, synX) |> activation.()
  end
end
