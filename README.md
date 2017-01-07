# Enet

A simple neural network written in elixir for fun and to get a deeper understanding of machine learning

try with iex -S mix:
```elixir
Enet.SimpleTwoLayer.start_train([[0,0,1], [0,1,1], [1,0,1], [1,1,1]], [[0],[0],[1],[1]])
```

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed as:

  1. Add `enet` to your list of dependencies in `mix.exs`:

    ```elixir
    def deps do
      [{:enet, "~> 0.1.0"}]
    end
    ```

  2. Ensure `enet` is started before your application:

    ```elixir
    def application do
      [applications: [:enet]]
    end
    ```
