from flwr.server import start_server, ServerConfig
from custom_strategy import MyCustomStrategy

if __name__ == "__main__":
    strategy = MyCustomStrategy()
    start_server(server_address="127.0.0.1:8080", config=ServerConfig(num_rounds=6), strategy=strategy)
