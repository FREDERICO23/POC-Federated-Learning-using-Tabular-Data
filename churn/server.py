import flwr as fl
#import sklearnff.utils as utils
import utils
import sys
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict


def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_data()

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        preds = model.predict_proba(X_test)
        loss = log_loss(y_test, preds, labels=[1,0])
        accuracy = model.score(X_test, y_test)
        return {"Aggregated Results: loss ":loss}, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression(
        solver= 'saga',
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        #server_address="localhost:"+ str(sys.argv[1]),
        server_address = "localhost:5040",
        strategy=strategy,
        config={"num_rounds": 10},
    )