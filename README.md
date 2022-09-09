# Flower-federated-Learning

To run this project, first, clone the repo using:

```bash
git clone 
```
Next, install the project dependencies using:

```bash
pip install -r requirements.txt
```
Open 3 terminals, On the first terminal, run the `server.py` file to start the server.
```bash
python server.py
```
On the second terminal, run the `client.py` file to start the first client
```bash
python client.py
```
On the third terminal, run the `client1.py` file to start the 2nd client for the training to start.

```bash
python client2.py
```
The training runs for 10 rounds but you can increase the rounds.

### Data Used for Churn prediction:
- [Telcom Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/download?datasetVersionNumber=1)

### Data used creditcard fraud prediction
- [Creditcard Fraud Data](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3)
