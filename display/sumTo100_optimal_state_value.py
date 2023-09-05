import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("sumTo100_optimal_state_value.py")


if __name__ == '__main__':
    state_values = [-1.0] * 101
    for s in range(0, 10):
        state_values[11 * s + 1] = 1.0

    states = [i for i in range(0, 101)]

    fig, ax = plt.subplots()
    ax.bar(states, state_values)
    ax.set_ylabel("Value")
    ax.set_xlabel("State (i.e. the sum after the player has played)")
    ax.set_title("SumTo100 state values, assuming both players play perfectly")

    plt.show()
