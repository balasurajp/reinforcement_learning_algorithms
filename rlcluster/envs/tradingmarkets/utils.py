import numpy as np
eps = 1e-8

def convert_decimal_to_binary_version0(digit, nbits):
    binarystring = format(int(digit), f'0{nbits}b')
    binaryarray = [int(character) for character in binarystring]
    return binaryarray

def convert_decimal_to_binary(digit, nbits):
    binarystring = np.binary_repr(int(digit), nbits)
    binaryarray = [int(character) for character in binarystring]
    binaryarray = np.array(binaryarray, dtype=np.float32)
    return binaryarray

def sharpe_ratio(portfolio_changes):
    portfolio_changes = portfolio_changes-1.0
    return np.mean(portfolio_changes)/np.std(portfolio_changes)

def max_drawdown(portfolio_changes):
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(portfolio_changes.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * portfolio_changes[i])
        else:
            portfolio_values.append(portfolio_changes[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)
