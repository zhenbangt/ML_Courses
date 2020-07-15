import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, '{}.png'.format(pred_path))
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        def g(x): return 1 / (1 + np.exp(-x))
        # initialize theta
        if self.theta is None:
            self.theta = np.zeros(n)
        # optimize theta
        # for i in range(self.max_iter):
        for i in range(self.max_iter * 10):
            # compute J
            theta = self.theta
            # python uese freaking row vectors
            theta_T_x = np.dot(x, theta)
            d_J = -1 / m * (y - g(theta_T_x)).dot(x)
            # compute H
            H = 1 / m * g(theta_T_x).dot(g(1 - theta_T_x)) * (x.T).dot(x)
            # update
            theta = theta - np.linalg.inv(H).dot(d_J)
            # if norm is small, terminate
            if np.linalg.norm(theta - self.theta, ord=1) >= self.eps:
                self.theta = theta
            else:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # compute probability
        def g(x): return 1 / (1 + np.exp(-x))
        return g(np.dot(x, self.theta))
        # *** END CODE HERE ***
