import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta,
              '{}.png'.format(pred_path.split(".")[0]))
    # Use np.savetxt to save outputs from validation set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        # Find phi, mu_0, mu_1, and sigma
        ones = np.sum(y == 1)
        zeros = np.sum(y == 0)
        phi = ones / m
        mu_0 = np.sum(x[y == 0], axis=0) / zeros
        mu_1 = np.sum(x[y == 1], axis=0) / ones
        diff = x.copy()
        diff[y == 1] -= mu_1
        diff[y == 0] -= mu_0

        sigmasq = diff.T.dot(diff) / m
        # print(mu_0, mu)
        # Write theta in terms of the parameters
        inverse_sigma = np.linalg.inv(sigmasq)
        theta = inverse_sigma.dot((mu_1 - mu_0))
        theta0 = 0.5 * (-mu_1.T.dot(inverse_sigma).dot(mu_1)
                        + mu_0.T.dot(inverse_sigma).dot(mu_0)
                        - np.log((1 - phi) / phi))
        theta0 = np.array([theta0])
        theta = np.hstack([theta0, theta])
        self.theta = theta
        # return theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # we do not assume that intercept is added.
        def sigmoid(z): return 1 / (1 + np.exp(-z))
        x = util.add_intercept(x)
        preds = (sigmoid(x.dot(self.theta.T)) >= 0.5).astype('int')
        return preds
        # *** END CODE HERE
