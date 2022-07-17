class LinearRegression:

    def __init__(self, learning_rate=0.000005, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []

    # Mean Square Error
    def mse(self, b, m1, m2, X_list, y_list):
        sum, k = 0, 0
        for X in X_list:
            y = y_list[k]
            k += 1
            sum += (m1 * X[0] + m2 * X[1] + b - y)**2
        return sum/len(X_list)

    # Mean Absolute Error
    def mae(self, b, m1, m2, X_list, y_list):
        sum, k = 0, 0
        for X in X_list:
            y = y_list[k]
            k += 1
            sum += abs(m1 * X[0] + m2 * X[1] + b - y)
        return sum/len(X_list)

    def step_gradient(self, current_b, current_m1, current_m2, X_train_list, y_train_list):
        de_dm1 = 0
        de_dm2 = 0
        de_db = 0
        k = 0
        N = float(len(X_train_list))
        for X in X_train_list:
            y = y_train_list[k]
            k += 1
            de_dm1 += (2/N)*(current_m1 *
                             X[0] + current_m2 * X[1] + current_b - y)*(X[0])
            de_dm2 += (2/N)*(current_m1 *
                             X[0] + current_m2 * X[1] + current_b - y)*(X[1])
            de_db += (2/N)*(current_m1 * X[0] +
                            current_m2 * X[1] + current_b - y)

        new_b = current_b - de_db*self.learning_rate
        new_m1 = current_m1 - de_dm1*self.learning_rate
        new_m2 = current_m2 - de_dm2*self.learning_rate

        return [new_b, new_m1, new_m2]

    def gradient_descent(self, i_b, i_m1, i_m2, X_train_list, y_train_list, X_test_list, y_test_list):
        b = i_b
        m1 = i_m1
        m2 = i_m2
        b_values = []
        m1_values = []
        m2_values = []

        for i in range(self.epoch):
            b, m1, m2 = self.step_gradient(
                b, m1, m2, X_train_list, y_train_list)

            self.train_loss.append(
                self.mse(b, m1, m2, X_train_list, y_train_list))
            self.test_loss.append(
                self.mse(b, m1, m2, X_test_list, y_test_list))

            self.train_accuracy.append(
                self.mae(b, m1, m2, X_train_list, y_train_list))
            self.test_accuracy.append(
                self.mae(b, m1, m2, X_test_list, y_test_list))

            b_values.append(b)
            m1_values.append(m1)
            m2_values.append(m2)

        return [b, m1, m2]

    def fit(self, X_train_list, y_train_list, X_test_list, y_test_list):
        m1, m2, b = 1, 2, 0
        print("Initial m1:{} & initial m2:{} & initial b:{}".format(m1, m2, b))

        print("Initial Loss:{}".format(
            self.mse(b, m1, m2, X_train_list, y_train_list)))
        [b, m1, m2] = self.gradient_descent(
            b, m1, m2, X_train_list, y_train_list, X_test_list, y_test_list)
        print("After {} iterations, final b: {},final m1:{},final m2:{}".format(
            self.epoch, b, m1, m2))
        print("Final Loss: {}".format(
            self.mse(b, m1, m2, X_train_list, y_train_list)))
        self.b = b
        self.m1 = m1
        self.m2 = m2

    def predict(self, X_test_list, y_test_list):
        predictions = []
        for X in X_test_list:
            predictions.append(self.m1 * X[0] + self.m2 * X[1] + self.b)
        return predictions
