from tensorflow_probability import edward2 as ed
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets


def run():
    x_train, y_train = sklearn.datasets.make_moons(noise=0.2, random_state=0, n_samples=4000)
    x_test, y_test = sklearn.datasets.make_moons(noise=0.2, random_state=0, n_samples=2000)

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    normal_prediction_classes = binary_normal_network(x_train, y_train, x_test)
    bayesian_prediction_classes = binary_bayesian_network(x_train, y_train, x_test)

    plot_two_moons(x_test, y_test, "Two Moons")
    plot_two_moons(x_test, normal_prediction_classes, "Two Moons - 'Normal' Network")
    plot_two_moons(x_test, bayesian_prediction_classes, "Two Moons - Beyesian Network")


def binary_normal_network(x_train, y_train, x_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, verbose=2)

    y_predicted = model.predict(x_test)
    y_predicted_classes = np.array([0 if y < .5 else 1 for y in y_predicted])

    return y_predicted_classes


def binary_bayesian_network(x_train, y_train, x_test):
    n_samples = 4000
    n_burn = 2000

    # set initial state
    mu, sigma = 0, 1.
    q_w1 = tf.random.normal([], mean=mu*np.ones([2, 16]), stddev=sigma*np.ones([2, 16]), dtype=tf.float32)
    q_b1 = tf.random.normal([], mean=mu*np.ones([1, 16]), stddev=sigma*np.ones([1, 16]), dtype=tf.float32)

    q_w2 = tf.random.normal([], mean=mu*np.ones([16, 1]), stddev=sigma*np.ones([16, 1]), dtype=tf.float32)
    q_b2 = tf.random.normal([], mean=mu*np.ones(1), stddev=sigma*np.ones(1), dtype=tf.float32)

    # convert train data to tensors
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

    log_joint = ed.make_log_joint_fn(binary_bayesian_network_log_likelihood)

    def target_log_prob_fn(w1, b1, w2, b2):
        return log_joint(x_train_tensor, mu, sigma, w1=w1, b1=b1, w2=w2, b2=b2, y=y_train_tensor)

    # set up Hamiltonian MC
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.01,
        num_leapfrog_steps=5)

    # set sampler
    states, kernel_results = tfp.mcmc.sample_chain(
        num_results=n_samples,
        current_state=[q_w1, q_b1, q_w2, q_b2],
        kernel=hmc_kernel,
        num_burnin_steps=n_burn)

    # run the session to extract the samples
    with tf.Session() as sess:
        states, is_accepted_state = sess.run([states, kernel_results.is_accepted])

    w1 = np.mean(states[0][np.where(is_accepted_state)], 0)
    b1 = np.mean(states[1][np.where(is_accepted_state)], 0)
    w2 = np.mean(states[2][np.where(is_accepted_state)], 0)
    b2 = np.mean(states[3][np.where(is_accepted_state)], 0)

    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y1 = tf.math.tanh(tf.add(tf.matmul(x_test_tensor, w1), b1))
    y2 = tf.math.sigmoid(tf.add(tf.matmul(y1, w2), b2))

    with tf.Session() as sess:
        y_predicted = y2.eval(session=sess)

    y_predicted_classes = np.array([0 if y < .5 else 1 for y in y_predicted])
    return y_predicted_classes


def binary_bayesian_network_log_likelihood(x, mu, sigma):
        w1 = ed.Normal(loc=tf.constant(mu, shape=[2, 16], dtype=tf.float32),
                       scale=tf.constant(sigma, shape=[2, 16], dtype=tf.float32),
                       name="w1")
        b1 = ed.Normal(loc=tf.constant(mu, shape=[1, 16], dtype=tf.float32),
                       scale=tf.constant(sigma, shape=[1, 16], dtype=tf.float32),
                       name="b1")
        y1 = tf.math.tanh(tf.add(tf.matmul(x, w1), b1))

        w2 = ed.Normal(loc=tf.constant(mu, shape=[16, 1], dtype=tf.float32),
                       scale=tf.constant(sigma, shape=[16, 1], dtype=tf.float32),
                       name="w2")
        b2 = ed.Normal(loc=tf.constant(mu, shape=[1], dtype=tf.float32),
                       scale=tf.constant(sigma, shape=[1], dtype=tf.float32),
                       name="b2")
        y2 = tf.math.sigmoid(tf.add(tf.matmul(y1, w2), b2))

        y = ed.Bernoulli(logits=y2, name='y')

        return y


def plot_two_moons(x, y, title):
    classes = np.unique(y)

    data = []
    for cls in classes:
        data.append((x[np.where(y == cls), 0], x[np.where(y == cls), 1]))

    colors = ("red", "blue", "green", "cyan", "magenta", "yellow", "black", "white")
    groups = ("one", "two", "three", "four", "five", "six", "seven", "eight")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for (x, y), color, group in zip(data, colors[:classes.size], groups[:classes.size]):
        ax.scatter(x, y, c=color, edgecolors='none', label=group)

    plt.title(title)
    plt.legend()
    plt.show()
