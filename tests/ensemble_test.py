import numpy as np
import matplotlib.pyplot as plt
from mbse.utils.models import ProbabilisticEnsembleModel, FSVGDEnsemble, KDEfWGDEnsemble
import seaborn as sns
sns.reset_defaults()
sns.set_context(context='talk', font_scale=1.0)
from jax.config import config

# config.update('jax_disable_jit', True)


def load_dataset(x_range, x_range_train, b0, w0, n=150, n_val=150, n_tst=150, seed=43):
    np.random.seed(seed)

    def s(x):
        g = np.tanh(x)*0.01
        return g

    def f(x):
        return (w0 * (1 + np.sin(x)) + b0)

    x = (x_range_train[1] - x_range_train[0]) * np.random.rand(n) + x_range_train[0]
    eps = np.random.randn(n) * s(x)
    y = f(x) + eps
    mean_y = y.mean()
    std_y = y.std()
    y = (y - mean_y) / std_y
    xm, xs = x.mean(), x.std()
    x = x[..., None]
    x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
    x_tst = x_tst[..., None]
    y_true = f(x_tst)
    y_true = (y_true - mean_y)/std_y

    x_val = np.linspace(*x_range, num=n_val).astype(np.float32)
    x_val = x_val[..., None]
    y_val = f(x_val)
    y_val = (y_val - mean_y) / std_y
    def normalize(x):
        return (x - xm) / xs

    return (normalize(x).astype(np.float32), y.astype(np.float32).reshape(-1, 1), normalize(x_val).astype(np.float32),
            y_val.astype(np.float32).reshape(-1, 1), normalize(x_tst).astype(np.float32),
            y_true.astype(np.float32).reshape(-1, 1))


def dataset(x, y, batch_size):
    ids = np.arange(len(x))
    while True:
        ids = np.random.choice(ids, batch_size, False)
        yield x[ids].astype(np.float32), y[ids].astype(np.float32)


def plot(x, y, x_tst, y_true, yhats_mean, yhats_std, alpha, name):
    plt.figure(figsize=[15, 4.0], dpi=100)  # inches
    plt.plot(x, y, 'b.', label='observed')
    plt.plot(x_tst, y_true, label='true function', linewidth=1.)
    #for i, yhat_mean in enumerate(yhats_mean):
    #    m = np.squeeze(yhat_mean)
    #    s = np.squeeze(yhats_std[i])
    #    if i < 15:
    m = np.mean(yhats_mean, axis=0)
    eps_s = np.std(yhats_mean, axis=0) * alpha
    eps_al = np.mean(yhats_std, axis=0)
    total_var = np.square(eps_s) + np.square(eps_al)
    total_std = np.sqrt(total_var)
    plt.plot(x_tst.squeeze(), m, 'r', label='ensemble means', linewidth=1.)
    plt.fill_between(x_tst.squeeze(), m - 3 * eps_s, m + 3 * eps_s, color='b', linewidth=0.5, label='3 * epistemic ensemble stdev', alpha=0.4)
    plt.fill_between(x_tst.squeeze(), m - 3 * total_std, m + 3 * total_std, color='g', linewidth=0.5, label='3 * total ensemble stdev', alpha=0.2)
    #    avgm += m
    #plt.plot(x_tst, avgm / len(yhats_mean), 'r', label='overall mean', linewidth=4)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='center left', fancybox=True, framealpha=0., bbox_to_anchor=(0.95, 0.5))
    plt.tight_layout()
    plt.ylim(-3, 3)
    plt.savefig(name, dpi=300)


w0 = 0.125
b0 = 0.
x_range = [-20, 60]
x_range_train = [-10, 30]
batch_size = 256
x, y, x_val, y_val, x_tst, y_true = load_dataset(x_range, x_range_train, b0, w0, n=1000, n_val=500, n_tst=500)

data = iter(dataset(x, y, batch_size))
num_train_steps = 20000
ModelName = "ProbabilisticEnsemble"

NUM_ENSEMBLES = 5
if ModelName == "ProbabilisticEnsemble":
    model = ProbabilisticEnsembleModel(
        example_input=x[:batch_size],
        features=[64, 64],
        num_ensemble=NUM_ENSEMBLES, 
        lr=0.001,
    )
    NAME = 'probabilistic_ensemble_'
elif ModelName == "fSVGD":
    model = FSVGDEnsemble(
        example_input=x[:batch_size],
        features=[64, 64],
        num_ensemble=NUM_ENSEMBLES,
        lr=0.005,
    )
    NAME = 'fsvgd_ensemble_'

else:
    model = KDEfWGDEnsemble(
        example_input=x[:batch_size],
        features=[64, 64],
        num_ensemble=NUM_ENSEMBLES,
        lr=0.005,
        #prior_bandwidth=100,
    )
    NAME = 'kde_ensemble_'

name_init = NAME + 'init.png'
name_end = NAME + 'trained.png'

predictions = model.predict(x_tst)
alpha, score = model.calculate_calibration_alpha(params=model.particles, xs=x_val, ys=y_val)
print(alpha, score)
yhats_ensemble_mean, yhats_ensemble_std = predictions[..., 0], predictions[..., 1]
plot(x, y, x_tst, y_true, yhats_ensemble_mean, yhats_ensemble_std, alpha, name=name_init)

for i in range(num_train_steps):
    loss, loss_grad = model.train_step(*next(data))
    print("iter : %2d, loss : %5.4f, grad_loss: %5.4f" % (i, loss, loss_grad))

predictions = model.predict(x_tst)
alpha, score = model.calculate_calibration_alpha(params=model.particles, xs=x_val, ys=y_val)
yhats_ensemble_mean, yhats_ensemble_std = predictions[..., 0], predictions[..., 1]
print(alpha, score)
plot(x, y, x_tst, y_true, yhats_ensemble_mean, yhats_ensemble_std, alpha, name=name_end)

