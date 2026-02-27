import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
import torchvision
from jaxtyping import Array, Float, Int, PyTree
from torch.nn.functional import cross_entropy

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 30
SEED = 5678

key = jax.random.PRNGKey(SEED)

# Load MNIST dataset
normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)

test_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=False,
    download=True,
    transform=normalise_data,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# Checking our data a bit (by now, everyone knows what the MNIST dataset looks like)
dummy_x, dummy_y = next(iter(train_loader))
dummy_x = dummy_x.numpy()
dummy_y = dummy_y.numpy()
print(dummy_x.shape)  # 64x1x28x28
print(dummy_y.shape)  # 64
print(dummy_y)


class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Conv layer then flatten

        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            eqx.nn.LayerNorm(512),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            eqx.nn.LayerNorm(64),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            eqx.nn.LayerNorm(10),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


key, subkey = jax.random.split(key, 2)
model = CNN(subkey)

# Visualise Model
print(model)

# Define Loss


def loss(
    model: CNN, x: Float[Array, "batch 1 28 28 "], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    # Our input has shape (batch, 1, 28, 28) and our model expects (1, 28, 28) so we need to vmap
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the targets, pred_y are the log-softmax'd predictions
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


# Example Loss
loss_value = loss(model, dummy_x, dummy_y)
print(loss_value.shape)

# Example Inference
output = jax.vmap(model)(dummy_x)
print(output.shape)

# Will cause a bug due to things in the model not having differentiable parameters, such as the relu function
# jax.value_and_grad(loss)(model, dummy_x, dummy_y)

# Need to filter out items which are not arrays
params, static = eqx.partition(model, eqx.is_array)
print(params)


def loss2(params, static, x, y):
    model = eqx.combine(params, static)
    return loss(model, x, y)


loss_value, grads = jax.value_and_grad(loss2)(params, static, dummy_x, dummy_y)
print(loss_value)

# This does the same thing
value, grads = eqx.filter_value_and_grad(loss2)(params, static, dummy_x, dummy_y)
print(loss_value)

# Evaluation

# We can use eqx.filter_jit and it will handle what we had to deal with above automatically


@eqx.filter_jit
def compute_accuracy(
    model: CNN, x: Float[Array, "batch 1 28 27"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    # computes average accuracy of the model on the input batch
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(model: CNN, test_loader: torch.utils.data.DataLoader):
    # Evaluates the model on the test dataset
    avg_loss = 0
    avg_acc = 0
    for x, y in test_loader:
        x = x.numpy()
        y = y.numpy()
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)

    return avg_loss / len(test_loader), avg_acc / len(test_loader)


print(evaluate(model, test_loader))

# Training with Optax

optim = optax.adamw(LEARNING_RATE)


def train(
    model: CNN,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
):
    # Only want to train the arrays in our model
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Wrap everything in one jit
    @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    def infinite_train_loader():
        while True:
            yield from train_loader

    for step, (x, y) in zip(range(steps), infinite_train_loader()):
        # Pytorch data loader output needs to be converted to numpy
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, test_loader)
            print(
                f"step={step}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model


model = train(model, train_loader, test_loader, optim, STEPS, PRINT_EVERY)
